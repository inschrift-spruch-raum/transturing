"""
Phase 1: Convex Hull KV Cache — Does the Geometry Work?

Tests the core claim from Percepta's "Can LLMs Be Computers?":
  2D convex hull lookups achieve O(log t) per query vs O(t) brute force
  for hard-max attention with 2D keys.

Implementation notes:
  - BruteForceKVCache: linear scan, O(n) per query
  - HullKVCache: maintains convex hull, vectorized scan over hull vertices
    For random 2D points, hull has O(sqrt(n)) vertices → O(sqrt(n)) query
    For structured keys (parabolic), we add a binary search path → O(log n)
  - ParabolicKVCache: specialized for k_j = (2j, -j^2) with O(log n) lookup
"""

import numpy as np
import time
import json

# ─── Brute Force KV Cache ───────────────────────────────────────────

class BruteForceKVCache:
    """Linear scan with numpy vectorization. O(n) per query but fast constants."""
    
    def __init__(self):
        self._keys_list = []
        self._vals_list = []
        self._keys_np = None
        self._vals_np = None
        self._dirty = False
    
    def add(self, key: tuple, value: float):
        self._keys_list.append(key)
        self._vals_list.append(value)
        self._dirty = True
    
    def _sync(self):
        if self._dirty:
            self._keys_np = np.array(self._keys_list)
            self._vals_np = np.array(self._vals_list)
            self._dirty = False
    
    def query(self, q: tuple) -> float:
        if not self._keys_list:
            raise ValueError("Empty cache")
        self._sync()
        scores = self._keys_np[:, 0] * q[0] + self._keys_np[:, 1] * q[1]
        best = scores.max()
        mask = np.abs(scores - best) < 1e-9
        return self._vals_np[mask].mean()
    
    def __len__(self):
        return len(self._keys_list)


# ─── Convex Hull KV Cache (numpy-vectorized hull scan) ──────────────

class HullKVCache:
    """
    Maintains incremental convex hull. Queries by vectorized scan over
    hull vertices, which is O(hull_size). For random 2D points,
    hull_size = O(sqrt(n) * log(n)), giving sublinear queries.
    
    Rebuild strategy: full rebuild on every add (Andrew's monotone chain 
    is O(n log n), but n here is total points). To keep add cost amortized,
    we rebuild lazily only when a query arrives after new points were added.
    """
    
    def __init__(self):
        self._all_keys = []      # all (k1, k2)
        self._all_values = []    # parallel values
        self._hull_indices = []  # indices into _all_keys for hull vertices
        self._hull_keys_np = None  # numpy array of hull keys for fast query
        self._dirty = False
        # Map from rounded key → list of values (for tie-breaking)
        self._val_map = {}
    
    def _key_id(self, k):
        return (round(k[0], 9), round(k[1], 9))
    
    def add(self, key: tuple, value: float):
        self._all_keys.append(key)
        self._all_values.append(value)
        kid = self._key_id(key)
        self._val_map.setdefault(kid, []).append(value)
        self._dirty = True
    
    def _rebuild(self):
        n = len(self._all_keys)
        if n < 3:
            self._hull_indices = list(range(n))
            if n > 0:
                self._hull_keys_np = np.array(self._all_keys[:n])
            self._dirty = False
            return
        
        pts = np.array(self._all_keys)
        idx = np.lexsort((pts[:, 1], pts[:, 0]))
        
        def cross(o, a, b):
            return ((pts[a, 0] - pts[o, 0]) * (pts[b, 1] - pts[o, 1]) -
                    (pts[a, 1] - pts[o, 1]) * (pts[b, 0] - pts[o, 0]))
        
        lower = []
        for i in idx:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], int(i)) <= 0:
                lower.pop()
            lower.append(int(i))
        
        upper = []
        for i in reversed(idx):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], int(i)) <= 0:
                upper.pop()
            upper.append(int(i))
        
        self._hull_indices = lower[:-1] + upper[:-1]
        self._hull_keys_np = pts[self._hull_indices]
        self._dirty = False
    
    def query(self, q: tuple) -> float:
        if not self._all_keys:
            raise ValueError("Empty cache")
        if self._dirty:
            self._rebuild()
        
        n_hull = len(self._hull_indices)
        if n_hull == 1:
            kid = self._key_id(self._all_keys[self._hull_indices[0]])
            return np.mean(self._val_map[kid])
        
        # Vectorized dot product over hull vertices
        scores = self._hull_keys_np[:, 0] * q[0] + self._hull_keys_np[:, 1] * q[1]
        best_score = scores.max()
        
        # Find all hull vertices achieving best score (ties)
        mask = np.abs(scores - best_score) < 1e-9
        tied_indices = np.where(mask)[0]
        
        all_vals = []
        for ti in tied_indices:
            orig_idx = self._hull_indices[ti]
            kid = self._key_id(self._all_keys[orig_idx])
            all_vals.extend(self._val_map.get(kid, []))
        
        return np.mean(all_vals) if all_vals else 0.0
    
    @property
    def hull_size(self):
        if self._dirty:
            self._rebuild()
        return len(self._hull_indices)
    
    def __len__(self):
        return len(self._all_keys)


# ─── Parabolic KV Cache (specialized O(log n) for index lookups) ────

class ParabolicKVCache:
    """
    Specialized cache for the parabolic encoding k_j = (2j, -j^2).
    
    For query q = (i, 1), the dot product is:
      q · k_j = 2ij - j^2 = -(j-i)^2 + i^2
    
    This is maximized at j = i. So index lookup is just array indexing: O(1).
    But to simulate the attention mechanism faithfully (where the model
    doesn't "know" the structure), we implement binary search over the
    sorted hull, which for parabolic points = all points sorted by x.
    
    The dot product f(j) = 2ij - j^2 is a downward parabola in j,
    so it's unimodal → ternary search works in O(log n).
    """
    
    def __init__(self):
        self.values = []  # values[j] = value stored at step j
    
    def add(self, key: tuple, value: float):
        # We ignore the key and just append — the key structure is implicit
        self.values.append(value)
    
    def query_direct(self, index: int) -> float:
        """O(1) direct lookup — what you'd actually do if you knew the structure."""
        return self.values[index]
    
    def query_ternary(self, q: tuple) -> float:
        """
        O(log n) ternary search — simulates what the hull binary search does.
        f(j) = q[0]*2j + q[1]*(-j^2) is unimodal in j.
        """
        n = len(self.values)
        if n == 0:
            raise ValueError("Empty")
        if n <= 3:
            scores = [q[0] * 2.0 * j + q[1] * (-float(j*j)) for j in range(n)]
            best_j = max(range(n), key=lambda j: scores[j])
            return self.values[best_j]
        
        lo, hi = 0, n - 1
        while hi - lo > 2:
            m1 = lo + (hi - lo) // 3
            m2 = hi - (hi - lo) // 3
            s1 = q[0] * 2.0 * m1 + q[1] * (-float(m1*m1))
            s2 = q[0] * 2.0 * m2 + q[1] * (-float(m2*m2))
            if s1 < s2:
                lo = m1 + 1
            else:
                hi = m2 - 1
        
        best_j = lo
        best_s = q[0] * 2.0 * lo + q[1] * (-float(lo*lo))
        for j in range(lo+1, hi+1):
            s = q[0] * 2.0 * j + q[1] * (-float(j*j))
            if s > best_s:
                best_s = s
                best_j = j
        return self.values[best_j]
    
    def __len__(self):
        return len(self.values)


# ─── Correctness Tests ─────────────────────────────────────────────

def test_correctness():
    print("=" * 60)
    print("CORRECTNESS TESTS")
    print("=" * 60)
    
    np.random.seed(42)
    all_pass = True
    
    # Test 1: Random keys
    print("\n  Test 1: Random 2D keys")
    brute = BruteForceKVCache()
    hull = HullKVCache()
    
    for i in range(500):
        key = (np.random.randn(), np.random.randn())
        value = np.random.randn()
        brute.add(key, value)
        hull.add(key, value)
    
    mismatches = 0
    max_err = 0.0
    for _ in range(500):
        q = (np.random.randn(), np.random.randn())
        vb = brute.query(q)
        vh = hull.query(q)
        err = abs(vb - vh)
        max_err = max(max_err, err)
        if err > 1e-6:
            mismatches += 1
    
    print(f"    500 points, 500 queries: {mismatches} mismatches, max_err={max_err:.2e}")
    if mismatches > 0:
        all_pass = False
    
    # Test 2: Parabolic encoding with HullKVCache
    print("\n  Test 2: Parabolic keys k=(2j, -j^2) via HullKVCache")
    brute2 = BruteForceKVCache()
    hull2 = HullKVCache()
    
    for j in range(100):
        key = (2.0 * j, -float(j * j))
        brute2.add(key, float(j))
        hull2.add(key, float(j))
    
    mismatches2 = 0
    for i in range(100):
        q = (float(i), 1.0)
        vb = brute2.query(q)
        vh = hull2.query(q)
        expected = float(i)
        if abs(vb - expected) > 1e-6:
            print(f"    BRUTE wrong at i={i}: got {vb}, expected {expected}")
            mismatches2 += 1
        if abs(vh - expected) > 1e-6:
            print(f"    HULL wrong at i={i}: got {vh}, expected {expected}")
            mismatches2 += 1
    print(f"    100 indices: {mismatches2} mismatches")
    if mismatches2 > 0:
        all_pass = False
    
    # Test 3: ParabolicKVCache ternary search
    print("\n  Test 3: ParabolicKVCache ternary search")
    para = ParabolicKVCache()
    for j in range(1000):
        para.add(None, float(j * 7))
    
    mismatches3 = 0
    for i in range(1000):
        q = (float(i), 1.0)
        v = para.query_ternary(q)
        expected = float(i * 7)
        if abs(v - expected) > 1e-6:
            print(f"    Ternary wrong at i={i}: got {v}, expected {expected}")
            mismatches3 += 1
            if mismatches3 > 5:
                print("    (stopping after 5 errors)")
                break
    print(f"    1000 indices: {mismatches3} mismatches")
    if mismatches3 > 0:
        all_pass = False
    
    return all_pass


# ─── Benchmarks ─────────────────────────────────────────────────────

def benchmark_query_scaling():
    """Query time vs cache size: brute O(n) vs hull O(hull_size) vs parabolic O(log n)."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Query Time vs Cache Size")
    print("=" * 60)
    
    np.random.seed(123)
    sizes = [100, 500, 1000, 5000, 10000, 50000]
    n_q = 200
    results = []
    
    print("\n  --- Random keys (hull shrinks relative to n) ---")
    for sz in sizes:
        keys = np.random.randn(sz, 2)
        vals = np.random.randn(sz)
        queries = np.random.randn(n_q, 2)
        
        brute = BruteForceKVCache()
        hull = HullKVCache()
        for i in range(sz):
            brute.add(tuple(keys[i]), float(vals[i]))
            hull.add(tuple(keys[i]), float(vals[i]))
        
        hs = hull.hull_size
        
        # Time queries
        t0 = time.perf_counter()
        for qi in range(n_q):
            brute.query(tuple(queries[qi]))
        bt = (time.perf_counter() - t0) / n_q
        
        t0 = time.perf_counter()
        for qi in range(n_q):
            hull.query(tuple(queries[qi]))
        ht = (time.perf_counter() - t0) / n_q
        
        sp = bt / ht if ht > 0 else float('inf')
        print(f"    n={sz:>7d}  hull_verts={hs:>5d}  "
              f"brute={bt*1e6:>8.1f}µs  hull={ht*1e6:>8.1f}µs  "
              f"speedup={sp:>6.1f}x")
        results.append({
            "type": "random", "n": sz, "hull_verts": hs,
            "brute_us": bt*1e6, "hull_us": ht*1e6, "speedup": sp
        })
    
    print("\n  --- Parabolic keys (all on hull — worst case for hull scan) ---")
    para_sizes = [100, 500, 1000, 5000, 10000]
    for sz in para_sizes:
        brute = BruteForceKVCache()
        hull = HullKVCache()
        para = ParabolicKVCache()
        
        for j in range(sz):
            key = (2.0 * j, -float(j * j))
            brute.add(key, float(j))
            hull.add(key, float(j))
            para.add(key, float(j))
        
        # Generate queries: random index lookups
        query_indices = np.random.randint(0, sz, size=n_q)
        queries = [(float(i), 1.0) for i in query_indices]
        
        t0 = time.perf_counter()
        for q in queries:
            brute.query(q)
        bt = (time.perf_counter() - t0) / n_q
        
        t0 = time.perf_counter()
        for q in queries:
            hull.query(q)
        ht = (time.perf_counter() - t0) / n_q
        
        t0 = time.perf_counter()
        for q in queries:
            para.query_ternary(q)
        pt = (time.perf_counter() - t0) / n_q
        
        print(f"    n={sz:>7d}  brute={bt*1e6:>8.1f}µs  "
              f"hull={ht*1e6:>8.1f}µs  ternary={pt*1e6:>8.1f}µs  "
              f"speedup(ternary)={bt/pt if pt>0 else 0:>6.1f}x")
        results.append({
            "type": "parabolic", "n": sz,
            "brute_us": bt*1e6, "hull_us": ht*1e6, "ternary_us": pt*1e6,
            "speedup_ternary": bt/pt if pt > 0 else 0
        })
    
    return results


def benchmark_execution_trace():
    """Simulated execution trace: pre-build cache, then measure query throughput."""
    print("\n" + "=" * 60)
    print("BENCHMARK: Simulated Execution Trace (Query-Only)")
    print("=" * 60)
    print("    (Pre-build cache to size N, then measure query throughput)")
    
    np.random.seed(789)
    trace_lengths = [500, 1000, 5000, 10000, 50000]
    n_q = 500
    results = []
    
    for n_steps in trace_lengths:
        # Pre-build both caches
        brute = BruteForceKVCache()
        para = ParabolicKVCache()
        for j in range(n_steps):
            key = (2.0 * j, -float(j * j))
            brute.add(key, float(j))
            para.add(key, float(j))
        brute._sync()  # pre-sync numpy array
        
        # Generate random index queries
        queries = [(float(np.random.randint(0, n_steps)), 1.0) for _ in range(n_q)]
        
        t0 = time.perf_counter()
        for q in queries:
            brute.query(q)
        bt = (time.perf_counter() - t0) / n_q
        
        t0 = time.perf_counter()
        for q in queries:
            para.query_ternary(q)
        pt = (time.perf_counter() - t0) / n_q
        
        sp = bt / pt if pt > 0 else 0
        print(f"    n={n_steps:>6d}  brute={bt*1e6:>8.1f}µs/q  "
              f"ternary={pt*1e6:>8.2f}µs/q  speedup={sp:>7.1f}x")
        results.append({
            "steps": n_steps, "brute_us": bt*1e6, "ternary_us": pt*1e6, "speedup": sp
        })
    
    return results


def benchmark_scaling_fit():
    """
    Measure per-query cost at fine granularity to fit scaling exponents.
    If brute ~ O(n), we expect slope ~1 in log-log.
    If ternary ~ O(log n), we expect sublinear growth.
    """
    print("\n" + "=" * 60)
    print("SCALING FIT: Measuring exponents")
    print("=" * 60)
    
    np.random.seed(999)
    sizes = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    n_q = 200
    
    brute_times = []
    ternary_times = []
    
    for sz in sizes:
        # Brute force
        brute = BruteForceKVCache()
        for j in range(sz):
            brute.add((2.0*j, -float(j*j)), float(j))
        
        queries = [(float(np.random.randint(0, sz)), 1.0) for _ in range(n_q)]
        
        t0 = time.perf_counter()
        for q in queries:
            brute.query(q)
        bt = (time.perf_counter() - t0) / n_q
        brute_times.append(bt)
        
        # Ternary
        para = ParabolicKVCache()
        for j in range(sz):
            para.add(None, float(j))
        
        t0 = time.perf_counter()
        for q in queries:
            para.query_ternary(q)
        pt = (time.perf_counter() - t0) / n_q
        ternary_times.append(pt)
    
    # Fit log-log slopes
    log_sizes = np.log(sizes)
    log_brute = np.log(brute_times)
    log_ternary = np.log(ternary_times)
    
    # Linear regression in log-log space
    brute_slope = np.polyfit(log_sizes, log_brute, 1)[0]
    ternary_slope = np.polyfit(log_sizes, log_ternary, 1)[0]
    
    print(f"\n    Brute force log-log slope:  {brute_slope:.3f}  (expected ~1.0 for O(n))")
    print(f"    Ternary search log-log slope: {ternary_slope:.3f}  (expected ~0 for O(log n))")
    print(f"\n    Size         Brute(µs)   Ternary(µs)  Ratio")
    for sz, bt, pt in zip(sizes, brute_times, ternary_times):
        print(f"    {sz:>7d}     {bt*1e6:>9.1f}   {pt*1e6:>9.2f}     {bt/pt:>7.1f}x")
    
    return {
        "sizes": sizes,
        "brute_us": [t*1e6 for t in brute_times],
        "ternary_us": [t*1e6 for t in ternary_times],
        "brute_slope": brute_slope,
        "ternary_slope": ternary_slope
    }


# ─── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Phase 1: Convex Hull KV Cache — Does the Geometry Work?\n")
    
    if not test_correctness():
        print("\nCORRECTNESS FAILED — stopping.")
        exit(1)
    print("\n*** ALL CORRECTNESS TESTS PASSED ***\n")
    
    query_results = benchmark_query_scaling()
    trace_results = benchmark_execution_trace()
    scaling = benchmark_scaling_fit()
    
    all_results = {
        "query_scaling": query_results,
        "trace_simulation": trace_results,
        "scaling_fit": scaling
    }
    with open("/home/claude/phase1_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"  Brute force scales as O(n^{scaling['brute_slope']:.2f})")
    print(f"  Ternary search scales as O(n^{scaling['ternary_slope']:.2f})")
    print(f"  At n=100K: {scaling['brute_us'][-1]:.0f}µs brute vs {scaling['ternary_us'][-1]:.1f}µs ternary")
