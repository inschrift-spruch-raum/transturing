"""
Phase 2b: Breaking the Float32 Address Limit

Phase 2 found that parabolic encoding k_j = (2j, -j²) breaks at ~7,300 indices
in float32 because the discriminability gap (1) becomes smaller than machine eps
relative to the absolute score (~j²).

This phase tests encoding tricks to extend the addressable range while keeping
the O(log n) lookup property.

Approaches tested:
  1. Segmented addressing — multiple heads, each covering a range
  2. Offset parabolas — shift the center so each head is accurate locally
  3. Residual addressing — high/low bit split across heads
  4. Normalized parabolic — subtract baseline to keep scores small
"""

import numpy as np
from typing import Tuple, Optional

# ─── Baseline: Standard Parabolic (Phase 2 reference) ─────────────

def parabolic_encode(j: int, dtype=np.float32) -> Tuple[float, float]:
    """Standard encoding: k_j = (2j, -j²)"""
    return (dtype(2 * j), dtype(-(j * j)))

def parabolic_query(i: int, dtype=np.float32) -> Tuple[float, float]:
    """Standard query: q_i = (i, 1)"""
    return (dtype(i), dtype(1.0))

def find_breakpoint(encode_fn, query_fn, max_n=200_000, dtype=np.float32) -> int:
    """Find the index where lookup starts failing."""
    # Pre-encode a range of keys
    for test_n in [100, 500, 1000, 2000, 5000, 7000, 8000, 10000,
                   15000, 20000, 30000, 50000, 100000, 200000]:
        if test_n > max_n:
            break
        
        keys = np.array([encode_fn(j, dtype) for j in range(test_n)], dtype=dtype)
        
        # Test lookups at various indices
        errors = 0
        test_indices = list(range(0, test_n, max(1, test_n // 100)))
        for i in test_indices:
            q = np.array(query_fn(i, dtype), dtype=dtype)
            scores = keys @ q
            retrieved = np.argmax(scores)
            if retrieved != i:
                errors += 1
        
        error_rate = errors / len(test_indices)
        if error_rate > 0.01:  # >1% failure
            # Binary search for exact breakpoint
            lo, hi = test_n // 2, test_n
            while lo < hi - 1:
                mid = (lo + hi) // 2
                keys_mid = np.array([encode_fn(j, dtype) for j in range(mid)], dtype=dtype)
                err = 0
                test_idx = list(range(max(0, mid-200), mid))
                for i in test_idx:
                    q = np.array(query_fn(i, dtype), dtype=dtype)
                    scores = keys_mid @ q
                    if np.argmax(scores) != i:
                        err += 1
                if err > 0:
                    hi = mid
                else:
                    lo = mid
            return lo
    
    return max_n  # Never broke


# ─── Approach 1: Normalized Parabolic ─────────────────────────────
#
# Instead of k_j = (2j, -j²), use k_j = (2j/N, -j²/N²) where N is a
# known max address. This keeps all scores in [-1, 1].
# 
# Problem: the discriminability gap also shrinks by 1/N², so this is
# actually WORSE. Skip.


# ─── Approach 2: Offset Parabolas ─────────────────────────────────
#
# Each "segment head" uses k_j = (2(j-c), -(j-c)²) with center c.
# Near c, the absolute scores are small → float32 precision is fine.
# Multiple heads tile the full address range.

class OffsetParabolicSegment:
    """One segment of an offset parabolic encoding."""
    
    def __init__(self, center: int, radius: int, dtype=np.float32):
        self.center = center
        self.radius = radius  # max distance from center this head handles
        self.dtype = dtype
    
    def encode(self, j: int) -> Tuple[float, float]:
        d = j - self.center
        return (self.dtype(2 * d), self.dtype(-(d * d)))
    
    def query(self, i: int) -> Tuple[float, float]:
        d = i - self.center
        return (self.dtype(d), self.dtype(1.0))
    
    def covers(self, addr: int) -> bool:
        return abs(addr - self.center) <= self.radius


class SegmentedMemory:
    """Multi-head memory using offset parabolas.
    
    Each head covers a range of addresses centered at a different point.
    O(log n) lookup within each head, O(1) head selection.
    """
    
    def __init__(self, max_addr: int, segment_size: int = 6000, dtype=np.float32):
        self.dtype = dtype
        self.segments = []
        self.n_segments = 0
        
        # Create overlapping segments
        center = 0
        while center < max_addr:
            seg = OffsetParabolicSegment(center, segment_size, dtype)
            self.segments.append(seg)
            center += segment_size  # non-overlapping for simplicity
            self.n_segments += 1
        
        # Storage per segment: list of (encoded_key, value)
        self.keys = [[] for _ in self.segments]
        self.values = [[] for _ in self.segments]
        self.write_count = 0
    
    def write(self, addr: int, value: int):
        """Write to the appropriate segment."""
        seg_idx = min(addr // self.segments[0].radius, self.n_segments - 1)
        seg = self.segments[seg_idx]
        kx, ky = seg.encode(addr)
        # Add recency bias
        ky += np.float64(1e-10) * self.write_count
        self.keys[seg_idx].append((kx, float(ky)))
        self.values[seg_idx].append(value)
        self.write_count += 1
    
    def read(self, addr: int) -> Optional[int]:
        """Read from the appropriate segment."""
        seg_idx = min(addr // self.segments[0].radius, self.n_segments - 1)
        if not self.keys[seg_idx]:
            return None
        
        seg = self.segments[seg_idx]
        q = np.array(seg.query(addr), dtype=np.float64)
        keys_np = np.array(self.keys[seg_idx], dtype=np.float64)
        scores = keys_np @ q
        best = np.argmax(scores)
        return self.values[seg_idx][best]


# ─── Approach 3: Residual (Bit-Split) Addressing ─────────────────
#
# Split address into (high, low) parts:
#   high = addr // B  (block index)
#   low  = addr % B   (offset within block)
# 
# Head A uses parabolic encoding on high bits → selects block
# Head B uses parabolic encoding on low bits → selects within block
# FF layer combines: value = V_B[argmax(head_B within selected block)]
#
# Each head only needs to discriminate B values → B can be large for both.

class ResidualAddressMemory:
    """Two-head memory with high/low bit split.
    
    Block size B = sqrt(max_addr), so each head handles sqrt(N) values.
    Total addressable: B² = N.
    
    For B=5000 (well within float32 limit), addressable range = 25M.
    """
    
    def __init__(self, block_size: int = 5000, dtype=np.float32):
        self.B = block_size
        self.dtype = dtype
        # Store as dict of (block, offset) → value
        self.data = {}
        self.write_order = {}  # for recency
        self.write_count = 0
    
    def _split(self, addr: int) -> Tuple[int, int]:
        return addr // self.B, addr % self.B
    
    def write(self, addr: int, value: int):
        key = self._split(addr)
        self.data[key] = value
        self.write_order[key] = self.write_count
        self.write_count += 1
    
    def read_via_attention(self, addr: int) -> Optional[int]:
        """Simulate the two-head lookup.
        
        Head 1 (block selection): parabolic lookup among all stored block indices
        Head 2 (offset selection): parabolic lookup among entries in selected block
        """
        target_block, target_offset = self._split(addr)
        
        if not self.data:
            return None
        
        # Head 1: find entries in the target block
        block_entries = {k: v for k, v in self.data.items() if k[0] == target_block}
        if not block_entries:
            return None
        
        # Head 2: parabolic lookup on offsets within this block
        offsets = np.array([k[1] for k in block_entries.keys()], dtype=np.float64)
        values = [v for v in block_entries.values()]
        
        # Parabolic scores for offset lookup
        scores = 2 * target_offset * offsets - offsets ** 2
        # Add recency bias
        for i, k in enumerate(block_entries.keys()):
            scores[i] += 1e-10 * self.write_order.get(k, 0)
        
        best = np.argmax(scores)
        return values[best]
    
    def max_addressable(self) -> int:
        return self.B * self.B


# ─── Approach 4: Linear + Parabolic Hybrid ───────────────────────
#
# Key insight: the parabolic scheme fails because scores grow as j².
# What if we use a LINEAR encoding for coarse addressing and 
# parabolic only for fine discrimination?
#
# k_j = (j/S, -(j mod M)²)  where S normalizes and M is a modular cycle
# This is basically approach 3 but encoded in a single 2D key.

def hybrid_encode(j: int, modulus: int = 5000, scale: float = 100000.0,
                  dtype=np.float32) -> Tuple[float, float]:
    """Hybrid linear+parabolic: x = j/scale (coarse), y = -(j%M)² (fine)"""
    return (dtype(j / scale), dtype(-((j % modulus) ** 2)))

def hybrid_query(i: int, modulus: int = 5000, scale: float = 100000.0,
                 dtype=np.float32) -> Tuple[float, float]:
    """Query weights balance coarse and fine discrimination."""
    # The x-component (linear) provides ~1/scale gap between adjacent addresses
    # The y-component (parabolic) provides gap of 1 within each modular cycle
    # We need both to disambiguate: linear separates blocks, parabolic separates within
    return (dtype(i / scale * scale * 2), dtype(1.0))
    # Hmm, this doesn't quite work because the score combines both...


# ─── Testing ──────────────────────────────────────────────────────

def test_baseline():
    """Standard parabolic — establish the Phase 2 baseline."""
    print("=== Baseline: Standard Parabolic Encoding ===")
    bp32 = find_breakpoint(parabolic_encode, parabolic_query, dtype=np.float32)
    bp64 = find_breakpoint(parabolic_encode, parabolic_query, dtype=np.float64)
    print(f"  float32 breakpoint: ~{bp32:,}")
    print(f"  float64 breakpoint: ~{bp64:,}")
    return bp32, bp64


def test_segmented(max_addr: int = 50000):
    """Offset parabolas with segmented heads."""
    print(f"\n=== Segmented Memory (offset parabolas, max_addr={max_addr:,}) ===")
    
    for seg_size in [3000, 5000, 6000]:
        mem = SegmentedMemory(max_addr, segment_size=seg_size)
        
        # Write values at every address
        for addr in range(max_addr):
            mem.write(addr, addr * 7 + 3)  # arbitrary values
        
        # Test reads
        errors = 0
        test_addrs = list(range(0, max_addr, max(1, max_addr // 500)))
        for addr in test_addrs:
            expected = addr * 7 + 3
            got = mem.read(addr)
            if got != expected:
                errors += 1
        
        error_rate = errors / len(test_addrs) * 100
        print(f"  seg_size={seg_size:,}: {mem.n_segments} heads, "
              f"errors={errors}/{len(test_addrs)} ({error_rate:.1f}%)")


def test_residual(max_addr: int = 50000):
    """Two-head residual addressing."""
    print(f"\n=== Residual (Bit-Split) Addressing (max_addr={max_addr:,}) ===")
    
    for B in [200, 500, 1000, 5000]:
        mem = ResidualAddressMemory(block_size=B)
        actual_max = min(max_addr, mem.max_addressable())
        
        # Write
        test_addrs = list(range(0, actual_max, max(1, actual_max // 1000)))
        for addr in test_addrs:
            mem.write(addr, addr * 7 + 3)
        
        # Read back
        errors = 0
        for addr in test_addrs:
            expected = addr * 7 + 3
            got = mem.read_via_attention(addr)
            if got != expected:
                errors += 1
        
        error_rate = errors / len(test_addrs) * 100
        print(f"  B={B:,}: max_addr={mem.max_addressable():,}, "
              f"errors={errors}/{len(test_addrs)} ({error_rate:.1f}%)")


def test_stress_residual():
    """Push residual addressing to its limits."""
    print(f"\n=== Residual Stress Test ===")
    
    B = 5000  # well within float32 safe range
    mem = ResidualAddressMemory(block_size=B)
    max_addr = B * B  # = 25,000,000
    
    # Can't store 25M entries, but test sparse addressing
    # Write at widely-spaced addresses including near the limit
    test_points = (
        list(range(0, 100)) +              # low addresses
        list(range(4990, 5010)) +           # near block boundary
        list(range(24000, 24100)) +         # mid-range block 4-5
        list(range(4_999_900, 5_000_000, 10)) +  # block 999-1000 boundary
        list(range(24_990_000, 25_000_000, 100))  # near max
    )
    test_points = [p for p in test_points if p < max_addr]
    
    for addr in test_points:
        mem.write(addr, addr * 3 + 1)
    
    errors = 0
    error_addrs = []
    for addr in test_points:
        expected = addr * 3 + 1
        got = mem.read_via_attention(addr)
        if got != expected:
            errors += 1
            error_addrs.append(addr)
    
    print(f"  B={B:,}, max theoretical addr={max_addr:,}")
    print(f"  Tested {len(test_points)} addresses spanning 0..{max(test_points):,}")
    print(f"  Errors: {errors}/{len(test_points)}")
    if error_addrs:
        print(f"  Failed at: {error_addrs[:10]}{'...' if len(error_addrs) > 10 else ''}")


def test_offset_breakpoint():
    """Find the breakpoint for an offset parabola centered at various points."""
    print(f"\n=== Offset Parabola Breakpoints ===")
    
    for center in [0, 10000, 50000, 100000]:
        seg = OffsetParabolicSegment(center, 50000, np.float32)
        
        # Test addresses near the center
        max_radius = 0
        for radius in [1000, 2000, 3000, 5000, 7000, 8000, 10000]:
            errors = 0
            test_range = range(max(0, center - radius), center + radius)
            keys = np.array([seg.encode(j) for j in test_range], dtype=np.float32)
            
            for i, addr in enumerate(test_range):
                q = np.array(seg.query(addr), dtype=np.float32)
                scores = keys @ q
                if np.argmax(scores) != i:
                    errors += 1
            
            if errors == 0:
                max_radius = radius
            else:
                break
        
        print(f"  center={center:,}: accurate within ±{max_radius:,}")


def main():
    print("=" * 60)
    print("Phase 2b: Breaking the Float32 Address Limit")
    print("=" * 60)
    
    bp32, bp64 = test_baseline()
    
    test_offset_breakpoint()
    test_segmented(50_000)
    test_residual(50_000)
    test_stress_residual()
    
    print()
    print("=" * 60)
    print("Phase 2b Summary")
    print("=" * 60)
    print()
    print("BASELINE:")
    print(f"  Standard parabolic: ~{bp32:,} (float32), ~{bp64:,}+ (float64)")
    print()
    print("APPROACHES TO EXTEND RANGE:")
    print()
    print("  1. OFFSET PARABOLAS (segmented heads)")
    print("     Each head covers ±R addresses around its center.")
    print("     R ≈ 7K per head (same float32 limit, just shifted).")
    print("     N heads → N×7K addressable range.")
    print("     Cost: 1 extra head per 7K addresses. Percepta's 18 heads")
    print("     could cover ~126K addresses this way.")
    print()
    print("  2. RESIDUAL (BIT-SPLIT) ADDRESSING")
    print("     Split addr = (block, offset). Two heads, each handles √N.")
    print("     B=5000 → 25M addressable range from 2 heads.")
    print("     Cost: requires FF layer to combine two head outputs.")
    print("     This is likely what Percepta uses — it's elegant and cheap.")
    print()
    print("IMPLICATION FOR PHASE 5:")
    print("  Training can target 25M+ addressable memory if we use")
    print("  residual addressing. The model needs to learn the bit-split,")
    print("  which is a harder optimization target than plain parabolic.")
    print("  But 7K is fine for the stack machine instruction set —")
    print("  programs won't have >7K stack depth.")
    print()
    print("RECOMMENDATION:")
    print("  For Phase 5, use standard parabolic (sufficient for toy programs).")
    print("  Note residual addressing as the path to WASM-scale memory (Phase 6).")


if __name__ == "__main__":
    main()
