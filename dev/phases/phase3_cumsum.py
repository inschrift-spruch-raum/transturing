"""
Phase 3: Cumulative Sum Attention — Tracking Running State

The claim: if all keys are identical, attention averages all values uniformly.
Multiply by position t to recover the cumulative sum.

Used for: instruction pointer, stack depth, call-stack depth —
all tracked as running sums of delta increments.

Tests:
1. Correctness of cumsum recovery
2. Numerical drift over long sequences
3. Stack depth simulation (push/pop deltas)
4. Interaction with the hull cache (do they compose on separate heads?)
"""

import numpy as np
import json


def cumsum_via_attention(deltas):
    """
    Simulate the attention-based cumsum trick.
    
    All keys identical → softmax gives uniform weights → attention = mean(values[0:t]).
    Multiply by t → sum(values[0:t]) = cumulative sum.
    
    Returns cumsum at each step.
    """
    n = len(deltas)
    result = np.zeros(n)
    running_sum = 0.0
    
    for t in range(n):
        running_sum += deltas[t]
        # Attention averages all values up to and including t
        mean_val = np.mean(deltas[:t+1])
        # Multiply by count to recover sum
        recovered = mean_val * (t + 1)
        result[t] = recovered
    
    return result


def cumsum_via_attention_vectorized(deltas):
    """Vectorized version for benchmarking."""
    n = len(deltas)
    cum = np.cumsum(deltas)
    positions = np.arange(1, n + 1, dtype=np.float64)
    # mean at step t = cum[t] / (t+1), then multiply by (t+1) → cum[t]
    # This is trivially exact in float64, but the point is to simulate
    # what happens when the mean is computed via attention (with float errors)
    means = cum / positions
    recovered = means * positions
    return recovered


def test_basic_correctness():
    print("=" * 60)
    print("TEST 1: Basic Correctness")
    print("=" * 60)
    
    # Simple case
    deltas = np.array([1.0, 2.0, 3.0, -1.0, 5.0])
    expected = np.cumsum(deltas)
    recovered = cumsum_via_attention(deltas)
    
    print(f"\n  Deltas:    {deltas.tolist()}")
    print(f"  Expected:  {expected.tolist()}")
    print(f"  Recovered: {recovered.tolist()}")
    print(f"  Max error: {np.max(np.abs(recovered - expected)):.2e}")
    
    # Push/pop sequence (+1/-1)
    deltas2 = np.array([1, 1, 1, -1, 1, -1, -1, 1, 1, -1], dtype=np.float64)
    expected2 = np.cumsum(deltas2)
    recovered2 = cumsum_via_attention(deltas2)
    
    print(f"\n  Push/pop:  {deltas2.astype(int).tolist()}")
    print(f"  Expected:  {expected2.astype(int).tolist()}")
    print(f"  Recovered: {recovered2.tolist()}")
    print(f"  Max error: {np.max(np.abs(recovered2 - expected2)):.2e}")
    
    return np.max(np.abs(recovered - expected)) < 1e-10


def test_numerical_drift():
    print("\n" + "=" * 60)
    print("TEST 2: Numerical Drift Over Long Sequences")
    print("=" * 60)
    
    print("\n  The trick: cumsum = mean(deltas[0:t]) × t")
    print("  Error source: mean involves division, then multiplication")
    print("  For exact arithmetic this is perfect; in float it drifts.\n")
    
    results = []
    
    for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
        print(f"  {dtype_name}:")
        
        for N in [100, 1000, 10000, 100000]:
            np.random.seed(42)
            # Random +1/-1 deltas (stack push/pop)
            deltas = np.random.choice([-1.0, 1.0], size=N).astype(dtype)
            
            true_cumsum = np.cumsum(deltas.astype(np.float64))  # ground truth in f64
            
            # Simulate attention-based recovery in target dtype
            cum = np.cumsum(deltas)
            positions = np.arange(1, N + 1, dtype=dtype)
            means = cum / positions
            recovered = means * positions
            
            errors = np.abs(recovered.astype(np.float64) - true_cumsum)
            max_err = float(errors.max())
            mean_err = float(errors.mean())
            # Where does error first exceed 0.5 (wrong integer)?
            wrong = np.where(errors > 0.5)[0]
            first_wrong = int(wrong[0]) if len(wrong) > 0 else N
            
            print(f"    N={N:>7d}  max_err={max_err:>10.4f}  mean_err={mean_err:>10.6f}  "
                  f"first_wrong_int={first_wrong:>7d}")
            
            results.append({
                "dtype": dtype_name, "N": N, "max_err": max_err,
                "mean_err": mean_err, "first_wrong_int": first_wrong
            })
    
    return results


def test_realistic_stack():
    print("\n" + "=" * 60)
    print("TEST 3: Realistic Stack Depth Tracking")
    print("=" * 60)
    
    print("\n  Simulating WASM execution: stack depth changes by small integers")
    print("  i32.const → +1, i32.add → -1, call → +N, return → -N\n")
    
    np.random.seed(123)
    
    for N in [1000, 10000, 100000]:
        # Generate realistic-ish stack deltas
        deltas = np.zeros(N)
        for i in range(N):
            r = np.random.random()
            if r < 0.4:    deltas[i] = 1    # push
            elif r < 0.7:  deltas[i] = -1   # pop/consume
            elif r < 0.85: deltas[i] = 2    # call with args
            elif r < 0.95: deltas[i] = -2   # return
            else:          deltas[i] = 0    # nop
        
        true_cumsum = np.cumsum(deltas)
        
        for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
            d = deltas.astype(dtype)
            cum = np.cumsum(d)
            pos = np.arange(1, N + 1, dtype=dtype)
            recovered = (cum / pos) * pos
            
            errors = np.abs(recovered.astype(np.float64) - true_cumsum)
            max_err = float(errors.max())
            
            # Can we recover the integer stack depth?
            rounded = np.round(recovered.astype(np.float64))
            int_errors = int(np.sum(rounded != true_cumsum))
            
            print(f"    N={N:>7d}  {dtype_name}  max_float_err={max_err:>8.4f}  "
                  f"int_round_errors={int_errors:>6d}/{N}")


def test_alternative_cumsum():
    print("\n" + "=" * 60)
    print("TEST 4: Alternative — Prefix Sum via Causal Attention")
    print("=" * 60)
    
    print("\n  Instead of mean×t, use causal attention with uniform weights")
    print("  directly summing (not averaging). This avoids the divide-multiply.")
    print("  In a transformer: use a head with constant keys, value = delta,")
    print("  and scale the output by t (position-dependent scaling).\n")
    print("  But there's a simpler perspective: the model just needs to know")
    print("  the CURRENT stack depth, not recompute from scratch each step.")
    print("  One attention head can look at the PREVIOUS trace token's depth.")
    print("  Then: depth[t] = depth[t-1] + delta[t]")
    print("  This is O(1) per step — no cumsum needed!\n")
    
    # Simulate the "look at previous step" approach
    N = 100000
    np.random.seed(42)
    deltas = np.random.choice([-1.0, 0.0, 1.0], size=N)
    
    # Approach 1: cumsum via mean×t (the blog's described method)
    true = np.cumsum(deltas)
    cum32 = np.cumsum(deltas.astype(np.float32))
    pos32 = np.arange(1, N+1, dtype=np.float32)
    method1 = (cum32 / pos32) * pos32
    err1 = np.abs(method1.astype(np.float64) - true)
    
    # Approach 2: sequential (look at previous step)
    # In a transformer this is just attending to position t-1
    depth = np.zeros(N, dtype=np.float32)
    depth[0] = np.float32(deltas[0])
    for i in range(1, N):
        depth[i] = depth[i-1] + np.float32(deltas[i])
    err2 = np.abs(depth.astype(np.float64) - true)
    
    print(f"  N=100K, float32:")
    print(f"    Method 1 (mean×t): max_err={err1.max():.4f}, int_errors={int(np.sum(np.round(method1.astype(np.float64)) != true))}")
    print(f"    Method 2 (prev+δ): max_err={err2.max():.4f}, int_errors={int(np.sum(np.round(depth.astype(np.float64)) != true))}")
    print(f"\n  → Sequential lookback is more numerically stable")
    print(f"    and only needs 1 attention head looking at position t-1")


if __name__ == "__main__":
    print("Phase 3: Cumulative Sum via Attention\n")
    
    test_basic_correctness()
    drift_results = test_numerical_drift()
    test_realistic_stack()
    test_alternative_cumsum()
    
    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)
    
    # Find float32 drift threshold
    f32 = [r for r in drift_results if r["dtype"] == "float32"]
    f64 = [r for r in drift_results if r["dtype"] == "float64"]
    
    f32_first_wrong = min(r["first_wrong_int"] for r in f32)
    f64_first_wrong = min(r["first_wrong_int"] for r in f64)
    
    print(f"  Cumsum via mean×t:")
    print(f"    float32: first integer error at step ~{f32_first_wrong:,}")
    print(f"    float64: first integer error at step ~{f64_first_wrong:,}")
    print(f"  Sequential lookback (depth[t] = depth[t-1] + δ):")
    print(f"    More stable, O(1) per step, needs only 1 attention head")
    print(f"  Percepta likely uses sequential lookback, not the mean×t trick")
    
    with open("/home/claude/phase3_results.json", "w") as f:
        json.dump({"drift": drift_results}, f, indent=2)
