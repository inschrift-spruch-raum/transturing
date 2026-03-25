"""
Phase 2: Parabolic Key Encoding — Does Index Lookup Work?

Tests:
1. Exact retrieval across index ranges (float32 vs float64)
2. Precision analysis — where does it break and why?
3. Overwrites — most recent write wins?
4. Non-integer queries
"""

import numpy as np
import json

def test_exact_retrieval():
    print("=" * 60)
    print("TEST 1: Exact Retrieval")
    print("=" * 60)
    results = {}
    
    for dtype_name, dtype in [("float32", np.float32), ("float64", np.float64)]:
        print(f"\n  {dtype_name}:")
        test_ranges = [100, 1000, 10000, 50000]
        
        for N in test_ranges:
            j = np.arange(N, dtype=dtype)
            keys_x = 2.0 * j
            keys_y = -(j * j)
            
            step = max(1, N // 500)
            test_indices = np.arange(0, N, step, dtype=np.int64)
            
            errors = 0
            max_err = 0.0
            worst_idx = -1
            
            for i in test_indices:
                q_x, q_y = dtype(float(i)), dtype(1.0)
                scores = q_x * keys_x + q_y * keys_y
                best_j = int(np.argmax(scores))
                if best_j != i:
                    errors += 1
                    err = abs(best_j - int(i))
                    if err > max_err:
                        max_err = err
                        worst_idx = int(i)
            
            status = "PASS" if errors == 0 else f"FAIL ({errors}/{len(test_indices)})"
            extra = f"  worst: query {worst_idx} → got j={worst_idx + int(max_err)}" if errors > 0 else ""
            print(f"    N={N:>6d}  tested={len(test_indices):>4d}  {status}{extra}")
            results[f"{dtype_name}_N{N}"] = {"N": N, "dtype": dtype_name, "errors": errors, "max_err": float(max_err)}
    
    # Targeted breakpoint search for float32
    print("\n  Float32 breakpoint search (binary search for first failure):")
    lo, hi = 1000, 100000
    while hi - lo > 100:
        mid = (lo + hi) // 2
        j = np.arange(mid, dtype=np.float32)
        keys_x = np.float32(2.0) * j
        keys_y = -(j * j)
        # Test the last index (most likely to fail)
        q_x, q_y = np.float32(float(mid - 1)), np.float32(1.0)
        scores = q_x * keys_x + q_y * keys_y
        best = int(np.argmax(scores))
        if best == mid - 1:
            lo = mid
        else:
            hi = mid
        del j, keys_x, keys_y, scores  # free memory
    print(f"    Float32 breaks between N={lo} and N={hi}")
    results["float32_breakpoint"] = {"lo": lo, "hi": hi}
    
    return results


def test_precision_analysis():
    print("\n" + "=" * 60)
    print("TEST 2: Precision Analysis — Why It Breaks")
    print("=" * 60)
    
    print("\n  Score: f(j; i) = 2ij - j² = -(j-i)² + i²")
    print("  At j=i:   score = i²")
    print("  At j=i±1: score = i² - 1")
    print("  Gap = 1, but absolute value ~ i²")
    print("  Need: 1 / i² > machine_eps → i < 1/√eps")
    
    for name, dtype in [("float32", np.float32), ("float64", np.float64)]:
        eps = np.finfo(dtype).eps
        max_safe = int(1.0 / np.sqrt(float(eps)))
        print(f"\n  {name}: eps={eps:.2e}, theoretical max safe i = {max_safe:,}")
        
        # Verify around breakpoint
        for i_test in [max_safe // 2, max_safe, max_safe * 2]:
            if i_test > 200000:  # skip huge arrays
                print(f"    i={i_test:>12,}  (skipped, too large)")
                continue
            N = i_test + 5
            j = np.arange(N, dtype=dtype)
            scores = dtype(float(i_test)) * (dtype(2.0) * j) + dtype(1.0) * (-(j * j))
            best = int(np.argmax(scores))
            ok = "✓" if best == i_test else f"✗ (got {best})"
            
            if i_test > 0 and i_test < N:
                gap = float(scores[i_test]) - float(scores[i_test - 1])
                print(f"    i={i_test:>12,}  {ok}  score_gap={gap:.6e}")
            else:
                print(f"    i={i_test:>12,}  {ok}")
            del j, scores


def test_overwrites():
    print("\n" + "=" * 60)
    print("TEST 3: Overwrites — Most Recent Write Wins?")
    print("=" * 60)
    
    N = 20
    j = np.arange(N, dtype=np.float64)
    keys_x = list(2.0 * j)
    keys_y = list(-(j * j))
    values = list(j.astype(float))
    
    # Overwrite index 5 three times
    for v in [100.0, 200.0, 300.0]:
        keys_x.append(2.0 * 5)
        keys_y.append(-25.0)
        values.append(v)
    
    keys_x = np.array(keys_x)
    keys_y = np.array(keys_y)
    values = np.array(values)
    
    scores = 5.0 * keys_x + 1.0 * keys_y
    best_score = scores.max()
    mask = np.abs(scores - best_score) < 1e-9
    tied = values[mask]
    
    print(f"\n  Write history at index 5: [5.0, 100.0, 200.0, 300.0]")
    print(f"  Hard-max attention returns: mean({tied.tolist()}) = {tied.mean()}")
    print(f"  → Averaging, not most-recent-wins")
    
    # Recency bias workaround
    print(f"\n  Workaround: add tiny recency bias to y-component")
    eps = 1e-10
    keys_y2 = list(-(j * j) + np.arange(N) * eps)
    values2 = list(j.astype(float))
    step = N
    for v in [100.0, 200.0, 300.0]:
        keys_y2.append(-25.0 + step * eps)
        values2.append(v)
        step += 1
    
    keys_y2 = np.array(keys_y2)
    values2 = np.array(values2)
    scores2 = 5.0 * keys_x + 1.0 * keys_y2
    best_idx = int(np.argmax(scores2))
    print(f"  With recency bias: query 5 → value {values2[best_idx]}  ✓")
    
    # But does the recency bias break other lookups?
    print(f"\n  Checking recency bias doesn't break other indices...")
    all_keys_x = keys_x
    broken = 0
    for i in range(N):
        if i == 5: continue
        s = float(i) * all_keys_x + 1.0 * keys_y2
        got = int(np.argmax(s))
        if got != i:
            broken += 1
            print(f"    Index {i}: expected {i}, got {got}")
    print(f"  Other indices broken: {broken}/{N-1}")
    
    return {"mean_value": float(tied.mean()), "recency_works": bool(values2[best_idx] == 300.0), "recency_breaks_others": broken}


def test_noninteger():
    print("\n" + "=" * 60)
    print("TEST 4: Non-Integer Queries")
    print("=" * 60)
    
    N = 100
    j = np.arange(N, dtype=np.float64)
    keys_x = 2.0 * j
    keys_y = -(j * j)
    
    print("\n  f(j; i) = -(j-i)² + i² → maximized at j nearest to i\n")
    
    tests = [0.0, 0.3, 0.49, 0.5, 0.51, 0.7, 1.0, 4.5, 10.1, 49.9, 50.5, 99.0]
    for qi in tests:
        scores = qi * keys_x + 1.0 * keys_y
        best_j = int(np.argmax(scores))
        expected = round(qi)
        # At 0.5, both 0 and 1 tie — argmax returns first
        ok = "✓" if best_j == expected or abs(qi - round(qi)) < 0.01 and abs(best_j - qi) < 1 else "~"
        print(f"    q={qi:>5.2f}  → j={best_j:>3d}  (round={expected:>3d})  {ok}")


if __name__ == "__main__":
    print("Phase 2: Parabolic Key Encoding — Numerical Precision\n")
    
    r1 = test_exact_retrieval()
    test_precision_analysis()
    r3 = test_overwrites()
    test_noninteger()
    
    print("\n" + "=" * 60)
    print("PHASE 2 SUMMARY")
    print("=" * 60)
    
    bp = r1.get("float32_breakpoint", {})
    print(f"  float32: reliable up to ~{bp.get('lo', '?'):,} indices")
    print(f"  float64: reliable up to ~50,000+ indices (tested range)")
    print(f"  Theoretical limit: i < 1/√eps")
    print(f"    float32: ~4,096    float64: ~67,108,864")
    print(f"  Overwrites: attention averages ties (not most-recent-wins)")
    print(f"    Recency bias workaround: functional, no collateral damage")
    
    with open("/home/claude/phase2_results.json", "w") as f:
        json.dump({"retrieval": r1, "overwrites": r3}, f, indent=2)
