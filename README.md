# llm-as-computer

Testing [Percepta's claim](https://percepta.ai/blog/can-llms-be-computers) that transformers can execute programs internally via 2D convex hull attention, achieving O(log t) per-step decoding over million-step execution traces.

## What is this?

Percepta published a blog post (Mar 11, 2026) describing how they embedded a WebAssembly interpreter inside a vanilla transformer by:

1. Restricting attention head dimension to 2D
2. Using parabolic key encoding k_j = (2j, −j²) for memory lookups
3. Replacing linear attention scans with convex hull queries for O(log t) decoding
4. Tracking state (instruction pointer, stack depth) via cumulative sum attention

They released no code or weights. This repo independently tests whether the core primitives work as described.

## Status

| Phase | Description | Status | Key Finding |
|-------|------------|--------|-------------|
| 1 | Convex hull KV cache scaling | ✅ Done | O(log t) confirmed via ternary search. 35× speedup at 50K steps. |
| 2 | Parabolic key encoding precision | ✅ Done | Float32 breaks at index ~4K (revised). Float64 safe to ~200K+. |
| 2b | Extended addressing | ✅ Done | Residual (bit-split) addressing: 25M range from 2 heads. |
| 3 | Cumulative sum via attention | ✅ Done | Rock-solid — zero integer errors at 100K in float32. |
| 4 | Hand-wired stack machine | ✅ Done | Primitives compose. 10/10 test programs execute correctly via attention only. |
| 5 | Trained micro-executor | 🔲 Todo | Can gradient descent discover the optimal attention structure? |
| 6 | WASM fragment execution | 🔲 Stretch | Fibonacci inside a transformer. |

## Files

```
RD-PLAN.md                  # Full R&D plan (6 phases)
FINDINGS.md                 # Combined findings from Phases 1-4, 2b
phase1_hull_cache.py        # Convex hull vs brute force benchmarks
phase2_parabolic.py         # Parabolic encoding precision tests
phase2b_address_limits.py   # Extended addressing exploration
phase3_cumsum.py            # Cumulative sum stability tests
phase4_stack_machine.py     # Stack machine via attention primitives
viz/phase1-results.jsx      # Phase 1 interactive visualization (React)
```

## Running

```bash
python3 phase1_hull_cache.py       # ~60s, benchmarks query scaling
python3 phase2_parabolic.py        # ~10s, finds float32 breakpoint
python3 phase3_cumsum.py           # ~10s, tests numerical drift
python3 phase4_stack_machine.py    # ~1s, stack machine composition test
```

Requires: numpy, scipy (for convex hull verification only — the fast path uses ternary search).

## Key Takeaways

- **The geometry works.** Parabolic keys + ternary search give exact O(log n) index lookup.
- **Float32 is the bottleneck — but solvable.** ~4K addressable indices per parabolic head (revised down from initial 7K measurement). Residual (bit-split) addressing with 2 heads extends this to 25M, likely what Percepta uses.
- **Cumsum is not the weak link.** Both the mean×t trick and sequential lookback are stable far beyond expected.
- **"Convex hull" is slightly misleading.** The speed comes from exploiting unimodal structure via binary/ternary search, not from hull size. For parabolic keys, all points lie on the hull.
- **The primitives compose.** Phase 4 proves that parabolic indexing, recency bias, and sequential state tracking work together as a stack machine executor — same memory primitive addresses both program memory and stack memory without interference.
- **The FF layer is the real challenge.** Attention heads have clean roles; the opcode-dependent routing in the feed-forward network is where model capacity actually goes.
