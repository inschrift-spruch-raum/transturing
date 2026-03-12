# Percepta "Can LLMs Be Computers?" — R&D Findings: Phases 1-3

## Context
Testing core primitives from Percepta's blog post (Mar 11, 2026) about embedding
a WASM executor inside a transformer via 2D convex hull attention.
Blog: https://percepta.ai/blog/can-llms-be-computers

---

## Phase 1: Convex Hull KV Cache — Does the Geometry Work?

**Result: YES — the O(log t) scaling claim holds.**

### Key findings:
- Ternary search over parabolic keys scales as O(n^0.18) in log-log (consistent with O(log n))
- Brute-force numpy scales as O(n^0.53) (sublinear due to SIMD, but still O(n) in operations)
- Speedup: 3.5× at n=100, 35× at n=50K. Extrapolates to ~100-200× at 1M steps.
- Crossover point: ~1-2K entries (below that, brute numpy is faster due to constant overhead)

### Important nuance:
- For **random** 2D keys, convex hull is tiny (8-17 vertices for 50K points) — hull scan is O(1) by accident
- For **parabolic** keys (the actual use case), ALL points lie on the hull — hull scan degrades to O(n)
- The O(log n) requires **ternary/binary search exploiting unimodal structure**, not just hull maintenance
- Percepta's "HullKVCache" name is slightly misleading — the win comes from structured search, not hull size

---

## Phase 2: Parabolic Key Encoding — Numerical Precision

**Result: Works in float64 for any practical trace. Float32 breaks around index 7,300.**

### The encoding: k_j = (2j, -j²), query q = (i, 1)
- Score function: f(j; i) = 2ij - j² = -(j-i)² + i²
- Maximized exactly at j = i. Gap between correct and nearest wrong = 1.
- But absolute score ~ i², so need relative precision 1/i² > machine_eps

### Breakpoints:
| dtype   | machine eps | theoretical limit (1/√eps) | measured breakpoint |
|---------|------------|---------------------------|---------------------|
| float32 | 1.19e-07   | ~2,896                    | ~7,283*             |
| float64 | 2.22e-16   | ~67,108,864               | >50,000 (all pass)  |

*Measured breakpoint higher than theory because numpy's vectorized ops have better intermediate precision than single-element math.

### Overwrite semantics:
- Hard-max attention **averages** tied keys (not most-recent-wins)
- Workaround: add tiny recency bias ε·step to y-component of key
- Tested: recency bias doesn't break other index lookups ✓

### Implication for Percepta:
- They almost certainly use float32 for inference speed (their demo runs on CPU)
- Float32 limit of ~7K indices means memory addresses are bounded
- For a WASM VM, 7K memory cells is very limiting — they likely use some encoding trick
  (e.g., multi-head addressing where different heads cover different address ranges)
- Alternatively they might use bfloat16 or mixed precision

---

## Phase 2b: Breaking the Float32 Address Limit

**Result: Residual (bit-split) addressing extends range to 25M+ from just 2 attention heads.**

### Motivation
Phase 2 found ~7K addressable indices in float32. For WASM-scale execution (Phase 6), this is far too limiting. Explored three workarounds.

### Re-measured baseline
More rigorous testing found the safe breakpoint is actually ~4K (not ~7.3K as in Phase 2). Phase 2's measurement was optimistic because numpy's vectorized ops have better intermediate precision than the worst case.

### Approaches tested

| Approach | Mechanism | Addressable | Heads | Errors |
|----------|-----------|-------------|-------|--------|
| Standard parabolic | k=(2j, -j²) | ~4K | 1 | 0% up to limit |
| Offset parabolas | k=(2(j-c), -(j-c)²), tiled | 3K × N_heads | N | 0% with 3K segments |
| **Residual (bit-split)** | **block=addr//B, offset=addr%B** | **B² (=25M for B=5K)** | **2** | **0%** |

### Residual addressing detail
- Split address into (block_index, offset_within_block)
- Head A: parabolic lookup on block indices → selects which block
- Head B: parabolic lookup on offsets within selected block → selects entry
- FF layer combines both heads' outputs
- B=5000 is well within float32 safe range → 25M addressable range
- Stress tested: 330 addresses spanning 0..25M, zero errors

### Key insight
This is likely what Percepta uses. Their d_model=36 with 18 heads can dedicate 2 heads to block/offset addressing and still have 16 heads for other roles. The FF layer routing for the combination adds modest complexity.

### Impact on later phases
- Phase 5 (training): standard parabolic is sufficient — toy programs won't need >4K stack depth
- Phase 6 (WASM): residual addressing is the path to realistic memory sizes
- Training challenge: the model must learn the bit-split decomposition, which is a harder optimization target than plain parabolic

---

## Phase 3: Cumulative Sum via Attention

**Result: Surprisingly robust. No integer errors at 100K steps even in float32.**

### The mechanism:
- All keys identical → softmax gives uniform weights → attention output = mean(values[0:t])
- Multiply by position t → recovers cumulative sum

### Numerical stability:
| N       | float32 max_err | float32 int_errors | float64 max_err |
|---------|----------------|--------------------|-----------------|
| 1,000   | 0.0000         | 0                  | 0.0000          |
| 10,000  | 0.0001         | 0                  | 0.0000          |
| 100,000 | 0.0010         | 0                  | 0.0000          |

Even with realistic WASM-like deltas (±1, ±2), zero integer errors at 100K in float32.

### Alternative approach discovered:
Sequential lookback: depth[t] = depth[t-1] + delta[t]
- Only needs 1 attention head attending to position t-1
- O(1) per step (vs O(n) for the mean×t trick under standard attention)
- Equally stable in practice
- Percepta likely uses this rather than the mean×t method described in the blog

---

## Summary: Primitive Viability

| Primitive          | Works? | Limit                    | Notes                          |
|-------------------|--------|--------------------------|--------------------------------|
| Hull query O(log t)| ✓     | Always (given structure)  | Ternary search, not hull scan  |
| Parabolic indexing | ✓     | ~7K (f32), ~67M (f64)   | Biggest practical constraint   |
| Cumulative sum     | ✓     | 100K+ (f32), unlimited (f64) | Very robust                |
| Overwrite/recency  | ✓     | Needs bias trick         | Works without breaking others  |

---

## Phase 4: Minimal Stack Machine via Attention

**Result: YES — the primitives compose. 10/10 test programs execute identically in the attention executor and the reference interpreter.**

### Architecture

Two parallel executors: a traditional `ReferenceExecutor` (normal stack machine) and an `AttentionExecutor` that uses ONLY attention primitives:

| Component | Primitive Used | Role |
|-----------|---------------|------|
| Program memory | Parabolic indexing | Fetch opcode/arg at instruction pointer |
| Stack memory | Parabolic indexing + recency bias | Read/write stack values by address |
| Instruction pointer | Sequential lookback | Counts completed steps |
| Stack pointer | Sequential lookback | Tracks push/pop deltas |

### Instruction set: PUSH, POP, ADD, DUP, HALT

### Trace format
Each instruction emits 4 tokens: `[OPCODE, ARG, SP, TOP]`
Full sequence: `[PROG_START, op0, arg0, ..., PROG_END, TRACE_START, step0_tokens, ...]`

### Test results

| Test | Program | Expected | Result |
|------|---------|----------|--------|
| basic_add | PUSH 3, PUSH 5, ADD, HALT | 8 | ✓ |
| push_halt | PUSH 42, HALT | 42 | ✓ |
| push_pop | PUSH 10, PUSH 20, POP, HALT | 10 | ✓ |
| dup_add | PUSH 7, DUP, ADD, HALT | 14 | ✓ |
| multi_add | PUSH 1..3, ADD, ADD, HALT | 6 | ✓ |
| stack_depth | PUSH 1..3, POP, POP, HALT | 1 | ✓ |
| overwrite | PUSH 5, POP, PUSH 9, HALT | 9 | ✓ |
| complex | PUSH 10,20,30, ADD, DUP, ADD, HALT | 100 | ✓ |
| many_pushes | PUSH 1..10, ADD×9, HALT | 55 | ✓ |
| alternating | interleaved PUSH/ADD, HALT | 10 | ✓ |

All traces match token-for-token between reference and attention executors.

### Key findings

1. **Parabolic indexing is the workhorse.** Used for two independent memory systems (program fetch AND stack addressing) with no interference. The key insight: different address spaces just need different key populations.

2. **Recency bias is essential for stack correctness.** The `overwrite` test proves this — PUSH 5 then POP then PUSH 9 both write to stack address 1. Without recency bias (ε·write_count in the y-component), the lookup would average the two writes. With it, the most recent write wins.

3. **Sequential lookback is simpler than cumsum for IP/SP tracking.** Phase 3 found that attending to position t-1 is both simpler and cheaper than the mean×t trick. Phase 4 confirms: IP and SP just need "previous value + delta."

4. **The FF layer is the hard part.** The attention heads have clean, separable roles. But the feed-forward network must do opcode-dependent routing:
   - PUSH → route ARG to stack write
   - ADD → read two stack values, compute sum, write result
   - POP → just decrement SP
   - This conditional logic requires either a deep-enough FF network or a second transformer layer where Layer 2 can condition on Layer 1's opcode retrieval.

5. **Head assignment for a real transformer.** Minimum 4 heads (IP fetch, ARG fetch, stack read, SP track). A realistic implementation needs 6: add an opcode-recall head (so later tokens in a step know which opcode they're executing) and a secondary stack-read head (ADD needs two stack values simultaneously).

### Implications for Percepta's claims

- The basic claim checks out: attention can implement addressable memory and state tracking
- Their d_model=36, n_heads=18, n_layers=7 architecture is probably not all functional — many heads likely provide redundancy, error correction, or handle edge cases in WASM execution
- The FF routing for complex opcodes (WASM has ~200) is where most of the model capacity goes — not the attention lookups
- float32 precision limit (~7K addresses) from Phase 2 constrains their addressable memory space, confirming our prediction that they need an encoding trick for larger WASM memories

### What's NOT proven yet

- We built the attention executor as a Python simulation, not as actual PyTorch weight matrices
- The simulation proves the information flow is correct, but doesn't prove a finite-width FF network can implement the routing
- Phase 5 (training) will test whether gradient descent discovers this structure on its own

---

## Updated Summary: All Phases

| Phase | Question | Answer | Key Constraint |
|-------|----------|--------|----------------|
| 1 | Does hull query scale O(log t)? | Yes | Ternary search required, not hull scan |
| 2 | Does parabolic indexing work? | Yes | float32 limit ~4K indices (revised down) |
| 2b | Can we extend the address limit? | Yes | Residual addressing: 25M from 2 heads |
| 3 | Is cumsum via attention stable? | Yes | 100K+ steps in float32 |
| 4 | Do the primitives compose? | Yes | FF routing is the bottleneck, not attention |

## Next: Phase 5 (Train a Micro-Executor)
- Generate (program, trace) dataset for the stack instruction set
- Train a d_model=36 transformer from scratch on next-token prediction
- Key question: does gradient descent discover the parabolic encoding?
- If it discovers a DIFFERENT structure that also works, that's more interesting than confirmation

## Files
- phase1_hull_cache.py — Hull cache benchmarks
- phase2_parabolic.py — Precision tests
- phase2b_address_limits.py — Extended addressing exploration
- phase3_cumsum.py — Cumulative sum tests
- phase4_stack_machine.py — Stack machine composition test
- viz/phase1-results.jsx — Phase 1 visualization (React)
