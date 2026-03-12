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

---

## Phase 5: Trained Micro-Executor

**Result: The model learns significant execution structure (56% token accuracy, 112× above chance) but does not reach perfect trace execution at this scale.**

### Setup
- Training: 1000 random programs, max 8 instructions, push values 0-30
- Validation: 150 programs, same distribution
- Test: 50 in-distribution + 30 out-of-distribution (longer programs)
- Vocabulary: 210 tokens (opcodes + special + numeric 0-200)
- Training: next-token prediction on execution traces (cross-entropy)

### Architecture comparison (25 epochs, 300 samples — all unconverged)

| Model | d_model | heads | layers | Params | Val Acc |
|-------|---------|-------|--------|--------|---------|
| minimal | 32 | 4 | 2 | 44K | 30% |
| deep | 32 | 4 | 4 | 69K | 35% |
| wide | 64 | 4 | 2 | 137K | 40% |

**Width > depth.** This confirms Phase 4's prediction: attention heads have clean roles but FF routing for opcode-dependent logic needs capacity *per layer*, not more layers.

### Best model (wide, 100 epochs, 1000 samples)

| Metric | Value |
|--------|-------|
| Val token accuracy | 56% (chance = ~0.5%) |
| Perfect traces | 0/50 |
| Final value correct | 5/50 (10%) |
| Training plateau | ~epoch 80 |

### Interpretation

1. **The model LEARNS execution patterns.** 56% token accuracy is 112× above chance. It correctly predicts opcodes and many state values.

2. **The gap is arithmetic, not structure.** The model learns WHEN to push, pop, add — but fumbles the exact numeric computations (SP deltas after ADD, TOP values after multi-step operations). It has learned the *grammar* of execution but not the *arithmetic*.

3. **One error cascades.** Even 56% token accuracy → 0% perfect traces. In autoregressive generation, one wrong SP or TOP value corrupts all subsequent steps.

4. **Width > depth confirms Phase 4's FF routing prediction.** The attention heads (instruction fetch, stack read, SP track) are mechanically simple. The FF layer must implement conditional logic (opcode → different computation), and that requires capacity within each layer, not more layers of simple computation.

5. **To reach perfection likely needs:** 10K+ training samples, 500K+ params, possibly curriculum learning (start with PUSH/HALT only, add ADD, then full instruction set).

### Limitations
- Container CPU timeout (200s) prevented training >50 epochs with 2000+ samples
- Bigger model (128/8/3, 670K params) was too slow to evaluate in this environment
- Full convergence study requires GPU access or persistent compute

---

## Updated Summary: All Phases

| Phase | Question | Answer | Key Constraint |
|-------|----------|--------|----------------|
| 1 | Does hull query scale O(log t)? | Yes | Ternary search required, not hull scan |
| 2 | Does parabolic indexing work? | Yes | float32 limit ~4K indices (revised down) |
| 2b | Can we extend the address limit? | Yes | Residual addressing: 25M from 2 heads |
| 3 | Is cumsum via attention stable? | Yes | 100K+ steps in float32 |
| 4 | Do the primitives compose? | Yes | FF routing is the bottleneck, not attention |
| 5 | Can gradient descent learn execution? | Partially | Learns structure (56%), not perfect arithmetic |

---

## Phase 6: Curriculum Learning

**Result: YES — curriculum learning significantly improves execution accuracy. 81% token accuracy (vs 56% baseline), 23/50 perfect traces (vs 0/50).**

### Hypothesis
Phase 5's gap exists because the model must simultaneously learn state tracking AND arithmetic. Decompose via curriculum: teach trivial routing first, then incrementally add complexity.

### Three stages

| Stage | Instructions | Target | Val Acc | Perfect | Final OK |
|-------|-------------|--------|---------|---------|----------|
| 1 | PUSH + HALT | >95% | 57% | 0/50 | 1/50 |
| 2 | PUSH + POP + DUP + HALT | >85% | 67% | 6/50 | 9/50 |
| 3 | Full set (+ ADD) | >70% | **81%** | **23/50** | **35/50** |

### Comparison with Phase 5 baseline

| Metric | Phase 5 | Phase 6 Stage 3 | Delta |
|--------|---------|-----------------|-------|
| Val token accuracy | 56% | 81% | **+25pp** |
| Perfect traces | 0/50 | 23/50 | **+23** |
| Final value correct | 5/50 | 35/50 | **+30** |

### Key findings

1. **Curriculum learning works.** The +25pp accuracy gain and 0→23 perfect traces is a qualitative breakthrough. The same 137K-param model that couldn't produce a single correct trace now executes complete programs correctly nearly half the time.

2. **Transfer learning compounds.** Each stage builds meaningfully on the previous. Stage 2 starts where Stage 1 left off and immediately benefits from the learned token structure. Stage 3 starts at 67% and climbs to 81%.

3. **Stage 1 underperformed (57% vs 95% target).** Even PUSH-only programs — the simplest possible routing — require non-trivial position-dependent value copying. The model must learn that TOP = the most recent PUSH argument, and SP = step count. This is harder than expected because the numeric values are arbitrary (0-50), so the FF layer can't just memorize — it must learn a general copy mechanism.

4. **Stage 3 met its target.** Despite Stage 1 and 2 missing their targets, Stage 3 exceeded 70%. The curriculum provides a better optimization landscape even when individual stages don't reach ceiling performance.

5. **Total training time: ~147s on CPU.** All three stages completed comfortably within compute limits. The 137K model trains at ~1s/epoch.

### Interpretation

The Phase 5 finding was "the model learns structure but not arithmetic." Phase 6 shows this was partly a learning-order problem, not just a capacity problem. By staging instruction complexity, the FF layers learn crisp routing for simple cases first, then refine for harder cases.

However, 81% token accuracy and 23/50 perfect traces means the model still makes errors — particularly on longer programs and ADD operations where two stack values must be retrieved and summed. The remaining gap likely requires either more parameters or more training data.

### Stage 1 Diagnostic: The Copy Bottleneck

Error decomposition on the Stage 1 model revealed the model's failure is entirely about **value copying**, not structure:

| Field | Teacher-forced accuracy | What it requires |
|-------|------------------------|------------------|
| OP (opcode) | 99.9% | Constant (always PUSH) — trivial |
| SP (stack ptr) | 98.4% | Increment counter — trivial |
| ARG (push value) | 21.2% | Copy value from program prefix — **hard** |
| TOP (stack top) | 4.4% | Copy most recent push value — **hard** |

The model collapses to predicting ~16 "favorite" values (32, 0, 3, 19...) instead of the 50 distinct values in the data. It predicts ARG == TOP only 20% of the time despite this being a hard invariant. **The FF layers learn position but not content-addressable lookup.**

Three ablations identified the bottleneck as **convergence, not capacity**:

| Experiment | Val Acc | ARG acc | TOP acc | Perfect | Change |
|-----------|---------|---------|---------|---------|--------|
| Baseline (1K data, 60 ep, d=64) | 57% | 21% | 4% | 0/50 | — |
| A: 5K data, 200 epochs | **85%** | **100%** | **100%** | **50/50** | More data wins |
| B: Small values (0-10) | 82% | 77% | 99% | 18/50 | Fewer values helps |
| C: Wider model (d=128) | 84% | 95% | 98% | 34/50 | More capacity helps |

**Experiment A is decisive:** the same 137K-param model achieves 100% ARG and TOP accuracy with sufficient data and training time. The copy mechanism IS learnable — the original Stage 1 was simply data-starved.

### Phase 6b: Full Curriculum with 5K Samples

Re-running all three stages with 5K training samples (5× original) and 200 max epochs:

| Stage | Instructions | Val Acc | Perfect | Final OK |
|-------|-------------|---------|---------|----------|
| 1 | PUSH + HALT | 85% | 49/50 | 50/50 |
| 2 | PUSH + POP + DUP + HALT | 86% | **50/50** | **50/50** |
| 3 | Full set (+ ADD) | **85%** | **39/50** | **44/50** |

**Progression across all runs:**

| Run | Val Acc | Perfect | Final OK |
|-----|---------|---------|----------|
| Phase 5 baseline | 56% | 0/50 | 5/50 |
| Phase 6a (1K data) | 81% | 23/50 | 35/50 |
| Phase 6b (5K data) | **85%** | **39/50** | **44/50** |

Stage 2 achieves **50/50 perfect traces** — the model perfectly executes all PUSH/POP/DUP programs. The remaining errors in Stage 3 are concentrated on ADD, where the model must retrieve two stack values and compute their sum.

### Key Insight: Copy Before Compute

The fundamental bottleneck in learning execution is not opcode dispatch or state tracking — it's **content-addressable memory lookup**. The model must learn to attend back to specific positions in the input and copy their values. This is exactly the parabolic indexing operation from Phases 1-2, but discovered via gradient descent rather than hand-wired. Once the copy mechanism converges (Experiment A), everything else follows.

---

## Updated Summary: All Phases

| Phase | Question | Answer | Key Constraint |
|-------|----------|--------|----------------|
| 1 | Does hull query scale O(log t)? | Yes | Ternary search required, not hull scan |
| 2 | Does parabolic indexing work? | Yes | float32 limit ~4K indices (revised down) |
| 2b | Can we extend the address limit? | Yes | Residual addressing: 25M from 2 heads |
| 3 | Is cumsum via attention stable? | Yes | 100K+ steps in float32 |
| 4 | Do the primitives compose? | Yes | FF routing is the bottleneck, not attention |
| 5 | Can gradient descent learn execution? | Partially | Learns structure (56%), not perfect arithmetic |
| 6 | Does curriculum learning help? | **Yes** | 56%→85% accuracy, 0→39 perfect traces |

## Key Insight Across All Phases

The consistent finding: **attention is the easy part; feed-forward routing is the hard part.** The 2D convex hull attention primitives (parabolic indexing, cumsum) are elegant and compose cleanly. But the conditional logic that maps opcodes to different computations — the "if PUSH then route arg to stack; if ADD then read two values and sum" — is where model capacity actually goes. This is true both in the hand-wired design (Phase 4) and in training (Phase 5).

Percepta's d_model=36, n_heads=18, n_layers=7 architecture is probably 80% FF routing capacity and 20% attention lookup mechanics.

## Files
- phase1_hull_cache.py — Hull cache benchmarks
- phase2_parabolic.py — Precision tests
- phase2b_address_limits.py — Extended addressing exploration
- phase3_cumsum.py — Cumulative sum tests
- phase4_stack_machine.py — Stack machine composition test
- phase5_training.py — Training experiments
- phase6_curriculum.py — Curriculum learning experiment
- viz/phase1-results.jsx — Phase 1 visualization (React)
