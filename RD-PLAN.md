# R&D Plan: Prototyping 2D Convex Hull Attention for In-Model Execution

**Context:** The Percepta blog post "Can LLMs Be Computers?" claims that restricting transformer attention heads to 2D enables O(log t) decoding via convex hull queries, making million-step execution traces feasible inside a transformer. No code or weights are published. This plan tests the core claims from first principles.

**Environment:** Claude skills container — CPU-only, Python 3.12, PyTorch available, scipy available. No GPU. This is fine; the whole point of their approach is CPU-friendly execution.

---

## Phase 1: Convex Hull KV Cache — Does the Geometry Work?

**Goal:** Validate that 2D convex hull lookups actually achieve O(log t) per query vs O(t) brute force, and measure the constant-factor overhead.

**Tasks:**
1. Implement a `BruteForceKVCache` that stores 2D keys and values, answers hard-max attention queries by linear scan.
2. Implement a `HullKVCache` that maintains an incremental convex hull (Graham scan or Andrew's monotone chain), answers max-dot-product queries via binary search on the hull.
3. Benchmark both on synthetic key streams of length 1K → 1M, measuring wall-clock time per query.
4. Plot scaling curves. The blog claims log vs linear; verify this and measure where the crossover happens (hull maintenance has overhead that may dominate at small t).

**Key question answered:** Is the convex hull approach actually faster in practice on CPU, and at what trace length does it break even?

**Estimated effort:** ~200 lines of Python, 1 session.

---

## Phase 2: Parabolic Key Encoding — Does Index Lookup Work?

**Goal:** Verify that the encoding k_j = (2j, -j²) with query q = (i, 1) correctly retrieves the value stored at index i via hard-max attention.

**Tasks:**
1. Implement the parabolic encoding scheme.
2. Test exact retrieval: store values at indices 0..N, query each index, verify 100% accuracy.
3. Test with overwrites: store at index i multiple times (simulating memory writes), verify that the *most recent* write wins. This requires augmenting keys with a recency dimension — examine whether 2D is sufficient or if they need a tie-breaking scheme.
4. Test numerical stability: as j grows large, -j² dominates. Check at what index range floating-point precision breaks the scheme. Try float32 vs float64.

**Key question answered:** Does the encoding actually work reliably as a memory lookup primitive, and what are its numerical limits?

**Estimated effort:** ~100 lines, quick.

---

## Phase 2b: Extended Addressing (added after Phase 4)

**Goal:** The float32 limit (~4K addresses, revised from initial ~7K) constrains WASM-scale execution. Find encoding tricks to extend addressable range.

**Result:** Residual (bit-split) addressing solves this. Split addr = (block, offset), each resolved by a separate head. B=5000 → 25M addressable range from 2 heads, zero errors.

**Impact:** Phase 5 can use standard parabolic (toy programs stay well within 4K). Phase 6 should use residual addressing for realistic WASM memory.

---

## Phase 3: Cumulative Sum Attention — Tracking Running State

**Goal:** Verify the claim that attention can compute cumulative sums (used for instruction pointer, stack depth, etc.).

**Tasks:**
1. Implement the uniform-key trick: all keys identical, values are deltas, attention averages all values, multiply by t to recover the sum.
2. Test on synthetic delta streams simulating stack push/pop (+1/-1 deltas).
3. Measure numerical drift over long sequences — cumulative sum via averaging will accumulate float errors.
4. Check whether this composes with the hull cache or requires separate heads (it should, since cumsum uses 0D/1D keys).

**Key question answered:** Is attention-based cumulative sum numerically viable for long traces?

**Estimated effort:** ~80 lines, quick.

---

## Phase 4: Minimal Stack Machine via Attention

**Goal:** Build a hand-wired transformer (weights set analytically, not trained) that executes a trivial instruction set using the primitives from Phases 1-3.

**Instruction set (deliberately minimal):**
- `PUSH <val>` — push a value onto the stack
- `POP` — pop top of stack
- `ADD` — pop two, push sum
- `HALT` — stop

**Tasks:**
1. Design the token vocabulary and trace format. Each instruction → a fixed number of tokens (they claim ≤5).
2. Assign attention heads to roles: stack-top lookup (parabolic keys), stack-depth tracking (cumsum), instruction-pointer tracking (cumsum).
3. Hand-wire the weight matrices (W_Q, W_K, W_V, W_O, FF weights) so the transformer produces the correct next token given the trace prefix.
4. Run the model on test programs: `PUSH 3, PUSH 5, ADD, HALT` → should produce trace ending with value 8.
5. Use the HullKVCache from Phase 1 for decoding.

**This is the hard phase.** The blog glosses over how attention heads compose across layers to implement control flow. Expect to iterate.

**Key question answered:** Can you actually hand-wire a transformer to execute even a trivial program, and does the hull cache make it fast?

**Estimated effort:** ~500 lines, likely 2-3 sessions with debugging.

---

## Phase 5: Trained Micro-Executor

**Goal:** Instead of hand-wiring, *train* a small transformer to learn execution of the Phase 4 instruction set from trace examples.

**Tasks:**
1. Generate a dataset of (program, execution_trace) pairs for the minimal stack machine.
2. Train a small transformer (start with d_model=32, n_heads=4, n_layers=2 — matching Phase 4's head assignment) on next-token prediction over traces.
3. Evaluate: does it learn perfect execution? On how many instructions? Where does it break?
4. Compare decoding speed: standard KV cache vs hull cache on the trained model.
5. Inspect learned attention patterns — do the heads discover the parabolic encoding or something else?

**Updated notes from Phase 4:**
- The FF routing for opcode-dependent logic is the hardest part to learn — the attention patterns are clean.
- Standard parabolic encoding suffices (programs stay within 4K stack depth).
- Start small: d_model=32 should be enough given Phase 4's 4-head decomposition.
- The scientifically interesting outcome: does it discover parabolic encoding, or something else that also works?

**Key question answered:** Can a tiny transformer actually *learn* to be an executor from data, and does it converge to the theoretically optimal attention structure?

**Estimated effort:** ~400 lines + training time. CPU-only training of a 32-dim model should be fast (minutes, not hours). This is the most interesting phase — it tests whether the theoretical construction is learnable or purely a proof of existence.

---

## Phase 6: Curriculum Learning (Complete)

**Goal:** Test whether staged instruction complexity improves learnability over Phase 5's flat training.

**Result:** YES. Curriculum learning improves accuracy from 56% → 85%, perfect traces from 0/50 → 39/50.

**Key findings:**
- **Copy bottleneck (solved):** Model couldn't copy values from program memory with 1K samples. 5K samples + 200 epochs → 100% copy accuracy. Convergence, not capacity.
- **Non-arithmetic execution (solved):** Stage 2 (PUSH/POP/DUP) achieves 50/50 perfect traces.
- **Two-operand retrieval (frontier):** ADD requires reading two different stack values simultaneously. Model gets 97% on DUP+ADD but 3% on PUSH a, PUSH b, ADD (a≠b). Doubling heads (h=4→8) at same d_model doesn't help.
- **Architectural limit at ~85% val accuracy** for d=64, 2-layer, regardless of head count.

**Files:** phase6_curriculum.py, phase6_results.json, phase6b_results.json

---

## Phase 7 (Next): Two-Operand Retrieval

**Goal:** Crack the ADD a+b problem — enable the model to read two different stack values and compute their sum.

**Hypotheses to test (in priority order):**

1. **More layers (L=4):** Layer 1 retrieves one operand into the residual stream, layer 2 retrieves the second and computes. The current L=2 architecture may not have enough sequential depth to do retrieve-then-retrieve-then-compute.

2. **Larger d_model with proportional heads (d=128, h=8):** Keep d_head=16 while providing enough heads. The d=128/h=8 experiment was started but not completed — would give 476K params.

3. **SP-relative positional encoding:** Add stack-pointer-relative position information to the embedding so attention can more easily address "the value 2 positions below SP" without learning it from scratch.

4. **Intermediate trace tokens:** Add a "partial result" token in the ADD trace step — e.g., `[ADD, operand_a, operand_b, sum, sp, top]` instead of `[ADD, 0, sp, top]`. This gives the model a chance to retrieve each operand into a separate token before combining.

**Key question answered:** Is the two-operand retrieval problem solvable with scale (more layers/width), or does it require architectural changes?

---

## Phase 8 (Stretch): WASM Fragment Execution

**Goal:** If Phase 7 succeeds at perfecting the toy instruction set, attempt a small subset of real WASM opcodes (i32.const, i32.add, i32.store, i32.load, br_if).

**Prerequisites:** Near-perfect execution of the current 5-opcode instruction set. Phase 2b's residual addressing for larger memory.

**Key question answered:** Does the approach scale beyond toy instruction sets toward the blog's claims?

---

## Success Criteria

| Phase | Success looks like | Status |
|-------|-------------------|--------|
| 1 | Clear log vs linear scaling plot with crossover point identified | Done |
| 2 | 100% retrieval accuracy up to some numerical limit, limit characterized | Done |
| 3 | Cumsum within ±1 of true value over 100K+ steps | Done |
| 4 | Hand-wired transformer correctly executes 10+ test programs | Done |
| 5 | Trained model achieves >99% token accuracy on held-out programs | Partial (85%) |
| 6 | Curriculum learning improves over Phase 5 baseline | Done (56%→85%) |
| 7 | Model correctly computes ADD a+b for arbitrary a≠b | Open |
| 8 | Fibonacci(10) executes correctly inside the transformer | Stretch |

## Dependencies & Risks

- **scipy.spatial** for convex hull operations — should be available, fall back to pure Python if not.
- **Numerical precision** is the biggest unknown. The parabolic encoding and cumsum tricks both depend on exact arithmetic in a float32 world. Phase 2 will reveal whether this is a real problem or a theoretical concern.
- **Hand-wiring weights (Phase 4)** is tedious and error-prone. The blog doesn't explain the full construction. Expect to reverse-engineer some of the design from the architectural constraints.
- **Training (Phase 5)** on CPU is fine for this model size, but if the loss landscape is tricky, may need more hyperparameter search than is comfortable in this environment.

## Recommended Execution Order

Phases 1-3 (done): Building blocks. Phase 4 (done): Composition. Phase 5 (done): Learnability baseline. Phase 6 (done): Curriculum learning + deep diagnostics.

**Phase 7 is the critical next step.** The two-operand retrieval problem is cleanly isolated and testable. Hypothesis 1 (more layers) is the quickest to test — just change n_layers from 2 to 4 and re-run the Phase 6b curriculum. If that doesn't work, hypothesis 4 (intermediate trace tokens) is the most architecturally interesting — it changes what the model needs to learn rather than just scaling up.

**The overarching research question has shifted:** We've proven that transformers CAN learn execution (Phases 4-6). The question is now: what's the minimum architecture that achieves PERFECT execution? The answer will tell us how Percepta's d=36/h=18/L=7 architecture is partitioned between lookup mechanics and routing capacity.
