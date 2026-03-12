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

## Phase 6 (Stretch): WASM Fragment Execution

**Goal:** If Phases 4-5 succeed, attempt a small subset of real WASM opcodes (i32.const, i32.add, i32.store, i32.load, br_if).

**Tasks:**
1. Compile a trivial C function (e.g., fibonacci) to WASM via Emscripten or hand-write WASM bytecode.
2. Extend the tokenizer and trace format to cover the additional opcodes.
3. Train or hand-wire the model to execute the WASM fragment.

**Key question answered:** Does the approach scale beyond toy instruction sets toward something resembling the blog's claims?

**Estimated effort:** Significant. Only attempt if earlier phases go well.

---

## Success Criteria

| Phase | Success looks like |
|-------|-------------------|
| 1 | Clear log vs linear scaling plot with crossover point identified |
| 2 | 100% retrieval accuracy up to some numerical limit, limit characterized |
| 3 | Cumsum within ±1 of true value over 100K+ steps |
| 4 | Hand-wired transformer correctly executes 10+ test programs |
| 5 | Trained model achieves >99% token accuracy on held-out programs |
| 6 | Fibonacci(10) executes correctly inside the transformer |

## Dependencies & Risks

- **scipy.spatial** for convex hull operations — should be available, fall back to pure Python if not.
- **Numerical precision** is the biggest unknown. The parabolic encoding and cumsum tricks both depend on exact arithmetic in a float32 world. Phase 2 will reveal whether this is a real problem or a theoretical concern.
- **Hand-wiring weights (Phase 4)** is tedious and error-prone. The blog doesn't explain the full construction. Expect to reverse-engineer some of the design from the architectural constraints.
- **Training (Phase 5)** on CPU is fine for this model size, but if the loss landscape is tricky, may need more hyperparameter search than is comfortable in this environment.

## Recommended Execution Order

Start with Phases 1-3 in parallel (they're independent). Phase 4 depends on all three. Phase 5 depends on Phase 4's tokenizer/trace design but not on the hand-wired weights. Phase 6 is optional.

Phases 1-3 are the "can we reproduce the building blocks" check. Phase 4 is the "can we compose them" check. Phase 5 is the "is this learnable" check — and the most scientifically interesting.
