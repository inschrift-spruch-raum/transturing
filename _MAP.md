# llm-as-computer/
*Core files + documentation. Phase exploration scripts in [dev/phases/](./dev/phases/).*

## Subdirectories

- **dev/phases/** — Research phase scripts (phases 1–20) and result JSON files
- **src/** — Mojo implementation experiments
- **viz/** — React visualizations

## Core Modules

### isa.py — Instruction Set Architecture
- `Instruction` :24 — Dataclass: opcode + optional value
- `program(*instrs)` :37 — Convenience constructor
- `TraceStep` :71 — Single execution step record
- `Trace` :83 — Full execution trace
- Constants: `D_MODEL=51`, `TOKENS_PER_STEP=4`, dimension mappings (DIM_OPCODE, DIM_VALUE, etc.)
- Opcodes: PUSH, POP, ADD, SUB, DUP, HALT, JZ, JNZ, NOP, SWAP, OVER, ROT, MUL, DIVMOD, REM, EQZ, LT, GT, LE, GE, EQ, NE, AND, OR, XOR, SHL, SHR, CLZ, POPCNT, MAX, ABS, CLAMP, SELECT

### executor.py — Compiled Transformer Executor
- `NumPyExecutor` :55 — NumPy reference implementation (Phase 11)
- `CompiledModel(nn.Module)` :456 — Real PyTorch compiled transformer (Phase 12). Weight matrices compiled from ISA.
- `TorchExecutor` :881 — High-level executor wrapping CompiledModel

### programs.py — Test Programs & Algorithm Library
- 10 basic test programs (test_basic through test_alternating)
- `ALL_TESTS` :87 — Registry of all basic tests
- Algorithm generators: `make_fibonacci`, `make_power_of_2`, `make_sum_1_to_n`, `make_multiply`, `make_is_even`, `make_factorial`, `make_gcd`
- Native-op generators: `make_native_multiply`, `make_native_divmod`, `make_native_remainder`, `make_native_is_even`, `make_native_max`, `make_native_abs`, `make_native_clamp`
- Bitwise: `make_bitwise_binary`, `make_popcount_loop`

### assembler.py — Structured WASM → ISA Compiler
- `compile_structured(wasm_instrs)` :52 — Compiles structured WASM (block/loop/if/br) to flat ISA instructions with jump resolution

### wat_parser.py — WAT Text Format Parser
- `parse_wat(text, append_halt=True)` :508 — Parses WebAssembly text format into ISA instructions. Supports locals, globals, memory, structured control flow.

### c_pipeline.py — C → WAT → ISA Pipeline
- `compile_c_to_wat(c_code, ...)` :168 — Compiles C source to WAT via Emscripten
- `compile_c(c_code, ...)` :253 — Full C → ISA compilation
- `compile_and_run(c_code, ...)` :395 — Compile + execute end-to-end

### test_consolidated.py — Integration Tests
- `test_numpy_equivalence()` :48 — NumPy executor vs reference interpreter
- `test_torch_equivalence()` :253 — PyTorch compiled model vs reference
- `test_new_np_vs_new_pt()` :356 — Cross-executor consistency

### test_wat_parser.py — WAT Parser Tests
- Test cases: Phase 4 programs, Fibonacci, factorial, bubble sort, sum, power, if/else, nested blocks, arithmetic, comparison chains, bitwise ops

## Documentation

- **README.md** — Project overview and status
- **WRITEUP.md** — Full narrative writeup
- **FINDINGS.md** — Detailed per-phase findings
- **RD-PLAN.md** — Research plan and evolution
- **CLAUDE.md** — Project instructions for Claude Code
