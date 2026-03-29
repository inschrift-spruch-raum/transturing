# dev/phases/ — Research Phase Scripts

20 standalone exploration scripts + 6 result JSON files. Each phase isolates a transformer primitive, tests it numerically, then composes with prior phases.

## STRUCTURE

```
dev/phases/
├── phase1_hull_cache.py        # Convex hull KV cache
├── phase2_parabolic.py         # Parabolic key encoding
├── phase2b_address_limits.py   # Extended addressing past float32
├── phase3_cumsum.py            # Cumulative sum via attention
├── phase4_stack_machine.py     # Hand-wired stack machine (imported by 11-14)
├── phase5_training.py          # Trained micro-executor
├── phase6_curriculum.py        # Curriculum learning
├── phase7_percepta_arch.py     # Percepta architecture training
├── phase8_microop_traces.py    # Micro-op diagnostics
├── phase9_weighted_arithmetic.py   # Weighted loss experiments
├── phase10_digit_decomposition.py  # Digit decomposition (exploratory)
├── phase11_compile_executor.py     # Compiled executor (numpy)
├── phase12_percepta_model.py       # PyTorch compiled transformer
├── phase13_isa_completeness.py     # ISA completeness + SWAP/OVER/ROT
├── phase14_extended_isa.py         # Full 55-opcode ISA (2885 lines — largest)
├── phase15_local_variables.py      # LOCAL.GET/SET/TEE
├── phase16_linear_memory.py        # Heap memory LOAD/STORE
├── phase17_function_calls.py       # CALL/RETURN + recursive factorial
├── phase18_integration_tests.py    # Bubble sort, recursive fib, multi-function
├── phase19_structured_assembler.py # Structured control flow testing
├── phase20_type_masking_tests.py   # i32 overflow masking tests
├── phase6_results.json ... phase10_results.json  # Training result data
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Find a phase's test functions | `phaseN_*.py` → `test_*` functions | Each has hand-rolled pass/fail counters |
| Find phase findings | `../FINDINGS.md` | Comprehensive per-phase analysis |
| Find import chain for a phase | Top of file, `import phaseN_*` | Phases 11-14 import from phase4 |
| Find training results | `phaseN_results.json` | Phases 6, 7, 8, 9, 10 have JSON data |

## CONVENTIONS

- **Every phase file is self-contained** — own test harness, own `main()`, run directly: `uv run dev/phases/phaseN_*.py`
- **No cross-phase imports except phase4** — Phases 11-14 import `phase4_stack_machine` for reference trace generation. `test_consolidated.py` imports `phase14_extended_isa` directly.
- **sys.path manipulation required** — Phase files add `os.path.dirname(os.path.abspath(__file__))` to `sys.path` for local imports.
- **Two executor classes per compiled phase** — `PhaseNExecutor` (numpy) + `PhaseNPyTorchExecutor` (PyTorch), both must produce identical traces.
- **Test pattern:** `passed = 0; failed = 0; ... ; print(f"{passed} passed, {failed} failed"); sys.exit(1 if failed else 0)`

## ANTI-PATTERNS

- **Do NOT change phase4 exports** — Phases 11-14 depend on `ReferenceExecutor`, `program`, `Instruction` from phase4.
- **Do NOT change phase14 exports** — `test_consolidated.py` imports `Phase14Executor` and `Phase14PyTorchExecutor` directly.
- **Do NOT assume training phases (5-10) are authoritative** — Training path was a productive wrong turn. Compiled phases (11+) are the correct approach.
- **Phase 14 is 2885 lines** — Use `docs/reference/api.md` for function index, then targeted line ranges.
