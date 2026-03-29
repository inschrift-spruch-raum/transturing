# src/ — Mojo Backend & Benchmarks

Mojo port of the 55-opcode compiled executor, plus benchmarks comparing Mojo vs Python performance.

## STRUCTURE

```
src/
├── executor.mojo        # Mojo port of full 55-opcode executor
├── percepta_exec        # Compiled Mojo binary (gitignored)
├── .gitignore           # Ignores percepta_exec
├── benchmark.py         # Mojo vs NumPy micro-benchmarks (137 lines)
├── benchmarks.py        # Substantial programs: FNV-1a, bubble sort, primes (408 lines)
├── llm_vs_native.py     # LLM-executor vs native Python comparison (221 lines)
└── run_mojo_tests.py    # Mojo executor test runner (450 lines)
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Understand Mojo executor | `executor.mojo` | Direct port of Python ISA |
| Run Mojo benchmarks | `src/benchmark.py` | Requires Mojo installed |
| Run Mojo tests | `src/run_mojo_tests.py` | Full ISA test suite via Mojo binary |
| Compare Python vs Mojo | `src/llm_vs_native.py` | Honest perf comparison |
| Find benchmark programs | `src/benchmarks.py` | FNV-1a hash, bubble sort, sieve |

## CONVENTIONS

- **Mojo binary built from source** — `executor.mojo` compiles to `percepta_exec`. Not pre-built.
- **Falls back to Python** — If Mojo unavailable, all scripts degrade gracefully to Python executor.
- **Imports from root** — `isa`, `executor`, `programs` imported directly (root on sys.path).
- **Subprocess for Mojo** — Mojo binary invoked via `subprocess.run()` with JSON input/output.

## ANTI-PATTERNS

- **Do NOT commit `percepta_exec`** — It's in `.gitignore`. Build locally.
- **Mojo bench subprocess timeout: 300s** — Long-running benchmarks need extended timeout.
- **Mojo is NOT pre-installed** — Requires skill at `/mnt/skills/user/llm-as-computer/` or manual install.
