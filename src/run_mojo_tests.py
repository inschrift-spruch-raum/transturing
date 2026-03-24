"""Runner: invoke the Mojo executor binary and compare against NumPyExecutor.

Usage:
    python src/run_mojo_tests.py           # run all tests
    python src/run_mojo_tests.py --bench   # include benchmark
    python src/run_mojo_tests.py -v        # verbose (show PASS lines)

The Mojo binary (percepta_exec) must be built first:
    cd src && mojo build executor.mojo -o percepta_exec
"""

import subprocess
import sys
import os
import time

# Add parent dir to path so we can import isa, executor, programs, etc.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from isa import (
    Instruction,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT,
    OP_SUB, OP_JZ, OP_JNZ, OP_NOP,
    OP_EQ, OP_LT_S, OP_GT_S,
    OP_AND, OP_OR, OP_XOR,
    OP_SHL, OP_SHR_U, OP_ROTL, OP_ROTR,
    OP_DIV_S, OP_REM_S,
    OP_TRAP,
)
from executor import NumPyExecutor
from programs import (
    ALL_TESTS,
    make_fibonacci, make_power_of_2, make_sum_1_to_n, make_multiply, make_is_even,
    make_native_multiply, make_native_divmod, make_native_remainder,
    make_native_is_even, make_factorial, make_gcd,
    make_compare_eqz, make_compare_binary,
    make_native_max, make_native_abs, make_native_clamp,
    make_bitwise_binary, make_popcount_loop, make_bit_extract,
    make_native_clz, make_native_ctz, make_native_popcnt,
    make_native_abs_unary, make_native_neg,
    make_select, make_select_max, make_log2_floor, make_is_power_of_2,
)


BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "percepta_exec")


# ─── Core runner ─────────────────────────────────────────────────

def instr_to_tokens(prog: list) -> list[str]:
    """Flatten program to [op, arg, op, arg, ...] strings."""
    tokens = []
    for instr in prog:
        tokens.append(str(instr.op))
        tokens.append(str(instr.arg))
    return tokens


def run_mojo(prog: list) -> tuple[int | None, list[tuple]]:
    """Run prog via Mojo binary. Returns (result, [(op,arg,sp,top), ...])."""
    tokens = instr_to_tokens(prog)
    r = subprocess.run([BINARY] + tokens, capture_output=True, text=True, timeout=10)
    trace, final = [], None
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("RESULT:"):
            final = int(line.split(":")[1].strip())
        elif line:
            parts = line.split()
            if len(parts) == 4:
                trace.append(tuple(int(x) for x in parts))
    return final, trace


def run_numpy(prog: list) -> tuple[int | None, list[tuple]]:
    """Run prog via NumPyExecutor. Returns (result, [(op,arg,sp,top), ...])."""
    tr = NumPyExecutor().execute(prog)
    steps = [(s.op, s.arg, s.sp, s.top) for s in tr.steps]
    final = steps[-1][3] if steps else 0
    return final, steps


# ─── Comparison helpers ───────────────────────────────────────────

def compare_program(name: str, prog: list, verbose: bool = False,
                    expect_trap: bool = False) -> bool:
    """Run both executors and check for trace-level match."""
    try:
        mojo_result, mojo_trace = run_mojo(prog)
        np_result,   np_trace   = run_numpy(prog)
    except subprocess.TimeoutExpired:
        print(f"  FAIL [{name}]: Mojo timed out")
        return False
    except Exception as e:
        print(f"  FAIL [{name}]: exception: {e}")
        return False

    if expect_trap:
        mojo_trapped = bool(mojo_trace) and mojo_trace[-1][0] == OP_TRAP
        np_trapped   = bool(np_trace)   and np_trace[-1][0]   == OP_TRAP
        ok = mojo_trapped and np_trapped
        if not ok:
            print(f"  FAIL [{name}]: trap mismatch — mojo_trapped={mojo_trapped}, numpy_trapped={np_trapped}")
        elif verbose:
            print(f"  PASS [{name}]: both trapped as expected")
        return ok

    if mojo_result != np_result:
        print(f"  FAIL [{name}]: result mismatch — mojo={mojo_result}, numpy={np_result}")
        if verbose:
            for i, (ms, ns) in enumerate(zip(mojo_trace[:5], np_trace[:5])):
                print(f"    step {i}: mojo={ms} numpy={ns}")
        return False

    if len(mojo_trace) != len(np_trace):
        print(f"  FAIL [{name}]: trace length mismatch — mojo={len(mojo_trace)}, numpy={len(np_trace)}")
        return False

    for i, (ms, ns) in enumerate(zip(mojo_trace, np_trace)):
        if ms != ns:
            print(f"  FAIL [{name}]: step {i} mismatch — mojo={ms}, numpy={ns}")
            return False

    if verbose:
        print(f"  PASS [{name}]: result={mojo_result}, {len(mojo_trace)} steps")
    return True


def run_group(label: str, tests: list, verbose: bool = False) -> tuple[int, int]:
    """Run a list of (name, prog[, expect_trap]) tests. Return (passed, total)."""
    passed = 0
    for entry in tests:
        name, prog = entry[0], entry[1]
        trap = len(entry) > 2 and entry[2]
        if compare_program(name, prog, verbose=verbose, expect_trap=trap):
            passed += 1
    total = len(tests)
    if not verbose:
        print(f"  {passed}/{total}")
    return passed, total


# ─── Build the full test list (mirrors test_consolidated.py) ──────

def build_all_tests() -> list[tuple]:
    """Return flat list of (name, prog) or (name, prog, True) for traps."""
    tests = []

    # Phase 4 base (from programs.ALL_TESTS)
    for name, fn in ALL_TESTS:
        prog, _ = fn()
        tests.append((name, prog))

    # Phase 11 extended
    tests += [
        ("sub_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_SUB), Instruction(OP_HALT)]),
        ("loop_countdown",
         [Instruction(OP_PUSH, 3), Instruction(OP_DUP),
          Instruction(OP_PUSH, 1), Instruction(OP_SUB),
          Instruction(OP_DUP), Instruction(OP_JNZ, 1),
          Instruction(OP_HALT)]),
        ("jz_taken",
         [Instruction(OP_PUSH, 0), Instruction(OP_JZ, 3),
          Instruction(OP_HALT),
          Instruction(OP_PUSH, 42), Instruction(OP_HALT)]),
    ]

    # Phase 13 algorithms
    for name, prog, _ in [
        ("fib(10)",      *make_fibonacci(10)),
        ("fib(7)",       *make_fibonacci(7)),
        ("sum(1..10)",   *make_sum_1_to_n(10)),
        ("power(2^5)",   *make_power_of_2(5)),
        ("mul(7,8)",     *make_multiply(7, 8)),
        ("is_even(10)",  *make_is_even(10)),
        ("is_even(7)",   *make_is_even(7)),
    ]:
        tests.append((name, prog))

    # Phase 14 arithmetic
    for name, prog, _ in [
        ("native_mul(7,8)",       *make_native_multiply(7, 8)),
        ("native_mul(0,5)",       *make_native_multiply(0, 5)),
        ("native_divmod(3,10)",   *make_native_divmod(3, 10)),
        ("native_rem(3,10)",      *make_native_remainder(3, 10)),
        ("factorial(5)",          *make_factorial(5)),
        ("factorial(10)",         *make_factorial(10)),
        ("gcd(12,8)",             *make_gcd(12, 8)),
        ("gcd(100,75)",           *make_gcd(100, 75)),
        ("native_is_even(42)",    *make_native_is_even(42)),
        ("native_is_even(15)",    *make_native_is_even(15)),
    ]:
        tests.append((name, prog))

    # Phase 14 comparisons
    for name, prog, _ in [
        ("eqz(0)",          *make_compare_eqz(0)),
        ("eqz(5)",          *make_compare_eqz(5)),
        ("eq(5,5)",         *make_compare_binary(OP_EQ, 5, 5)),
        ("lt_s(3,7)",       *make_compare_binary(OP_LT_S, 3, 7)),
        ("gt_s(10,2)",      *make_compare_binary(OP_GT_S, 10, 2)),
        ("max(3,7)",        *make_native_max(3, 7)),
        ("max(10,2)",       *make_native_max(10, 2)),
        ("abs(42)",         *make_native_abs(42)),
        ("clamp(5,0,10)",   *make_native_clamp(5, 0, 10)),
        ("clamp(15,0,10)",  *make_native_clamp(15, 0, 10)),
    ]:
        tests.append((name, prog))

    # Phase 14 bitwise
    for name, prog, _ in [
        ("and(0xFF,0x0F)",  *make_bitwise_binary(OP_AND,   0xFF,  0x0F)),
        ("or(0xF0,0x0F)",   *make_bitwise_binary(OP_OR,    0xF0,  0x0F)),
        ("xor(0xFF,0xFF)",  *make_bitwise_binary(OP_XOR,   0xFF,  0xFF)),
        ("shl(1,4)",        *make_bitwise_binary(OP_SHL,   1,     4)),
        ("shr_u(16,1)",     *make_bitwise_binary(OP_SHR_U, 16,    1)),
        ("rotl(1,1)",       *make_bitwise_binary(OP_ROTL,  1,     1)),
        ("rotr(2,1)",       *make_bitwise_binary(OP_ROTR,  2,     1)),
        ("popcount(255)",   *make_popcount_loop(255)),
        ("bit(0xFF,4)",     *make_bit_extract(0xFF, 4)),
    ]:
        tests.append((name, prog))

    # Phase 14 unary + parametric
    for name, prog, _ in [
        ("clz(0)",           *make_native_clz(0)),
        ("clz(255)",         *make_native_clz(255)),
        ("ctz(8)",           *make_native_ctz(8)),
        ("popcnt(255)",      *make_native_popcnt(255)),
        ("abs_native(42)",   *make_native_abs_unary(42)),
        ("abs_native(-7)",   *make_native_abs_unary(-7)),
        ("neg(5)",           *make_native_neg(5)),
        ("neg(-3)",          *make_native_neg(-3)),
        ("select(10,20,1)",  *make_select(10, 20, 1)),
        ("select(10,20,0)",  *make_select(10, 20, 0)),
        ("select_max(10,25)", *make_select_max(10, 25)),
        ("log2(8)",          *make_log2_floor(8)),
        ("ispow2(8)",        *make_is_power_of_2(8)),
        ("ispow2(7)",        *make_is_power_of_2(7)),
    ]:
        tests.append((name, prog))

    # Trap tests (div/rem by zero)
    tests += [
        ("div_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_DIV_S), Instruction(OP_HALT)], True),
        ("rem_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_REM_S), Instruction(OP_HALT)], True),
    ]

    return tests


def build_tier2_tests() -> list[tuple]:
    """Return Tier-2 programs from phase15-18 PHASE*_TESTS collections."""
    tests = []

    phase_modules = [
        ("phase15_local_variables",   "PHASE15_TESTS", "p15"),
        ("phase16_linear_memory",     "PHASE16_TESTS", "p16"),
        ("phase17_function_calls",    "PHASE17_TESTS", "p17"),
        ("phase18_integration_tests", "INTEGRATION_TESTS", "p18"),
    ]

    for mod_name, attr_name, prefix in phase_modules:
        try:
            import importlib
            m = importlib.import_module(mod_name)
            collection = getattr(m, attr_name, None)
            if collection:
                for name, fn in collection:
                    prog, _ = fn()
                    tests.append((f"{prefix}/{name}", prog))
        except (ImportError, Exception):
            pass

        # Also try PHASE*_TRAP_TESTS (e.g. phase17 has return-without-call)
        try:
            import importlib
            m = importlib.import_module(mod_name)
            trap_attr = attr_name.replace("_TESTS", "_TRAP_TESTS")
            trap_coll = getattr(m, trap_attr, None)
            if trap_coll:
                for name, fn in trap_coll:
                    result = fn()
                    # Trap fn may return plain list[Instruction] or (prog, expected)
                    prog = result[0] if isinstance(result, tuple) else result
                    tests.append((f"{prefix}/{name}_trap", prog, True))
        except (ImportError, Exception):
            pass

    return tests


def build_structured_tests() -> list[tuple]:
    """Return phase19 structured control flow + phase20 i32 masking tests."""
    tests = []

    # Phase 19: structured assembler programs
    try:
        import phase19_structured_assembler as p19
        for name, fn in p19.STRUCTURED_TESTS:
            prog, _ = fn()
            tests.append((f"p19/{name}", prog))
    except (ImportError, Exception):
        pass

    # Phase 20: i32 masking tests
    try:
        import phase20_type_masking_tests as p20
        for name, fn in p20.MASKING_TESTS:
            prog, _ = fn()
            tests.append((f"p20/{name}", prog))
    except (ImportError, Exception):
        pass

    return tests


# ─── Benchmark ───────────────────────────────────────────────────

def benchmark_mojo(prog: list, n: int = 200) -> float:
    """Return median in-process execution time in µs via --repeat N."""
    tokens = instr_to_tokens(prog)
    r = subprocess.run(
        [BINARY, "--repeat", str(n)] + tokens,
        capture_output=True, text=True, timeout=30,
    )
    for line in r.stdout.splitlines():
        if line.startswith("TIMING_NS:"):
            return int(line.split(":")[1].strip()) / 1000.0  # ns → µs
    # Fallback: subprocess timing
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        subprocess.run([BINARY] + tokens, capture_output=True, timeout=5)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e6


def benchmark_numpy(prog: list, n: int = 50) -> float:
    """Return median wall-clock time in µs for NumPyExecutor on prog."""
    ex = NumPyExecutor()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        ex.execute(prog)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2] * 1e6


# ─── Entry point ─────────────────────────────────────────────────

def main():
    do_bench  = "--bench"   in sys.argv
    verbose   = "--verbose" in sys.argv or "-v" in sys.argv

    if not os.path.isfile(BINARY):
        print(f"ERROR: Mojo binary not found: {BINARY}")
        print("  Build with: cd src && mojo build executor.mojo -o percepta_exec")
        sys.exit(1)

    print(f"Mojo binary: {BINARY}")
    print()

    total_passed = 0
    total_tests  = 0

    # ── Suite 1: ALL_TESTS (~50 programs) ──
    print("=== ALL_TESTS (programs.py + test_consolidated programs) ===")
    suite1 = build_all_tests()
    p, t = 0, 0
    for entry in suite1:
        name, prog = entry[0], entry[1]
        trap = len(entry) > 2 and entry[2]
        ok = compare_program(name, prog, verbose=verbose, expect_trap=trap)
        p += ok; t += 1
    print(f"  {p}/{t} passed")
    total_passed += p; total_tests += t

    # ── Suite 2: Tier-2 programs ──
    print()
    print("=== Tier-2: phase15-18 programs ===")
    suite2 = build_tier2_tests()
    if not suite2:
        print("  (no ALL_*_TESTS lists found in phase15-18 — skipped)")
    else:
        p, t = 0, 0
        for entry in suite2:
            name, prog = entry[0], entry[1]
            ok = compare_program(name, prog, verbose=verbose)
            p += ok; t += 1
        print(f"  {p}/{t} passed")
        total_passed += p; total_tests += t

    # ── Suite 3: Structured CF + i32 masking (phase19-20) ──
    print()
    print("=== Structured control flow + i32 masking (phase19-20) ===")
    suite3 = build_structured_tests()
    if not suite3:
        print("  (no phase19/phase20 tests found — skipped)")
    else:
        p, t = 0, 0
        for entry in suite3:
            name, prog = entry[0], entry[1]
            trap = len(entry) > 2 and entry[2]
            ok = compare_program(name, prog, verbose=verbose, expect_trap=trap)
            p += ok; t += 1
        print(f"  {p}/{t} passed")
        total_passed += p; total_tests += t

    # ── Benchmark ──
    if do_bench:
        print()
        print("=== Benchmark ===")
        from programs import make_fibonacci
        prog_fib, _ = make_fibonacci(10)
        mojo_us  = benchmark_mojo(prog_fib)
        numpy_us = benchmark_numpy(prog_fib)
        speedup = numpy_us / mojo_us if mojo_us > 0 else float('inf')
        print(f"  fib(10): Mojo={mojo_us:.1f} µs, NumPy={numpy_us:.0f} µs  "
              f"({speedup:.0f}× speedup)")

        # Countdown (14 steps, target <5 µs per issue #40)
        countdown_prog = [
            Instruction(OP_PUSH, 3), Instruction(OP_DUP),
            Instruction(OP_PUSH, 1), Instruction(OP_SUB),
            Instruction(OP_DUP), Instruction(OP_JNZ, 1),
            Instruction(OP_HALT),
        ]
        mojo_cd = benchmark_mojo(countdown_prog)
        status = "PASS" if mojo_cd < 5.0 else "FAIL"
        print(f"  countdown(14 steps): Mojo={mojo_cd:.1f} µs  "
              f"(target: <5 µs) [{status}]")

    print()
    print(f"Overall: {total_passed}/{total_tests} passed")
    if total_passed < total_tests:
        sys.exit(1)


if __name__ == "__main__":
    main()
