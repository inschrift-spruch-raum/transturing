"""Benchmark: Mojo executor vs NumPy executor — µs/step, not subprocess time.

Measures actual execution time by:
  - Mojo:  invoking percepta_exec --repeat N and reading the TIMING_NS output
            (runs the program N times inside one process, reports median ns)
  - NumPy: timing NumPyExecutor().execute() directly in Python

Reports µs/program and µs/step for both, plus speedup ratio.

Usage:
    python src/benchmark.py
    python src/benchmark.py --repeat 500    # more samples (default: 200)
    python src/benchmark.py --also-test     # verify correctness first
"""

import subprocess
import sys
import os
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from executor import NumPyExecutor
from src.benchmarks import BENCHMARKS

BINARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "percepta_exec")


# ─── Timing helpers ──────────────────────────────────────────────

def instr_to_tokens(prog) -> list[str]:
    return [str(x) for instr in prog for x in (instr.op, instr.arg)]


def time_mojo(prog, repeat: int) -> float:
    """Run prog inside one Mojo process N times. Return median ns."""
    tokens = instr_to_tokens(prog)
    r = subprocess.run(
        [BINARY, "--repeat", str(repeat)] + tokens,
        capture_output=True, text=True, timeout=120,
    )
    for line in r.stdout.splitlines():
        if line.startswith("TIMING_NS:"):
            return float(line.split(":")[1].strip())
    raise RuntimeError(f"No TIMING_NS in output:\n{r.stdout}\n{r.stderr}")


def time_numpy(prog, repeat: int) -> float:
    """Run NumPyExecutor N times. Return median ns."""
    ex = NumPyExecutor()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        ex.execute(prog, max_steps=50000)
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return float(samples[repeat // 2])


def count_steps(prog) -> int:
    """Count execution steps via NumPyExecutor (reference)."""
    ex = NumPyExecutor()
    trace = ex.execute(prog, max_steps=50000)
    return len(trace.steps)


# ─── Correctness check ───────────────────────────────────────────

def verify_mojo(prog, expected) -> bool:
    """Run Mojo in normal mode and check result."""
    tokens = instr_to_tokens(prog)
    r = subprocess.run([BINARY] + tokens, capture_output=True, text=True, timeout=30)
    for line in r.stdout.splitlines():
        if line.startswith("RESULT:"):
            got = int(line.split(":")[1].strip())
            return got == expected
    return False


# ─── Main ────────────────────────────────────────────────────────

def main():
    repeat     = 200
    also_test  = "--also-test" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--repeat="):
            repeat = int(arg.split("=")[1])
        elif arg == "--repeat" and sys.argv.index(arg) + 1 < len(sys.argv):
            repeat = int(sys.argv[sys.argv.index(arg) + 1])

    if not os.path.isfile(BINARY):
        print(f"ERROR: Mojo binary not found: {BINARY}")
        print("  Build: cd src && mojo build executor.mojo -o percepta_exec")
        sys.exit(1)

    print(f"Mojo executor vs NumPy executor  ({repeat} samples each)\n")

    if also_test:
        print("Verifying correctness first...")
        for name, prog, expected, desc in BENCHMARKS:
            ok = verify_mojo(prog, expected)
            print(f"  {'OK' if ok else 'FAIL'}  {name}")
        print()

    # Header
    w = 16
    print(f"{'Program':<{w}}  {'Steps':>6}  "
          f"{'Mojo µs':>9}  {'NumPy µs':>9}  "
          f"{'Mojo ns/step':>13}  {'NumPy ns/step':>14}  {'Speedup':>8}")
    print("-" * 85)

    for name, prog, expected, desc in BENCHMARKS:
        steps = count_steps(prog)

        mojo_ns  = time_mojo(prog, repeat)
        numpy_ns = time_numpy(prog, repeat)

        mojo_us       = mojo_ns  / 1000
        numpy_us      = numpy_ns / 1000
        mojo_ns_step  = mojo_ns  / steps
        numpy_ns_step = numpy_ns / steps
        speedup       = numpy_ns / mojo_ns

        print(f"{name:<{w}}  {steps:>6}  "
              f"{mojo_us:>9.1f}  {numpy_us:>9.1f}  "
              f"{mojo_ns_step:>13.1f}  {numpy_ns_step:>14.1f}  "
              f"{speedup:>7.1f}×")

    print()
    print("Note: Mojo time = in-process execution only (--repeat N, median of N runs).")
    print("      NumPy time = Python execute() call only, no subprocess overhead.")
    print("      Lower is better. Speedup = NumPy/Mojo.")


if __name__ == "__main__":
    main()
