"""Million-step execution benchmarks and scaling validation (Issue #52).

Measures wall-clock time, throughput (steps/s), and memory usage at 10K, 100K,
and 1M+ step scales using the skill's Mojo executor (with Python fallback).

Usage:
    python benchmark_scaling.py
    python benchmark_scaling.py --max-tier 2   # skip 1M tier
"""

import sys
import os
import time
import tracemalloc

# Skill path MUST come first to shadow repo's programs.py/isa.py (which need torch)
sys.path.insert(0, '/mnt/skills/user/llm-as-computer/src')

from programs import make_fibonacci, make_sum_1_to_n, make_multiply
from runner import setup, _run_mojo, _run_python, MOJO_BIN
from isa_lite import Instruction, OP_HALT


# ─── Benchmark definitions ───────────────────────────────────────

def make_countdown(n):
    """Simple countdown loop: PUSH n, SUB 1, JNZ loop, HALT. ~3n steps."""
    from isa_lite import program
    prog = program(
        ('PUSH', n),    # 0: counter
        ('PUSH', 1),    # 1
        ('SUB',),       # 2: counter - 1
        ('DUP',),       # 3: copy for test
        ('JNZ', 1),     # 4: loop back
        ('HALT',),      # 5
    )
    return prog, 0


TIERS = {
    1: "10K",
    2: "100K",
    3: "1M",
}

BENCHMARKS = [
    # (name, tier, generator_call)
    # Tier 1: ~10K steps
    ("sum_1to1K",       1, lambda: make_sum_1_to_n(1000)),
    ("fib_1000",        1, lambda: make_fibonacci(1000)),
    ("countdown_3K",    1, lambda: make_countdown(3000)),

    # Tier 2: ~100K steps
    ("sum_1to10K",      2, lambda: make_sum_1_to_n(10000)),
    ("fib_10000",       2, lambda: make_fibonacci(10000)),
    ("countdown_30K",   2, lambda: make_countdown(30000)),

    # Tier 3: ~1M steps
    ("sum_1to100K",     3, lambda: make_sum_1_to_n(100000)),
    ("fib_100000",      3, lambda: make_fibonacci(100000)),
    ("countdown_300K",  3, lambda: make_countdown(300000)),
]


# ─── Runners ─────────────────────────────────────────────────────

def count_steps_python(prog, max_steps=2_000_000):
    """Count steps via Python fallback (no subprocess)."""
    lines, result = _run_python(prog, max_steps)
    return len(lines), result


def time_mojo(prog, max_steps=2_000_000, repeat=50):
    """Time Mojo execution via --repeat mode (no trace I/O).

    Returns (median_exec_ns, result_or_None).
    Uses --repeat for pure execution timing without trace output overhead.
    Also does one normal run to get the result for correctness checking.
    """
    from runner import _instrs_to_tokens
    import subprocess
    tokens = _instrs_to_tokens(prog).split()

    # 1) Benchmark: --repeat N → reports median ns
    cmd_bench = [MOJO_BIN, "--repeat", str(repeat)] + tokens
    r = subprocess.run(cmd_bench, capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        raise RuntimeError(f"Mojo bench failed: {r.stderr[:200]}")

    median_ns = None
    for line in r.stdout.strip().split('\n'):
        if line.startswith("TIMING_NS:"):
            median_ns = int(line.split(":")[1].strip())
    if median_ns is None:
        raise RuntimeError(f"No TIMING_NS in output: {r.stdout[:200]}")

    # 2) One normal run for result (only for small programs — large ones
    #    produce huge trace output through subprocess I/O)
    result = None
    if len(prog) < 50 and repeat >= 10:
        cmd_run = [MOJO_BIN] + tokens
        try:
            r2 = subprocess.run(cmd_run, capture_output=True, text=True, timeout=60)
            for line in r2.stdout.strip().split('\n'):
                if line.startswith("RESULT:"):
                    result = int(line.split(":")[1].strip())
        except subprocess.TimeoutExpired:
            pass  # skip result check for programs that take too long

    return median_ns, result


def time_python(prog, max_steps=2_000_000):
    """Time Python fallback execution. Returns (steps, wall_ns, result)."""
    t0 = time.perf_counter_ns()
    lines, result = _run_python(prog, max_steps)
    wall_ns = time.perf_counter_ns() - t0
    return len(lines), wall_ns, result


def measure_memory_python(prog, max_steps=2_000_000):
    """Measure peak memory of Python execution."""
    tracemalloc.start()
    _run_python(prog, max_steps)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak  # bytes


# ─── Main ────────────────────────────────────────────────────────

def fmt_time(ns):
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns/1e3:.1f} µs"
    if ns < 1_000_000_000:
        return f"{ns/1e6:.1f} ms"
    return f"{ns/1e9:.2f} s"


def fmt_mem(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024**2:
        return f"{b/1024:.1f} KB"
    return f"{b/1024**2:.1f} MB"


def main():
    max_tier = 3
    if "--max-tier" in sys.argv:
        idx = sys.argv.index("--max-tier")
        max_tier = int(sys.argv[idx + 1])

    has_mojo = os.path.exists(MOJO_BIN)
    if not has_mojo:
        print("Mojo binary not found — running setup...")
        setup()
        has_mojo = os.path.exists(MOJO_BIN)

    print(f"Mojo: {'available' if has_mojo else 'NOT available (Python only)'}")
    print()

    # ─── Results table ────────────────────────────────────────────
    results = []

    for name, tier, gen_fn in BENCHMARKS:
        if tier > max_tier:
            continue

        prog, expected = gen_fn()
        print(f"--- {name} (tier {TIERS[tier]}, {len(prog)} instructions) ---")

        # Python execution + timing
        try:
            py_steps, py_ns, py_result = time_python(prog, max_steps=2_000_000)
            py_mem = measure_memory_python(prog, max_steps=2_000_000)
            py_steps_per_sec = py_steps * 1e9 / py_ns if py_ns > 0 else 0
            print(f"  Python: {py_steps:>10,d} steps | {fmt_time(py_ns):>10s} | "
                  f"{py_steps_per_sec:>12,.0f} steps/s | mem {fmt_mem(py_mem)}")
        except Exception as e:
            print(f"  Python: FAILED — {e}")
            py_steps, py_ns, py_result, py_mem, py_steps_per_sec = 0, 0, None, 0, 0

        # Mojo execution + timing (--repeat mode, pure execution time)
        mojo_ns = 0
        mojo_steps_per_sec = 0
        speedup = 0
        mj_result = None
        # Use fewer repeats for large programs to avoid timeout
        mojo_repeat = 50 if tier <= 1 else (10 if tier == 2 else 3)
        if has_mojo:
            try:
                mojo_ns, mj_result = time_mojo(prog, repeat=mojo_repeat)
                mojo_steps_per_sec = py_steps * 1e9 / mojo_ns if mojo_ns > 0 else 0
                speedup = py_ns / mojo_ns if mojo_ns > 0 else 0
                match = "OK" if mj_result == py_result else f"MISMATCH({mj_result})"
                print(f"  Mojo:   {py_steps:>10,d} steps | {fmt_time(mojo_ns):>10s} | "
                      f"{mojo_steps_per_sec:>12,.0f} steps/s | speedup {speedup:.1f}× | {match}")
            except Exception as e:
                print(f"  Mojo:   FAILED — {e}")

        results.append({
            "name": name,
            "tier": TIERS[tier],
            "instructions": len(prog),
            "steps": py_steps,
            "py_time_ns": py_ns,
            "py_steps_per_sec": py_steps_per_sec,
            "py_mem_bytes": py_mem,
            "mojo_time_ns": mojo_ns,
            "mojo_steps_per_sec": mojo_steps_per_sec,
            "speedup": speedup,
        })
        print()

    # ─── Summary table ────────────────────────────────────────────
    print()
    print("=" * 100)
    print("SCALING BENCHMARK RESULTS")
    print("=" * 100)
    w = 18
    print(f"{'Program':<{w}} {'Tier':>5} {'Steps':>10} "
          f"{'Python':>10} {'Mojo':>10} {'Speedup':>8} "
          f"{'Py steps/s':>12} {'Mj steps/s':>12} {'Py mem':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['name']:<{w}} {r['tier']:>5} {r['steps']:>10,d} "
              f"{fmt_time(r['py_time_ns']):>10} {fmt_time(r['mojo_time_ns']):>10} "
              f"{r['speedup']:>7.1f}× "
              f"{r['py_steps_per_sec']:>12,.0f} {r['mojo_steps_per_sec']:>12,.0f} "
              f"{fmt_mem(r['py_mem_bytes']):>8}")
    print("=" * 100)

    # ─── Scaling analysis ─────────────────────────────────────────
    print()
    print("SCALING ANALYSIS (time per step):")
    for r in results:
        py_ns_per_step = r['py_time_ns'] / r['steps'] if r['steps'] > 0 else 0
        mojo_ns_per_step = r['mojo_time_ns'] / r['steps'] if r['steps'] > 0 else 0
        print(f"  {r['name']:<{w}} {r['steps']:>10,d} steps | "
              f"Python: {py_ns_per_step:>8.1f} ns/step | "
              f"Mojo: {mojo_ns_per_step:>8.1f} ns/step")


if __name__ == "__main__":
    main()
