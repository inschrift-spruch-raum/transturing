"""Phase 20: Integration tests — structured control flow + i32 type masking.

Validates that i32 overflow semantics are applied correctly when arithmetic
ops are embedded inside structured control flow (BLOCK/LOOP/IF/ELSE/BR).

Each test produces results that would be WRONG without MASK32 wrapping, and is
verified on both NumPyExecutor and TorchExecutor.

Tests
-----
1. i32_add_overflow        PUSH 0xFFFFFFFF; PUSH 1; ADD → 0   (classic wrap)
2. i32_mul_overflow        PUSH 0x10000; PUSH 0x10000; MUL → 0  (32-bit product wraps)
3. i32_sub_wrap            PUSH 0; PUSH 1; SUB → 0xFFFFFFFF   (underflow wraps)
4. i32_neg_positive        PUSH 1; NEG → 0xFFFFFFFF            (-1 as unsigned i32)
5. loop_add_overflow       Loop adds 1 to 0xFFFFFFFE (3 iters), expects final top = 1
6. if_overflow_branch      IF selects between overflow ADD and plain PUSH; verify masked
7. br_table_overflow       BR_TABLE dispatches after overflow comparison produces 0
8. overflow_in_else        ELSE branch computes (0xFFFFFFFF + 2) → 1

All eight programs verify: trace_top == expected on NumPy AND PyTorch.
A ninth test confirms both executors produce identical traces on all programs.

Issue: #37 — Tier 3 Chunk 2: type system + float docs + integration tests
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isa import compare_traces
from executor import NumPyExecutor, TorchExecutor
from assembler import compile_structured

MASK32 = 0xFFFFFFFF


# ─── Program builders ────────────────────────────────────────────────────────

def prog_add_overflow():
    """PUSH 0xFFFFFFFF; PUSH 1; ADD → 0 (wraps to zero).

    Without masking: 0x100000000. With masking: 0.
    """
    prog = compile_structured([
        ('PUSH', 0xFFFFFFFF),
        ('PUSH', 1),
        ('ADD',),
        ('HALT',),
    ])
    return prog, 0


def prog_mul_overflow():
    """PUSH 0x10000; PUSH 0x10000; MUL → 0 (0x100000000 wraps to 0).

    Without masking: 4294967296. With masking: 0.
    """
    prog = compile_structured([
        ('PUSH', 0x10000),
        ('PUSH', 0x10000),
        ('MUL',),
        ('HALT',),
    ])
    return prog, 0


def prog_sub_wrap():
    """PUSH 0; PUSH 1; SUB → 0xFFFFFFFF (0 - 1 wraps in i32).

    Stack: va=1 (top), vb=0 (below).  result = vb - va = 0 - 1 = -1.
    Without masking: -1. With masking: 0xFFFFFFFF = 4294967295.
    """
    prog = compile_structured([
        ('PUSH', 0),
        ('PUSH', 1),
        ('SUB',),
        ('HALT',),
    ])
    return prog, 0xFFFFFFFF


def prog_neg_positive():
    """PUSH 1; NEG → 0xFFFFFFFF (-1 stored as unsigned i32).

    Without masking: -1. With masking: 4294967295.
    """
    prog = compile_structured([
        ('PUSH', 1),
        ('NEG',),
        ('HALT',),
    ])
    return prog, 0xFFFFFFFF


def prog_loop_add_overflow():
    """Loop: start at 0xFFFFFFFE, add 1 three times, check result = 1.

    Iterations:
      0xFFFFFFFE + 1 = 0xFFFFFFFF  (still ≠ 0, loop continues)
      0xFFFFFFFF + 1 = 0x00000000  (wrap, == 0 so EQZ triggers... but we want 3 iters)
      Actually let's use a counter-driven loop for determinism.

    Approach: counter in local 0 (3 iterations), accumulator in local 1 (start 0xFFFFFFFE).
    Each iteration: acc = acc + 1; counter -= 1; exit when counter == 0.
    Final acc = (0xFFFFFFFE + 3) & MASK32 = 1.
    """
    prog = compile_structured([
        ('PUSH', 3),
        ('LOCAL.SET', 0),          # counter = 3
        ('PUSH', 0xFFFFFFFE),
        ('LOCAL.SET', 1),          # acc = 0xFFFFFFFE

        ('BLOCK',),
          ('LOOP',),
            ('LOCAL.GET', 0),      # counter
            ('EQZ',),
            ('BR_IF', 1),          # exit BLOCK when counter == 0

            ('LOCAL.GET', 1),      # acc
            ('PUSH', 1),
            ('ADD',),              # acc + 1 (wraps on 3rd iter: 0xFFFFFFFF+1=0)
            ('LOCAL.SET', 1),

            ('LOCAL.GET', 0),
            ('PUSH', 1),
            ('SUB',),              # counter - 1
            ('LOCAL.SET', 0),

            ('BR', 0),             # continue LOOP
          ('END',),
        ('END',),

        ('LOCAL.GET', 1),          # push final acc
        ('HALT',),
    ])
    # 0xFFFFFFFE + 1 = 0xFFFFFFFF
    # 0xFFFFFFFF + 1 = 0x00000000 (wrap)
    # 0x00000000 + 1 = 0x00000001
    return prog, 1


def prog_if_overflow_branch():
    """IF selects overflow ADD vs plain PUSH; verify masked result selected.

    condition = 1 (true) → enter IF body.
    IF body: PUSH 0xFFFFFFFF; PUSH 2; ADD → 1 (wraps).
    Expected top: 1.

    Without masking the ADD, result would be 0x100000001.
    """
    prog = compile_structured([
        ('PUSH', 1),               # condition = true
        ('IF',),
          ('PUSH', 0xFFFFFFFF),
          ('PUSH', 2),
          ('ADD',),                # (0xFFFFFFFF + 2) & MASK32 = 1
        ('ELSE',),
          ('PUSH', 999),           # unreachable
        ('END',),
        ('HALT',),
    ])
    return prog, 1


def prog_br_table_overflow():
    """BR_TABLE dispatch driven by an overflow comparison result.

    Compute (0xFFFFFFFF + 1) & MASK32 = 0.
    Use 0 as index into BR_TABLE [0, 1] default=2.
    case 0 → push 10, exit; case 1 → push 20, exit; default → push 99, halt.
    Expected: index=0 → push 10 → top = 10.

    Pattern mirrors phase19's prog_br_table_switch (HALT is at done_end;
    matched cases BR out of done BLOCK to HALT; default jumps to done_end=HALT).
    PUSH 99 never executes for in-range indices.

    This proves that overflow-masked results can drive control flow correctly.
    """
    prog = compile_structured([
        # Compute overflow: result = 0
        ('PUSH', 0xFFFFFFFF),
        ('PUSH', 1),
        ('ADD',),                  # (0xFFFFFFFF + 1) & MASK32 = 0
        # index is now on stack

        ('BLOCK',),                # done  (depth 2 = default target)
          ('BLOCK',),              # case1 (depth 1)
            ('BLOCK',),            # case0 (depth 0)
              ('BR_TABLE', [0, 1], 2),
            ('END',),              # case0 END → case0 body
            ('PUSH', 10),
            ('BR', 1),             # exit done BLOCK → HALT at done_end
          ('END',),                # case1 END → case1 body
          ('PUSH', 20),
          ('BR', 0),               # exit done BLOCK → HALT at done_end
        ('END',),                  # done END = position of HALT
        ('HALT',),                 # matched cases land here; default also lands here
    ])
    return prog, 10


def prog_overflow_in_else():
    """ELSE branch computes (0xFFFFFFFF + 2) & MASK32 = 1.

    Condition is false → ELSE body executes.
    Expected top: 1.
    """
    prog = compile_structured([
        ('PUSH', 0),               # condition = false
        ('IF',),
          ('PUSH', 999),           # skipped
        ('ELSE',),
          ('PUSH', 0xFFFFFFFF),
          ('PUSH', 2),
          ('ADD',),                # (0xFFFFFFFF + 2) & MASK32 = 1
        ('END',),
        ('HALT',),
    ])
    return prog, 1


# ─── All test cases ───────────────────────────────────────────────────────────

MASKING_TESTS = [
    ("i32_add_overflow",    prog_add_overflow),
    ("i32_mul_overflow",    prog_mul_overflow),
    ("i32_sub_wrap",        prog_sub_wrap),
    ("i32_neg_positive",    prog_neg_positive),
    ("loop_add_overflow",   prog_loop_add_overflow),
    ("if_overflow_branch",  prog_if_overflow_branch),
    ("br_table_overflow",   prog_br_table_overflow),
    ("overflow_in_else",    prog_overflow_in_else),
]


# ─── Test runners ─────────────────────────────────────────────────────────────

def test_masking_programs(verbose=False):
    """Run all masking programs on both executors; check expected top value."""
    print("=" * 60)
    print("Phase 20: i32 Masking + Structured CF — Program Tests")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = len(MASKING_TESTS) * 2  # numpy + pytorch

    for name, builder in MASKING_TESTS:
        prog, expected = builder()

        for label, exc in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = exc.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            passed += ok
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  {name:<30s}  expected={expected:>12}  got={top}")

        if verbose:
            prog_len = len(prog)
            print(f"    flat length: {prog_len} instructions")

    print(f"\n  Programs: {passed}/{total} passed")
    return passed == total


def test_trace_match():
    """NumPy and PyTorch must produce identical traces on all masking programs."""
    print("=" * 60)
    print("Phase 20: Trace Match (numpy == pytorch)")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    matched = 0
    for name, builder in MASKING_TESTS:
        prog, _ = builder()
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        ok, diff = compare_traces(np_trace, pt_trace)
        matched += ok
        status = "MATCH" if ok else "DIFF "
        print(f"  {status}  {name}")
        if not ok and diff:
            print(f"         first diff: {diff}")

    print(f"\n  Trace match: {matched}/{len(MASKING_TESTS)}")
    return matched == len(MASKING_TESTS)


def test_without_masking_would_fail():
    """Verify that 'expected' values are genuinely different without masking.

    Checks that the naive (unmasked) result != expected for overflow tests.
    This confirms these tests would have caught missing masking.
    """
    print("=" * 60)
    print("Phase 20: Masking Necessity Check")
    print("=" * 60)

    # For each overflow test, compute unmasked result and confirm it differs
    unmasked_cases = [
        ("i32_add_overflow",  0xFFFFFFFF + 1,     0),           # unmasked=0x100000000, expected=0
        ("i32_mul_overflow",  0x10000 * 0x10000,  0),           # unmasked=4294967296, expected=0
        ("i32_sub_wrap",      0 - 1,              0xFFFFFFFF),  # unmasked=-1, expected=4294967295
        ("i32_neg_positive",  -1,                 0xFFFFFFFF),  # unmasked=-1, expected=4294967295
        ("if_overflow_branch", 0xFFFFFFFF + 2,    1),           # unmasked=0x100000001, expected=1
        ("overflow_in_else",  0xFFFFFFFF + 2,     1),           # unmasked=0x100000001, expected=1
    ]

    passed = True
    for name, unmasked, expected in unmasked_cases:
        differs = (unmasked != expected)
        passed = passed and differs
        status = "OK  " if differs else "SKIP"
        print(f"  {status}  {name:<30s}  unmasked={unmasked:>14}  expected={expected:>12}")

    print(f"\n  All overflow tests have unmasked≠expected: {passed}")
    return passed


def test_regression():
    """Existing test_consolidated and phase19 tests must still pass."""
    print("=" * 60)
    print("Phase 20: Regression — consolidated + phase19")
    print("=" * 60)

    from programs import ALL_TESTS
    from phase19_structured_assembler import STRUCTURED_TESTS

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0
    failures = []

    for name, builder in ALL_TESTS + STRUCTURED_TESTS:
        prog, expected = builder()
        for label, exc in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = exc.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            passed += ok
            total += 1
            if not ok:
                failures.append((label, name, expected, top))

    print(f"  Regression: {passed}/{total} passed")
    for label, name, expected, top in failures:
        print(f"  FAIL  {label:5s}  {name}  expected={expected}  got={top}")
    print()
    return passed == total


def main():
    results = []
    results.append(test_masking_programs())
    print()
    results.append(test_trace_match())
    print()
    results.append(test_without_masking_would_fail())
    print()
    results.append(test_regression())

    all_pass = all(results)
    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
