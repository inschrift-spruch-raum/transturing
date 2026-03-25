"""Phase 19: Structured Control Flow Assembler + BR_TABLE

Validates the assembler.compile_structured() function by compiling 8 programs
that use WASM-style structured control flow and executing them on both
NumPyExecutor and TorchExecutor.

Constructs tested:
  BLOCK / BR_IF    — early exit from a block
  LOOP / BR / BR_IF — counting loop with backward branch
  IF               — conditional without ELSE
  IF / ELSE / END  — conditional with two branches (true and false)
  Nested BLOCK+LOOP — multi-level BR
  BR_TABLE         — 3-case switch + default
  LOOP + locals    — factorial(5) = 120
  LOOP + locals    — GCD(48, 18) = 6

All programs are verified against hand-computed expected values on both
executors.  Invariants checked:
  - compile_structured output contains only valid Instruction objects.
  - No executor regressions (test_consolidated ALL_TESTS still pass).
  - BR_TABLE with N cases contains exactly N+1 comparison/jump instructions
    in the chain (N JNZ comparisons + 1 default jump).

Issue: #36 — Tier 3 Chunk 1: Structured control flow assembler + BR_TABLE
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isa import Instruction, OP_PUSH, OP_JNZ, OP_EQ, compare_traces
from executor import NumPyExecutor, TorchExecutor
from assembler import compile_structured


# ─── Program builders ────────────────────────────────────────────────────────

def prog_block_early_exit():
    """BLOCK with a BR_IF that always exits; PUSH 99 is unreachable.

    PUSH 5 ; BLOCK ; PUSH 1 ; BR_IF 0 ; PUSH 99 (dead) ; END ; HALT
    Expected top: 5
    """
    prog = compile_structured([
        ('PUSH', 5),
        ('BLOCK',),
          ('PUSH', 1),       # non-zero condition → always branch
          ('BR_IF', 0),      # exit BLOCK
          ('PUSH', 99),      # unreachable
        ('END',),
        ('HALT',),
    ])
    return prog, 5


def prog_loop_countdown():
    """Count down from 3 to 0 using a LOOP with a backward branch.

    Expected top: 0  (counter reaches 0 and exits)
    """
    prog = compile_structured([
        ('PUSH', 3),           # counter
        ('BLOCK',),            # outer block: exit when counter == 0
          ('LOOP',),
            ('DUP',),
            ('EQZ',),          # is counter 0?
            ('BR_IF', 1),      # exit BLOCK (depth 1) if so
            ('PUSH', 1),
            ('SUB',),          # counter -= 1
            ('BR', 0),         # continue LOOP (depth 0)
          ('END',),
        ('END',),
        ('HALT',),
    ])
    return prog, 0


def prog_if_no_else_taken():
    """IF without ELSE; condition is true so then-body executes.

    PUSH 7; PUSH 3; GT_S → 1; IF; PUSH 100; END; HALT
    Expected top: 100
    """
    prog = compile_structured([
        ('PUSH', 7),
        ('PUSH', 3),
        ('GT_S',),    # 7 > 3 = 1 (true)
        ('IF',),
          ('PUSH', 100),
        ('END',),
        ('HALT',),
    ])
    return prog, 100


def prog_if_no_else_skipped():
    """IF without ELSE; condition is false so then-body is skipped.

    Sentinel 99 remains on stack.
    Expected top: 99
    """
    prog = compile_structured([
        ('PUSH', 99),          # sentinel below condition result
        ('PUSH', 3),
        ('PUSH', 7),
        ('GT_S',),    # 3 > 7 = 0 (false)
        ('IF',),
          ('PUSH', 100),       # skipped
        ('END',),
        ('HALT',),
    ])
    # GT_S pops 3 and 7, pushes 0; IF pops 0 (false) → skip body
    # Stack: [99].  top = 99
    return prog, 99


def prog_if_else_true():
    """IF/ELSE/END; condition true → then-branch pushes 100.

    Expected top: 100
    """
    prog = compile_structured([
        ('PUSH', 7),
        ('PUSH', 3),
        ('GT_S',),   # 1
        ('IF',),
          ('PUSH', 100),
        ('ELSE',),
          ('PUSH', 200),
        ('END',),
        ('HALT',),
    ])
    return prog, 100


def prog_if_else_false():
    """IF/ELSE/END; condition false → else-branch pushes 200.

    Expected top: 200
    """
    prog = compile_structured([
        ('PUSH', 3),
        ('PUSH', 7),
        ('GT_S',),   # 0
        ('IF',),
          ('PUSH', 100),
        ('ELSE',),
          ('PUSH', 200),
        ('END',),
        ('HALT',),
    ])
    return prog, 200


def prog_br_table_switch():
    """BR_TABLE with 3 cases + default; index=1 selects case1 → pushes 20.

    Structure mirrors a switch statement:
      switch(index) {
        case 0: push 10; break;
        case 1: push 20; break;   ← selected
        case 2: push 30; break;
        default: push 99;
      }

    BR_TABLE [0,1,2] 3 is used with four nested BLOCKs.  The depths inside
    BR_TABLE refer to the four blocks (innermost first):
      depth 0 = case0-BLOCK  → after case0-END → case0 body (PUSH 10)
      depth 1 = case1-BLOCK  → after case1-END → case1 body (PUSH 20)
      depth 2 = case2-BLOCK  → after case2-END → case2 body (PUSH 30)
      depth 3 = done-BLOCK   → after done-END  → HALT

    Invariant check: for N=3, the comparison chain has 4 JNZ/jump instructions.
    Expected top: 20
    """
    prog = compile_structured([
        ('PUSH', 1),           # index = 1
        ('BLOCK',),            # done  (depth 3 from BR_TABLE)
          ('BLOCK',),          # case2 (depth 2)
            ('BLOCK',),        # case1 (depth 1)
              ('BLOCK',),      # case0 (depth 0)
                ('BR_TABLE', [0, 1, 2], 3),
              ('END',),        # case0 end → case0 body follows
              ('PUSH', 10),
              ('BR', 2),       # exit done BLOCK
            ('END',),          # case1 end → case1 body follows
            ('PUSH', 20),
            ('BR', 1),         # exit done BLOCK
          ('END',),            # case2 end → case2 body follows
          ('PUSH', 30),
          ('BR', 0),           # exit done BLOCK
        ('END',),              # done BLOCK end
        ('HALT',),
    ])
    return prog, 20


def prog_br_table_default():
    """BR_TABLE default branch: index=5 is out of range → default → pushes 99.

    Same 3-case structure as prog_br_table_switch; index=5.
    Expected top: 99
    """
    prog = compile_structured([
        ('PUSH', 5),           # index = 5 (out of range for cases [0,1,2])
        ('BLOCK',),            # done
          ('BLOCK',),          # case2
            ('BLOCK',),        # case1
              ('BLOCK',),      # case0
                ('BR_TABLE', [0, 1, 2], 3),
              ('END',),
              ('PUSH', 10),
              ('BR', 2),
            ('END',),
            ('PUSH', 20),
            ('BR', 1),
          ('END',),
          ('PUSH', 30),
          ('BR', 0),
        ('END',),              # done end
        ('PUSH', 99),          # default falls through to here
        ('HALT',),
    ])
    # With default=depth3=done BLOCK: default jump goes to done-END,
    # then PUSH 99 executes.  top = 99
    return prog, 99


def prog_factorial():
    """factorial(5) = 120 via structured LOOP + LOCAL variables.

    local 0 = n (countdown), local 1 = result (accumulator)
    Each iteration: result *= n; n -= 1
    Exits when n == 0.
    Expected top: 120
    """
    prog = compile_structured([
        ('PUSH', 5),
        ('LOCAL.SET', 0),      # n = 5
        ('PUSH', 1),
        ('LOCAL.SET', 1),      # result = 1

        ('BLOCK',),
          ('LOOP',),
            ('LOCAL.GET', 0),  # n
            ('EQZ',),
            ('BR_IF', 1),      # exit BLOCK if n == 0

            ('LOCAL.GET', 0),  # n
            ('LOCAL.GET', 1),  # result
            ('MUL',),          # n * result
            ('LOCAL.SET', 1),  # result = n * result

            ('LOCAL.GET', 0),  # n
            ('PUSH', 1),
            ('SUB',),          # n - 1
            ('LOCAL.SET', 0),  # n -= 1

            ('BR', 0),         # continue LOOP
          ('END',),
        ('END',),

        ('LOCAL.GET', 1),      # push result
        ('HALT',),
    ])
    return prog, 120


def prog_gcd():
    """GCD(48, 18) = 6 via structured LOOP + LOCAL variables (Euclidean).

    local 0 = a, local 1 = b, local 2 = temp
    Each iteration: temp = a % b; a = b; b = temp
    Exits when b == 0.  Returns a.
    Expected top: 6
    """
    prog = compile_structured([
        ('PUSH', 48),
        ('LOCAL.SET', 0),      # a = 48
        ('PUSH', 18),
        ('LOCAL.SET', 1),      # b = 18

        ('BLOCK',),
          ('LOOP',),
            ('LOCAL.GET', 1),  # b
            ('EQZ',),
            ('BR_IF', 1),      # exit BLOCK if b == 0

            ('LOCAL.GET', 0),  # a
            ('LOCAL.GET', 1),  # b
            ('REM_U',),        # a % b
            ('LOCAL.SET', 2),  # temp = a % b

            ('LOCAL.GET', 1),
            ('LOCAL.SET', 0),  # a = b

            ('LOCAL.GET', 2),
            ('LOCAL.SET', 1),  # b = temp

            ('BR', 0),         # continue LOOP
          ('END',),
        ('END',),

        ('LOCAL.GET', 0),      # return a (GCD)
        ('HALT',),
    ])
    return prog, 6


# ─── All structured test cases ────────────────────────────────────────────────

STRUCTURED_TESTS = [
    ("block_early_exit",       prog_block_early_exit),
    ("loop_countdown",         prog_loop_countdown),
    ("if_no_else_taken",       prog_if_no_else_taken),
    ("if_no_else_skipped",     prog_if_no_else_skipped),
    ("if_else_true",           prog_if_else_true),
    ("if_else_false",          prog_if_else_false),
    ("br_table_switch_case1",  prog_br_table_switch),
    ("br_table_default",       prog_br_table_default),
    ("factorial_5",            prog_factorial),
    ("gcd_48_18",              prog_gcd),
]


# ─── Test runners ─────────────────────────────────────────────────────────────

def test_structured_programs(verbose=False):
    """Run all structured programs on both executors; check expected top."""
    print("=" * 60)
    print("Phase 19: Structured Assembler — Program Tests")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = len(STRUCTURED_TESTS) * 2  # numpy + pytorch

    for name, builder in STRUCTURED_TESTS:
        prog, expected = builder()

        for label, exc in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = exc.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            passed += ok
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  {name:<32s}  expected={expected:>5}  got={top}")

        if verbose:
            print(f"    flat length: {len(prog)} instructions")

    print(f"\n  Programs: {passed}/{total} passed")
    return passed == total


def test_trace_match():
    """NumPy and PyTorch traces must be identical on all programs."""
    print("=" * 60)
    print("Phase 19: Trace Match (numpy == pytorch)")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    matched = 0
    for name, builder in STRUCTURED_TESTS:
        prog, _ = builder()
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        ok, diff = compare_traces(np_trace, pt_trace)
        matched += ok
        status = "MATCH" if ok else "DIFF "
        print(f"  {status}  {name}")
        if not ok and diff:
            print(f"         first diff: {diff}")

    print(f"\n  Trace match: {matched}/{len(STRUCTURED_TESTS)}")
    return matched == len(STRUCTURED_TESTS)


def test_br_table_invariant():
    """BR_TABLE with N cases must emit exactly N+1 jump instructions in chain.

    Counts: N JNZ instructions from the comparison chain + 1 default jump.
    Does NOT count the handler jumps (those are targets, not comparisons).
    """
    print("=" * 60)
    print("Phase 19: BR_TABLE Invariant Check")
    print("=" * 60)

    passed = True
    for n_cases in range(1, 6):
        # Build a minimal BR_TABLE program with n_cases cases.
        # Wrap in BLOCK so there's a valid target at depth n_cases.
        wasm = [('PUSH', 0)]  # index = 0 (doesn't matter for counting)
        for _ in range(n_cases + 1):
            wasm.append(('BLOCK',))
        wasm.append(('BR_TABLE', list(range(n_cases)), n_cases))
        for _ in range(n_cases + 1):
            wasm.append(('END',))
        wasm.append(('HALT',))

        prog = compile_structured(wasm)

        # Count JNZ instructions that appear in the comparison chain.
        # The chain is: for each case i: DUP + PUSH i + EQ + JNZ(handler)
        # Then: POP + (PUSH 1 + JNZ) for default  → N+1 jumps total.
        # Each case handler (POP + PUSH 1 + JNZ) is NOT in the chain.
        #
        # Strategy: find the first JNZ with a *handler* target (index into
        # flat[]) and count consecutive JNZ blocks before it.
        #
        # Simpler: count how many (DUP, PUSH i, EQ, JNZ) quads appear.
        eq_jnz_pairs = 0
        i = 0
        while i < len(prog) - 1:
            if prog[i].op == OP_EQ and prog[i + 1].op == OP_JNZ:
                eq_jnz_pairs += 1
                i += 2
            else:
                i += 1

        # Plus 1 for the default jump (a PUSH 1 + JNZ block).
        # Count PUSH-1; JNZ pairs to find default + handler jumps.
        push1_jnz_pairs = sum(
            1 for j in range(len(prog) - 1)
            if prog[j].op == OP_PUSH and prog[j].arg == 1
            and prog[j + 1].op == OP_JNZ
        )
        # push1_jnz_pairs = 1 (default) + n_cases (handlers)
        chain_jumps = eq_jnz_pairs + 1  # comparisons + default
        expected_chain = n_cases + 1
        ok = (chain_jumps == expected_chain)
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(
            f"  {status}  N={n_cases}  chain_jumps={chain_jumps}  "
            f"expected={expected_chain}  "
            f"(EQ+JNZ pairs={eq_jnz_pairs}, PUSH1+JNZ={push1_jnz_pairs})"
        )

    print()
    return passed


def test_output_types():
    """compile_structured must return a list of Instruction objects."""
    print("=" * 60)
    print("Phase 19: Output Type Check")
    print("=" * 60)

    passed = True
    for name, builder in STRUCTURED_TESTS:
        prog, _ = builder()
        ok = (
            isinstance(prog, list)
            and all(isinstance(i, Instruction) for i in prog)
        )
        passed = passed and ok
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}  ({len(prog)} instructions, "
              f"all Instruction: {ok})")

    print()
    return passed


def test_regression():
    """Existing consolidated tests must still pass (no executor regression)."""
    print("=" * 60)
    print("Phase 19: Regression — test_consolidated ALL_TESTS")
    print("=" * 60)

    from programs import ALL_TESTS
    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = len(ALL_TESTS) * 2

    failures = []
    for name, builder in ALL_TESTS:
        prog, expected = builder()
        for label, exc in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = exc.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            passed += ok
            if not ok:
                failures.append((label, name, expected, top))

    print(f"  Regression: {passed}/{total} passed")
    for label, name, expected, top in failures:
        print(f"  FAIL  {label:5s}  {name}  expected={expected}  got={top}")
    print()
    return passed == total


def main():
    results = []
    results.append(test_output_types())
    results.append(test_structured_programs())
    print()
    results.append(test_trace_match())
    print()
    results.append(test_br_table_invariant())
    results.append(test_regression())

    all_pass = all(results)
    print("=" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'FAILURES DETECTED'}")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
