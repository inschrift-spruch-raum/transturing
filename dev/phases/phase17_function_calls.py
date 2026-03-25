"""Phase 17: Function calls (CALL/RETURN + call stack).

Adds a call stack parabolic address space with 2 opcodes and 1 new attention head
(Head 9). Part of Tier 2 Chunk 3 (Issue #26).

Imports from the restructured isa.py + executor.py modules.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isa import (
    Instruction, Trace, TraceStep,
    compare_traces, test_algorithm, test_trap_algorithm,
    D_MODEL, N_OPCODES,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT,
    OP_SUB, OP_JZ, OP_JNZ, OP_NOP,
    OP_SWAP, OP_OVER, OP_ROT,
    OP_MUL, OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE,
    OP_I32_LOAD, OP_I32_STORE,
    OP_CALL, OP_RETURN, OP_TRAP,
    OP_NAMES,
)
from executor import NumPyExecutor, CompiledModel, TorchExecutor
from programs import ALL_TESTS, make_fibonacci, make_power_of_2, make_sum_1_to_n


# ─── Test Programs ──────────────────────────────────────────────

def test_simple_call():
    """Function at addr 5 pushes 42 and returns. Main calls it. → 42"""
    prog = [
        # Main
        Instruction(OP_CALL, 5),       # 0: call function at addr 5
        Instruction(OP_HALT),          # 1: halt with return value
        Instruction(OP_NOP),           # 2: padding
        Instruction(OP_NOP),           # 3: padding
        Instruction(OP_NOP),           # 4: padding
        # Function at addr 5
        Instruction(OP_PUSH, 42),      # 5: push return value
        Instruction(OP_RETURN),        # 6: return
    ]
    return prog, 42


def test_call_with_args():
    """Push arg, call function that doubles it via stack. → 2×arg"""
    prog = [
        # Main: push arg=15, call double
        Instruction(OP_PUSH, 15),      # 0
        Instruction(OP_CALL, 4),       # 1: call double()
        Instruction(OP_HALT),          # 2
        Instruction(OP_NOP),           # 3: padding
        # double() at addr 4: DUP, ADD, RETURN
        Instruction(OP_DUP),           # 4
        Instruction(OP_ADD),           # 5
        Instruction(OP_RETURN),        # 6
    ]
    return prog, 30


def test_locals_scoping():
    """Main sets local 0 = 10, calls function that sets local 0 = 20.
    After return, main reads local 0 → should be 10 (function's local is isolated).
    """
    prog = [
        # Main: set local[0] = 10
        Instruction(OP_PUSH, 10),      # 0
        Instruction(OP_LOCAL_SET, 0),   # 1
        # Call function
        Instruction(OP_CALL, 8),       # 2
        # After return: drop return value, read local[0]
        Instruction(OP_POP),           # 3
        Instruction(OP_LOCAL_GET, 0),   # 4: should be 10
        Instruction(OP_HALT),          # 5
        Instruction(OP_NOP),           # 6: padding
        Instruction(OP_NOP),           # 7: padding
        # Function at addr 8: set local[0] = 20, return
        Instruction(OP_PUSH, 20),      # 8
        Instruction(OP_LOCAL_SET, 0),   # 9: this writes to function's local[0]
        Instruction(OP_LOCAL_GET, 0),   # 10: push function's local[0] = 20
        Instruction(OP_RETURN),        # 11
    ]
    return prog, 10


def test_nested_calls():
    """f calls g calls h, each returns a value. f()+g()+h() = 1+2+3 = 6."""
    prog = [
        # Main: call f, halt
        Instruction(OP_CALL, 3),       # 0: call f()
        Instruction(OP_HALT),          # 1
        Instruction(OP_NOP),           # 2
        # f() at addr 3: push 1, call g(), add results, return
        Instruction(OP_PUSH, 1),       # 3
        Instruction(OP_CALL, 9),       # 4: call g()
        Instruction(OP_ADD),           # 5: 1 + g()
        Instruction(OP_RETURN),        # 6
        Instruction(OP_NOP),           # 7
        Instruction(OP_NOP),           # 8
        # g() at addr 9: push 2, call h(), add results, return
        Instruction(OP_PUSH, 2),       # 9
        Instruction(OP_CALL, 15),      # 10: call h()
        Instruction(OP_ADD),           # 11: 2 + h()
        Instruction(OP_RETURN),        # 12
        Instruction(OP_NOP),           # 13
        Instruction(OP_NOP),           # 14
        # h() at addr 15: push 3, return
        Instruction(OP_PUSH, 3),       # 15
        Instruction(OP_RETURN),        # 16
    ]
    return prog, 6  # 1 + (2 + 3) = 6


def test_recursive_factorial():
    """Recursive factorial: fact(5) = 120.
    fact(n): if n <= 1 return 1; else return n * fact(n-1)
    """
    prog = [
        # Main: push n=5, call fact
        Instruction(OP_PUSH, 5),       # 0
        Instruction(OP_CALL, 3),       # 1: call fact(5)
        Instruction(OP_HALT),          # 2
        # fact(n) at addr 3:
        # n is on stack top
        Instruction(OP_LOCAL_TEE, 0),   # 3: local[0] = n, keep n on stack
        Instruction(OP_PUSH, 1),       # 4
        Instruction(OP_SUB),           # 5: n - 1
        Instruction(OP_DUP),           # 6: dup (n-1)
        Instruction(OP_JZ, 13),        # 7: if n-1 == 0, goto base case
        # Recursive case: push n-1, call fact, multiply
        Instruction(OP_CALL, 3),       # 8: call fact(n-1)
        Instruction(OP_LOCAL_GET, 0),   # 9: push n
        Instruction(OP_MUL),           # 10: n * fact(n-1)
        Instruction(OP_RETURN),        # 11
        Instruction(OP_NOP),           # 12: padding
        # Base case (addr 13): n-1 == 0, so n == 1
        Instruction(OP_POP),           # 13: drop the 0
        Instruction(OP_PUSH, 1),       # 14: return 1
        Instruction(OP_RETURN),        # 15
    ]
    return prog, 120


def test_recursive_factorial_10():
    """Recursive factorial: fact(10) = 3628800."""
    prog = [
        Instruction(OP_PUSH, 10),      # 0
        Instruction(OP_CALL, 3),       # 1
        Instruction(OP_HALT),          # 2
        # fact(n) — same as above
        Instruction(OP_LOCAL_TEE, 0),   # 3
        Instruction(OP_PUSH, 1),       # 4
        Instruction(OP_SUB),           # 5
        Instruction(OP_DUP),           # 6
        Instruction(OP_JZ, 13),        # 7
        Instruction(OP_CALL, 3),       # 8
        Instruction(OP_LOCAL_GET, 0),   # 9
        Instruction(OP_MUL),           # 10
        Instruction(OP_RETURN),        # 11
        Instruction(OP_NOP),           # 12
        Instruction(OP_POP),           # 13
        Instruction(OP_PUSH, 1),       # 14
        Instruction(OP_RETURN),        # 15
    ]
    return prog, 3628800


def test_return_without_call():
    """RETURN without matching CALL should trap."""
    prog = [
        Instruction(OP_PUSH, 42),
        Instruction(OP_RETURN),
    ]
    return prog


def test_locals_across_functions():
    """Each function uses its own locals independently."""
    prog = [
        # Main: set local[0]=100, local[1]=200
        Instruction(OP_PUSH, 100),     # 0
        Instruction(OP_LOCAL_SET, 0),   # 1
        Instruction(OP_PUSH, 200),     # 2
        Instruction(OP_LOCAL_SET, 1),   # 3
        # Call function
        Instruction(OP_CALL, 10),      # 4
        # After return: read main's locals and sum
        Instruction(OP_POP),           # 5: drop return value
        Instruction(OP_LOCAL_GET, 0),   # 6: should be 100
        Instruction(OP_LOCAL_GET, 1),   # 7: should be 200
        Instruction(OP_ADD),           # 8: 300
        Instruction(OP_HALT),          # 9
        # Function at addr 10: set its own local[0]=999, local[1]=888
        Instruction(OP_PUSH, 999),     # 10
        Instruction(OP_LOCAL_SET, 0),   # 11
        Instruction(OP_PUSH, 888),     # 12
        Instruction(OP_LOCAL_SET, 1),   # 13
        Instruction(OP_LOCAL_GET, 0),   # 14: function's local[0] = 999
        Instruction(OP_RETURN),        # 15
    ]
    return prog, 300


# ─── All Phase 17 Tests ──────────────────────────────────────────

PHASE17_TESTS = [
    ("simple_call",           test_simple_call),
    ("call_with_args",        test_call_with_args),
    ("locals_scoping",        test_locals_scoping),
    ("nested_calls",          test_nested_calls),
    ("recursive_fact_5",      test_recursive_factorial),
    ("recursive_fact_10",     test_recursive_factorial_10),
    ("locals_across_funcs",   test_locals_across_functions),
]

PHASE17_TRAP_TESTS = [
    ("return_without_call",   test_return_without_call),
]


# ─── Test Runners ────────────────────────────────────────────────

def test_function_calls():
    """Test all CALL/RETURN programs on both executors."""
    print("=" * 60)
    print("Phase 17: Function Calls (CALL/RETURN)")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0

    for name, test_fn in PHASE17_TESTS:
        prog, expected = test_fn()
        ok, steps = test_algorithm(name, prog, expected, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1
        total += 1

    for name, test_fn in PHASE17_TRAP_TESTS:
        prog = test_fn()
        ok = test_trap_algorithm(name, prog, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Phase 17 tests: {passed}/{total} passed")
    return passed == total


def test_regression():
    """Verify all Phase 4-16 tests still pass."""
    print("\n" + "=" * 60)
    print("Regression: Phase 4-16 Tests")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0

    # Phase 4 basic tests
    for name, test_fn in ALL_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(name, prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 11 extended tests
    ext_tests = [
        ("sub_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_SUB), Instruction(OP_HALT)], 7),
        ("loop_countdown",
         [Instruction(OP_PUSH, 3), Instruction(OP_DUP),
          Instruction(OP_PUSH, 1), Instruction(OP_SUB),
          Instruction(OP_DUP), Instruction(OP_JNZ, 1),
          Instruction(OP_HALT)], 0),
    ]
    for name, prog, expected in ext_tests:
        ok, _ = test_algorithm(name, prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 13 algorithms
    p13_algos = [
        ("fib(10)", *make_fibonacci(10)),
        ("sum(1..10)", *make_sum_1_to_n(10)),
        ("power(2^5)", *make_power_of_2(5)),
    ]
    for name, prog, expected in p13_algos:
        ok, _ = test_algorithm(name, prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 15 local variable tests
    from phase15_local_variables import PHASE15_TESTS
    for name, test_fn in PHASE15_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(f"p15_{name}", prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 16 memory tests
    from phase16_linear_memory import PHASE16_TESTS
    for name, test_fn in PHASE16_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(f"p16_{name}", prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Regression: {passed}/{total} passed")
    return passed == total


def test_model_summary():
    """Verify model architecture: 10 heads, correct D_MODEL and N_OPCODES."""
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)

    model = CompiledModel()
    heads = [
        ("Head 0: prog opcode", model.head_prog_op),
        ("Head 1: prog arg", model.head_prog_arg),
        ("Head 2: stack SP", model.head_stack_a),
        ("Head 3: stack SP-1", model.head_stack_b),
        ("Head 4: stack SP-2", model.head_stack_c),
        ("Head 5: local value", model.head_local_val),
        ("Head 6: local addr", model.head_local_addr),
        ("Head 7: heap value", model.head_heap_val),
        ("Head 8: heap addr", model.head_heap_addr),
        ("Head 9: call stack", model.head_call_stack),
    ]

    total_params = 0
    for name, head in heads:
        p = sum(param.numel() for param in head.parameters())
        total_params += p
        print(f"  {name}: {p} params")

    buf_params = model.M_top.numel() + model.sp_deltas.numel()
    total_params += buf_params
    print(f"  FF dispatch buffers: {buf_params} params")
    print(f"  Total: {total_params} compiled parameters")
    print(f"  D_MODEL: {D_MODEL}")
    print(f"  N_OPCODES: {N_OPCODES}")
    print(f"  Active heads: {len(heads)}")
    print(f"  Address spaces: program, stack, locals, heap, call stack")

    checks = []

    ok = len(heads) == 10
    checks.append(ok)
    print(f"\n  {'PASS' if ok else 'FAIL'}  Head count = {len(heads)} (expected 10)")

    ok = D_MODEL == 51
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  D_MODEL = {D_MODEL} (expected 51)")

    ok = N_OPCODES == 55
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  N_OPCODES = {N_OPCODES} (expected 55)")

    ok = model.sp_deltas.shape[0] == N_OPCODES
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  sp_deltas size = {model.sp_deltas.shape[0]} (expected {N_OPCODES})")

    ok = model.M_top.shape == (N_OPCODES, 6)
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  M_top shape = {tuple(model.M_top.shape)} (expected ({N_OPCODES}, 6))")

    return all(checks)


def test_invariants():
    """Verify all invariants from the issue spec."""
    print("\n" + "=" * 60)
    print("Invariant Checks")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()
    checks = []

    # 1. CALL saves and RETURN restores the correct return address
    prog, expected = test_simple_call()
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    ok = np_trace.steps[-1].top == 42 and pt_trace.steps[-1].top == 42
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  CALL/RETURN restores return address")

    # 2. Each function gets isolated locals
    prog, expected = test_locals_scoping()
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    ok = np_trace.steps[-1].top == 10 and pt_trace.steps[-1].top == 10
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  Locals isolation (caller={np_trace.steps[-1].top}, expected=10)")

    # 3. Recursive factorial produces correct results
    prog, expected = test_recursive_factorial()
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    ok = np_trace.steps[-1].top == 120 and pt_trace.steps[-1].top == 120
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  fact(5)={np_trace.steps[-1].top} (expected 120)")

    prog, expected = test_recursive_factorial_10()
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    ok = np_trace.steps[-1].top == 3628800 and pt_trace.steps[-1].top == 3628800
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  fact(10)={np_trace.steps[-1].top} (expected 3628800)")

    # 4. RETURN without CALL traps
    prog = test_return_without_call()
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_trapped = np_trace.steps[-1].op == OP_TRAP
    pt_trapped = pt_trace.steps[-1].op == OP_TRAP
    ok = np_trapped and pt_trapped
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  RETURN without CALL traps (np={np_trapped}, pt={pt_trapped})")

    # 5. All traces match (numpy vs pytorch)
    trace_match_ok = True
    for name, test_fn in PHASE17_TESTS:
        prog, expected = test_fn()
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, detail = compare_traces(np_trace, pt_trace)
        if not match:
            print(f"  FAIL  Trace mismatch for {name}: {detail}")
            trace_match_ok = False
    checks.append(trace_match_ok)
    print(f"  {'PASS' if trace_match_ok else 'FAIL'}  All traces match (numpy vs pytorch)")

    print(f"\n  Invariants: {sum(checks)}/{len(checks)} passed")
    return all(checks)


def main():
    print("=" * 60)
    print("Phase 17: Function Calls — Full Test Suite")
    print("=" * 60)
    print()

    t0 = time.time()

    results = []
    results.append(("Function call ops", test_function_calls()))
    results.append(("Regression (Phase 4-16)", test_regression()))
    results.append(("Model summary", test_model_summary()))
    results.append(("Invariants", test_invariants()))

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  {status}  {name}")

    print(f"\n  Time: {elapsed:.2f}s")

    if all_pass:
        print("\n  All Phase 17 tests pass. Function calls fully operational.")
    else:
        print("\n  FAILURES detected. See details above.")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
