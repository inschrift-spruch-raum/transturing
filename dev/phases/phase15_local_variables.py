"""Phase 15: Local variables address space (LOCAL.GET/SET/TEE).

Adds a new parabolic address space for local variables with 3 opcodes
and 2 new attention heads (Heads 5-6).

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
    OP_NAMES,
)
from executor import NumPyExecutor, CompiledModel, TorchExecutor
from programs import ALL_TESTS, make_fibonacci, make_power_of_2, make_sum_1_to_n


# ─── Test Programs ──────────────────────────────────────────────

def test_set_and_get():
    """PUSH 42 / LOCAL.SET 0 / LOCAL.GET 0 / HALT → 42"""
    prog = [
        Instruction(OP_PUSH, 42),
        Instruction(OP_LOCAL_SET, 0),
        Instruction(OP_LOCAL_GET, 0),
        Instruction(OP_HALT),
    ]
    return prog, 42


def test_swap_via_locals():
    """Swap two values using locals instead of ROT gymnastics.
    PUSH 10 / PUSH 20 / LOCAL.SET 0 / LOCAL.SET 1 / LOCAL.GET 0 / LOCAL.GET 1 / HALT
    → stack has [20, 10] (swapped), top = 10
    """
    prog = [
        Instruction(OP_PUSH, 10),   # stack: [10]
        Instruction(OP_PUSH, 20),   # stack: [10, 20]
        Instruction(OP_LOCAL_SET, 0),  # pop 20 → local[0], stack: [10]
        Instruction(OP_LOCAL_SET, 1),  # pop 10 → local[1], stack: []
        Instruction(OP_LOCAL_GET, 0),  # push local[0]=20, stack: [20]
        Instruction(OP_LOCAL_GET, 1),  # push local[1]=10, stack: [20, 10]
        Instruction(OP_HALT),
    ]
    return prog, 10


def test_tee():
    """PUSH 99 / LOCAL.TEE 0 / LOCAL.GET 0 / ADD / HALT → 198 (99+99)"""
    prog = [
        Instruction(OP_PUSH, 99),
        Instruction(OP_LOCAL_TEE, 0),  # copy 99 to local[0], stack still [99]
        Instruction(OP_LOCAL_GET, 0),  # push local[0]=99, stack: [99, 99]
        Instruction(OP_ADD),           # 99 + 99 = 198
        Instruction(OP_HALT),
    ]
    return prog, 198


def test_multiple_locals():
    """Use locals 0-3 in a computation: (10 + 20) * (30 + 40) via MUL is nonlinear,
    so use addition: local[0]=10, local[1]=20, local[2]=30, local[3]=40
    result = local[0] + local[1] + local[2] + local[3] = 100
    """
    prog = [
        Instruction(OP_PUSH, 10),
        Instruction(OP_LOCAL_SET, 0),
        Instruction(OP_PUSH, 20),
        Instruction(OP_LOCAL_SET, 1),
        Instruction(OP_PUSH, 30),
        Instruction(OP_LOCAL_SET, 2),
        Instruction(OP_PUSH, 40),
        Instruction(OP_LOCAL_SET, 3),
        # Now sum them: local[0] + local[1] + local[2] + local[3]
        Instruction(OP_LOCAL_GET, 0),  # push 10
        Instruction(OP_LOCAL_GET, 1),  # push 20
        Instruction(OP_ADD),           # 30
        Instruction(OP_LOCAL_GET, 2),  # push 30
        Instruction(OP_ADD),           # 60
        Instruction(OP_LOCAL_GET, 3),  # push 40
        Instruction(OP_ADD),           # 100
        Instruction(OP_HALT),
    ]
    return prog, 100


def test_overwrite():
    """PUSH 1 / LOCAL.SET 0 / PUSH 2 / LOCAL.SET 0 / LOCAL.GET 0 / HALT → 2"""
    prog = [
        Instruction(OP_PUSH, 1),
        Instruction(OP_LOCAL_SET, 0),
        Instruction(OP_PUSH, 2),
        Instruction(OP_LOCAL_SET, 0),
        Instruction(OP_LOCAL_GET, 0),
        Instruction(OP_HALT),
    ]
    return prog, 2


def test_unset_local():
    """LOCAL.GET of unset local should return 0."""
    prog = [
        Instruction(OP_LOCAL_GET, 5),  # never set → 0
        Instruction(OP_HALT),
    ]
    return prog, 0


def test_tee_preserves_stack():
    """LOCAL.TEE should leave the stack unchanged."""
    prog = [
        Instruction(OP_PUSH, 77),
        Instruction(OP_LOCAL_TEE, 0),  # stack still [77]
        Instruction(OP_HALT),          # top = 77
    ]
    return prog, 77


def test_locals_with_stack_ops():
    """Mix locals and stack operations."""
    prog = [
        Instruction(OP_PUSH, 5),
        Instruction(OP_DUP),            # stack: [5, 5]
        Instruction(OP_LOCAL_SET, 0),    # pop 5 → local[0], stack: [5]
        Instruction(OP_PUSH, 3),         # stack: [5, 3]
        Instruction(OP_ADD),             # stack: [8]
        Instruction(OP_LOCAL_GET, 0),    # push local[0]=5, stack: [8, 5]
        Instruction(OP_ADD),             # stack: [13]
        Instruction(OP_HALT),
    ]
    return prog, 13


def test_accumulator_loop():
    """Use a local as a loop accumulator: sum 1+2+3+4+5 = 15.
    local[0] = accumulator, local[1] = counter
    """
    prog = [
        # Init: acc=0, counter=5
        Instruction(OP_PUSH, 0),
        Instruction(OP_LOCAL_SET, 0),    # local[0] = 0 (acc)
        Instruction(OP_PUSH, 5),
        Instruction(OP_LOCAL_SET, 1),    # local[1] = 5 (counter)
        # Loop body (ip=4):
        Instruction(OP_LOCAL_GET, 0),    # push acc
        Instruction(OP_LOCAL_GET, 1),    # push counter
        Instruction(OP_ADD),             # acc + counter
        Instruction(OP_LOCAL_SET, 0),    # store new acc
        # Decrement counter
        Instruction(OP_LOCAL_GET, 1),    # push counter
        Instruction(OP_PUSH, 1),
        Instruction(OP_SUB),             # counter - 1
        Instruction(OP_LOCAL_TEE, 1),    # store and keep on stack
        Instruction(OP_JNZ, 4),          # loop if counter != 0
        # Done: push result
        Instruction(OP_LOCAL_GET, 0),    # push acc
        Instruction(OP_HALT),
    ]
    return prog, 15


# ─── All Phase 15 Tests ──────────────────────────────────────────

PHASE15_TESTS = [
    ("set_and_get",         test_set_and_get),
    ("swap_via_locals",     test_swap_via_locals),
    ("tee",                 test_tee),
    ("multiple_locals",     test_multiple_locals),
    ("overwrite",           test_overwrite),
    ("unset_local",         test_unset_local),
    ("tee_preserves_stack", test_tee_preserves_stack),
    ("locals_with_stack",   test_locals_with_stack_ops),
    ("accumulator_loop",    test_accumulator_loop),
]


# ─── Test Runners ────────────────────────────────────────────────

def test_local_variables():
    """Test all LOCAL.GET/SET/TEE programs on both executors."""
    print("=" * 60)
    print("Phase 15: Local Variables (LOCAL.GET/SET/TEE)")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0

    for name, test_fn in PHASE15_TESTS:
        prog, expected = test_fn()
        ok, steps = test_algorithm(name, prog, expected, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Phase 15 tests: {passed}/{total} passed")
    return passed == total


def test_regression():
    """Verify all Phase 4-14 tests still pass with expanded D_MODEL."""
    print("\n" + "=" * 60)
    print("Regression: Phase 4-14 Tests")
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
        ("jz_taken",
         [Instruction(OP_PUSH, 0), Instruction(OP_JZ, 3),
          Instruction(OP_HALT),
          Instruction(OP_PUSH, 42), Instruction(OP_HALT)], 42),
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

    print(f"\n  Regression: {passed}/{total} passed")
    return passed == total


def test_model_summary():
    """Verify model architecture: current heads, D_MODEL, N_OPCODES."""
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
    ]
    # Include heap heads if they exist (Phase 16+)
    if hasattr(model, 'head_heap_val'):
        heads.append(("Head 7: heap value", model.head_heap_val))
        heads.append(("Head 8: heap addr", model.head_heap_addr))

    total_params = 0
    for name, head in heads:
        p = sum(param.numel() for param in head.parameters())
        total_params += p
        print(f"  {name}: {p} params")

    # Count buffer params
    buf_params = model.M_top.numel() + model.sp_deltas.numel()
    total_params += buf_params
    print(f"  FF dispatch buffers: {buf_params} params")
    print(f"  Total: {total_params} compiled parameters")
    print(f"  D_MODEL: {D_MODEL}")
    print(f"  N_OPCODES: {N_OPCODES}")
    print(f"  Active heads: {len(heads)}")

    checks = []

    # Verify minimum head count (at least 7 for locals)
    ok = len(heads) >= 7
    checks.append(ok)
    print(f"\n  {'PASS' if ok else 'FAIL'}  Head count = {len(heads)} (expected >= 7)")

    # Verify D_MODEL
    ok = D_MODEL >= 42
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  D_MODEL = {D_MODEL} (expected >= 42)")

    # Verify N_OPCODES
    ok = N_OPCODES >= 45
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  N_OPCODES = {N_OPCODES} (expected >= 45)")

    # Verify sp_deltas size
    ok = model.sp_deltas.shape[0] == N_OPCODES
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  sp_deltas size = {model.sp_deltas.shape[0]} (expected {N_OPCODES})")

    # Verify M_top rows match N_OPCODES
    ok = model.M_top.shape[0] == N_OPCODES
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  M_top rows = {model.M_top.shape[0]} (expected {N_OPCODES})")

    return all(checks)


def test_invariants():
    """Verify all invariants from the issue spec."""
    print("\n" + "=" * 60)
    print("Invariant Checks")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()
    checks = []

    # 1. LOCAL.GET of an unset local returns 0
    prog = [Instruction(OP_LOCAL_GET, 99), Instruction(OP_HALT)]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 0 and pt_top == 0
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  Unset local returns 0 (np={np_top}, pt={pt_top})")

    # 2. LOCAL.SET + LOCAL.GET roundtrip
    prog = [
        Instruction(OP_PUSH, 42),
        Instruction(OP_LOCAL_SET, 0),
        Instruction(OP_LOCAL_GET, 0),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 42 and pt_top == 42
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  SET+GET roundtrip (np={np_top}, pt={pt_top})")

    # 3. LOCAL.TEE leaves stack unchanged AND stores the value
    prog = [
        Instruction(OP_PUSH, 55),
        Instruction(OP_LOCAL_TEE, 0),
        Instruction(OP_LOCAL_GET, 0),  # should be 55
        Instruction(OP_ADD),           # 55 + 55 = 110
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 110 and pt_top == 110
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  TEE preserves stack + stores (np={np_top}, pt={pt_top})")

    # 4. Overwrite returns new value
    prog = [
        Instruction(OP_PUSH, 1),
        Instruction(OP_LOCAL_SET, 0),
        Instruction(OP_PUSH, 999),
        Instruction(OP_LOCAL_SET, 0),
        Instruction(OP_LOCAL_GET, 0),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 999 and pt_top == 999
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  Overwrite returns new value (np={np_top}, pt={pt_top})")

    # 5. NumPy and PyTorch produce identical traces for all phase 15 test programs
    trace_match_ok = True
    for name, test_fn in PHASE15_TESTS:
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
    print("Phase 15: Local Variables — Full Test Suite")
    print("=" * 60)
    print()

    t0 = time.time()

    results = []
    results.append(("Local variable ops", test_local_variables()))
    results.append(("Regression (Phase 4-14)", test_regression()))
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
        print("\n  All Phase 15 tests pass. Local variables fully operational.")
    else:
        print("\n  FAILURES detected. See details above.")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
