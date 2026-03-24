"""Phase 16: Linear memory address space (I32.LOAD/STORE + width variants).

Adds a heap parabolic address space with 8 opcodes and 2 new attention heads
(Heads 7-8). Part of Tier 2 Chunk 2 (Issue #25).

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
    OP_I32_LOAD8_U, OP_I32_LOAD8_S,
    OP_I32_LOAD16_U, OP_I32_LOAD16_S,
    OP_I32_STORE8, OP_I32_STORE16,
    OP_NAMES,
)
from executor import NumPyExecutor, CompiledModel, TorchExecutor
from programs import ALL_TESTS, make_fibonacci, make_power_of_2, make_sum_1_to_n


# ─── Test Programs ──────────────────────────────────────────────

def test_store_and_load():
    """Store 42 at addr 0, load it back. → 42"""
    prog = [
        Instruction(OP_PUSH, 0),       # addr
        Instruction(OP_PUSH, 42),      # val
        Instruction(OP_I32_STORE),     # memory[0] = 42
        Instruction(OP_PUSH, 0),       # addr
        Instruction(OP_I32_LOAD),      # load memory[0]
        Instruction(OP_HALT),
    ]
    return prog, 42


def test_multiple_addresses():
    """Store different values at addresses 0, 1, 2, read them back and sum."""
    prog = [
        # Store 10 at addr 0
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 10),
        Instruction(OP_I32_STORE),
        # Store 20 at addr 1
        Instruction(OP_PUSH, 1),
        Instruction(OP_PUSH, 20),
        Instruction(OP_I32_STORE),
        # Store 30 at addr 2
        Instruction(OP_PUSH, 2),
        Instruction(OP_PUSH, 30),
        Instruction(OP_I32_STORE),
        # Load and sum: 10 + 20 + 30 = 60
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_PUSH, 1),
        Instruction(OP_I32_LOAD),
        Instruction(OP_ADD),
        Instruction(OP_PUSH, 2),
        Instruction(OP_I32_LOAD),
        Instruction(OP_ADD),
        Instruction(OP_HALT),
    ]
    return prog, 60


def test_uninitialized_load():
    """Load from untouched address → 0."""
    prog = [
        Instruction(OP_PUSH, 99),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    return prog, 0


def test_overwrite_memory():
    """Store 1 then 2 at addr 0; load should return 2."""
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 1),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 2),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    return prog, 2


def test_store8_mask():
    """Store 0x1FF via STORE8 → should mask to 0xFF."""
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x1FF),
        Instruction(OP_I32_STORE8),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    return prog, 0xFF


def test_store16_mask():
    """Store 0x1FFFF via STORE16 → should mask to 0xFFFF."""
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x1FFFF),
        Instruction(OP_I32_STORE16),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    return prog, 0xFFFF


def test_load8_u():
    """Store 0x180, LOAD8_U → 0x80 (128)."""
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x180),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD8_U),
        Instruction(OP_HALT),
    ]
    return prog, 0x80  # 128


def test_load8_s():
    """Store 0x180, LOAD8_S → sign-extend 0x80 = -128."""
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x180),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD8_S),
        Instruction(OP_HALT),
    ]
    return prog, -128


def test_load16_u():
    """Store 0x18000, LOAD16_U → 0x8000 (32768)."""
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x18000),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD16_U),
        Instruction(OP_HALT),
    ]
    return prog, 0x8000  # 32768


def test_load16_s():
    """Store 0x18000, LOAD16_S → sign-extend 0x8000 = -32768."""
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x18000),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD16_S),
        Instruction(OP_HALT),
    ]
    return prog, -32768


def test_array_sum():
    """Store [10, 20, 30] at addrs 0-2, sum via loop with locals."""
    prog = [
        # Store array
        Instruction(OP_PUSH, 0),          # 0
        Instruction(OP_PUSH, 10),         # 1
        Instruction(OP_I32_STORE),        # 2
        Instruction(OP_PUSH, 1),          # 3
        Instruction(OP_PUSH, 20),         # 4
        Instruction(OP_I32_STORE),        # 5
        Instruction(OP_PUSH, 2),          # 6
        Instruction(OP_PUSH, 30),         # 7
        Instruction(OP_I32_STORE),        # 8
        # Init: acc=0 (local[0]), i=0 (local[1])
        Instruction(OP_PUSH, 0),          # 9
        Instruction(OP_LOCAL_SET, 0),     # 10: acc = 0
        Instruction(OP_PUSH, 0),          # 11
        Instruction(OP_LOCAL_SET, 1),     # 12: i = 0
        # Loop (ip=13):
        Instruction(OP_LOCAL_GET, 1),     # 13: push i
        Instruction(OP_PUSH, 3),          # 14: push len
        Instruction(OP_SUB),             # 15: i - len (0 when i == len)
        Instruction(OP_JZ, 28),           # 16: if i == len, exit loop
        # Load memory[i], add to acc
        Instruction(OP_LOCAL_GET, 1),     # 17: push i
        Instruction(OP_I32_LOAD),         # 18: load memory[i]
        Instruction(OP_LOCAL_GET, 0),     # 19: push acc
        Instruction(OP_ADD),              # 20: acc + memory[i]
        Instruction(OP_LOCAL_SET, 0),     # 21: acc = acc + memory[i]
        # i++
        Instruction(OP_LOCAL_GET, 1),     # 22: push i
        Instruction(OP_PUSH, 1),          # 23
        Instruction(OP_ADD),              # 24: i + 1
        Instruction(OP_LOCAL_SET, 1),     # 25: i = i + 1
        Instruction(OP_PUSH, 1),          # 26
        Instruction(OP_JNZ, 13),          # 27: unconditional jump to loop
        # Done
        Instruction(OP_LOCAL_GET, 0),     # 28: push acc
        Instruction(OP_HALT),             # 29
    ]
    return prog, 60


def test_memory_with_locals():
    """Use locals and memory together: store via stack, load via local index."""
    prog = [
        # Store 100 at addr 5
        Instruction(OP_PUSH, 5),
        Instruction(OP_PUSH, 100),
        Instruction(OP_I32_STORE),
        # Set local[0] = 5 (the address)
        Instruction(OP_PUSH, 5),
        Instruction(OP_LOCAL_SET, 0),
        # Load from address stored in local[0]
        Instruction(OP_LOCAL_GET, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    return prog, 100


# ─── All Phase 16 Tests ──────────────────────────────────────────

PHASE16_TESTS = [
    ("store_and_load",      test_store_and_load),
    ("multiple_addresses",  test_multiple_addresses),
    ("uninitialized_load",  test_uninitialized_load),
    ("overwrite_memory",    test_overwrite_memory),
    ("store8_mask",         test_store8_mask),
    ("store16_mask",        test_store16_mask),
    ("load8_u",             test_load8_u),
    ("load8_s",             test_load8_s),
    ("load16_u",            test_load16_u),
    ("load16_s",            test_load16_s),
    ("array_sum",           test_array_sum),
    ("memory_with_locals",  test_memory_with_locals),
]


# ─── Test Runners ────────────────────────────────────────────────

def test_linear_memory():
    """Test all I32.LOAD/STORE programs on both executors."""
    print("=" * 60)
    print("Phase 16: Linear Memory (I32.LOAD/STORE + width variants)")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0

    for name, test_fn in PHASE16_TESTS:
        prog, expected = test_fn()
        ok, steps = test_algorithm(name, prog, expected, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Phase 16 tests: {passed}/{total} passed")
    return passed == total


def test_regression():
    """Verify all Phase 4-15 tests still pass."""
    print("\n" + "=" * 60)
    print("Regression: Phase 4-15 Tests")
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

    # Phase 15 local variable tests
    from phase15_local_variables import PHASE15_TESTS
    for name, test_fn in PHASE15_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(f"p15_{name}", prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Regression: {passed}/{total} passed")
    return passed == total


def test_model_summary():
    """Verify model architecture: 9 heads, correct D_MODEL and N_OPCODES."""
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
    ]

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

    # Verify head count
    ok = len(heads) == 9
    checks.append(ok)
    print(f"\n  {'PASS' if ok else 'FAIL'}  Head count = {len(heads)} (expected 9)")

    # Verify D_MODEL
    ok = D_MODEL == 45
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  D_MODEL = {D_MODEL} (expected 45)")

    # Verify N_OPCODES
    ok = N_OPCODES == 53
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  N_OPCODES = {N_OPCODES} (expected 53)")

    # Verify sp_deltas size
    ok = model.sp_deltas.shape[0] == N_OPCODES
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  sp_deltas size = {model.sp_deltas.shape[0]} (expected {N_OPCODES})")

    # Verify M_top size
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

    # 1. STORE then LOAD roundtrips exactly
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 42),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 42 and pt_top == 42
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  STORE+LOAD roundtrip (np={np_top}, pt={pt_top})")

    # 2. LOAD from untouched address returns 0
    prog = [
        Instruction(OP_PUSH, 999),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 0 and pt_top == 0
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  Untouched addr returns 0 (np={np_top}, pt={pt_top})")

    # 3. STORE8 masks to low 8 bits
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x1FF),
        Instruction(OP_I32_STORE8),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 0xFF and pt_top == 0xFF
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  STORE8 masks (np={np_top}, pt={pt_top}, expected=255)")

    # 4. STORE16 masks to low 16 bits
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x1FFFF),
        Instruction(OP_I32_STORE16),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 0xFFFF and pt_top == 0xFFFF
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  STORE16 masks (np={np_top}, pt={pt_top}, expected=65535)")

    # 5. LOAD8_S sign-extends (0x80 → -128)
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x80),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD8_S),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == -128 and pt_top == -128
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  LOAD8_S sign-extends (np={np_top}, pt={pt_top}, expected=-128)")

    # 6. LOAD8_U zero-extends (0x80 → 128)
    prog = [
        Instruction(OP_PUSH, 0),
        Instruction(OP_PUSH, 0x80),
        Instruction(OP_I32_STORE),
        Instruction(OP_PUSH, 0),
        Instruction(OP_I32_LOAD8_U),
        Instruction(OP_HALT),
    ]
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    np_top = np_trace.steps[-1].top
    pt_top = pt_trace.steps[-1].top
    ok = np_top == 128 and pt_top == 128
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  LOAD8_U zero-extends (np={np_top}, pt={pt_top}, expected=128)")

    # 7. All traces match (numpy vs pytorch)
    trace_match_ok = True
    for name, test_fn in PHASE16_TESTS:
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
    print("Phase 16: Linear Memory — Full Test Suite")
    print("=" * 60)
    print()

    t0 = time.time()

    results = []
    results.append(("Linear memory ops", test_linear_memory()))
    results.append(("Regression (Phase 4-15)", test_regression()))
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
        print("\n  All Phase 16 tests pass. Linear memory fully operational.")
    else:
        print("\n  FAILURES detected. See details above.")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
