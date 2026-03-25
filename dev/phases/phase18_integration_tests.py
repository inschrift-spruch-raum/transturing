"""Phase 18: Integration tests — locals + linear memory + function calls.

Tests exercising all three Tier 2 memory spaces together, plus full regression.
Part of Tier 2 Chunk 4 (Issue #27). This chunk is TESTS ONLY — no new opcodes
or architecture changes.
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
    OP_CALL, OP_RETURN, OP_TRAP,
    OP_GT_S, OP_LT_S, OP_GE_S,
    OP_NAMES,
)
from executor import NumPyExecutor, CompiledModel, TorchExecutor
from programs import (
    ALL_TESTS, make_fibonacci, make_power_of_2, make_sum_1_to_n,
    make_factorial,
)


# ─── Integration Test Programs ──────────────────────────────────

def test_bubble_sort_3():
    """Sort 3 elements [3, 1, 2] using locals + memory + loops → min=1."""
    prog = [
        # Store array [3, 1, 2] in memory
        Instruction(OP_PUSH, 0), Instruction(OP_PUSH, 3), Instruction(OP_I32_STORE),  # 0-2
        Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 1), Instruction(OP_I32_STORE),  # 3-5
        Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 2), Instruction(OP_I32_STORE),  # 6-8
        # Outer loop: pass = local[0], start at 0
        Instruction(OP_PUSH, 0),           # 9
        Instruction(OP_LOCAL_SET, 0),      # 10: pass=0
        # Outer loop (ip=11):
        Instruction(OP_LOCAL_GET, 0),      # 11: push pass
        Instruction(OP_PUSH, 2),           # 12
        Instruction(OP_SUB),              # 13: pass - 2
        Instruction(OP_JZ, 59),            # 14: if pass==2, done → ip=59
        # Inner: j = local[1], start at 0
        Instruction(OP_PUSH, 0),           # 15
        Instruction(OP_LOCAL_SET, 1),      # 16: j=0
        # Inner loop (ip=17):
        Instruction(OP_LOCAL_GET, 1),      # 17: push j
        Instruction(OP_PUSH, 2),           # 18: n-1=2
        Instruction(OP_SUB),              # 19: j - 2
        Instruction(OP_JZ, 50),            # 20: if j==2, inner done → ip=50
        # Compare mem[j] and mem[j+1]
        Instruction(OP_LOCAL_GET, 1),      # 21: j
        Instruction(OP_I32_LOAD),          # 22: mem[j]
        Instruction(OP_LOCAL_SET, 2),      # 23: local[2] = mem[j]
        Instruction(OP_LOCAL_GET, 1),      # 24: j
        Instruction(OP_PUSH, 1),           # 25
        Instruction(OP_ADD),               # 26: j+1
        Instruction(OP_I32_LOAD),          # 27: mem[j+1]
        Instruction(OP_LOCAL_SET, 3),      # 28: local[3] = mem[j+1]
        # if mem[j] > mem[j+1], swap
        Instruction(OP_LOCAL_GET, 2),      # 29: mem[j]
        Instruction(OP_LOCAL_GET, 3),      # 30: mem[j+1]
        Instruction(OP_GT_S),              # 31: mem[j] > mem[j+1]?
        Instruction(OP_JZ, 44),            # 32: if not, skip swap → ip=44
        # Swap: mem[j] = local[3], mem[j+1] = local[2]
        Instruction(OP_LOCAL_GET, 1),      # 33: j (addr)
        Instruction(OP_LOCAL_GET, 3),      # 34: mem[j+1] (val)
        Instruction(OP_I32_STORE),         # 35: mem[j] = mem[j+1]
        Instruction(OP_LOCAL_GET, 1),      # 36: j
        Instruction(OP_PUSH, 1),           # 37
        Instruction(OP_ADD),               # 38: j+1 (addr)
        Instruction(OP_LOCAL_GET, 2),      # 39: mem[j] (val)
        Instruction(OP_I32_STORE),         # 40: mem[j+1] = mem[j]
        Instruction(OP_PUSH, 1),           # 41
        Instruction(OP_JNZ, 44),           # 42: jump to j++ → ip=44
        Instruction(OP_NOP),               # 43: padding
        # j++ (ip=44):
        Instruction(OP_LOCAL_GET, 1),      # 44
        Instruction(OP_PUSH, 1),           # 45
        Instruction(OP_ADD),               # 46
        Instruction(OP_LOCAL_SET, 1),      # 47: j++
        Instruction(OP_PUSH, 1),           # 48
        Instruction(OP_JNZ, 17),           # 49: back to inner loop
        # Inner done (ip=50): pass++
        Instruction(OP_LOCAL_GET, 0),      # 50
        Instruction(OP_PUSH, 1),           # 51
        Instruction(OP_ADD),               # 52
        Instruction(OP_LOCAL_SET, 0),      # 53: pass++
        Instruction(OP_PUSH, 1),           # 54
        Instruction(OP_JNZ, 11),           # 55: back to outer loop
        Instruction(OP_NOP),               # 56
        Instruction(OP_NOP),               # 57
        Instruction(OP_NOP),               # 58
        # Done (ip=59): load mem[0] (should be 1)
        Instruction(OP_PUSH, 0),           # 59
        Instruction(OP_I32_LOAD),          # 60
        Instruction(OP_HALT),              # 61
    ]
    return prog, 1


def test_recursive_fib_with_locals():
    """Recursive Fibonacci using CALL/RETURN + locals.
    fib(n): if n<=1 return n; return fib(n-1) + fib(n-2)
    fib(7) = 13
    """
    prog = [
        # Main
        Instruction(OP_PUSH, 7),       # 0
        Instruction(OP_CALL, 3),       # 1
        Instruction(OP_HALT),          # 2
        # fib(n) at addr 3:
        Instruction(OP_LOCAL_TEE, 0),   # 3: local[0] = n, keep on stack
        Instruction(OP_PUSH, 2),       # 4
        Instruction(OP_LT_S),          # 5: n < 2?
        Instruction(OP_JZ, 10),        # 6: if not, recursive case
        # Base case: n < 2, return n
        Instruction(OP_LOCAL_GET, 0),   # 7
        Instruction(OP_RETURN),        # 8
        Instruction(OP_NOP),           # 9
        # Recursive case (ip=10): fib(n-1) + fib(n-2)
        Instruction(OP_LOCAL_GET, 0),   # 10: n
        Instruction(OP_PUSH, 1),       # 11
        Instruction(OP_SUB),           # 12: n-1
        Instruction(OP_CALL, 3),       # 13: fib(n-1)
        Instruction(OP_LOCAL_SET, 1),   # 14: local[1] = fib(n-1)
        Instruction(OP_LOCAL_GET, 0),   # 15: n
        Instruction(OP_PUSH, 2),       # 16
        Instruction(OP_SUB),           # 17: n-2
        Instruction(OP_CALL, 3),       # 18: fib(n-2)
        Instruction(OP_LOCAL_GET, 1),   # 19: fib(n-1)
        Instruction(OP_ADD),           # 20: fib(n-1) + fib(n-2)
        Instruction(OP_RETURN),        # 21
    ]
    return prog, 13


def test_array_sum_via_function():
    """Store array in memory, call sum(base, len) function.
    Uses locals for loop counter and accumulator.
    """
    prog = [
        # Store [10, 20, 30, 40, 50] at addrs 0-4
        Instruction(OP_PUSH, 0), Instruction(OP_PUSH, 10), Instruction(OP_I32_STORE),  # 0-2
        Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 20), Instruction(OP_I32_STORE),  # 3-5
        Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 30), Instruction(OP_I32_STORE),  # 6-8
        Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 40), Instruction(OP_I32_STORE),  # 9-11
        Instruction(OP_PUSH, 4), Instruction(OP_PUSH, 50), Instruction(OP_I32_STORE),  # 12-14
        # Call sum(base_addr=0, length=5): push args, call
        Instruction(OP_PUSH, 0),           # 15: base_addr
        Instruction(OP_PUSH, 5),           # 16: length
        Instruction(OP_CALL, 19),          # 17: call sum
        Instruction(OP_HALT),              # 18
        # sum(base, len) at addr 19:
        # Stack has: base, len (len on top)
        Instruction(OP_LOCAL_SET, 1),      # 19: local[1] = len
        Instruction(OP_LOCAL_SET, 0),      # 20: local[0] = base
        Instruction(OP_PUSH, 0),           # 21: acc = 0
        Instruction(OP_LOCAL_SET, 2),      # 22: local[2] = acc
        Instruction(OP_PUSH, 0),           # 23: i = 0
        Instruction(OP_LOCAL_SET, 3),      # 24: local[3] = i
        # Loop (ip=25):
        Instruction(OP_LOCAL_GET, 3),      # 25: i
        Instruction(OP_LOCAL_GET, 1),      # 26: len
        Instruction(OP_SUB),              # 27: len - i (actually i - len due to SUB semantics)
        Instruction(OP_JZ, 42),            # 28: if i == len, exit → ip=42
        # Load mem[base + i]
        Instruction(OP_LOCAL_GET, 0),      # 29: base
        Instruction(OP_LOCAL_GET, 3),      # 30: i
        Instruction(OP_ADD),               # 31: base + i
        Instruction(OP_I32_LOAD),          # 32: mem[base+i]
        Instruction(OP_LOCAL_GET, 2),      # 33: acc
        Instruction(OP_ADD),               # 34: acc + mem[base+i]
        Instruction(OP_LOCAL_SET, 2),      # 35: acc = acc + val
        # i++
        Instruction(OP_LOCAL_GET, 3),      # 36
        Instruction(OP_PUSH, 1),           # 37
        Instruction(OP_ADD),               # 38
        Instruction(OP_LOCAL_SET, 3),      # 39: i++
        Instruction(OP_PUSH, 1),           # 40
        Instruction(OP_JNZ, 25),           # 41: loop back
        # Return acc (ip=42):
        Instruction(OP_LOCAL_GET, 2),      # 42: push acc
        Instruction(OP_RETURN),            # 43
    ]
    return prog, 150  # 10+20+30+40+50


def test_memory_stack_via_functions():
    """Implement push/pop on a secondary memory-based stack using functions.
    Uses local[0] as stack pointer (memory address).
    """
    prog = [
        # Init: memory stack pointer = 100 (store in local[0])
        Instruction(OP_PUSH, 100),
        Instruction(OP_LOCAL_SET, 0),      # 1: local[0] = msp = 100
        # mem_push(42): addr=local[0], store val, increment msp
        Instruction(OP_LOCAL_GET, 0),      # 2: msp
        Instruction(OP_PUSH, 42),          # 3: val
        Instruction(OP_I32_STORE),         # 4: mem[100] = 42
        Instruction(OP_LOCAL_GET, 0),      # 5
        Instruction(OP_PUSH, 1),           # 6
        Instruction(OP_ADD),               # 7
        Instruction(OP_LOCAL_SET, 0),      # 8: msp = 101
        # mem_push(99)
        Instruction(OP_LOCAL_GET, 0),      # 9
        Instruction(OP_PUSH, 99),          # 10
        Instruction(OP_I32_STORE),         # 11: mem[101] = 99
        Instruction(OP_LOCAL_GET, 0),      # 12
        Instruction(OP_PUSH, 1),           # 13
        Instruction(OP_ADD),               # 14
        Instruction(OP_LOCAL_SET, 0),      # 15: msp = 102
        # mem_pop(): decrement msp, load mem[msp]
        Instruction(OP_LOCAL_GET, 0),      # 16
        Instruction(OP_PUSH, 1),           # 17
        Instruction(OP_SUB),               # 18: msp - 1 = 101
        Instruction(OP_LOCAL_TEE, 0),      # 19: msp = 101, keep on stack
        Instruction(OP_I32_LOAD),          # 20: mem[101] = 99
        Instruction(OP_LOCAL_SET, 1),      # 21: local[1] = popped val (99)
        # mem_pop() again
        Instruction(OP_LOCAL_GET, 0),      # 22
        Instruction(OP_PUSH, 1),           # 23
        Instruction(OP_SUB),               # 24: msp - 1 = 100
        Instruction(OP_LOCAL_TEE, 0),      # 25: msp = 100
        Instruction(OP_I32_LOAD),          # 26: mem[100] = 42
        # Add the two popped values: 99 + 42 = 141
        Instruction(OP_LOCAL_GET, 1),      # 27: 99
        Instruction(OP_ADD),               # 28: 42 + 99 = 141
        Instruction(OP_HALT),              # 29
    ]
    return prog, 141


def test_multi_function_max():
    """Store values in memory, call max() in a loop to find global max.
    Tests: locals scoping across calls, memory + locals + calls.
    """
    prog = [
        # Store [7, 3, 9, 1, 5] at addrs 0-4
        Instruction(OP_PUSH, 0), Instruction(OP_PUSH, 7), Instruction(OP_I32_STORE),   # 0-2
        Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 3), Instruction(OP_I32_STORE),   # 3-5
        Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 9), Instruction(OP_I32_STORE),   # 6-8
        Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 1), Instruction(OP_I32_STORE),   # 9-11
        Instruction(OP_PUSH, 4), Instruction(OP_PUSH, 5), Instruction(OP_I32_STORE),   # 12-14
        # Init: best = mem[0], i = 1
        Instruction(OP_PUSH, 0),           # 15
        Instruction(OP_I32_LOAD),          # 16: mem[0] = 7
        Instruction(OP_LOCAL_SET, 0),      # 17: local[0] = best = 7
        Instruction(OP_PUSH, 1),           # 18
        Instruction(OP_LOCAL_SET, 1),      # 19: local[1] = i = 1
        # Loop (ip=20):
        Instruction(OP_LOCAL_GET, 1),      # 20: i
        Instruction(OP_PUSH, 5),           # 21: n
        Instruction(OP_SUB),              # 22: n - i
        Instruction(OP_JZ, 38),            # 23: if i==n, done
        # Call max(best, mem[i])
        Instruction(OP_LOCAL_GET, 0),      # 24: best
        Instruction(OP_LOCAL_GET, 1),      # 25: i
        Instruction(OP_I32_LOAD),          # 26: mem[i]
        Instruction(OP_CALL, 44),          # 27: call max(a, b)
        Instruction(OP_LOCAL_SET, 0),      # 28: best = max result
        # i++
        Instruction(OP_LOCAL_GET, 1),      # 29
        Instruction(OP_PUSH, 1),           # 30
        Instruction(OP_ADD),               # 31
        Instruction(OP_LOCAL_SET, 1),      # 32: i++
        Instruction(OP_PUSH, 1),           # 33
        Instruction(OP_JNZ, 20),           # 34: loop
        Instruction(OP_NOP),               # 35
        Instruction(OP_NOP),               # 36
        Instruction(OP_NOP),               # 37
        # Done (ip=38): push best
        Instruction(OP_LOCAL_GET, 0),      # 38
        Instruction(OP_HALT),              # 39
        Instruction(OP_NOP),               # 40
        Instruction(OP_NOP),               # 41
        Instruction(OP_NOP),               # 42
        Instruction(OP_NOP),               # 43
        # max(a, b) at addr 44: stack has [a, b], b on top
        Instruction(OP_LOCAL_SET, 0),      # 44: local[0] = b
        Instruction(OP_LOCAL_SET, 1),      # 45: local[1] = a
        Instruction(OP_LOCAL_GET, 1),      # 46: a
        Instruction(OP_LOCAL_GET, 0),      # 47: b
        Instruction(OP_GT_S),              # 48: a > b?
        Instruction(OP_JZ, 52),            # 49: if not, return b
        Instruction(OP_LOCAL_GET, 1),      # 50: return a
        Instruction(OP_RETURN),            # 51
        Instruction(OP_LOCAL_GET, 0),      # 52: return b
        Instruction(OP_RETURN),            # 53
    ]
    return prog, 9  # max(7, 3, 9, 1, 5) = 9


def test_recursive_fib_10():
    """Recursive fib(10) = 55. Same code as fib(7) but higher n."""
    prog = [
        Instruction(OP_PUSH, 10),      # 0
        Instruction(OP_CALL, 3),       # 1
        Instruction(OP_HALT),          # 2
        # fib(n) — same as test_recursive_fib_with_locals
        Instruction(OP_LOCAL_TEE, 0),   # 3
        Instruction(OP_PUSH, 2),       # 4
        Instruction(OP_LT_S),          # 5
        Instruction(OP_JZ, 10),        # 6
        Instruction(OP_LOCAL_GET, 0),   # 7
        Instruction(OP_RETURN),        # 8
        Instruction(OP_NOP),           # 9
        Instruction(OP_LOCAL_GET, 0),   # 10
        Instruction(OP_PUSH, 1),       # 11
        Instruction(OP_SUB),           # 12
        Instruction(OP_CALL, 3),       # 13
        Instruction(OP_LOCAL_SET, 1),   # 14
        Instruction(OP_LOCAL_GET, 0),   # 15
        Instruction(OP_PUSH, 2),       # 16
        Instruction(OP_SUB),           # 17
        Instruction(OP_CALL, 3),       # 18
        Instruction(OP_LOCAL_GET, 1),   # 19
        Instruction(OP_ADD),           # 20
        Instruction(OP_RETURN),        # 21
    ]
    return prog, 55


# ─── All Integration Tests ───────────────────────────────────────

INTEGRATION_TESTS = [
    ("bubble_sort_3",          test_bubble_sort_3),
    ("recursive_fib_7",        test_recursive_fib_with_locals),
    ("recursive_fib_10",       test_recursive_fib_10),
    ("array_sum_via_func",     test_array_sum_via_function),
    ("memory_stack",           test_memory_stack_via_functions),
    ("multi_func_max",         test_multi_function_max),
]


# ─── Test Runners ────────────────────────────────────────────────

def test_integration():
    """Run all integration tests on both executors."""
    print("=" * 60)
    print("Phase 18: Integration Tests (locals + memory + calls)")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0

    for name, test_fn in INTEGRATION_TESTS:
        prog, expected = test_fn()
        ok, steps = test_algorithm(name, prog, expected, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Integration tests: {passed}/{total} passed")
    return passed == total


def test_full_regression():
    """Full regression: ALL prior phases."""
    print("\n" + "=" * 60)
    print("Full Regression: Phase 4-17")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    passed = 0
    total = 0

    # Phase 4 basic
    for name, test_fn in ALL_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(name, prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 11 extended
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

    # Phase 14 native factorial
    p14_algos = [
        ("factorial(7)", *make_factorial(7)),
    ]
    for name, prog, expected in p14_algos:
        ok, _ = test_algorithm(name, prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 15 locals
    from phase15_local_variables import PHASE15_TESTS
    for name, test_fn in PHASE15_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(f"p15_{name}", prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 16 memory
    from phase16_linear_memory import PHASE16_TESTS
    for name, test_fn in PHASE16_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(f"p16_{name}", prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    # Phase 17 function calls
    from phase17_function_calls import PHASE17_TESTS, PHASE17_TRAP_TESTS
    for name, test_fn in PHASE17_TESTS:
        prog, expected = test_fn()
        ok, _ = test_algorithm(f"p17_{name}", prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1
    for name, test_fn in PHASE17_TRAP_TESTS:
        prog = test_fn()
        ok = test_trap_algorithm(f"p17_{name}", prog, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Full regression: {passed}/{total} passed")
    return passed == total


def test_step_comparison():
    """Compare step counts: recursive fib vs Phase 13's loop-based fib."""
    print("\n" + "=" * 60)
    print("Step Count Comparison")
    print("=" * 60)

    np_exec = NumPyExecutor()

    # Loop-based fib(10) from Phase 13
    loop_prog, loop_expected = make_fibonacci(10)
    loop_trace = np_exec.execute(loop_prog)
    loop_steps = len(loop_trace.steps)

    # Recursive fib(10) from integration tests
    rec_prog, rec_expected = test_recursive_fib_10()
    rec_trace = np_exec.execute(rec_prog)
    rec_steps = len(rec_trace.steps)

    print(f"  Loop fib(10):      {loop_steps:>6} steps, result={loop_trace.steps[-1].top}")
    print(f"  Recursive fib(10): {rec_steps:>6} steps, result={rec_trace.steps[-1].top}")
    print(f"  Ratio: {rec_steps/loop_steps:.1f}x (recursive is more expensive)")

    ok = (loop_trace.steps[-1].top == 55 and rec_trace.steps[-1].top == 55)
    print(f"\n  {'PASS' if ok else 'FAIL'}  Both produce correct result (55)")
    return ok


def test_architecture_summary():
    """Final architecture summary after Tier 2."""
    print("\n" + "=" * 60)
    print("Architecture Summary (Post-Tier 2)")
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

    total_params = sum(
        sum(p.numel() for p in h.parameters()) for _, h in heads
    ) + model.M_top.numel() + model.sp_deltas.numel()

    print(f"  Active heads:     {len(heads)} of 18 slots (8 reserved)")
    print(f"  Opcodes:          {N_OPCODES}")
    print(f"  D_MODEL:          {D_MODEL}")
    print(f"  Parameters:       {total_params}")
    print(f"  Address spaces:   program, stack, locals, heap, call stack")
    print(f"  Remaining heads:  {18 - len(heads)} for Tier 3+")

    checks = []

    ok = len(heads) == 10
    checks.append(ok)
    print(f"\n  {'PASS' if ok else 'FAIL'}  Head allocation: 10 of 18 active")

    ok = N_OPCODES == 55
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  Opcode count: {N_OPCODES} (Tier 1 + Tier 2)")

    ok = D_MODEL == 51
    checks.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  D_MODEL: {D_MODEL}")

    return all(checks)


def test_trace_match():
    """Verify numpy and pytorch produce identical traces for ALL integration tests."""
    print("\n" + "=" * 60)
    print("Trace-Level Match Verification")
    print("=" * 60)

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()
    all_ok = True

    for name, test_fn in INTEGRATION_TESTS:
        prog, expected = test_fn()
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, detail = compare_traces(np_trace, pt_trace)
        status = "PASS" if match else "FAIL"
        if not match:
            all_ok = False
        print(f"  {status}  {name}: {len(np_trace.steps)} steps, {'match' if match else detail}")

    print(f"\n  {'PASS' if all_ok else 'FAIL'}  All integration traces match")
    return all_ok


def main():
    print("=" * 60)
    print("Phase 18: Tier 2 Integration Tests — Full Suite")
    print("=" * 60)
    print()

    t0 = time.time()

    results = []
    results.append(("Integration tests", test_integration()))
    results.append(("Full regression (Phase 4-17)", test_full_regression()))
    results.append(("Step count comparison", test_step_comparison()))
    results.append(("Architecture summary", test_architecture_summary()))
    results.append(("Trace-level match", test_trace_match()))

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
        print("\n  All Tier 2 integration tests pass.")
        print("  Tier 2 complete: locals + linear memory + function calls.")
    else:
        print("\n  FAILURES detected. See details above.")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
