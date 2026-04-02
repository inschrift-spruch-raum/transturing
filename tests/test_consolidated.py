"""Executor correctness tests: verify NumPy and PyTorch backends produce expected results.

Imports from the core package (transturing).
Cross-validates NumPy vs PyTorch traces for internal consistency.
"""

import pytest

from transturing.backends.numpy_backend import NumPyExecutor
from transturing.backends.torch_backend import TorchExecutor
from transturing.core.isa import (
    OP_AND,
    OP_DIV_S,
    OP_DUP,
    OP_EQ,
    OP_GE_S,
    OP_GT_S,
    OP_HALT,
    OP_JNZ,
    OP_JZ,
    OP_LE_S,
    OP_LT_S,
    OP_NE,
    OP_OR,
    OP_PUSH,
    OP_REM_S,
    OP_REM_U,
    OP_ROTL,
    OP_ROTR,
    OP_SHL,
    OP_SHR_S,
    OP_SHR_U,
    OP_SUB,
    OP_TRAP,
    OP_XOR,
    Instruction,
    compare_traces,
)
from transturing.core.programs import (
    ALL_TESTS,
    make_bit_extract,
    make_bitwise_binary,
    make_compare_binary,
    make_compare_eqz,
    make_factorial,
    make_fibonacci,
    make_gcd,
    make_is_even,
    make_is_power_of_2,
    make_log2_floor,
    make_multiply,
    make_native_abs,
    make_native_abs_unary,
    make_native_clamp,
    make_native_clz,
    make_native_ctz,
    make_native_divmod,
    make_native_is_even,
    make_native_max,
    make_native_multiply,
    make_native_neg,
    make_native_popcnt,
    make_native_remainder,
    make_popcount_loop,
    make_power_of_2,
    make_select,
    make_select_max,
    make_sum_1_to_n,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def np_exec():
    return NumPyExecutor()


@pytest.fixture
def pt_exec():
    return TorchExecutor()


# ---------------------------------------------------------------------------
# Test-case collectors
# ---------------------------------------------------------------------------


def _phase4_tests():
    """Phase 4: ALL_TESTS entries.  Each yields (name, prog, expected)."""
    for name, test_fn in ALL_TESTS:
        prog, expected = test_fn()
        yield (name, prog, expected)


def _phase11_extended():
    """Phase 11 extended instruction tests."""
    return [
        (
            "sub_basic",
            [
                Instruction(OP_PUSH, 10),
                Instruction(OP_PUSH, 3),
                Instruction(OP_SUB),
                Instruction(OP_HALT),
            ],
            7,
        ),
        (
            "loop_countdown",
            [
                Instruction(OP_PUSH, 3),
                Instruction(OP_DUP),
                Instruction(OP_PUSH, 1),
                Instruction(OP_SUB),
                Instruction(OP_DUP),
                Instruction(OP_JNZ, 1),
                Instruction(OP_HALT),
            ],
            0,
        ),
        (
            "jz_taken",
            [
                Instruction(OP_PUSH, 0),
                Instruction(OP_JZ, 3),
                Instruction(OP_HALT),
                Instruction(OP_PUSH, 42),
                Instruction(OP_HALT),
            ],
            42,
        ),
    ]


def _phase13_algos_numpy():
    """Phase 13 algorithms used in NumPy equivalence tests."""
    return [
        ("fib(10)", *make_fibonacci(10)),
        ("fib(7)", *make_fibonacci(7)),
        ("sum(1..10)", *make_sum_1_to_n(10)),
        ("power(2^5)", *make_power_of_2(5)),
        ("mul(7,8)", *make_multiply(7, 8)),
        ("is_even(10)", *make_is_even(10)),
        ("is_even(7)", *make_is_even(7)),
    ]


def _phase13_algos_torch():
    """Phase 13 algorithms used in Torch equivalence tests."""
    return [
        ("fib(10)", *make_fibonacci(10)),
        ("sum(1..10)", *make_sum_1_to_n(10)),
        ("power(2^5)", *make_power_of_2(5)),
        ("mul(7,8)", *make_multiply(7, 8)),
    ]


def _phase13_algos_consistency():
    """Phase 13 algorithms used in internal consistency tests."""
    return [
        ("fib(10)", *make_fibonacci(10)),
        ("sum(1..15)", *make_sum_1_to_n(15)),
        ("power(2^7)", *make_power_of_2(7)),
        ("mul(12,10)", *make_multiply(12, 10)),
    ]


def _phase14_arith():
    """Phase 14 arithmetic tests."""
    return [
        ("native_mul(7,8)", *make_native_multiply(7, 8)),
        ("native_mul(0,5)", *make_native_multiply(0, 5)),
        ("native_divmod(3,10)", *make_native_divmod(3, 10)),
        ("native_rem(3,10)", *make_native_remainder(3, 10)),
        ("factorial(5)", *make_factorial(5)),
        ("factorial(10)", *make_factorial(10)),
        ("gcd(12,8)", *make_gcd(12, 8)),
        ("gcd(100,75)", *make_gcd(100, 75)),
        ("native_is_even(42)", *make_native_is_even(42)),
        ("native_is_even(15)", *make_native_is_even(15)),
    ]


def _phase14_cmp():
    """Phase 14 comparison tests."""
    return [
        ("eqz(0)", *make_compare_eqz(0)),
        ("eqz(5)", *make_compare_eqz(5)),
        ("eq(5,5)", *make_compare_binary(OP_EQ, 5, 5)),
        ("lt_s(3,7)", *make_compare_binary(OP_LT_S, 3, 7)),
        ("gt_s(10,2)", *make_compare_binary(OP_GT_S, 10, 2)),
        ("max(3,7)", *make_native_max(3, 7)),
        ("max(10,2)", *make_native_max(10, 2)),
        ("abs(42)", *make_native_abs(42)),
        ("clamp(5,0,10)", *make_native_clamp(5, 0, 10)),
        ("clamp(15,0,10)", *make_native_clamp(15, 0, 10)),
    ]


def _phase14_bit():
    """Phase 14 bitwise tests."""
    return [
        ("and(0xFF,0x0F)", *make_bitwise_binary(OP_AND, 0xFF, 0x0F)),
        ("or(0xF0,0x0F)", *make_bitwise_binary(OP_OR, 0xF0, 0x0F)),
        ("xor(0xFF,0xFF)", *make_bitwise_binary(OP_XOR, 0xFF, 0xFF)),
        ("shl(1,4)", *make_bitwise_binary(OP_SHL, 1, 4)),
        ("shr_u(16,1)", *make_bitwise_binary(OP_SHR_U, 16, 1)),
        ("rotl(1,1)", *make_bitwise_binary(OP_ROTL, 1, 1)),
        ("rotr(2,1)", *make_bitwise_binary(OP_ROTR, 2, 1)),
        ("popcount(255)", *make_popcount_loop(255)),
        ("bit(0xFF,4)", *make_bit_extract(0xFF, 4)),
    ]


def _phase14_unary():
    """Phase 14 unary + parametric tests."""
    return [
        ("clz(0)", *make_native_clz(0)),
        ("clz(255)", *make_native_clz(255)),
        ("ctz(8)", *make_native_ctz(8)),
        ("popcnt(255)", *make_native_popcnt(255)),
        ("abs_native(42)", *make_native_abs_unary(42)),
        ("abs_native(-7)", *make_native_abs_unary(-7)),
        ("neg(5)", *make_native_neg(5)),
        ("neg(-3)", *make_native_neg(-3)),
        ("select(10,20,1)", *make_select(10, 20, 1)),
        ("select(10,20,0)", *make_select(10, 20, 0)),
        ("select_max(10,25)", *make_select_max(10, 25)),
        ("log2(8)", *make_log2_floor(8)),
        ("ispow2(8)", *make_is_power_of_2(8)),
        ("ispow2(7)", *make_is_power_of_2(7)),
    ]


def _phase14_torch():
    """Phase 14 tests used in Torch equivalence."""
    return [
        ("native_mul(12,10)", *make_native_multiply(12, 10)),
        ("factorial(7)", *make_factorial(7)),
        ("gcd(48,36)", *make_gcd(48, 36)),
        ("eqz(0)", *make_compare_eqz(0)),
        ("gt_s(10,2)", *make_compare_binary(OP_GT_S, 10, 2)),
        ("max(10,25)", *make_native_max(10, 25)),
        ("and(0xAA,0x55)", *make_bitwise_binary(OP_AND, 0xAA, 0x55)),
        ("popcount(15)", *make_popcount_loop(15)),
        ("clz(1)", *make_native_clz(1)),
        ("popcnt_native(7)", *make_native_popcnt(7)),
        ("abs_native(-7)", *make_native_abs_unary(-7)),
        ("neg(5)", *make_native_neg(5)),
        ("select(10,20,1)", *make_select(10, 20, 1)),
        ("select_max(25,10)", *make_select_max(25, 10)),
        ("log2(1024)", *make_log2_floor(1024)),
        ("ispow2(1024)", *make_is_power_of_2(1024)),
    ]


def _phase14_consistency():
    """Phase 14 full-coverage tests for internal consistency."""
    return [
        ("native_mul(100,200)", *make_native_multiply(100, 200)),
        ("factorial(10)", *make_factorial(10)),
        ("gcd(17,13)", *make_gcd(17, 13)),
        ("native_is_even(100)", *make_native_is_even(100)),
        ("eq(5,5)", *make_compare_binary(OP_EQ, 5, 5)),
        ("ne(3,7)", *make_compare_binary(OP_NE, 3, 7)),
        ("le_s(3,7)", *make_compare_binary(OP_LE_S, 3, 7)),
        ("ge_s(10,2)", *make_compare_binary(OP_GE_S, 10, 2)),
        ("max(5,5)", *make_native_max(5, 5)),
        ("clamp(0,5,10)", *make_native_clamp(0, 5, 10)),
        ("or(0xAA,0x55)", *make_bitwise_binary(OP_OR, 0xAA, 0x55)),
        ("xor(12,10)", *make_bitwise_binary(OP_XOR, 12, 10)),
        ("shl(0xFF,4)", *make_bitwise_binary(OP_SHL, 0xFF, 4)),
        ("shr_s(16,1)", *make_bitwise_binary(OP_SHR_S, 16, 1)),
        ("popcount(0xFFFF)", *make_popcount_loop(0xFFFF)),
        ("bit(42,3)", *make_bit_extract(42, 3)),
        ("clz(65536)", *make_native_clz(65536)),
        ("ctz(1024)", *make_native_ctz(1024)),
        ("popcnt_native(1023)", *make_native_popcnt(1023)),
        ("abs_native(0)", *make_native_abs_unary(0)),
        ("neg(1000)", *make_native_neg(1000)),
        ("select(5,9,-1)", *make_select(5, 9, -1)),
        ("select_max(7,7)", *make_select_max(7, 7)),
        ("log2(255)", *make_log2_floor(255)),
        ("ispow2(0)", *make_is_power_of_2(0)),
    ]


def _trap_progs_numpy():
    """Trap test programs for NumPy equivalence."""
    return [
        (
            "div_by_zero",
            [
                Instruction(OP_PUSH, 10),
                Instruction(OP_PUSH, 0),
                Instruction(OP_DIV_S),
                Instruction(OP_HALT),
            ],
        ),
        (
            "rem_by_zero",
            [
                Instruction(OP_PUSH, 10),
                Instruction(OP_PUSH, 0),
                Instruction(OP_REM_S),
                Instruction(OP_HALT),
            ],
        ),
    ]


def _trap_progs_torch():
    """Trap test programs for Torch equivalence."""
    return [
        (
            "div_by_zero",
            [
                Instruction(OP_PUSH, 10),
                Instruction(OP_PUSH, 0),
                Instruction(OP_DIV_S),
                Instruction(OP_HALT),
            ],
        ),
        (
            "rem_by_zero",
            [
                Instruction(OP_PUSH, 10),
                Instruction(OP_PUSH, 0),
                Instruction(OP_REM_U),
                Instruction(OP_HALT),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Helper for id generation
# ---------------------------------------------------------------------------


def _id_fn(val):
    """Use the test name as the pytest id when val is a string."""
    return val if isinstance(val, str) else None


# ---------------------------------------------------------------------------
# NumPy equivalence tests (old Phase14Executor vs new NumPyExecutor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,prog,expected", list(_phase4_tests()), ids=_id_fn)
def test_numpy_phase4(np_exec, name, prog, expected):
    trace = np_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase11_extended(), ids=_id_fn)
def test_numpy_phase11(np_exec, name, prog, expected):
    trace = np_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase13_algos_numpy(), ids=_id_fn)
def test_numpy_phase13(np_exec, name, prog, expected):
    trace = np_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase14_arith(), ids=_id_fn)
def test_numpy_phase14_arith(np_exec, name, prog, expected):
    trace = np_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase14_cmp(), ids=_id_fn)
def test_numpy_phase14_cmp(np_exec, name, prog, expected):
    trace = np_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase14_bit(), ids=_id_fn)
def test_numpy_phase14_bit(np_exec, name, prog, expected):
    trace = np_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase14_unary(), ids=_id_fn)
def test_numpy_phase14_unary(np_exec, name, prog, expected):
    trace = np_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog", _trap_progs_numpy(), ids=_id_fn)
def test_numpy_trap(np_exec, name, prog):
    trace = np_exec.execute(prog)
    trapped = trace.steps and trace.steps[-1].op == OP_TRAP
    assert trapped, "Executor did not trap"


# ---------------------------------------------------------------------------
# Torch equivalence tests (old Phase14PyTorchExecutor vs new TorchExecutor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,prog,expected", list(_phase4_tests()), ids=_id_fn)
def test_torch_phase4(pt_exec, name, prog, expected):
    trace = pt_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase13_algos_torch(), ids=_id_fn)
def test_torch_phase13(pt_exec, name, prog, expected):
    trace = pt_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase14_torch(), ids=_id_fn)
def test_torch_phase14(pt_exec, name, prog, expected):
    trace = pt_exec.execute(prog)
    assert trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog", _trap_progs_torch(), ids=_id_fn)
def test_torch_trap(pt_exec, name, prog):
    trace = pt_exec.execute(prog)
    trapped = trace.steps and trace.steps[-1].op == OP_TRAP
    assert trapped, "Executor did not trap"


# ---------------------------------------------------------------------------
# Internal consistency tests (new NumPyExecutor vs new TorchExecutor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,prog,expected", list(_phase4_tests()), ids=_id_fn)
def test_consistency_phase4(np_exec, pt_exec, name, prog, expected):
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    match, detail = compare_traces(np_trace, pt_trace)
    assert match, f"Trace mismatch: {detail}"
    assert np_trace.steps[-1].top == expected
    assert pt_trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase13_algos_consistency(), ids=_id_fn)
def test_consistency_phase13(np_exec, pt_exec, name, prog, expected):
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    match, detail = compare_traces(np_trace, pt_trace)
    assert match, f"Trace mismatch: {detail}"
    assert np_trace.steps[-1].top == expected
    assert pt_trace.steps[-1].top == expected


@pytest.mark.parametrize("name,prog,expected", _phase14_consistency(), ids=_id_fn)
def test_consistency_phase14(np_exec, pt_exec, name, prog, expected):
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)
    match, detail = compare_traces(np_trace, pt_trace)
    assert match, f"Trace mismatch: {detail}"
    assert np_trace.steps[-1].top == expected
    assert pt_trace.steps[-1].top == expected
