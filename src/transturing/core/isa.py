"""
ISA definition for the compiled transformer stack machine.

Core, zero-dependency module containing:
  - Types: Instruction, Trace, TraceStep
  - Constants: D_MODEL, N_OPCODES, MASK32, TOKENS_PER_STEP, all DIM_* layout
  - Opcodes: OP_PUSH through OP_SELECT, OP_TRAP
  - Maps: OP_NAMES, OPCODE_DIM_MAP, OPCODE_IDX, NONLINEAR_OPS
  - Math helpers: trunc_div, trunc_rem, bitwise ops
  - Test utilities: compare_traces, test_algorithm, test_trap_algorithm
  - Program builder: program()

Torch-specific items (CompiledAttentionHead, TokenVocab, embed_* functions, DTYPE, EPS)
live in transturing.backends.torch_backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abc import ExecutorBackend

logger = logging.getLogger(__name__)

# ─── Types ──────────────────────────────────────────────────────────

type WasmInstr = tuple[str] | tuple[str, int] | tuple[str, list[int], int]


@dataclass
class Instruction:
    """A single ISA instruction with opcode and optional argument."""

    op: int
    arg: int = 0

    def __repr__(self) -> str:
        """Return human-readable instruction string."""
        name = OP_NAMES.get(self.op, f"?{self.op}")
        if self.op in (
            OP_PUSH,
            OP_JZ,
            OP_JNZ,
            OP_LOCAL_GET,
            OP_LOCAL_SET,
            OP_LOCAL_TEE,
            OP_CALL,
        ):
            return f"{name} {self.arg}"
        return name


def program(*instrs: Instruction | WasmInstr) -> list[Instruction]:
    """Build a program from instruction tuples or Instruction objects."""
    result: list[Instruction] = []
    _name_to_op: dict[str, int] = {
        "PUSH": 1,
        "POP": 2,
        "ADD": 3,
        "DUP": 4,
        "HALT": 5,
        "SUB": 6,
        "JZ": 7,
        "JNZ": 8,
        "NOP": 9,
        "SWAP": 10,
        "OVER": 11,
        "ROT": 12,
        "MUL": 13,
        "DIV_S": 14,
        "DIV_U": 15,
        "REM_S": 16,
        "REM_U": 17,
        "EQZ": 18,
        "EQ": 19,
        "NE": 20,
        "LT_S": 21,
        "LT_U": 22,
        "GT_S": 23,
        "GT_U": 24,
        "LE_S": 25,
        "LE_U": 26,
        "GE_S": 27,
        "GE_U": 28,
        "AND": 29,
        "OR": 30,
        "XOR": 31,
        "SHL": 32,
        "SHR_S": 33,
        "SHR_U": 34,
        "ROTL": 35,
        "ROTR": 36,
        "CLZ": 37,
        "CTZ": 38,
        "POPCNT": 39,
        "ABS": 40,
        "NEG": 41,
        "SELECT": 42,
        "LOCAL.GET": 43,
        "LOCAL.SET": 44,
        "LOCAL.TEE": 45,
        "I32.LOAD": 46,
        "I32.STORE": 47,
        "I32.LOAD8_U": 48,
        "I32.LOAD8_S": 49,
        "I32.LOAD16_U": 50,
        "I32.LOAD16_S": 51,
        "I32.STORE8": 52,
        "I32.STORE16": 53,
        "CALL": 54,
        "RETURN": 55,
    }
    for instr in instrs:
        if isinstance(instr, Instruction):
            result.append(instr)
            continue
        name = instr[0].upper()
        # Handle three WasmInstr shapes: (op,), (op, arg), (op, labels, default)
        match instr:
            case (_, _, default):
                arg = default
            case (_, a):
                arg = a
            case _:
                arg = 0
        op = _name_to_op[name]
        result.append(Instruction(op, arg))
    return result


@dataclass
class TraceStep:
    """One instruction's execution record."""

    op: int
    arg: int
    sp: int  # stack pointer AFTER execution
    top: int  # top-of-stack value AFTER execution

    def tokens(self) -> list[int]:
        """Return token representation of this step."""
        return [self.op, self.arg, self.sp, self.top]


@dataclass
class Trace:
    """Full execution trace: program prefix + step records."""

    program: list[Instruction]
    steps: list[TraceStep] = field(default_factory=list)

    def format_trace(self) -> str:
        """Human-readable trace."""
        lines: list[str] = []
        lines.append(f"Program: {' ; '.join(str(i) for i in self.program)}")
        lines.append(f"{'Step':>4}  {'Instruction':<10} {'SP':>3}  {'TOP':>5}")
        lines.append("-" * 35)
        for i, s in enumerate(self.steps):
            name = OP_NAMES.get(s.op, "?")
            instr_str = (
                f"{name} {s.arg}"
                if s.op
                in (
                    OP_PUSH,
                    OP_JZ,
                    OP_JNZ,
                    OP_LOCAL_GET,
                    OP_LOCAL_SET,
                    OP_LOCAL_TEE,
                    OP_CALL,
                )
                else name
            )
            lines.append(f"{i:4d}  {instr_str:<10} {s.sp:3d}  {s.top:5d}")
        return "\n".join(lines)


# ─── Constants ────────────────────────────────────────────────────

D_MODEL = 51

# Token roles (from phase4, used in training phases)
TOKENS_PER_STEP = 4


# ─── Embedding Dimension Layout (all 51 dims) ────────────────────

DIM_IS_PROG = 0
DIM_IS_STACK = 1
DIM_IS_STATE = 2
DIM_PROG_KEY_0 = 3
DIM_PROG_KEY_1 = 4
DIM_STACK_KEY_0 = 5
DIM_STACK_KEY_1 = 6
DIM_OPCODE = 7
DIM_VALUE = 8
DIM_IP = 9
DIM_SP = 10
DIM_ONE = 11
DIM_IS_PUSH = 12
DIM_IS_POP = 13
DIM_IS_ADD = 14
DIM_IS_DUP = 15
DIM_IS_HALT = 16
DIM_IS_SUB = 17
DIM_IS_JZ = 18
DIM_IS_JNZ = 19
DIM_IS_NOP = 20
DIM_IS_SWAP = 21
DIM_IS_OVER = 22
DIM_IS_ROT = 23
DIM_IS_MUL = 24
DIM_IS_DIV_S = 25
DIM_IS_DIV_U = 26
DIM_IS_REM_S = 27
DIM_IS_REM_U = 28
DIM_IS_EQZ = 29
DIM_IS_EQ = 30
DIM_IS_NE = 31
DIM_IS_LT = 32  # shared by LT_S and LT_U
DIM_IS_GT = 33  # shared by GT_S and GT_U
DIM_IS_LE = 34  # shared by LE_S and LE_U
DIM_IS_GE = 35  # shared by GE_S and GE_U

# Phase 15: local variables address space
DIM_IS_LOCAL = 36
DIM_LOCAL_KEY_0 = 37
DIM_LOCAL_KEY_1 = 38
DIM_IS_LOCAL_GET = 39
DIM_IS_LOCAL_SET = 40
DIM_IS_LOCAL_TEE = 41

# Phase 16: linear memory (heap) address space
DIM_IS_HEAP = 42
DIM_HEAP_KEY_0 = 43
DIM_HEAP_KEY_1 = 44

# Phase 17: call stack address space
DIM_IS_CALL_STACK = 45
DIM_CALL_KEY_0 = 46
DIM_CALL_KEY_1 = 47
DIM_CALL_RET_ADDR = 48
DIM_CALL_SAVED_SP = 49
DIM_CALL_LOCALS_BASE = 50


# ─── Opcodes ─────────────────────────────────────────────────────

# Phase 4 base
OP_PUSH = 1
OP_POP = 2
OP_ADD = 3
OP_DUP = 4
OP_HALT = 5

# Phase 11 extended
OP_SUB = 6
OP_JZ = 7
OP_JNZ = 8
OP_NOP = 9

# Phase 13 stack manipulation
OP_SWAP = 10
OP_OVER = 11
OP_ROT = 12

# Phase 14 Chunk 1: arithmetic
OP_MUL = 13
OP_DIV_S = 14
OP_DIV_U = 15
OP_REM_S = 16
OP_REM_U = 17

# Phase 14 Chunk 2: comparisons
OP_EQZ = 18
OP_EQ = 19
OP_NE = 20
OP_LT_S = 21
OP_LT_U = 22
OP_GT_S = 23
OP_GT_U = 24
OP_LE_S = 25
OP_LE_U = 26
OP_GE_S = 27
OP_GE_U = 28

# Phase 14 Chunk 3: bitwise
OP_AND = 29
OP_OR = 30
OP_XOR = 31
OP_SHL = 32
OP_SHR_S = 33
OP_SHR_U = 34
OP_ROTL = 35
OP_ROTR = 36

# Phase 14 Chunk 4: unary + parametric
OP_CLZ = 37
OP_CTZ = 38
OP_POPCNT = 39
OP_ABS = 40
OP_NEG = 41
OP_SELECT = 42

# Phase 15: local variables
OP_LOCAL_GET = 43
OP_LOCAL_SET = 44
OP_LOCAL_TEE = 45

# Phase 16: linear memory
OP_I32_LOAD = 46
OP_I32_STORE = 47
OP_I32_LOAD8_U = 48
OP_I32_LOAD8_S = 49
OP_I32_LOAD16_U = 50
OP_I32_LOAD16_S = 51
OP_I32_STORE8 = 52
OP_I32_STORE16 = 53

# Phase 17: function calls
OP_CALL = 54
OP_RETURN = 55

# Trap
OP_TRAP = 99

N_OPCODES = 55  # 53 base + 2 call ops


# ─── Maps ─────────────────────────────────────────────────────────

OP_NAMES = {
    OP_PUSH: "PUSH",
    OP_POP: "POP",
    OP_ADD: "ADD",
    OP_DUP: "DUP",
    OP_HALT: "HALT",
    OP_SUB: "SUB",
    OP_JZ: "JZ",
    OP_JNZ: "JNZ",
    OP_NOP: "NOP",
    OP_SWAP: "SWAP",
    OP_OVER: "OVER",
    OP_ROT: "ROT",
    OP_MUL: "MUL",
    OP_DIV_S: "DIV_S",
    OP_DIV_U: "DIV_U",
    OP_REM_S: "REM_S",
    OP_REM_U: "REM_U",
    OP_EQZ: "EQZ",
    OP_EQ: "EQ",
    OP_NE: "NE",
    OP_LT_S: "LT_S",
    OP_LT_U: "LT_U",
    OP_GT_S: "GT_S",
    OP_GT_U: "GT_U",
    OP_LE_S: "LE_S",
    OP_LE_U: "LE_U",
    OP_GE_S: "GE_S",
    OP_GE_U: "GE_U",
    OP_AND: "AND",
    OP_OR: "OR",
    OP_XOR: "XOR",
    OP_SHL: "SHL",
    OP_SHR_S: "SHR_S",
    OP_SHR_U: "SHR_U",
    OP_ROTL: "ROTL",
    OP_ROTR: "ROTR",
    OP_CLZ: "CLZ",
    OP_CTZ: "CTZ",
    OP_POPCNT: "POPCNT",
    OP_ABS: "ABS",
    OP_NEG: "NEG",
    OP_SELECT: "SELECT",
    OP_LOCAL_GET: "LOCAL.GET",
    OP_LOCAL_SET: "LOCAL.SET",
    OP_LOCAL_TEE: "LOCAL.TEE",
    OP_I32_LOAD: "I32.LOAD",
    OP_I32_STORE: "I32.STORE",
    OP_I32_LOAD8_U: "I32.LOAD8_U",
    OP_I32_LOAD8_S: "I32.LOAD8_S",
    OP_I32_LOAD16_U: "I32.LOAD16_U",
    OP_I32_LOAD16_S: "I32.LOAD16_S",
    OP_I32_STORE8: "I32.STORE8",
    OP_I32_STORE16: "I32.STORE16",
    OP_CALL: "CALL",
    OP_RETURN: "RETURN",
    OP_TRAP: "TRAP",
}

OPCODE_DIM_MAP = {
    OP_PUSH: DIM_IS_PUSH,
    OP_POP: DIM_IS_POP,
    OP_ADD: DIM_IS_ADD,
    OP_DUP: DIM_IS_DUP,
    OP_HALT: DIM_IS_HALT,
    OP_SUB: DIM_IS_SUB,
    OP_JZ: DIM_IS_JZ,
    OP_JNZ: DIM_IS_JNZ,
    OP_NOP: DIM_IS_NOP,
    OP_SWAP: DIM_IS_SWAP,
    OP_OVER: DIM_IS_OVER,
    OP_ROT: DIM_IS_ROT,
    OP_MUL: DIM_IS_MUL,
    OP_DIV_S: DIM_IS_DIV_S,
    OP_DIV_U: DIM_IS_DIV_U,
    OP_REM_S: DIM_IS_REM_S,
    OP_REM_U: DIM_IS_REM_U,
    OP_EQZ: DIM_IS_EQZ,
    OP_EQ: DIM_IS_EQ,
    OP_NE: DIM_IS_NE,
    OP_LT_S: DIM_IS_LT,
    OP_LT_U: DIM_IS_LT,
    OP_GT_S: DIM_IS_GT,
    OP_GT_U: DIM_IS_GT,
    OP_LE_S: DIM_IS_LE,
    OP_LE_U: DIM_IS_LE,
    OP_GE_S: DIM_IS_GE,
    OP_GE_U: DIM_IS_GE,
    OP_LOCAL_GET: DIM_IS_LOCAL_GET,
    OP_LOCAL_SET: DIM_IS_LOCAL_SET,
    OP_LOCAL_TEE: DIM_IS_LOCAL_TEE,
}

OPCODE_IDX = {
    OP_PUSH: 0,
    OP_POP: 1,
    OP_ADD: 2,
    OP_DUP: 3,
    OP_HALT: 4,
    OP_SUB: 5,
    OP_JZ: 6,
    OP_JNZ: 7,
    OP_NOP: 8,
    OP_SWAP: 9,
    OP_OVER: 10,
    OP_ROT: 11,
    OP_MUL: 12,
    OP_DIV_S: 13,
    OP_DIV_U: 14,
    OP_REM_S: 15,
    OP_REM_U: 16,
    OP_EQZ: 17,
    OP_EQ: 18,
    OP_NE: 19,
    OP_LT_S: 20,
    OP_LT_U: 21,
    OP_GT_S: 22,
    OP_GT_U: 23,
    OP_LE_S: 24,
    OP_LE_U: 25,
    OP_GE_S: 26,
    OP_GE_U: 27,
    OP_AND: 28,
    OP_OR: 29,
    OP_XOR: 30,
    OP_SHL: 31,
    OP_SHR_S: 32,
    OP_SHR_U: 33,
    OP_ROTL: 34,
    OP_ROTR: 35,
    OP_CLZ: 36,
    OP_CTZ: 37,
    OP_POPCNT: 38,
    OP_ABS: 39,
    OP_NEG: 40,
    OP_SELECT: 41,
    OP_LOCAL_GET: 42,
    OP_LOCAL_SET: 43,
    OP_LOCAL_TEE: 44,
    OP_I32_LOAD: 45,
    OP_I32_STORE: 46,
    OP_I32_LOAD8_U: 47,
    OP_I32_LOAD8_S: 48,
    OP_I32_LOAD16_U: 49,
    OP_I32_LOAD16_S: 50,
    OP_I32_STORE8: 51,
    OP_I32_STORE16: 52,
    OP_CALL: 53,
    OP_RETURN: 54,
}

NONLINEAR_OPS = {
    OP_MUL,
    OP_DIV_S,
    OP_DIV_U,
    OP_REM_S,
    OP_REM_U,
    OP_EQZ,
    OP_EQ,
    OP_NE,
    OP_LT_S,
    OP_LT_U,
    OP_GT_S,
    OP_GT_U,
    OP_LE_S,
    OP_LE_U,
    OP_GE_S,
    OP_GE_U,
    OP_AND,
    OP_OR,
    OP_XOR,
    OP_SHL,
    OP_SHR_S,
    OP_SHR_U,
    OP_ROTL,
    OP_ROTR,
    OP_CLZ,
    OP_CTZ,
    OP_POPCNT,
    OP_ABS,
    OP_NEG,
    OP_SELECT,
    OP_I32_LOAD8_U,
    OP_I32_LOAD8_S,
    OP_I32_LOAD16_U,
    OP_I32_LOAD16_S,
}


# ─── Math Helpers ─────────────────────────────────────────────────

MASK32 = 0xFFFFFFFF

# Named constants for i32/i16/i8 bit manipulation (avoids PLR2004 violations)
I32_SIGN_BIT = 0x80000000
I32_MODULO = 0x100000000
I8_SIGN_BIT = 0x80
I8_RANGE = 0x100
I16_SIGN_BIT = 0x8000
I16_RANGE = 0x10000

# CLZ threshold table — avoids inline magic numbers in clz32()
_CLZ_THRESHOLDS = [0x0000FFFF, 0x00FFFFFF, 0x0FFFFFFF, 0x3FFFFFFF, 0x7FFFFFFF]
_CLZ_SHIFTS = [16, 8, 4, 2, 1]


def trunc_div(b: int, a: int) -> int:
    """Signed integer division truncating toward zero (WASM semantics)."""
    return int(b / a)


def trunc_rem(b: int, a: int) -> int:
    """Signed remainder matching truncated division: b - trunc(b/a)*a."""
    return b - trunc_div(b, a) * a


def to_i32(val: int) -> int:
    """Cast to 32-bit signed integer from potentially float stack value."""
    return int(val) & MASK32


def shr_u(b: int, a: int) -> int:
    """Logical (unsigned) right shift of b by a positions."""
    return to_i32(b) >> (int(a) & 31)


def shr_s(b: int, a: int) -> int:
    """Arithmetic (signed) right shift of b by a positions."""
    val = to_i32(b)
    if val >= I32_SIGN_BIT:
        val -= I32_MODULO
    shift = int(a) & 31
    result = val >> shift
    return result & MASK32 if result < 0 else result


def rotl32(b: int, a: int) -> int:
    """Left-rotate b by a positions within 32-bit word."""
    val = to_i32(b)
    shift = int(a) & 31
    return ((val << shift) | (val >> (32 - shift))) & MASK32 if shift else val


def rotr32(b: int, a: int) -> int:
    """Right-rotate b by a positions within 32-bit word."""
    val = to_i32(b)
    shift = int(a) & 31
    return ((val >> shift) | (val << (32 - shift))) & MASK32 if shift else val


def clz32(val: int) -> int:
    """Count leading zeros in 32-bit representation."""
    v = to_i32(val)
    if v == 0:
        return 32
    n = 0
    for threshold, shift in zip(_CLZ_THRESHOLDS, _CLZ_SHIFTS, strict=True):
        if v <= threshold:
            n += shift
            v <<= shift
    return n


def ctz32(val: int) -> int:
    """Count trailing zeros in 32-bit representation."""
    v = to_i32(val)
    if v == 0:
        return 32
    n = 0
    if (v & 0x0000FFFF) == 0:
        n += 16
        v >>= 16
    if (v & 0x000000FF) == 0:
        n += 8
        v >>= 8
    if (v & 0x0000000F) == 0:
        n += 4
        v >>= 4
    if (v & 0x00000003) == 0:
        n += 2
        v >>= 2
    if (v & 0x00000001) == 0:
        n += 1
    return n


def popcnt32(val: int) -> int:
    """Count set bits in 32-bit representation."""
    return to_i32(val).bit_count()


def sign_extend_8(val: int) -> int:
    """Sign-extend an 8-bit value to a signed integer."""
    v = int(val) & 0xFF
    return v - I8_RANGE if v >= I8_SIGN_BIT else v


def sign_extend_16(val: int) -> int:
    """Sign-extend a 16-bit value to a signed integer."""
    v = int(val) & 0xFFFF
    return v - I16_RANGE if v >= I16_SIGN_BIT else v


# ─── Test Utilities ────────────────────────────────────────────────


@dataclass
class TestConfig:
    """Configuration bundle for test_algorithm / test_trap_algorithm."""

    name: str
    prog: list[Instruction]
    expected: int | None
    np_exec: ExecutorBackend
    pt_exec: ExecutorBackend
    verbose: bool = False


def compare_traces(trace_a: Trace, trace_b: Trace) -> tuple[bool, str]:
    """Compare two traces token by token. Returns (match, detail)."""
    if len(trace_a.steps) != len(trace_b.steps):
        return False, f"length mismatch: {len(trace_a.steps)} vs {len(trace_b.steps)}"
    for i, (a, b) in enumerate(zip(trace_a.steps, trace_b.steps, strict=True)):
        if a.tokens() != b.tokens():
            return False, f"step {i}: {a.tokens()} vs {b.tokens()}"
    return True, "match"


def test_algorithm(cfg: TestConfig) -> tuple[bool, int]:
    """Run an algorithm on both executors and verify."""
    np_trace = cfg.np_exec.execute(cfg.prog)
    pt_trace = cfg.pt_exec.execute(cfg.prog)

    np_top = np_trace.steps[-1].top if np_trace.steps else None
    pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
    match, detail = compare_traces(np_trace, pt_trace)

    np_ok = np_top == cfg.expected
    pt_ok = pt_top == cfg.expected
    all_ok = np_ok and pt_ok and match

    status = "PASS" if all_ok else "FAIL"
    logger.info(
        "  %s  %-30s  expected=%6s  numpy=%6s  torch=%6s  steps=%4d  trace_match=%s",
        status,
        cfg.name,
        cfg.expected,
        np_top,
        pt_top,
        len(np_trace.steps),
        "Y" if match else "N",
    )

    if not all_ok and cfg.verbose:
        if not match:
            logger.info("         Trace mismatch: %s", detail)
        if not np_ok:
            logger.info(
                "         NumPy wrong: got %s, expected %s", np_top, cfg.expected
            )
        if not pt_ok:
            logger.info(
                "         PyTorch wrong: got %s, expected %s", pt_top, cfg.expected
            )

    return all_ok, len(np_trace.steps)


def test_trap_algorithm(cfg: TestConfig) -> bool:
    """Run a program expected to TRAP on both executors. Returns True if both trap."""
    np_trace = cfg.np_exec.execute(cfg.prog)
    pt_trace = cfg.pt_exec.execute(cfg.prog)

    np_trapped: bool = len(np_trace.steps) > 0 and np_trace.steps[-1].op == OP_TRAP
    pt_trapped: bool = len(pt_trace.steps) > 0 and pt_trace.steps[-1].op == OP_TRAP
    match, detail = compare_traces(np_trace, pt_trace)
    all_ok: bool = np_trapped and pt_trapped and match

    status = "PASS" if all_ok else "FAIL"
    np_label = (
        f"TRAP@{len(np_trace.steps)}"
        if np_trapped
        else f"top={np_trace.steps[-1].top if np_trace.steps else '?'}"
    )
    pt_label = (
        f"TRAP@{len(pt_trace.steps)}"
        if pt_trapped
        else f"top={pt_trace.steps[-1].top if pt_trace.steps else '?'}"
    )
    logger.info(
        "  %s  %-30s  numpy=%10s  torch=%10s  trace_match=%s",
        status,
        cfg.name,
        np_label,
        pt_label,
        "Y" if match else "N",
    )

    if not all_ok and cfg.verbose:
        if not np_trapped:
            logger.info("         NumPy did not trap")
        if not pt_trapped:
            logger.info("         PyTorch did not trap")
        if not match:
            logger.info("         Trace mismatch: %s", detail)

    return all_ok
