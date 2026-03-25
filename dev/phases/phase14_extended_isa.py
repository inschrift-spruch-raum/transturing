"""
Phase 14: Extended ISA — Chunks 1-5: Arithmetic, Comparison, Bitwise, Unary & Parametric Operations + Integration Tests

Chunk 1 (Issue #11): 5 arithmetic opcodes:
  MUL   (13)  — a b → (a*b)
  DIV_S (14)  — a b → (b/a) signed, truncate toward zero. Trap if a==0.
  DIV_U (15)  — a b → (b/a) unsigned (same as DIV_S for positive ints)
  REM_S (16)  — a b → (b%a) signed, sign matches dividend
  REM_U (17)  — a b → (b%a) unsigned (same as REM_S for positive ints)

Chunk 2 (Issue #12): 11 comparison opcodes (all push 1=true or 0=false):
  EQZ   (18)  — a → (a==0 ? 1 : 0)       unary, sd=0
  EQ    (19)  — a b → (a==b ? 1 : 0)      sd=-1
  NE    (20)  — a b → (a≠b ? 1 : 0)       sd=-1
  LT_S  (21)  — a b → (b<a ? 1 : 0)       sd=-1, signed
  LT_U  (22)  — a b → (b<a ? 1 : 0)       sd=-1, unsigned (=LT_S for now)
  GT_S  (23)  — a b → (b>a ? 1 : 0)       sd=-1, signed
  GT_U  (24)  — a b → (b>a ? 1 : 0)       sd=-1, unsigned (=GT_S for now)
  LE_S  (25)  — a b → (b≤a ? 1 : 0)       sd=-1, signed
  LE_U  (26)  — a b → (b≤a ? 1 : 0)       sd=-1, unsigned (=LE_S for now)
  GE_S  (27)  — a b → (b≥a ? 1 : 0)       sd=-1, signed
  GE_U  (28)  — a b → (b≥a ? 1 : 0)       sd=-1, unsigned (=GE_S for now)

Chunk 3 (Issue #13): 8 bitwise opcodes (all binary, sd=-1):
  AND   (29)  — a b → (a & b)
  OR    (30)  — a b → (a | b)
  XOR   (31)  — a b → (a ^ b)
  SHL   (32)  — a b → (b << a)            shift count from top, masked to 0-31
  SHR_S (33)  — a b → (b >> a) signed     arithmetic right shift
  SHR_U (34)  — a b → (b >> a) unsigned   logical right shift
  ROTL  (35)  — a b → rotl(b, a)          32-bit left rotate
  ROTR  (36)  — a b → rotr(b, a)          32-bit right rotate

Chunk 4 (Issue #14): 5 unary ops + 1 parametric op:
  CLZ    (37)  — a → clz(a)               Count leading zeros (32-bit), sd=0
  CTZ    (38)  — a → ctz(a)               Count trailing zeros (32-bit), sd=0
  POPCNT (39)  — a → popcnt(a)            Population count (32-bit), sd=0
  ABS    (40)  — a → abs(a)               Absolute value, sd=0
  NEG    (41)  — a → -a                   Negate, sd=0
  SELECT (42)  — a b c → (c≠0 ? a : b)   Ternary select, sd=-2

Division by zero triggers OP_TRAP (99): executor appends a TraceStep with
op=OP_TRAP and breaks. The runner prints "TRAP: division by zero" instead
of a result.

Architecture note: MUL/DIV/REM, comparisons, and bitwise ops are NONLINEAR.
ADD/SUB can be expressed as linear combinations via M_top. MUL (val_a *
val_b), comparisons (conditional 0/1), and bitwise ops (integer logic)
cannot. The FF dispatch has:
  - M_top: linear routing matrix (handles PUSH through ROT)
  - Nonlinear override: explicit computation for arith + cmp + bitwise
This is actually more faithful to real transformer FF layers (which have
nonlinear activations) than the pure-linear M_top was.

D_MODEL=36 constraint: comparison ops share embedding dims for S/U pairs.
Bitwise ops (Chunk 3) use OPCODE_IDX for dispatch but do not have
dedicated one-hot embedding dims (D_MODEL=36 is fully allocated).
The compiled model routes via opcode_one_hot from OPCODE_IDX, not
embedding dims, so this works without D_MODEL expansion.

Chunk 4 unary ops (CLZ, CTZ, POPCNT, ABS, NEG) also lack dedicated
embedding dims and dispatch via OPCODE_IDX. SELECT reads three stack
values (sp, sp-1, sp-2) using Head 4 for SP-2, with sd=-2.

No new attention heads needed — all ops use the existing stack read/write
mechanism.

Part of Issue #8 (Tier 1 ISA expansion).
  Chunk 1: Issue #11 (closed). Chunk 2: Issue #12. Chunk 3: Issue #13.
  Chunk 4: Issue #14. Chunk 5: Issue #15 (integration tests).
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_stack_machine import (
    program, Instruction, ReferenceExecutor, Trace, TraceStep,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT, OP_NAMES,
    TOKENS_PER_STEP, ALL_TESTS,
)

from phase12_percepta_model import (
    PerceptaModel, PerceptaExtendedExecutor, CompiledAttentionHead,
    embed_program_token, embed_stack_entry, embed_state,
    D_MODEL, DTYPE, EPS, N_OPCODES as N_OPCODES_BASE,
    DIM_IS_PROG, DIM_IS_STACK, DIM_IS_STATE,
    DIM_PROG_KEY_0, DIM_PROG_KEY_1,
    DIM_STACK_KEY_0, DIM_STACK_KEY_1,
    DIM_OPCODE, DIM_VALUE, DIM_IP, DIM_SP, DIM_ONE,
    DIM_IS_PUSH, DIM_IS_POP, DIM_IS_ADD, DIM_IS_DUP, DIM_IS_HALT,
    DIM_IS_SUB, DIM_IS_JZ, DIM_IS_JNZ, DIM_IS_NOP,
    OP_SUB, OP_JZ, OP_JNZ, OP_NOP,
)

# Import Phase 13 components
from phase13_isa_completeness import (
    Phase13Executor, Phase13Model, Phase13PyTorchExecutor,
    embed_program_token_ext as embed_program_token_p13,
    OP_SWAP, OP_OVER, OP_ROT,
    DIM_IS_SWAP, DIM_IS_OVER, DIM_IS_ROT,
    OPCODE_DIM_MAP as OPCODE_DIM_MAP_P13,
    OPCODE_IDX as OPCODE_IDX_P13,
    OP_NAMES_P13,
    N_OPCODES as N_OPCODES_P13,
    compare_traces, test_algorithm,
    make_fibonacci, make_power_of_2, make_sum_1_to_n,
    make_multiply as make_multiply_p13,
    make_is_even as make_is_even_p13,
    fib,
)


# ─── New Opcodes ──────────────────────────────────────────────────

OP_MUL   = 13
OP_DIV_S = 14
OP_DIV_U = 15
OP_REM_S = 16
OP_REM_U = 17
OP_EQZ   = 18
OP_EQ    = 19
OP_NE    = 20
OP_LT_S  = 21
OP_LT_U  = 22
OP_GT_S  = 23
OP_GT_U  = 24
OP_LE_S  = 25
OP_LE_U  = 26
OP_GE_S  = 27
OP_GE_U  = 28
OP_AND   = 29
OP_OR    = 30
OP_XOR   = 31
OP_SHL   = 32
OP_SHR_S = 33
OP_SHR_U = 34
OP_ROTL  = 35
OP_ROTR  = 36
OP_CLZ    = 37
OP_CTZ    = 38
OP_POPCNT = 39
OP_ABS    = 40
OP_NEG    = 41
OP_SELECT = 42
OP_TRAP  = 99  # Division by zero exit condition

OP_NAMES_P14 = {
    **OP_NAMES_P13,
    OP_MUL:   "MUL",
    OP_DIV_S: "DIV_S",
    OP_DIV_U: "DIV_U",
    OP_REM_S: "REM_S",
    OP_REM_U: "REM_U",
    OP_EQZ:   "EQZ",
    OP_EQ:    "EQ",
    OP_NE:    "NE",
    OP_LT_S:  "LT_S",
    OP_LT_U:  "LT_U",
    OP_GT_S:  "GT_S",
    OP_GT_U:  "GT_U",
    OP_LE_S:  "LE_S",
    OP_LE_U:  "LE_U",
    OP_GE_S:  "GE_S",
    OP_GE_U:  "GE_U",
    OP_AND:   "AND",
    OP_OR:    "OR",
    OP_XOR:   "XOR",
    OP_SHL:   "SHL",
    OP_SHR_S: "SHR_S",
    OP_SHR_U: "SHR_U",
    OP_ROTL:  "ROTL",
    OP_ROTR:  "ROTR",
    OP_CLZ:    "CLZ",
    OP_CTZ:    "CTZ",
    OP_POPCNT: "POPCNT",
    OP_ABS:    "ABS",
    OP_NEG:    "NEG",
    OP_SELECT: "SELECT",
    OP_TRAP:  "TRAP",
}

# One-hot dimension assignments (continuing from Phase 13: SWAP=21, OVER=22, ROT=23)
# Chunk 1: arithmetic
DIM_IS_MUL   = 24
DIM_IS_DIV_S = 25
DIM_IS_DIV_U = 26
DIM_IS_REM_S = 27
DIM_IS_REM_U = 28

# Chunk 2: comparisons — S/U pairs share dims (D_MODEL=36, dims 29-35 available)
DIM_IS_EQZ   = 29
DIM_IS_EQ    = 30
DIM_IS_NE    = 31
DIM_IS_LT    = 32  # shared by LT_S and LT_U
DIM_IS_GT    = 33  # shared by GT_S and GT_U
DIM_IS_LE    = 34  # shared by LE_S and LE_U
DIM_IS_GE    = 35  # shared by GE_S and GE_U

OPCODE_DIM_MAP = {
    **OPCODE_DIM_MAP_P13,
    OP_MUL:   DIM_IS_MUL,
    OP_DIV_S: DIM_IS_DIV_S,
    OP_DIV_U: DIM_IS_DIV_U,
    OP_REM_S: DIM_IS_REM_S,
    OP_REM_U: DIM_IS_REM_U,
    OP_EQZ:   DIM_IS_EQZ,
    OP_EQ:    DIM_IS_EQ,
    OP_NE:    DIM_IS_NE,
    OP_LT_S:  DIM_IS_LT,
    OP_LT_U:  DIM_IS_LT,
    OP_GT_S:  DIM_IS_GT,
    OP_GT_U:  DIM_IS_GT,
    OP_LE_S:  DIM_IS_LE,
    OP_LE_U:  DIM_IS_LE,
    OP_GE_S:  DIM_IS_GE,
    OP_GE_U:  DIM_IS_GE,
}

OPCODE_IDX = {
    **OPCODE_IDX_P13,
    OP_MUL:   12,
    OP_DIV_S: 13,
    OP_DIV_U: 14,
    OP_REM_S: 15,
    OP_REM_U: 16,
    OP_EQZ:   17,
    OP_EQ:    18,
    OP_NE:    19,
    OP_LT_S:  20,
    OP_LT_U:  21,
    OP_GT_S:  22,
    OP_GT_U:  23,
    OP_LE_S:  24,
    OP_LE_U:  25,
    OP_GE_S:  26,
    OP_GE_U:  27,
    OP_AND:   28,
    OP_OR:    29,
    OP_XOR:   30,
    OP_SHL:   31,
    OP_SHR_S: 32,
    OP_SHR_U: 33,
    OP_ROTL:  34,
    OP_ROTR:  35,
    OP_CLZ:    36,
    OP_CTZ:    37,
    OP_POPCNT: 38,
    OP_ABS:    39,
    OP_NEG:    40,
    OP_SELECT: 41,
}

N_OPCODES = 42  # 12 base + 5 arith + 11 cmp + 8 bitwise + 5 unary + 1 parametric

# Which opcodes are nonlinear (can't be expressed via M_top linear routing)
NONLINEAR_OPS = {OP_MUL, OP_DIV_S, OP_DIV_U, OP_REM_S, OP_REM_U,
                 OP_EQZ, OP_EQ, OP_NE,
                 OP_LT_S, OP_LT_U, OP_GT_S, OP_GT_U,
                 OP_LE_S, OP_LE_U, OP_GE_S, OP_GE_U,
                 OP_AND, OP_OR, OP_XOR,
                 OP_SHL, OP_SHR_S, OP_SHR_U,
                 OP_ROTL, OP_ROTR,
                 OP_CLZ, OP_CTZ, OP_POPCNT, OP_ABS, OP_NEG, OP_SELECT}


# ─── Signed division/remainder (truncate toward zero) ─────────────

def _trunc_div(b, a):
    """Signed integer division truncating toward zero (WASM semantics).
    Python's // rounds toward negative infinity; we want C-style truncation.
    """
    return int(b / a)  # float division then truncate


def _trunc_rem(b, a):
    """Signed remainder matching truncated division: b - trunc(b/a)*a.
    Sign of result matches dividend b (WASM i32.rem_s semantics).
    """
    return b - _trunc_div(b, a) * a


# ─── Bitwise helpers (32-bit word semantics) ─────────────────────

MASK32 = 0xFFFFFFFF

def _to_i32(val):
    """Cast to 32-bit signed integer from potentially float stack value."""
    return int(val) & MASK32

def _shr_u(b, a):
    """Logical (unsigned) right shift of b by a positions.
    b is treated as unsigned 32-bit; result is unsigned.
    """
    return (_to_i32(b) >> (int(a) & 31))

def _shr_s(b, a):
    """Arithmetic (signed) right shift of b by a positions.
    b is treated as signed 32-bit; sign bit is preserved.
    """
    val = _to_i32(b)
    # Convert to signed 32-bit
    if val >= 0x80000000:
        val -= 0x100000000
    shift = int(a) & 31
    result = val >> shift
    # Back to unsigned representation for storage
    return result & MASK32 if result < 0 else result

def _rotl32(b, a):
    """Left-rotate b by a positions within 32-bit word."""
    val = _to_i32(b)
    shift = int(a) & 31
    return ((val << shift) | (val >> (32 - shift))) & MASK32 if shift else val

def _rotr32(b, a):
    """Right-rotate b by a positions within 32-bit word."""
    val = _to_i32(b)
    shift = int(a) & 31
    return ((val >> shift) | (val << (32 - shift))) & MASK32 if shift else val


def _clz32(val):
    """Count leading zeros in 32-bit representation."""
    v = _to_i32(val)
    if v == 0:
        return 32
    n = 0
    if v <= 0x0000FFFF: n += 16; v <<= 16
    if v <= 0x00FFFFFF: n += 8;  v <<= 8
    if v <= 0x0FFFFFFF: n += 4;  v <<= 4
    if v <= 0x3FFFFFFF: n += 2;  v <<= 2
    if v <= 0x7FFFFFFF: n += 1
    return n

def _ctz32(val):
    """Count trailing zeros in 32-bit representation."""
    v = _to_i32(val)
    if v == 0:
        return 32
    n = 0
    if (v & 0x0000FFFF) == 0: n += 16; v >>= 16
    if (v & 0x000000FF) == 0: n += 8;  v >>= 8
    if (v & 0x0000000F) == 0: n += 4;  v >>= 4
    if (v & 0x00000003) == 0: n += 2;  v >>= 2
    if (v & 0x00000001) == 0: n += 1
    return n

def _popcnt32(val):
    """Population count (number of set bits) in 32-bit representation."""
    return bin(_to_i32(val)).count('1')

def embed_program_token_ext(pos, instr):
    """Create embedding for a program instruction (Phase 14 ISA)."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_PROG]     = 1.0
    emb[DIM_PROG_KEY_0]  = 2.0 * pos
    emb[DIM_PROG_KEY_1]  = -float(pos * pos)
    emb[DIM_OPCODE]      = float(instr.op)
    emb[DIM_VALUE]       = float(instr.arg)
    emb[DIM_ONE]         = 1.0
    dim = OPCODE_DIM_MAP.get(instr.op)
    if dim is not None:
        emb[dim] = 1.0
    return emb


# ─── NumPy Executor ───────────────────────────────────────────────

class Phase14Executor(Phase13Executor):
    """Compiled numpy executor with Phase 14 arithmetic ops.

    Adds MUL, DIV_S, DIV_U, REM_S, REM_U to Phase 13's executor.
    Division by zero produces OP_TRAP exit.
    """

    def execute(self, prog, max_steps=5000):
        trace = Trace(program=prog)

        prog_keys = np.array([(2.0*j, -float(j*j)) for j in range(len(prog))])
        prog_ops = np.array([instr.op for instr in prog])
        prog_args = np.array([instr.arg for instr in prog])

        stack_keys = []
        stack_vals = []
        write_count = 0
        eps = 1e-10

        ip = 0
        sp = 0

        def stack_write(addr, val):
            nonlocal write_count
            stack_keys.append((2.0*addr, -float(addr*addr) + eps*write_count))
            stack_vals.append(val)
            write_count += 1

        def stack_read(addr):
            if not stack_keys:
                return 0
            keys = np.array(stack_keys)
            q = np.array([addr, 1.0])
            scores = keys @ q
            best = np.argmax(scores)
            stored_addr = round(keys[best, 0] / 2.0)
            return stack_vals[best] if stored_addr == addr else 0

        for step in range(max_steps):
            if ip >= len(prog):
                break

            op = prog[ip].op
            arg = prog[ip].arg
            next_ip = ip + 1
            top = 0

            # ── Phase 4 base ops ──
            if op == OP_PUSH:
                sp += 1
                stack_write(sp, arg)
                top = arg
            elif op == OP_POP:
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_ADD:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = val_a + val_b
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SUB:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = val_b - val_a
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_DUP:
                val = stack_read(sp)
                sp += 1
                stack_write(sp, val)
                top = val

            # ── Phase 13 stack manipulation ──
            elif op == OP_SWAP:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                stack_write(sp, val_b)
                stack_write(sp - 1, val_a)
                top = val_b
            elif op == OP_OVER:
                val_b = stack_read(sp - 1)
                sp += 1
                stack_write(sp, val_b)
                top = val_b
            elif op == OP_ROT:
                val_top    = stack_read(sp)
                val_second = stack_read(sp - 1)
                val_third  = stack_read(sp - 2)
                stack_write(sp, val_third)
                stack_write(sp - 1, val_top)
                stack_write(sp - 2, val_second)
                top = val_third

            # ── Phase 14 arithmetic ──
            elif op == OP_MUL:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = val_a * val_b
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op in (OP_DIV_S, OP_DIV_U):
                val_a = stack_read(sp)
                if val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                val_b = stack_read(sp - 1)
                result = _trunc_div(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op in (OP_REM_S, OP_REM_U):
                val_a = stack_read(sp)
                if val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                val_b = stack_read(sp - 1)
                result = _trunc_rem(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result

            # ── Phase 14 Chunk 2: comparison ops ──
            elif op == OP_EQZ:
                val_a = stack_read(sp)
                result = 1 if val_a == 0 else 0
                stack_write(sp, result)  # sd=0, replaces top
                top = result
            elif op in (OP_EQ, OP_NE, OP_LT_S, OP_LT_U, OP_GT_S, OP_GT_U,
                        OP_LE_S, OP_LE_U, OP_GE_S, OP_GE_U):
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                if op in (OP_EQ,):
                    result = 1 if val_a == val_b else 0
                elif op in (OP_NE,):
                    result = 1 if val_a != val_b else 0
                elif op in (OP_LT_S, OP_LT_U):
                    result = 1 if val_b < val_a else 0
                elif op in (OP_GT_S, OP_GT_U):
                    result = 1 if val_b > val_a else 0
                elif op in (OP_LE_S, OP_LE_U):
                    result = 1 if val_b <= val_a else 0
                elif op in (OP_GE_S, OP_GE_U):
                    result = 1 if val_b >= val_a else 0
                sp -= 1
                stack_write(sp, result)
                top = result

            # ── Phase 14 Chunk 3: bitwise ops ──
            elif op == OP_AND:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _to_i32(val_a) & _to_i32(val_b)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_OR:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _to_i32(val_a) | _to_i32(val_b)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_XOR:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _to_i32(val_a) ^ _to_i32(val_b)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SHL:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = (_to_i32(val_b) << (int(val_a) & 31)) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SHR_S:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _shr_s(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SHR_U:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _shr_u(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_ROTL:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _rotl32(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_ROTR:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = _rotr32(val_b, val_a)
                sp -= 1
                stack_write(sp, result)
                top = result

            # ── Phase 14 Chunk 4: unary + parametric ops ──
            elif op == OP_CLZ:
                val_a = stack_read(sp)
                result = _clz32(val_a)
                stack_write(sp, result)  # sd=0, replaces top
                top = result
            elif op == OP_CTZ:
                val_a = stack_read(sp)
                result = _ctz32(val_a)
                stack_write(sp, result)
                top = result
            elif op == OP_POPCNT:
                val_a = stack_read(sp)
                result = _popcnt32(val_a)
                stack_write(sp, result)
                top = result
            elif op == OP_ABS:
                val_a = stack_read(sp)
                result = abs(int(val_a))
                stack_write(sp, result)
                top = result
            elif op == OP_NEG:
                val_a = stack_read(sp)
                result = (-int(val_a)) & MASK32
                stack_write(sp, result)
                top = result
            elif op == OP_SELECT:
                val_a = stack_read(sp)       # c (condition)
                val_b = stack_read(sp - 1)   # b (false value)
                val_c = stack_read(sp - 2)   # a (true value)
                result = val_c if val_a != 0 else val_b
                sp -= 2
                stack_write(sp, result)
                top = result

            # ── Control flow ──
            elif op == OP_JZ:
                cond = stack_read(sp)
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
                if cond == 0:
                    next_ip = arg
            elif op == OP_JNZ:
                cond = stack_read(sp)
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
                if cond != 0:
                    next_ip = arg
            elif op == OP_NOP:
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_HALT:
                top = stack_read(sp) if sp > 0 else 0
                trace.steps.append(TraceStep(op, arg, sp, top))
                break
            else:
                # Unknown opcode — treat as NOP
                top = stack_read(sp) if sp > 0 else 0

            trace.steps.append(TraceStep(op, arg, sp, top))
            ip = next_ip

        return trace


# ─── PyTorch Model ────────────────────────────────────────────────

class Phase14Model(Phase13Model):
    """Compiled transformer with Phase 14 arithmetic + comparison + bitwise ops.

    Extends Phase 13's FF dispatch with nonlinear computation for
    MUL, DIV_S, DIV_U, REM_S, REM_U (arithmetic),
    EQZ, EQ, NE, LT/GT/LE/GE_S/U (comparisons), and
    AND, OR, XOR, SHL, SHR_S, SHR_U, ROTL, ROTR (bitwise).

    These can't be expressed as linear routing via M_top:
      MUL = val_a * val_b (product, not linear combination)
      EQ = 1 if val_a == val_b else 0 (conditional, not linear)
      AND = int(val_a) & int(val_b) (integer logic, not linear)

    Architecture: M_top handles linear ops (PUSH through ROT).
    Nonlinear ops have M_top rows set to zero; their results come
    from explicit computation in forward(). The one-hot opcode
    vector selects which path contributes to the final top value.
    """

    def __init__(self, d_model=D_MODEL):
        # Skip Phase13Model.__init__(), build from nn.Module
        nn.Module.__init__(self)
        self.d_model = d_model

        # Heads 0-4: same as Phase 13
        self.head_prog_op  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_prog_arg = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_a  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_b  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1, use_bias_q=True)
        self.head_stack_c  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1, use_bias_q=True)

        # FF dispatch: 36 opcodes, 4 value inputs
        self.register_buffer('M_top', torch.zeros(N_OPCODES, 4, dtype=DTYPE))
        self.register_buffer('sp_deltas', torch.zeros(N_OPCODES, dtype=DTYPE))

        self._compile_weights()

    def _compile_weights(self):
        """Set all weight matrices analytically."""
        with torch.no_grad():
            # ── Heads 0-4: identical to Phase 13 ──

            # Head 0: Program opcode fetch
            W = torch.zeros(2, self.d_model)
            W[0, DIM_IP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_prog_op.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_PROG_KEY_0] = 1.0
            W[1, DIM_PROG_KEY_1] = 1.0
            self.head_prog_op.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_OPCODE] = 1.0
            self.head_prog_op.W_V.weight.copy_(W)

            # Head 1: Program argument fetch
            W = torch.zeros(2, self.d_model)
            W[0, DIM_IP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_prog_arg.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_PROG_KEY_0] = 1.0
            W[1, DIM_PROG_KEY_1] = 1.0
            self.head_prog_arg.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_prog_arg.W_V.weight.copy_(W)

            # Head 2: Stack read at SP
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_a.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_a.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_a.W_V.weight.copy_(W)

            # Head 3: Stack read at SP-1
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_b.W_Q.weight.copy_(W)
            b = torch.zeros(2)
            b[0] = -1.0
            self.head_stack_b.W_Q.bias.copy_(b)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_b.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_b.W_V.weight.copy_(W)

            # Head 4: Stack read at SP-2 (for ROT)
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_c.W_Q.weight.copy_(W)
            b = torch.zeros(2)
            b[0] = -2.0
            self.head_stack_c.W_Q.bias.copy_(b)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_c.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_c.W_V.weight.copy_(W)

            # ── FF dispatch: linear routing (Phase 13 ops) ──
            # M_top maps [arg, val_a, val_b, val_c] → candidate top per opcode
            #                         arg  va   vb   vc
            self.M_top[0]  = torch.tensor([ 1.,  0.,  0.,  0.])  # PUSH: top = arg
            self.M_top[1]  = torch.tensor([ 0.,  0.,  1.,  0.])  # POP:  top = val_b
            self.M_top[2]  = torch.tensor([ 0.,  1.,  1.,  0.])  # ADD:  top = va + vb
            self.M_top[3]  = torch.tensor([ 0.,  1.,  0.,  0.])  # DUP:  top = va
            self.M_top[4]  = torch.tensor([ 0.,  1.,  0.,  0.])  # HALT: top = va
            self.M_top[5]  = torch.tensor([ 0., -1.,  1.,  0.])  # SUB:  top = vb - va
            self.M_top[6]  = torch.tensor([ 0.,  0.,  1.,  0.])  # JZ:   top = vb
            self.M_top[7]  = torch.tensor([ 0.,  0.,  1.,  0.])  # JNZ:  top = vb
            self.M_top[8]  = torch.tensor([ 0.,  1.,  0.,  0.])  # NOP:  top = va
            self.M_top[9]  = torch.tensor([ 0.,  0.,  1.,  0.])  # SWAP: top = vb
            self.M_top[10] = torch.tensor([ 0.,  0.,  1.,  0.])  # OVER: top = vb
            self.M_top[11] = torch.tensor([ 0.,  0.,  0.,  1.])  # ROT:  top = vc

            # Nonlinear ops: M_top rows stay zero — results computed in forward()
            # self.M_top[12..35] = 0  (MUL..GE_U, AND..ROTR)

            # SP deltas:
            # Idx: PUSH POP  ADD  DUP HALT SUB  JZ  JNZ NOP SWAP OVER ROT
            #      MUL DIVS DIVU REMS REMU EQZ  EQ   NE LTS LTU GTS GTU LES LEU GES GEU
            #      AND  OR  XOR  SHL SHRS SHRU ROTL ROTR
            #      CLZ  CTZ  POPCNT ABS NEG SELECT
            self.sp_deltas.copy_(torch.tensor(
                [1., -1., -1., 1., 0., -1., -1., -1., 0., 0., 1., 0.,
                 -1., -1., -1., -1., -1.,
                 0.,  -1., -1., -1., -1., -1., -1., -1., -1., -1., -1.,
                 -1., -1., -1., -1., -1., -1., -1., -1.,
                 0., 0., 0., 0., 0., -2.]))

    def forward(self, query_emb, prog_embs, stack_embs):
        """Execute one step.

        Returns (opcode, arg, sp_delta, top, opcode_one_hot, val_a, val_b, val_c).
        """
        # Head 0: Fetch opcode
        opcode_val, _, _ = self.head_prog_op(query_emb, prog_embs)
        # Head 1: Fetch argument
        arg_val, _, _ = self.head_prog_arg(query_emb, prog_embs)

        # Head 2: Read stack[SP]
        if stack_embs.shape[0] > 0:
            val_a_raw, _, idx_a = self.head_stack_a(query_emb, stack_embs)
            stored_addr_a = round(stack_embs[idx_a, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp = round(query_emb[DIM_SP].item())
            val_a = val_a_raw[0] if stored_addr_a == queried_sp else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_a = torch.tensor(0.0, dtype=DTYPE)

        # Head 3: Read stack[SP-1]
        if stack_embs.shape[0] > 0:
            val_b_raw, _, idx_b = self.head_stack_b(query_emb, stack_embs)
            stored_addr_b = round(stack_embs[idx_b, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m1 = round(query_emb[DIM_SP].item()) - 1
            val_b = val_b_raw[0] if stored_addr_b == queried_sp_m1 else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_b = torch.tensor(0.0, dtype=DTYPE)

        # Head 4: Read stack[SP-2]
        if stack_embs.shape[0] > 0:
            val_c_raw, _, idx_c = self.head_stack_c(query_emb, stack_embs)
            stored_addr_c = round(stack_embs[idx_c, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m2 = round(query_emb[DIM_SP].item()) - 2
            val_c = val_c_raw[0] if stored_addr_c == queried_sp_m2 else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_c = torch.tensor(0.0, dtype=DTYPE)

        # Decode
        opcode = round(opcode_val[0].item())
        arg = round(arg_val[0].item())

        # FF Dispatch — linear path (Phase 13 ops)
        opcode_one_hot = torch.zeros(N_OPCODES, dtype=DTYPE)
        idx = OPCODE_IDX.get(opcode, -1)
        if idx >= 0:
            opcode_one_hot[idx] = 1.0

        values = torch.stack([
            torch.tensor(float(arg), dtype=DTYPE),
            val_a, val_b, val_c
        ])
        candidates = self.M_top @ values  # (N_OPCODES,)
        top_linear = (opcode_one_hot * candidates).sum()

        # FF Dispatch — nonlinear path (Phase 14 arithmetic + comparisons + bitwise)
        va = round(val_a.item())
        vb = round(val_b.item())

        nonlinear = torch.zeros(N_OPCODES, dtype=DTYPE)
        nonlinear[OPCODE_IDX[OP_MUL]] = float(va * vb)
        if va != 0:
            nonlinear[OPCODE_IDX[OP_DIV_S]] = float(_trunc_div(vb, va))
            nonlinear[OPCODE_IDX[OP_DIV_U]] = float(_trunc_div(vb, va))
            nonlinear[OPCODE_IDX[OP_REM_S]] = float(_trunc_rem(vb, va))
            nonlinear[OPCODE_IDX[OP_REM_U]] = float(_trunc_rem(vb, va))
        # else: zeros — executor handles the trap, model just returns 0

        # Comparison ops: produce 1.0 (true) or 0.0 (false)
        nonlinear[OPCODE_IDX[OP_EQZ]]  = 1.0 if va == 0 else 0.0
        nonlinear[OPCODE_IDX[OP_EQ]]   = 1.0 if va == vb else 0.0
        nonlinear[OPCODE_IDX[OP_NE]]   = 1.0 if va != vb else 0.0
        nonlinear[OPCODE_IDX[OP_LT_S]] = 1.0 if vb < va else 0.0
        nonlinear[OPCODE_IDX[OP_LT_U]] = 1.0 if vb < va else 0.0
        nonlinear[OPCODE_IDX[OP_GT_S]] = 1.0 if vb > va else 0.0
        nonlinear[OPCODE_IDX[OP_GT_U]] = 1.0 if vb > va else 0.0
        nonlinear[OPCODE_IDX[OP_LE_S]] = 1.0 if vb <= va else 0.0
        nonlinear[OPCODE_IDX[OP_LE_U]] = 1.0 if vb <= va else 0.0
        nonlinear[OPCODE_IDX[OP_GE_S]] = 1.0 if vb >= va else 0.0
        nonlinear[OPCODE_IDX[OP_GE_U]] = 1.0 if vb >= va else 0.0

        # Bitwise ops: integer logic on 32-bit values
        nonlinear[OPCODE_IDX[OP_AND]]   = float(_to_i32(va) & _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_OR]]    = float(_to_i32(va) | _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_XOR]]   = float(_to_i32(va) ^ _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_SHL]]   = float((_to_i32(vb) << (int(va) & 31)) & MASK32)
        nonlinear[OPCODE_IDX[OP_SHR_S]] = float(_shr_s(vb, va))
        nonlinear[OPCODE_IDX[OP_SHR_U]] = float(_shr_u(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTL]]  = float(_rotl32(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTR]]  = float(_rotr32(vb, va))

        # Unary ops: operate on val_a (top of stack), sd=0
        nonlinear[OPCODE_IDX[OP_CLZ]]    = float(_clz32(va))
        nonlinear[OPCODE_IDX[OP_CTZ]]    = float(_ctz32(va))
        nonlinear[OPCODE_IDX[OP_POPCNT]] = float(_popcnt32(va))
        nonlinear[OPCODE_IDX[OP_ABS]]    = float(abs(int(va)))
        nonlinear[OPCODE_IDX[OP_NEG]]    = float((-int(va)) & MASK32)

        # Parametric: SELECT — c≠0 ? a : b where c=va(sp), b=vb(sp-1), a=vc(sp-2)
        vc = round(val_c.item())
        nonlinear[OPCODE_IDX[OP_SELECT]] = float(vc if va != 0 else vb)

        top_nonlinear = (opcode_one_hot * nonlinear).sum()
        top = top_linear + top_nonlinear

        sp_delta = (opcode_one_hot * self.sp_deltas).sum()

        return (opcode, arg, int(sp_delta.item()), round(top.item()),
                opcode_one_hot, round(val_a.item()), round(val_b.item()), round(val_c.item()))


# ─── PyTorch Executor ─────────────────────────────────────────────

class Phase14PyTorchExecutor:
    """Executes programs using Phase14Model with full Phase 14 ISA."""

    def __init__(self, model=None):
        self.model = model or Phase14Model()
        self.model.eval()

    def execute(self, prog, max_steps=5000):
        trace = Trace(program=prog)

        prog_embs = torch.stack([
            embed_program_token_ext(i, instr)
            for i, instr in enumerate(prog)
        ])

        stack_embs_list = []
        write_count = 0
        ip = 0
        sp = 0

        with torch.no_grad():
            for step in range(max_steps):
                if ip >= len(prog):
                    break

                query = embed_state(ip, sp)
                stack_embs = (torch.stack(stack_embs_list)
                              if stack_embs_list
                              else torch.zeros(0, D_MODEL, dtype=DTYPE))

                opcode, arg, sp_delta, top, _, val_a, val_b, val_c = \
                    self.model.forward(query, prog_embs, stack_embs)

                if opcode == OP_HALT:
                    trace.steps.append(TraceStep(opcode, arg, sp, top))
                    break

                # Trap: division by zero
                if opcode in (OP_DIV_S, OP_DIV_U, OP_REM_S, OP_REM_U) and val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break

                # For JZ/JNZ: read condition BEFORE updating SP
                cond_val = None
                if opcode in (OP_JZ, OP_JNZ):
                    cond_val = val_a

                new_sp = sp + sp_delta

                # Stack writes per opcode
                if opcode in (OP_PUSH, OP_DUP, OP_OVER):
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode in (OP_ADD, OP_SUB, OP_MUL, OP_DIV_S, OP_DIV_U,
                                OP_REM_S, OP_REM_U,
                                OP_EQ, OP_NE,
                                OP_LT_S, OP_LT_U, OP_GT_S, OP_GT_U,
                                OP_LE_S, OP_LE_U, OP_GE_S, OP_GE_U,
                                OP_AND, OP_OR, OP_XOR,
                                OP_SHL, OP_SHR_S, OP_SHR_U,
                                OP_ROTL, OP_ROTR):
                    # All binary ops: pop two, push one at new_sp
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode in (OP_EQZ, OP_CLZ, OP_CTZ, OP_POPCNT, OP_ABS, OP_NEG):
                    # Unary: sd=0, replaces top at same sp
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_SELECT:
                    # Parametric: sd=-2, push result at new_sp
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_SWAP:
                    stack_embs_list.append(
                        embed_stack_entry(sp, val_b, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count))
                    write_count += 1
                elif opcode == OP_ROT:
                    stack_embs_list.append(
                        embed_stack_entry(sp, val_c, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 2, val_b, write_count))
                    write_count += 1

                trace.steps.append(TraceStep(opcode, arg, new_sp, top))
                sp = new_sp

                # IP update
                if opcode == OP_JZ:
                    ip = arg if cond_val == 0 else ip + 1
                elif opcode == OP_JNZ:
                    ip = arg if cond_val != 0 else ip + 1
                else:
                    ip += 1

        return trace


# ─── Program Generators ──────────────────────────────────────────

def make_native_multiply(a, b):
    """Compute a*b using native MUL. 4 instructions.

    Contrast with Phase 13's make_multiply: 18 instructions, O(b) steps.
    """
    return [
        Instruction(OP_PUSH, a),
        Instruction(OP_PUSH, b),
        Instruction(OP_MUL),
        Instruction(OP_HALT),
    ], a * b


def make_native_divmod(a, b):
    """Compute b/a and b%a. Returns (program, expected_quotient).

    Stack: PUSH b, PUSH a, DIV_S → quotient on top.
    """
    if a == 0:
        # Will trap
        return [
            Instruction(OP_PUSH, b),
            Instruction(OP_PUSH, a),
            Instruction(OP_DIV_S),
            Instruction(OP_HALT),
        ], None  # None signals expected trap
    return [
        Instruction(OP_PUSH, b),
        Instruction(OP_PUSH, a),
        Instruction(OP_DIV_S),
        Instruction(OP_HALT),
    ], _trunc_div(b, a)


def make_native_remainder(a, b):
    """Compute b%a using REM_S. Returns (program, expected_remainder)."""
    if a == 0:
        return [
            Instruction(OP_PUSH, b),
            Instruction(OP_PUSH, a),
            Instruction(OP_REM_S),
            Instruction(OP_HALT),
        ], None  # trap
    return [
        Instruction(OP_PUSH, b),
        Instruction(OP_PUSH, a),
        Instruction(OP_REM_S),
        Instruction(OP_HALT),
    ], _trunc_rem(b, a)


def make_native_is_even(n):
    """Test parity using native REM_S + JZ. ~7 instructions.

    Contrast with Phase 13's make_is_even: 17 instructions, O(n) steps.
    """
    prog = [
        Instruction(OP_PUSH, n),      # 0: n
        Instruction(OP_PUSH, 2),      # 1: 2
        Instruction(OP_REM_S),        # 2: n % 2
        Instruction(OP_JZ, 6),        # 3: if remainder == 0 → even
        # Odd
        Instruction(OP_PUSH, 0),      # 4
        Instruction(OP_HALT),         # 5
        # Even
        Instruction(OP_PUSH, 1),      # 6
        Instruction(OP_HALT),         # 7
    ]
    return prog, 1 if n % 2 == 0 else 0


def make_factorial(n):
    """Compute n! using native MUL.

    Stack layout: [result, counter]
    Each iteration: result *= counter, counter -= 1.

    Returns (program, expected_result).
    """
    if n <= 1:
        return [Instruction(OP_PUSH, 1), Instruction(OP_HALT)], 1

    prog = [
        Instruction(OP_PUSH, 1),      # 0: result = 1
        Instruction(OP_PUSH, n),      # 1: counter = n
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [result, counter, counter]
        Instruction(OP_JZ, 12),       # 3: if counter == 0 → done
        # Multiply: result *= counter
        Instruction(OP_DUP),          # 4: [result, counter, counter]
        Instruction(OP_ROT),          # 5: [counter, counter, result]
        Instruction(OP_MUL),          # 6: [counter, counter*result]
        Instruction(OP_SWAP),         # 7: [counter*result, counter]
        # Decrement counter
        Instruction(OP_PUSH, 1),      # 8
        Instruction(OP_SUB),          # 9: [counter*result, counter-1]
        # Loop
        Instruction(OP_PUSH, 1),      # 10
        Instruction(OP_JNZ, 2),       # 11
        # ── Done (addr 12) ──
        Instruction(OP_POP),          # 12: drop 0
        Instruction(OP_HALT),         # 13
    ]
    expected = 1
    for i in range(2, n + 1):
        expected *= i
    return prog, expected


def make_gcd(a, b):
    """Compute GCD(a, b) via Euclidean algorithm using native REM_S.

    Stack: [a, b]. Loop: if b==0 → done (a is GCD). Else: a, b = b, a%b.

    Returns (program, expected_result).
    """
    import math
    if a == 0 and b == 0:
        return [Instruction(OP_PUSH, 0), Instruction(OP_HALT)], 0

    prog = [
        Instruction(OP_PUSH, a),      # 0: a
        Instruction(OP_PUSH, b),      # 1: b
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [a, b, b]
        Instruction(OP_JZ, 10),       # 3: if b == 0 → done, result = a
        # a, b = b, a % b
        Instruction(OP_SWAP),         # 4: [b, a]   (note: OVER below reads b)
        Instruction(OP_OVER),         # 5: [b, a, b]
        Instruction(OP_REM_S),        # 6: [b, a%b]  (top=b, reads a and b)
        # Wait — REM_S pops top two: stack[sp]=b, stack[sp-1]=a → result = a%b
        # Actually: val_a = stack[sp] = b (the OVER'd copy), val_b = stack[sp-1] = a
        # result = val_b % val_a = a % b ✓
        # Stack now: [b, a%b]
        # Loop back
        Instruction(OP_PUSH, 1),      # 7
        Instruction(OP_JNZ, 2),       # 8: always jump
        Instruction(OP_NOP),          # 9: padding (never reached)
        # ── Done (addr 10) ──
        # Stack: [result, 0] after JZ popped the 0 copy
        Instruction(OP_POP),          # 10: drop 0
        Instruction(OP_HALT),         # 11
    ]
    return prog, math.gcd(a, b)


# ─── Comparison Program Generators ───────────────────────────────

def make_compare_eqz(a):
    """Test a == 0 using EQZ. Returns (program, expected_result)."""
    return [
        Instruction(OP_PUSH, a),
        Instruction(OP_EQZ),
        Instruction(OP_HALT),
    ], 1 if a == 0 else 0


def make_compare_binary(op, a, b):
    """Generic binary comparison: PUSH a, PUSH b, OP, HALT.

    Note: stack has a at sp-1, b at sp. The comparison semantics are
    b <op> a per the ISA spec (val_b <op> val_a).
    Wait — the spec says a b → (a==b) for EQ, but for LT_S it says
    a b → (b<a). Let's be careful:
      stack after PUSH a, PUSH b: sp points to b (top), sp-1 points to a.
      val_a = stack[sp] = b, val_b = stack[sp-1] = a.
      So for LT_S: result = (val_b < val_a) = (a < b).

    Returns (program, expected_result).
    """
    CMP_SEMANTICS = {
        OP_EQ:   lambda va, vb: vb == va,   # a == b → val_b==val_a
        OP_NE:   lambda va, vb: vb != va,
        OP_LT_S: lambda va, vb: vb < va,    # a < b → val_b < val_a
        OP_LT_U: lambda va, vb: vb < va,
        OP_GT_S: lambda va, vb: vb > va,
        OP_GT_U: lambda va, vb: vb > va,
        OP_LE_S: lambda va, vb: vb <= va,
        OP_LE_U: lambda va, vb: vb <= va,
        OP_GE_S: lambda va, vb: vb >= va,
        OP_GE_U: lambda va, vb: vb >= va,
    }
    # val_a = stack[sp] = b (second pushed), val_b = stack[sp-1] = a (first pushed)
    expected = 1 if CMP_SEMANTICS[op](b, a) else 0
    return [
        Instruction(OP_PUSH, a),
        Instruction(OP_PUSH, b),
        Instruction(op),
        Instruction(OP_HALT),
    ], expected


def make_native_max(a, b):
    """Compute max(a, b) using GT_S + JZ.

    If a > b: result = a. Else: result = b.
    Stack approach: push both, compare, branch.
    """
    expected = max(a, b)
    prog = [
        Instruction(OP_PUSH, a),      # 0
        Instruction(OP_PUSH, b),      # 1
        # Compare: is a > b?
        # Stack: [a, b]. val_a=b, val_b=a. GT_S → (val_b > val_a) → (a > b)
        Instruction(OP_GT_S),         # 2: [a>b ? 1 : 0]  ... wait, GT_S pops both
        # After GT_S: sp decreased by 1, result on stack. But we lost a and b!
        # Need to keep copies. Let me restructure.
    ]
    # Better approach: use DUP/OVER to keep copies
    prog = [
        Instruction(OP_PUSH, a),      # 0: [a]
        Instruction(OP_PUSH, b),      # 1: [a, b]
        Instruction(OP_OVER),         # 2: [a, b, a]
        Instruction(OP_OVER),         # 3: [a, b, a, b]
        # Stack top: val_a=b, val_b=a. GT_S: (a > b)?
        Instruction(OP_GT_S),         # 4: [a, b, (a>b)]
        Instruction(OP_JZ, 9),        # 5: if NOT(a>b) → b is max
        # a > b: drop b, keep a
        Instruction(OP_POP),          # 6: [a]
        Instruction(OP_HALT),         # 7
        Instruction(OP_NOP),          # 8: padding
        # b >= a: drop a (it's under b), keep b
        Instruction(OP_SWAP),         # 9: [b, a]
        Instruction(OP_POP),          # 10: [b]
        Instruction(OP_HALT),         # 11
    ]
    return prog, expected


def make_native_abs(n):
    """Compute abs(n) using LT_S comparison + conditional negate.

    For now, only works with values expressible in our ISA.
    Uses: if n < 0 then 0 - n else n.
    """
    expected = abs(n)
    prog = [
        Instruction(OP_PUSH, n),      # 0: [n]
        Instruction(OP_DUP),          # 1: [n, n]
        Instruction(OP_PUSH, 0),      # 2: [n, n, 0]
        # val_a=0, val_b=n. LT_S: (val_b < val_a) = (n < 0)?
        Instruction(OP_LT_S),        # 3: [n, (n<0)]
        Instruction(OP_JZ, 9),        # 4: if n >= 0 → already positive
        # n < 0: negate (0 - n)
        Instruction(OP_PUSH, 0),      # 5: [n, 0]
        Instruction(OP_SWAP),         # 6: [0, n]
        Instruction(OP_SUB),          # 7: [0-n] = [-n] = abs(n)
        Instruction(OP_HALT),         # 8
        # n >= 0: already the answer
        Instruction(OP_HALT),         # 9
    ]
    return prog, expected


def make_native_clamp(val, lo, hi):
    """Clamp val to [lo, hi] using comparisons.

    if val < lo: result = lo
    elif val > hi: result = hi
    else: result = val
    """
    expected = max(lo, min(val, hi))
    prog = [
        Instruction(OP_PUSH, val),    # 0: [val]
        # Check val < lo
        Instruction(OP_DUP),          # 1: [val, val]
        Instruction(OP_PUSH, lo),     # 2: [val, val, lo]
        # val_a=lo, val_b=val. LT_S: (val < lo)?
        Instruction(OP_LT_S),        # 3: [val, (val<lo)]
        Instruction(OP_JZ, 8),        # 4: if not(val<lo) → check upper
        # val < lo: replace with lo
        Instruction(OP_POP),          # 5: []
        Instruction(OP_PUSH, lo),     # 6: [lo]
        Instruction(OP_HALT),         # 7
        # Check val > hi
        Instruction(OP_DUP),          # 8: [val, val]
        Instruction(OP_PUSH, hi),     # 9: [val, val, hi]
        # val_a=hi, val_b=val. GT_S: (val > hi)?
        Instruction(OP_GT_S),         # 10: [val, (val>hi)]
        Instruction(OP_JZ, 15),       # 11: if not(val>hi) → val is in range
        # val > hi: replace with hi
        Instruction(OP_POP),          # 12: []
        Instruction(OP_PUSH, hi),     # 13: [hi]
        Instruction(OP_HALT),         # 14
        # val in range: keep it
        Instruction(OP_HALT),         # 15
    ]
    return prog, expected


# ─── Bitwise Program Generators ──────────────────────────────────

def make_bitwise_binary(op, a, b):
    """Generic bitwise binary: PUSH a, PUSH b, OP, HALT.

    Stack after pushes: sp points to b (top), sp-1 to a.
    val_a = stack[sp] = b, val_b = stack[sp-1] = a.

    For AND/OR/XOR: result = val_a OP val_b (commutative, order doesn't matter).
    For SHL:   result = val_b << val_a = a << b  (shift count from top = b)
    For SHR_S: result = val_b >> val_a = a >> b  (arithmetic)
    For SHR_U: result = val_b >> val_a = a >> b  (logical)
    For ROTL:  result = rotl(val_b, val_a) = rotl(a, b)
    For ROTR:  result = rotr(val_b, val_a) = rotr(a, b)

    Returns (program, expected_result).
    """
    # val_a = b (top of stack), val_b = a (second on stack)
    va, vb = b, a
    BITWISE_SEMANTICS = {
        OP_AND:   lambda va, vb: _to_i32(va) & _to_i32(vb),
        OP_OR:    lambda va, vb: _to_i32(va) | _to_i32(vb),
        OP_XOR:   lambda va, vb: _to_i32(va) ^ _to_i32(vb),
        OP_SHL:   lambda va, vb: (_to_i32(vb) << (int(va) & 31)) & MASK32,
        OP_SHR_S: lambda va, vb: _shr_s(vb, va),
        OP_SHR_U: lambda va, vb: _shr_u(vb, va),
        OP_ROTL:  lambda va, vb: _rotl32(vb, va),
        OP_ROTR:  lambda va, vb: _rotr32(vb, va),
    }
    expected = BITWISE_SEMANTICS[op](va, vb)
    return [
        Instruction(OP_PUSH, a),
        Instruction(OP_PUSH, b),
        Instruction(op),
        Instruction(OP_HALT),
    ], expected


def make_popcount_loop(n):
    """Count set bits of n using AND + SHR_U loop. O(bits) steps.

    Algorithm: count = 0; while n != 0: count += (n & 1); n >>= 1.
    Uses native bitwise ops — no POPCNT instruction yet (that's Tier 1 unary, Chunk 4).

    Returns (program, expected_result).
    """
    expected = bin(n & MASK32).count('1')
    prog = [
        Instruction(OP_PUSH, n),      # 0: n
        Instruction(OP_PUSH, 0),      # 1: count = 0
        # ── Loop (addr 2) ──
        Instruction(OP_SWAP),         # 2: [count, n] → [n, count]... wait
        # Rethink: keep [n, count] on stack.
        # Start: [n, count]. We need to test if n == 0.
    ]
    # Simpler layout: [n, count] with n below count
    prog = [
        Instruction(OP_PUSH, 0),      # 0: count = 0
        Instruction(OP_PUSH, n),      # 1: n  → stack: [count, n]
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [count, n, n]
        Instruction(OP_JZ, 13),       # 3: if n == 0 → done
        # Extract LSB: n & 1
        Instruction(OP_DUP),          # 4: [count, n, n]
        Instruction(OP_PUSH, 1),      # 5: [count, n, n, 1]
        Instruction(OP_AND),          # 6: [count, n, n&1]
        # Add LSB to count
        Instruction(OP_ROT),          # 7: [n, n&1, count]
        Instruction(OP_ADD),          # 8: [n, count+lsb]
        Instruction(OP_SWAP),         # 9: [count', n]
        # Shift n right by 1
        Instruction(OP_PUSH, 1),      # 10: [count', n, 1]
        Instruction(OP_SHR_U),        # 11: [count', n>>1]
        Instruction(OP_PUSH, 1),      # 12
        Instruction(OP_JNZ, 2),       # 13: always loop
        # ── Done (addr 14) ──  Wait, JZ at addr 3 jumps to 13, but 13 is JNZ...
    ]
    # Let me fix the addresses. JZ at addr 3 should jump past the loop.
    prog = [
        Instruction(OP_PUSH, 0),      # 0: count = 0
        Instruction(OP_PUSH, n),      # 1: n  → stack: [count, n]
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [count, n, n]
        Instruction(OP_JZ, 14),       # 3: if n == 0 → done (jump to addr 14)
        # Extract LSB: n & 1
        Instruction(OP_DUP),          # 4: [count, n, n]
        Instruction(OP_PUSH, 1),      # 5: [count, n, n, 1]
        Instruction(OP_AND),          # 6: [count, n, n&1]
        # Add LSB to count
        Instruction(OP_ROT),          # 7: [n, n&1, count]
        Instruction(OP_ADD),          # 8: [n, count+lsb]
        Instruction(OP_SWAP),         # 9: [count', n]
        # Shift n right by 1
        Instruction(OP_PUSH, 1),      # 10: [count', n, 1]
        Instruction(OP_SHR_U),        # 11: [count', n>>1]
        # Loop back
        Instruction(OP_PUSH, 1),      # 12
        Instruction(OP_JNZ, 2),       # 13: always jump (1 != 0)
        # ── Done (addr 14) ──
        # Stack: [count, 0] after JZ popped the 0 copy
        Instruction(OP_POP),          # 14: drop the 0 (from JZ's pop? no...)
    ]
    # Actually: JZ pops the condition. After JZ at addr 3 jumps to 14:
    # Stack before JZ: [count, n, n]. JZ pops n (the dup'd copy). Jump taken.
    # Stack at 14: [count, n=0].
    # Need to drop the 0 and keep count.
    prog = [
        Instruction(OP_PUSH, 0),      # 0: count = 0
        Instruction(OP_PUSH, n),      # 1: n  → stack: [count, n]
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [count, n, n]
        Instruction(OP_JZ, 14),       # 3: if n == 0 → done. Pops dup'd n.
        # Stack here: [count, n] (n != 0)
        Instruction(OP_DUP),          # 4: [count, n, n]
        Instruction(OP_PUSH, 1),      # 5: [count, n, n, 1]
        Instruction(OP_AND),          # 6: [count, n, n&1]
        Instruction(OP_ROT),          # 7: [n, n&1, count]
        Instruction(OP_ADD),          # 8: [n, count']
        Instruction(OP_SWAP),         # 9: [count', n]
        Instruction(OP_PUSH, 1),      # 10: [count', n, 1]
        Instruction(OP_SHR_U),        # 11: [count', n>>1]
        Instruction(OP_PUSH, 1),      # 12
        Instruction(OP_JNZ, 2),       # 13: always loop
        # ── Done (addr 14) ──
        # Stack: [count, 0]. JZ popped the dup'd copy; n=0 is still on stack.
        Instruction(OP_SWAP),         # 14: [0, count]
        Instruction(OP_POP),          # 15: drop 0  → wait, POP drops top and top becomes sp-1
    ]
    # POP decrements sp. After SWAP: [0, count]. sp points to count.
    # POP: sp-=1, top = stack[sp] = stack[sp_old - 1] = 0. That's wrong.
    # Let me just use SWAP + POP differently:
    # After JZ lands at 14: stack = [count, 0]. sp points to 0.
    # POP: sp -= 1, top = stack[sp] = count. That works!
    prog = [
        Instruction(OP_PUSH, 0),      # 0: count = 0
        Instruction(OP_PUSH, n),      # 1: n  → stack: [count, n]
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [count, n, n]
        Instruction(OP_JZ, 14),       # 3: if n == 0 → done
        Instruction(OP_DUP),          # 4: [count, n, n]
        Instruction(OP_PUSH, 1),      # 5: [count, n, n, 1]
        Instruction(OP_AND),          # 6: [count, n, n&1]
        Instruction(OP_ROT),          # 7: [n, n&1, count]
        Instruction(OP_ADD),          # 8: [n, count']
        Instruction(OP_SWAP),         # 9: [count', n]
        Instruction(OP_PUSH, 1),      # 10: [count', n, 1]
        Instruction(OP_SHR_U),        # 11: [count', n>>1]
        Instruction(OP_PUSH, 1),      # 12
        Instruction(OP_JNZ, 2),       # 13: always jump
        # ── Done (addr 14) ──
        Instruction(OP_POP),          # 14: drop n=0, top = count
        Instruction(OP_HALT),         # 15
    ]
    return prog, expected


def make_bit_extract(n, bit_pos):
    """Extract bit at position bit_pos from n. Result: 0 or 1.

    Algorithm: (n >> bit_pos) & 1

    Returns (program, expected_result).
    """
    expected = (_to_i32(n) >> (bit_pos & 31)) & 1
    prog = [
        Instruction(OP_PUSH, n),         # 0: n
        Instruction(OP_PUSH, bit_pos),   # 1: bit_pos
        Instruction(OP_SHR_U),           # 2: n >> bit_pos
        Instruction(OP_PUSH, 1),         # 3: 1
        Instruction(OP_AND),             # 4: (n >> bit_pos) & 1
        Instruction(OP_HALT),            # 5
    ]
    return prog, expected


# ─── Chunk 4: Unary + Parametric Program Generators ──────────────

def make_native_clz(n):
    """Count leading zeros of n using native CLZ. 3 instructions.

    Contrast with loop-based approach: O(32) steps.
    Returns (program, expected_result).
    """
    return [
        Instruction(OP_PUSH, n),
        Instruction(OP_CLZ),
        Instruction(OP_HALT),
    ], _clz32(n)

def make_native_ctz(n):
    """Count trailing zeros of n using native CTZ. 3 instructions."""
    return [
        Instruction(OP_PUSH, n),
        Instruction(OP_CTZ),
        Instruction(OP_HALT),
    ], _ctz32(n)

def make_native_popcnt(n):
    """Population count of n using native POPCNT. 3 instructions.

    Contrast with make_popcount_loop: O(bits) steps using AND + SHR_U loop.
    Returns (program, expected_result).
    """
    return [
        Instruction(OP_PUSH, n),
        Instruction(OP_POPCNT),
        Instruction(OP_HALT),
    ], _popcnt32(n)

def make_native_abs(n):
    """Absolute value using native ABS. 3 instructions.

    Contrast with make_native_abs (comparison-based, Chunk 2): ~8 instructions.
    Returns (program, expected_result).
    """
    return [
        Instruction(OP_PUSH, n),
        Instruction(OP_ABS),
        Instruction(OP_HALT),
    ], abs(int(n))

def make_native_neg(n):
    """Negate n using native NEG. Result is i32-masked (WASM overflow semantics)."""
    return [
        Instruction(OP_PUSH, n),
        Instruction(OP_NEG),
        Instruction(OP_HALT),
    ], (-int(n)) & 0xFFFFFFFF

def make_select(a, b, c):
    """SELECT: push a, b, c; SELECT pops all three → (c≠0 ? a : b).

    Stack before SELECT: [a, b, c] with c on top.
    Returns (program, expected_result).
    """
    expected = a if c != 0 else b
    return [
        Instruction(OP_PUSH, a),   # sp-2: a (true value)
        Instruction(OP_PUSH, b),   # sp-1: b (false value)
        Instruction(OP_PUSH, c),   # sp:   c (condition)
        Instruction(OP_SELECT),
        Instruction(OP_HALT),
    ], expected

def make_select_max(a, b):
    """Max of two numbers using GT_S + SELECT. 7 instructions.

    Stack: PUSH a, PUSH b, PUSH a, PUSH b, GT_S, SELECT, HALT.
    GT_S produces 1 if a > b (since b is TOS, a is second: checks a > b).
    SELECT: condition=GT result, false=b, true=a.

    Actually stack layout for SELECT is [true_val, false_val, cond]:
      PUSH a → [a]
      PUSH b → [a, b]
      PUSH a → [a, b, a]
      PUSH b → [a, b, a, b]
      GT_S   → [a, b, (a>b)?1:0]   (pops a,b from top, pushes comparison)
      SELECT → [max(a,b)]           (pops condition, false, true)
    """
    expected = max(a, b)
    prog = [
        Instruction(OP_PUSH, a),   # 0: [a]
        Instruction(OP_PUSH, b),   # 1: [a, b]
        Instruction(OP_PUSH, a),   # 2: [a, b, a]
        Instruction(OP_PUSH, b),   # 3: [a, b, a, b]
        Instruction(OP_GT_S),      # 4: [a, b, (a>b)]
        Instruction(OP_SELECT),    # 5: [result]
        Instruction(OP_HALT),      # 6
    ]
    return prog, expected

def make_log2_floor(n):
    """Floor of log2(n) using CLZ: 31 - CLZ(n). 5 instructions.

    Only valid for n > 0.
    Returns (program, expected_result).
    """
    if n <= 0:
        return [Instruction(OP_PUSH, 0), Instruction(OP_HALT)], 0
    expected = 31 - _clz32(n)
    prog = [
        Instruction(OP_PUSH, n),
        Instruction(OP_CLZ),
        Instruction(OP_PUSH, 31),
        Instruction(OP_SWAP),
        Instruction(OP_SUB),       # 31 - clz(n)
        Instruction(OP_HALT),
    ]
    return prog, expected

def make_is_power_of_2(n):
    """Check if n is a power of 2 using POPCNT. Result: 1 or 0.

    A positive integer is a power of 2 iff it has exactly one set bit.
    Returns (program, expected_result).
    """
    expected = 1 if (n > 0 and _popcnt32(n) == 1) else 0
    prog = [
        Instruction(OP_PUSH, n),
        Instruction(OP_POPCNT),     # number of set bits
        Instruction(OP_PUSH, 1),
        Instruction(OP_EQ),         # popcnt == 1 ?
        Instruction(OP_HALT),
    ]
    return prog, expected


# ─── Test Suite ───────────────────────────────────────────────────

def test_trap_algorithm(name, prog, np_exec, pt_exec, verbose=False):
    """Run a program expected to TRAP on both executors. Returns True if both trap."""
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)

    np_trapped = np_trace.steps and np_trace.steps[-1].op == OP_TRAP
    pt_trapped = pt_trace.steps and pt_trace.steps[-1].op == OP_TRAP
    match, detail = compare_traces(np_trace, pt_trace)
    all_ok = np_trapped and pt_trapped and match

    status = "PASS" if all_ok else "FAIL"
    np_label = f"TRAP@{len(np_trace.steps)}" if np_trapped else f"top={np_trace.steps[-1].top if np_trace.steps else '?'}"
    pt_label = f"TRAP@{len(pt_trace.steps)}" if pt_trapped else f"top={pt_trace.steps[-1].top if pt_trace.steps else '?'}"
    print(f"  {status}  {name:30s}  numpy={np_label:>10}  torch={pt_label:>10}  "
          f"trace_match={'Y' if match else 'N'}")

    if not all_ok and verbose:
        if not np_trapped:
            print(f"         NumPy did not trap")
        if not pt_trapped:
            print(f"         PyTorch did not trap")
        if not match:
            print(f"         Trace mismatch: {detail}")

    return all_ok


def test_arithmetic_unit():
    """Unit tests for MUL, DIV_S, DIV_U, REM_S, REM_U."""
    print("=" * 60)
    print("Test 1: Arithmetic Unit Tests")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    tests = [
        # (name, program, expected_top)
        ("mul_basic",
         [Instruction(OP_PUSH, 7), Instruction(OP_PUSH, 8),
          Instruction(OP_MUL), Instruction(OP_HALT)], 56),
        ("mul_zero",
         [Instruction(OP_PUSH, 0), Instruction(OP_PUSH, 5),
          Instruction(OP_MUL), Instruction(OP_HALT)], 0),
        ("mul_one",
         [Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 42),
          Instruction(OP_MUL), Instruction(OP_HALT)], 42),
        ("mul_large",
         [Instruction(OP_PUSH, 100), Instruction(OP_PUSH, 200),
          Instruction(OP_MUL), Instruction(OP_HALT)], 20000),

        ("div_s_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_DIV_S), Instruction(OP_HALT)], 3),  # 10/3 = 3
        ("div_s_exact",
         [Instruction(OP_PUSH, 12), Instruction(OP_PUSH, 4),
          Instruction(OP_DIV_S), Instruction(OP_HALT)], 3),  # 12/4 = 3
        ("div_s_one",
         [Instruction(OP_PUSH, 7), Instruction(OP_PUSH, 1),
          Instruction(OP_DIV_S), Instruction(OP_HALT)], 7),  # 7/1 = 7
        ("div_s_self",
         [Instruction(OP_PUSH, 5), Instruction(OP_PUSH, 5),
          Instruction(OP_DIV_S), Instruction(OP_HALT)], 1),  # 5/5 = 1

        ("div_u_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_DIV_U), Instruction(OP_HALT)], 3),

        ("rem_s_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_REM_S), Instruction(OP_HALT)], 1),  # 10%3 = 1
        ("rem_s_exact",
         [Instruction(OP_PUSH, 12), Instruction(OP_PUSH, 4),
          Instruction(OP_REM_S), Instruction(OP_HALT)], 0),  # 12%4 = 0
        ("rem_s_less_than",
         [Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 7),
          Instruction(OP_REM_S), Instruction(OP_HALT)], 3),  # 3%7 = 3

        ("rem_u_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_REM_U), Instruction(OP_HALT)], 1),
    ]

    passed = 0
    total = len(tests) * 2

    for name, prog, expected in tests:
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  {name:20s}  expected={expected:>8}  got={top}")

    # Verify traces match
    trace_match = 0
    for name, prog, _ in tests:
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, _ = compare_traces(np_trace, pt_trace)
        if match:
            trace_match += 1

    print(f"\n  Unit tests: {passed}/{total} passed")
    print(f"  Trace match: {trace_match}/{len(tests)} numpy==pytorch")
    return passed == total and trace_match == len(tests)


def test_division_by_zero():
    """Test that division/remainder by zero produces TRAP."""
    print("\n" + "=" * 60)
    print("Test 2: Division by Zero → TRAP")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    trap_tests = [
        ("div_s_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_DIV_S), Instruction(OP_HALT)]),
        ("div_u_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_DIV_U), Instruction(OP_HALT)]),
        ("rem_s_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_REM_S), Instruction(OP_HALT)]),
        ("rem_u_by_zero",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 0),
          Instruction(OP_REM_U), Instruction(OP_HALT)]),
        ("div_zero_by_zero",
         [Instruction(OP_PUSH, 0), Instruction(OP_PUSH, 0),
          Instruction(OP_DIV_S), Instruction(OP_HALT)]),
    ]

    passed = 0
    for name, prog in trap_tests:
        if test_trap_algorithm(name, prog, np_exec, pt_exec, verbose=True):
            passed += 1

    print(f"\n  Result: {passed}/{len(trap_tests)} passed")
    return passed == len(trap_tests)


def test_native_multiply():
    """Test native MUL multiply vs Phase 13's repeated addition."""
    print("\n" + "=" * 60)
    print("Test 3: Native Multiply (MUL)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    cases = [(0, 5), (5, 0), (1, 7), (3, 4), (7, 8), (12, 10), (100, 200)]
    passed = 0
    for a, b in cases:
        prog, expected = make_native_multiply(a, b)
        ok, steps = test_algorithm(f"mul({a},{b})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1
            # Report step count improvement
            if a > 0 and b > 0:
                print(f"         Native: {steps} steps (vs ~{2*min(a,b)*6 + 8} with repeated addition)")

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_native_division():
    """Test DIV_S and REM_S."""
    print("\n" + "=" * 60)
    print("Test 4: Native Division & Remainder")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    div_cases = [(3, 10), (4, 12), (1, 7), (5, 5), (7, 3), (100, 1000)]
    passed = 0

    print("  --- DIV_S ---")
    for a, b in div_cases:
        prog, expected = make_native_divmod(a, b)
        ok, _ = test_algorithm(f"div({b}/{a})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print("\n  --- REM_S ---")
    rem_cases = [(3, 10), (4, 12), (7, 3), (2, 15), (5, 5)]
    for a, b in rem_cases:
        prog, expected = make_native_remainder(a, b)
        ok, _ = test_algorithm(f"rem({b}%{a})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    total = len(div_cases) + len(rem_cases)
    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def test_native_is_even():
    """Test parity using REM_S — O(1) vs Phase 13's O(n)."""
    print("\n" + "=" * 60)
    print("Test 5: Native Parity (REM_S + JZ)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    cases = [0, 1, 2, 3, 4, 7, 10, 15, 20, 100]
    passed = 0
    for n in cases:
        prog, expected = make_native_is_even(n)
        label = "even" if expected else "odd"
        ok, steps = test_algorithm(f"is_even({n})→{label}", prog, expected, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_factorial():
    """Test factorial using native MUL."""
    print("\n" + "=" * 60)
    print("Test 6: Factorial (native MUL)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    cases = [0, 1, 2, 3, 4, 5, 7, 10]
    passed = 0
    for n in cases:
        prog, expected = make_factorial(n)
        ok, steps = test_algorithm(f"fact({n})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_gcd():
    """Test GCD using Euclidean algorithm with native REM_S."""
    print("\n" + "=" * 60)
    print("Test 7: GCD (Euclidean algorithm with REM_S)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    cases = [(12, 8), (100, 75), (7, 3), (15, 5), (17, 13), (48, 36), (1, 100)]
    passed = 0
    for a, b in cases:
        prog, expected = make_gcd(a, b)
        ok, steps = test_algorithm(f"gcd({a},{b})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_regression():
    """Verify all Phase 13 algorithms still work."""
    print("\n" + "=" * 60)
    print("Test 8: Regression (Phase 4 + Phase 11 + Phase 13)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    passed = 0
    total = 0

    # Phase 4 tests
    for name, test_fn in ALL_TESTS:
        prog, expected_top = test_fn()
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
        match, _ = compare_traces(np_trace, pt_trace)
        ok = (np_top == expected_top and pt_top == expected_top and match)
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:20s}  expected={expected_top:>5}  "
              f"numpy={np_top}  torch={pt_top}  match={'Y' if match else 'N'}")

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
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
        match, _ = compare_traces(np_trace, pt_trace)
        ok = (np_top == expected and pt_top == expected and match)
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:20s}  expected={expected:>5}  "
              f"numpy={np_top}  torch={pt_top}  match={'Y' if match else 'N'}")

    # Phase 13 algorithm suite
    p13_algos = [
        ("fib(10)", *make_fibonacci(10)),
        ("fib(7)", *make_fibonacci(7)),
        ("sum(1..10)", *make_sum_1_to_n(10)),
        ("power(2^5)", *make_power_of_2(5)),
    ]
    for name, prog, expected in p13_algos:
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
        match, _ = compare_traces(np_trace, pt_trace)
        ok = (np_top == expected and pt_top == expected and match)
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:20s}  expected={expected:>5}  "
              f"numpy={np_top}  torch={pt_top}  match={'Y' if match else 'N'}")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def test_model_summary():
    """Report model architecture."""
    print("\n" + "=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)

    model = Phase14Model()
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())

    print(f"  d_model:          {D_MODEL}")
    print(f"  n_active_heads:   5 (unchanged from Phase 13)")
    print(f"  ISA opcodes:      {N_OPCODES} (was {N_OPCODES_P13} in Phase 13)")
    print(f"  M_top shape:      {tuple(model.M_top.shape)}")
    print(f"  sp_deltas shape:  {tuple(model.sp_deltas.shape)}")
    print(f"  linear ops:       12 (PUSH through ROT)")
    print(f"  nonlinear ops:    30 (5 arith + 11 cmp + 8 bitwise + 5 unary + 1 parametric)")
    print(f"  trainable params: {total_params}")
    print(f"  buffer params:    {total_buffers}")
    print(f"  total compiled:   {total_params + total_buffers}")

    return True


def test_step_count_comparison():
    """Compare step counts: native ops vs Phase 13 repeated-op algorithms."""
    print("\n" + "=" * 60)
    print("Step Count Comparison: Native vs Repeated-Op")
    print("=" * 60)

    np_exec = Phase14Executor()

    comparisons = [
        ("multiply(7, 100)", make_native_multiply(7, 100)),
        ("multiply(12, 10)", make_native_multiply(12, 10)),
        ("is_even(100)",     make_native_is_even(100)),
        ("is_even(15)",      make_native_is_even(15)),
    ]

    # Phase 13 equivalents (import the old generators)
    from phase13_isa_completeness import make_multiply as make_slow_multiply
    from phase13_isa_completeness import make_is_even as make_slow_is_even

    slow_comparisons = [
        ("multiply(7, 100)", make_slow_multiply(7, 100)),
        ("multiply(12, 10)", make_slow_multiply(12, 10)),
        ("is_even(100)",     make_slow_is_even(100)),
        ("is_even(15)",      make_slow_is_even(15)),
    ]

    for (name, (fast_prog, _)), (_, (slow_prog, _)) in zip(comparisons, slow_comparisons):
        fast_trace = np_exec.execute(fast_prog)
        slow_trace = np_exec.execute(slow_prog)
        fast_steps = len(fast_trace.steps)
        slow_steps = len(slow_trace.steps)
        speedup = slow_steps / fast_steps if fast_steps > 0 else float('inf')
        print(f"  {name:25s}  native={fast_steps:>4} steps  "
              f"repeated={slow_steps:>4} steps  {speedup:.0f}× fewer steps")

    return True  # informational only


def test_comparison_unit():
    """Unit tests for all 11 comparison opcodes."""
    print("\n" + "=" * 60)
    print("Test 9: Comparison Unit Tests")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    # EQZ tests (unary)
    eqz_cases = [
        ("eqz(0)→1",  0, 1),
        ("eqz(1)→0",  1, 0),
        ("eqz(5)→0",  5, 0),
        ("eqz(-1)→0", -1, 0),
    ]

    passed = 0
    total = 0

    print("  --- EQZ (unary) ---")
    for name, val, expected in eqz_cases:
        prog, exp = make_compare_eqz(val)
        assert exp == expected, f"Generator bug: {name}"
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  {name:20s}  expected={expected}  got={top}")

    # Binary comparison tests
    binary_ops = [
        (OP_EQ,   "EQ"),
        (OP_NE,   "NE"),
        (OP_LT_S, "LT_S"),
        (OP_LT_U, "LT_U"),
        (OP_GT_S, "GT_S"),
        (OP_GT_U, "GT_U"),
        (OP_LE_S, "LE_S"),
        (OP_LE_U, "LE_U"),
        (OP_GE_S, "GE_S"),
        (OP_GE_U, "GE_U"),
    ]

    # Test pairs: (a, b) where a is pushed first, b second
    test_pairs = [
        (5, 5),    # equal
        (3, 7),    # a < b
        (10, 2),   # a > b
        (0, 0),    # both zero
        (0, 1),    # zero vs positive
    ]

    for op, op_name in binary_ops:
        print(f"\n  --- {op_name} ---")
        for a, b in test_pairs:
            prog, expected = make_compare_binary(op, a, b)
            name = f"{op_name}({a},{b})→{expected}"
            for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
                trace = executor.execute(prog)
                top = trace.steps[-1].top if trace.steps else None
                ok = (top == expected)
                if ok: passed += 1
                total += 1
                status = "PASS" if ok else "FAIL"
                print(f"  {status}  {label:5s}  {name:25s}  got={top}")

    # Verify trace match across all tests
    trace_pass = 0
    trace_total = 0
    for _, val, _ in eqz_cases:
        prog, _ = make_compare_eqz(val)
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, _ = compare_traces(np_trace, pt_trace)
        if match: trace_pass += 1
        trace_total += 1
    for op, _ in binary_ops:
        for a, b in test_pairs:
            prog, _ = make_compare_binary(op, a, b)
            np_trace = np_exec.execute(prog)
            pt_trace = pt_exec.execute(prog)
            match, _ = compare_traces(np_trace, pt_trace)
            if match: trace_pass += 1
            trace_total += 1

    print(f"\n  Unit tests: {passed}/{total} passed")
    print(f"  Trace match: {trace_pass}/{trace_total} numpy==pytorch")
    return passed == total and trace_pass == trace_total


def test_comparison_algorithms():
    """Test programs using comparisons: max, abs, clamp."""
    print("\n" + "=" * 60)
    print("Test 10: Comparison Algorithms (max, abs, clamp)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    passed = 0
    total = 0

    # max(a, b)
    print("  --- max ---")
    max_cases = [(3, 7), (10, 2), (5, 5), (0, 1), (100, 99)]
    for a, b in max_cases:
        prog, expected = make_native_max(a, b)
        ok, steps = test_algorithm(f"max({a},{b})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1
        total += 1

    # abs(n) — only test non-negative for now since our ISA uses positive ints primarily
    print("\n  --- abs ---")
    abs_cases = [0, 1, 5, 42, 100]
    for n in abs_cases:
        prog, expected = make_native_abs(n)
        ok, steps = test_algorithm(f"abs({n})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1
        total += 1

    # clamp(val, lo, hi)
    print("\n  --- clamp ---")
    clamp_cases = [
        (5, 0, 10),    # in range
        (15, 0, 10),   # above
        (0, 5, 10),    # below (note: 0 < 5 so clamped to 5)
        (3, 3, 7),     # at lower bound
        (7, 3, 7),     # at upper bound
        (50, 10, 20),  # way above
    ]
    for val, lo, hi in clamp_cases:
        prog, expected = make_native_clamp(val, lo, hi)
        ok, steps = test_algorithm(f"clamp({val},{lo},{hi})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1
        total += 1

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def test_bitwise_unit():
    """Unit tests for all 8 bitwise opcodes."""
    print("\n" + "=" * 60)
    print("Test 11: Bitwise Unit Tests")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    passed = 0
    total = 0

    # AND tests
    and_cases = [
        (0xFF, 0x0F, 0x0F),
        (0xFF, 0xFF, 0xFF),
        (0xFF, 0x00, 0x00),
        (0xAA, 0x55, 0x00),
        (12, 10, 8),       # 1100 & 1010 = 1000
    ]
    print("  --- AND ---")
    for a, b, expected in and_cases:
        prog, exp = make_bitwise_binary(OP_AND, a, b)
        assert exp == expected, f"Generator bug: AND({a},{b}) exp={expected} got={exp}"
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  AND({a:#x},{b:#x})→{expected:#x}  got={top}")

    # OR tests
    or_cases = [
        (0xF0, 0x0F, 0xFF),
        (0x00, 0x00, 0x00),
        (0xAA, 0x55, 0xFF),
        (12, 10, 14),       # 1100 | 1010 = 1110
    ]
    print("\n  --- OR ---")
    for a, b, expected in or_cases:
        prog, exp = make_bitwise_binary(OP_OR, a, b)
        assert exp == expected
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  OR({a:#x},{b:#x})→{expected:#x}  got={top}")

    # XOR tests
    xor_cases = [
        (0xFF, 0xFF, 0x00),
        (0xFF, 0x00, 0xFF),
        (0xAA, 0x55, 0xFF),
        (12, 10, 6),       # 1100 ^ 1010 = 0110
    ]
    print("\n  --- XOR ---")
    for a, b, expected in xor_cases:
        prog, exp = make_bitwise_binary(OP_XOR, a, b)
        assert exp == expected
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  XOR({a:#x},{b:#x})→{expected:#x}  got={top}")

    # SHL tests (PUSH a, PUSH b → b is shift count, a is value to shift)
    # make_bitwise_binary(OP_SHL, a, b): val_a=b, val_b=a → result = a << b
    shl_cases = [
        (1, 0, 1),         # 1 << 0 = 1
        (1, 1, 2),         # 1 << 1 = 2
        (1, 4, 16),        # 1 << 4 = 16
        (0xFF, 4, 0xFF0),  # 255 << 4 = 4080
        (1, 31, 0x80000000),  # 1 << 31
    ]
    print("\n  --- SHL ---")
    for a, b, expected in shl_cases:
        prog, exp = make_bitwise_binary(OP_SHL, a, b)
        assert exp == expected, f"Generator bug: SHL({a},{b}) exp={expected} got={exp}"
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  SHL({a},{b})→{expected}  got={top}")

    # SHR_U tests (logical shift right)
    # make_bitwise_binary(OP_SHR_U, a, b): val_a=b, val_b=a → result = a >> b
    shr_u_cases = [
        (16, 1, 8),        # 16 >> 1 = 8
        (255, 4, 15),      # 255 >> 4 = 15
        (0x80000000, 31, 1),  # high bit >> 31 = 1 (logical, no sign extension)
        (1, 0, 1),         # n >> 0 = n
    ]
    print("\n  --- SHR_U ---")
    for a, b, expected in shr_u_cases:
        prog, exp = make_bitwise_binary(OP_SHR_U, a, b)
        assert exp == expected, f"Generator bug: SHR_U({a},{b}) exp={expected} got={exp}"
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  SHR_U({a},{b})→{expected}  got={top}")

    # SHR_S tests (arithmetic shift right)
    # For positive values, same as SHR_U
    shr_s_cases = [
        (16, 1, 8),
        (255, 4, 15),
    ]
    print("\n  --- SHR_S (positive values) ---")
    for a, b, expected in shr_s_cases:
        prog, exp = make_bitwise_binary(OP_SHR_S, a, b)
        assert exp == expected, f"Generator bug: SHR_S({a},{b}) exp={expected} got={exp}"
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  SHR_S({a},{b})→{expected}  got={top}")

    # ROTL tests
    # make_bitwise_binary(OP_ROTL, a, b): val_a=b, val_b=a → rotl(a, b)
    rotl_cases = [
        (1, 1, 2),                # rotl(1, 1) = 2
        (0x80000000, 1, 1),       # rotl(high_bit, 1) = 1 (wraps around)
        (0xFF, 8, 0xFF00),        # rotl(0xFF, 8) = 0xFF00
    ]
    print("\n  --- ROTL ---")
    for a, b, expected in rotl_cases:
        prog, exp = make_bitwise_binary(OP_ROTL, a, b)
        assert exp == expected, f"Generator bug: ROTL({a},{b}) exp={expected} got={exp}"
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  ROTL({a:#x},{b})→{expected:#x}  got={top}")

    # ROTR tests
    # make_bitwise_binary(OP_ROTR, a, b): val_a=b, val_b=a → rotr(a, b)
    rotr_cases = [
        (2, 1, 1),                # rotr(2, 1) = 1
        (1, 1, 0x80000000),       # rotr(1, 1) = high bit
        (0xFF00, 8, 0xFF),        # rotr(0xFF00, 8) = 0xFF
    ]
    print("\n  --- ROTR ---")
    for a, b, expected in rotr_cases:
        prog, exp = make_bitwise_binary(OP_ROTR, a, b)
        assert exp == expected, f"Generator bug: ROTR({a},{b}) exp={expected} got={exp}"
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok: passed += 1
            total += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  ROTR({a:#x},{b})→{expected:#x}  got={top}")

    # Verify trace match across all bitwise tests
    all_cases = (
        [(OP_AND, a, b) for a, b, _ in and_cases] +
        [(OP_OR, a, b) for a, b, _ in or_cases] +
        [(OP_XOR, a, b) for a, b, _ in xor_cases] +
        [(OP_SHL, a, b) for a, b, _ in shl_cases] +
        [(OP_SHR_U, a, b) for a, b, _ in shr_u_cases] +
        [(OP_SHR_S, a, b) for a, b, _ in shr_s_cases] +
        [(OP_ROTL, a, b) for a, b, _ in rotl_cases] +
        [(OP_ROTR, a, b) for a, b, _ in rotr_cases]
    )
    trace_pass = 0
    trace_total = len(all_cases)
    for op, a, b in all_cases:
        prog, _ = make_bitwise_binary(op, a, b)
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, _ = compare_traces(np_trace, pt_trace)
        if match: trace_pass += 1

    print(f"\n  Unit tests: {passed}/{total} passed")
    print(f"  Trace match: {trace_pass}/{trace_total} numpy==pytorch")
    return passed == total and trace_pass == trace_total


def test_bitwise_algorithms():
    """Test programs combining bitwise ops: popcount, bit extract."""
    print("\n" + "=" * 60)
    print("Test 12: Bitwise Algorithms (popcount, bit extract)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    passed = 0
    total = 0

    # Popcount via loop
    print("  --- popcount (AND + SHR_U loop) ---")
    popcount_cases = [0, 1, 3, 7, 15, 255, 0xFFFF]
    for n in popcount_cases:
        prog, expected = make_popcount_loop(n)
        ok, steps = test_algorithm(f"popcount({n})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok:
            passed += 1
            print(f"         {steps} steps for {bin(n & MASK32).count('1')} set bits")
        total += 1

    # Bit extraction
    print("\n  --- bit extract (SHR_U + AND) ---")
    bit_cases = [
        (0xFF, 0), (0xFF, 4), (0xFF, 7), (0xFF, 8),
        (0x80000000, 31), (0x80000000, 30),
        (42, 0), (42, 1), (42, 3), (42, 5),
    ]
    for n, bit in bit_cases:
        prog, expected = make_bit_extract(n, bit)
        ok, steps = test_algorithm(f"bit({n:#x}[{bit}])", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1
        total += 1

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


# ─── Chunk 4 Tests ────────────────────────────────────────────────

def test_unary_unit():
    """Unit tests for CLZ, CTZ, POPCNT, ABS, NEG."""
    print("\n" + "=" * 60)
    print("Test: Unary ops unit (CLZ, CTZ, POPCNT, ABS, NEG)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    tests = [
        # CLZ: count leading zeros (32-bit)
        ("clz_zero",
         [Instruction(OP_PUSH, 0), Instruction(OP_CLZ), Instruction(OP_HALT)], 32),
        ("clz_one",
         [Instruction(OP_PUSH, 1), Instruction(OP_CLZ), Instruction(OP_HALT)], 31),
        ("clz_two",
         [Instruction(OP_PUSH, 2), Instruction(OP_CLZ), Instruction(OP_HALT)], 30),
        ("clz_128",
         [Instruction(OP_PUSH, 128), Instruction(OP_CLZ), Instruction(OP_HALT)], 24),
        ("clz_255",
         [Instruction(OP_PUSH, 255), Instruction(OP_CLZ), Instruction(OP_HALT)], 24),
        ("clz_65536",
         [Instruction(OP_PUSH, 65536), Instruction(OP_CLZ), Instruction(OP_HALT)], 15),

        # CTZ: count trailing zeros (32-bit)
        ("ctz_zero",
         [Instruction(OP_PUSH, 0), Instruction(OP_CTZ), Instruction(OP_HALT)], 32),
        ("ctz_one",
         [Instruction(OP_PUSH, 1), Instruction(OP_CTZ), Instruction(OP_HALT)], 0),
        ("ctz_two",
         [Instruction(OP_PUSH, 2), Instruction(OP_CTZ), Instruction(OP_HALT)], 1),
        ("ctz_eight",
         [Instruction(OP_PUSH, 8), Instruction(OP_CTZ), Instruction(OP_HALT)], 3),
        ("ctz_1024",
         [Instruction(OP_PUSH, 1024), Instruction(OP_CTZ), Instruction(OP_HALT)], 10),
        ("ctz_12",
         [Instruction(OP_PUSH, 12), Instruction(OP_CTZ), Instruction(OP_HALT)], 2),  # 0b1100

        # POPCNT: population count (32-bit)
        ("popcnt_zero",
         [Instruction(OP_PUSH, 0), Instruction(OP_POPCNT), Instruction(OP_HALT)], 0),
        ("popcnt_one",
         [Instruction(OP_PUSH, 1), Instruction(OP_POPCNT), Instruction(OP_HALT)], 1),
        ("popcnt_255",
         [Instruction(OP_PUSH, 255), Instruction(OP_POPCNT), Instruction(OP_HALT)], 8),
        ("popcnt_7",
         [Instruction(OP_PUSH, 7), Instruction(OP_POPCNT), Instruction(OP_HALT)], 3),
        ("popcnt_1023",
         [Instruction(OP_PUSH, 1023), Instruction(OP_POPCNT), Instruction(OP_HALT)], 10),

        # ABS: absolute value
        ("abs_positive",
         [Instruction(OP_PUSH, 42), Instruction(OP_ABS), Instruction(OP_HALT)], 42),
        ("abs_zero",
         [Instruction(OP_PUSH, 0), Instruction(OP_ABS), Instruction(OP_HALT)], 0),
        ("abs_negative",
         [Instruction(OP_PUSH, -7), Instruction(OP_ABS), Instruction(OP_HALT)], 7),
        ("abs_neg_one",
         [Instruction(OP_PUSH, -1), Instruction(OP_ABS), Instruction(OP_HALT)], 1),

        # NEG: negate (i32-masked per WASM overflow semantics)
        ("neg_positive",
         [Instruction(OP_PUSH, 5), Instruction(OP_NEG), Instruction(OP_HALT)], (-5) & 0xFFFFFFFF),
        ("neg_zero",
         [Instruction(OP_PUSH, 0), Instruction(OP_NEG), Instruction(OP_HALT)], 0),
        ("neg_negative",
         [Instruction(OP_PUSH, -3), Instruction(OP_NEG), Instruction(OP_HALT)], 3),
        ("neg_large",
         [Instruction(OP_PUSH, 1000), Instruction(OP_NEG), Instruction(OP_HALT)], (-1000) & 0xFFFFFFFF),
    ]

    passed = 0
    total = len(tests) * 2

    for name, prog, expected in tests:
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  {name:20s}  expected={expected:>8}  got={top}")

    # Verify traces match
    trace_match = 0
    for name, prog, _ in tests:
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, _ = compare_traces(np_trace, pt_trace)
        if match:
            trace_match += 1

    print(f"\n  Unit tests: {passed}/{total} passed")
    print(f"  Trace match: {trace_match}/{len(tests)} numpy==pytorch")
    return passed == total and trace_match == len(tests)


def test_select_unit():
    """Unit tests for SELECT parametric op."""
    print("\n" + "=" * 60)
    print("Test: SELECT parametric op")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    tests = [
        # SELECT: a b c → (c≠0 ? a : b)
        # Stack: PUSH a (true), PUSH b (false), PUSH c (cond)
        ("select_true",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 20),
          Instruction(OP_PUSH, 1), Instruction(OP_SELECT),
          Instruction(OP_HALT)], 10),
        ("select_false",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 20),
          Instruction(OP_PUSH, 0), Instruction(OP_SELECT),
          Instruction(OP_HALT)], 20),
        ("select_nonzero_cond",
         [Instruction(OP_PUSH, 100), Instruction(OP_PUSH, 200),
          Instruction(OP_PUSH, 42), Instruction(OP_SELECT),
          Instruction(OP_HALT)], 100),
        ("select_neg_cond",
         [Instruction(OP_PUSH, 5), Instruction(OP_PUSH, 9),
          Instruction(OP_PUSH, -1), Instruction(OP_SELECT),
          Instruction(OP_HALT)], 5),
        ("select_same_values",
         [Instruction(OP_PUSH, 7), Instruction(OP_PUSH, 7),
          Instruction(OP_PUSH, 0), Instruction(OP_SELECT),
          Instruction(OP_HALT)], 7),
    ]

    passed = 0
    total = len(tests) * 2

    for name, prog, expected in tests:
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  {name:20s}  expected={expected:>8}  got={top}")

    trace_match = 0
    for name, prog, _ in tests:
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, _ = compare_traces(np_trace, pt_trace)
        if match:
            trace_match += 1

    print(f"\n  Unit tests: {passed}/{total} passed")
    print(f"  Trace match: {trace_match}/{len(tests)} numpy==pytorch")
    return passed == total and trace_match == len(tests)


def test_unary_algorithms():
    """Test algorithm generators using Chunk 4 ops."""
    print("\n" + "=" * 60)
    print("Test: Unary + parametric algorithm programs")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()

    algos = [
        # CLZ generators
        ("clz(0)=32", *make_native_clz(0)),
        ("clz(1)=31", *make_native_clz(1)),
        ("clz(255)=24", *make_native_clz(255)),
        ("clz(65536)=15", *make_native_clz(65536)),

        # CTZ generators
        ("ctz(0)=32", *make_native_ctz(0)),
        ("ctz(1)=0", *make_native_ctz(1)),
        ("ctz(8)=3", *make_native_ctz(8)),
        ("ctz(1024)=10", *make_native_ctz(1024)),

        # POPCNT generators
        ("popcnt(0)=0", *make_native_popcnt(0)),
        ("popcnt(255)=8", *make_native_popcnt(255)),
        ("popcnt(7)=3", *make_native_popcnt(7)),

        # ABS generators
        ("abs(42)=42", *make_native_abs(42)),
        ("abs(-7)=7", *make_native_abs(-7)),
        ("abs(0)=0", *make_native_abs(0)),

        # NEG generators
        ("neg(5)=-5", *make_native_neg(5)),
        ("neg(-3)=3", *make_native_neg(-3)),
        ("neg(0)=0", *make_native_neg(0)),

        # SELECT generators
        ("select(10,20,1)=10", *make_select(10, 20, 1)),
        ("select(10,20,0)=20", *make_select(10, 20, 0)),
        ("select(5,9,-1)=5", *make_select(5, 9, -1)),

        # Composite: select_max
        ("max(10,25)=25", *make_select_max(10, 25)),
        ("max(25,10)=25", *make_select_max(25, 10)),
        ("max(7,7)=7", *make_select_max(7, 7)),

        # Composite: log2_floor
        ("log2(1)=0", *make_log2_floor(1)),
        ("log2(8)=3", *make_log2_floor(8)),
        ("log2(255)=7", *make_log2_floor(255)),
        ("log2(1024)=10", *make_log2_floor(1024)),

        # Composite: is_power_of_2
        ("ispow2(1)=1", *make_is_power_of_2(1)),
        ("ispow2(8)=1", *make_is_power_of_2(8)),
        ("ispow2(7)=0", *make_is_power_of_2(7)),
        ("ispow2(0)=0", *make_is_power_of_2(0)),
        ("ispow2(1024)=1", *make_is_power_of_2(1024)),
    ]

    passed = 0
    total = 0

    for name, prog, expected in algos:
        ok = test_algorithm(name, prog, expected, np_exec, pt_exec)
        if ok:
            passed += 1
        total += 1

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def test_step_count_chunk4():
    """Compare step counts: native POPCNT vs loop-based popcount."""
    print("\n" + "=" * 60)
    print("Test: Step count comparison (Chunk 4 ops)")
    print("=" * 60)

    np_exec = Phase14Executor()

    comparisons = [
        ("POPCNT(255)", make_native_popcnt(255), make_popcount_loop(255)),
        ("POPCNT(1023)", make_native_popcnt(1023), make_popcount_loop(1023)),
        ("POPCNT(7)", make_native_popcnt(7), make_popcount_loop(7)),
    ]

    all_faster = True
    for label, (native_prog, native_exp), (loop_prog, loop_exp) in comparisons:
        native_trace = np_exec.execute(native_prog)
        loop_trace = np_exec.execute(loop_prog)
        native_steps = len(native_trace.steps)
        loop_steps = len(loop_trace.steps)
        speedup = loop_steps / native_steps if native_steps > 0 else float('inf')

        native_top = native_trace.steps[-1].top if native_trace.steps else None
        loop_top = loop_trace.steps[-1].top if loop_trace.steps else None
        correct = (native_top == native_exp and loop_top == loop_exp)
        faster = native_steps < loop_steps

        status = "PASS" if (correct and faster) else "FAIL"
        if not (correct and faster):
            all_faster = False
        print(f"  {status}  {label:20s}  native={native_steps:>4} steps  "
              f"loop={loop_steps:>4} steps  speedup={speedup:.0f}×  "
              f"result={native_top}")

    return all_faster


def test_integration_chunk5():
    """Chunk 5 (Issue #15): Integration tests exercising new opcodes in combination.

    Five canonical programs that span the full Tier 1 ISA, run on both NumPy
    and PyTorch executors with trace-level match verification and step-count
    comparison against Phase 13 equivalents where applicable.
    """
    print("\n" + "=" * 60)
    print("Test: Chunk 5 Integration (Issue #15)")
    print("=" * 60)

    np_exec = Phase14Executor()
    pt_exec = Phase14PyTorchExecutor()
    p13_exec = Phase13Executor()

    all_pass = True
    step_comparisons = []

    # ── Program 1: Native multiply 7 * 8 = 56 ──
    print("\n  ── 1. Native multiply: 7 × 8 = 56 ──")
    prog, expected = make_native_multiply(7, 8)
    ok, steps = test_algorithm("native_mul(7,8)", prog, expected,
                               np_exec, pt_exec, verbose=True)
    if not ok: all_pass = False
    # Compare with Phase 13 repeated addition
    p13_prog, _ = make_multiply_p13(7, 8)
    p13_trace = p13_exec.execute(p13_prog)
    p13_steps = len(p13_trace.steps)
    step_comparisons.append(("multiply(7,8)", steps, p13_steps))
    print(f"         Steps: {steps} native vs {p13_steps} repeated-addition "
          f"({p13_steps // steps}× speedup)")

    # ── Program 2: Max of two numbers using GT_S + SELECT ──
    print("\n  ── 2. Max of two numbers: max(10, 25) = 25 ──")
    prog, expected = make_select_max(10, 25)
    ok, steps = test_algorithm("select_max(10,25)", prog, expected,
                               np_exec, pt_exec, verbose=True)
    if not ok: all_pass = False
    # Also test reversed and equal
    for a, b in [(25, 10), (7, 7), (0, 100), (99, 1)]:
        prog2, exp2 = make_select_max(a, b)
        ok2, _ = test_algorithm(f"select_max({a},{b})", prog2, exp2,
                                np_exec, pt_exec, verbose=True)
        if not ok2: all_pass = False

    # ── Program 3: is_even(42) → 1 ──
    print("\n  ── 3. is_even(42) → 1 ──")
    prog, expected = make_native_is_even(42)
    ok, steps = test_algorithm("is_even(42)", prog, expected,
                               np_exec, pt_exec, verbose=True)
    if not ok: all_pass = False
    # Compare with Phase 13 repeated subtraction approach
    p13_prog, _ = make_is_even_p13(42)
    p13_trace = p13_exec.execute(p13_prog)
    p13_steps = len(p13_trace.steps)
    step_comparisons.append(("is_even(42)", steps, p13_steps))
    print(f"         Steps: {steps} native vs {p13_steps} repeated-subtraction "
          f"({p13_steps // steps}× speedup)")

    # Also test odd
    prog_odd, exp_odd = make_native_is_even(43)
    ok_odd, _ = test_algorithm("is_even(43)→odd", prog_odd, exp_odd,
                               np_exec, pt_exec, verbose=True)
    if not ok_odd: all_pass = False

    # ── Program 4: Factorial(10) → 3628800 ──
    print("\n  ── 4. Factorial(10) → 3628800 ──")
    prog, expected = make_factorial(10)
    ok, steps = test_algorithm("factorial(10)", prog, expected,
                               np_exec, pt_exec, verbose=True)
    if not ok: all_pass = False
    step_comparisons.append(("factorial(10)", steps, None))
    print(f"         Steps: {steps} (no Phase 13 equivalent — MUL required)")

    # ── Program 5: Popcount(255) → 8 ──
    print("\n  ── 5. Popcount(255) → 8 ──")
    prog, expected = make_native_popcnt(255)
    ok, steps = test_algorithm("popcnt(255)", prog, expected,
                               np_exec, pt_exec, verbose=True)
    if not ok: all_pass = False
    # Compare with loop-based popcount
    loop_prog, loop_exp = make_popcount_loop(255)
    loop_trace = np_exec.execute(loop_prog)
    loop_steps = len(loop_trace.steps)
    step_comparisons.append(("popcnt(255)", steps, loop_steps))
    print(f"         Steps: {steps} native vs {loop_steps} loop-based "
          f"({loop_steps // steps}× speedup)")

    # ── Phase 13 regression check ──
    print("\n  ── Regression: Phase 13 algorithms on Phase 14 executor ──")
    regression_cases = [
        ("fib(10)",    make_fibonacci(10)),
        ("pow2(5)",    make_power_of_2(5)),
        ("sum(1..10)", make_sum_1_to_n(10)),
    ]
    for label, (rprog, rexp) in regression_cases:
        np_trace = np_exec.execute(rprog)
        pt_trace = pt_exec.execute(rprog)
        match, detail = compare_traces(np_trace, pt_trace)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
        ok = (np_top == rexp and pt_top == rexp and match)
        if not ok: all_pass = False
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label:20s}  expected={rexp:>10}  "
              f"numpy={np_top}  torch={pt_top}  trace_match={'Y' if match else 'N'}")

    # ── Step-count summary ──
    print("\n  ── Step-count summary ──")
    for label, native, old in step_comparisons:
        if old is not None:
            print(f"    {label:20s}  {native:>4} steps (was {old:>5})  "
                  f"{old // native:>4}× fewer steps")
        else:
            print(f"    {label:20s}  {native:>4} steps (new — no prior equivalent)")

    return all_pass


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 14: Extended ISA — Chunks 1-5: Arith, Cmp, Bitwise, Unary, Parametric + Integration")
    print("=" * 60)
    print(f"  Chunk 1 ops:   MUL, DIV_S, DIV_U, REM_S, REM_U")
    print(f"  Chunk 2 ops:   EQZ, EQ, NE, LT/GT/LE/GE_S/U")
    print(f"  Chunk 3 ops:   AND, OR, XOR, SHL, SHR_S, SHR_U, ROTL, ROTR")
    print(f"  Chunk 4 ops:   CLZ, CTZ, POPCNT, ABS, NEG, SELECT")
    print(f"  Chunk 5:       Integration tests (native mul, max, is_even, factorial, popcount)")
    print(f"  Trap opcode:   OP_TRAP ({OP_TRAP}) — division by zero")
    print(f"  Total ISA:     {N_OPCODES} opcodes")
    print(f"  Architecture:  Linear + nonlinear FF dispatch")
    print()

    t0 = time.time()
    results = []

    # Chunk 1 tests (arithmetic)
    results.append(("Arithmetic unit",     test_arithmetic_unit()))
    results.append(("Division by zero",    test_division_by_zero()))
    results.append(("Native multiply",     test_native_multiply()))
    results.append(("Native div/rem",      test_native_division()))
    results.append(("Native is_even",      test_native_is_even()))
    results.append(("Factorial",           test_factorial()))
    results.append(("GCD",                 test_gcd()))

    # Chunk 2 tests (comparisons)
    results.append(("Comparison unit",     test_comparison_unit()))
    results.append(("Comparison algos",    test_comparison_algorithms()))

    # Chunk 3 tests (bitwise)
    results.append(("Bitwise unit",        test_bitwise_unit()))
    results.append(("Bitwise algos",       test_bitwise_algorithms()))

    # Chunk 4 tests (unary + parametric)
    results.append(("Unary unit",          test_unary_unit()))
    results.append(("SELECT unit",         test_select_unit()))
    results.append(("Unary+param algos",   test_unary_algorithms()))
    results.append(("Step count (Chunk4)", test_step_count_chunk4()))

    # Chunk 5 tests (integration — Issue #15)
    results.append(("Integration (Chunk5)", test_integration_chunk5()))

    # Shared tests
    results.append(("Regression",          test_regression()))
    results.append(("Model summary",       test_model_summary()))
    results.append(("Step comparison",     test_step_count_comparison()))

    elapsed = time.time() - t0

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  {status}  {name}")

    print(f"\n  Time: {elapsed:.2f}s")

    if all_pass:
        print(f"\n  ✓ Phase 14 Chunks 1-5 complete: {N_OPCODES}-opcode ISA")
        print(f"    Arithmetic ops collapse O(n) repeated-op algorithms to O(1).")
        print(f"    Comparison ops enable native branching (max, abs, clamp).")
        print(f"    Bitwise ops enable bit manipulation (popcount, extract, masks).")
        print(f"    Unary ops add CLZ/CTZ/POPCNT/ABS/NEG — O(1) bit inspection.")
        print(f"    SELECT enables branchless ternary selection (sd=-2).")
        print(f"    Division by zero traps cleanly (OP_TRAP).")
        print(f"    Integration tests verify cross-opcode programs on both backends.")
        print(f"    Nonlinear FF dispatch extends the compiled transformer paradigm.")
        print(f"    All Phase 4/11/13 tests pass (full backward compatibility).")
    else:
        print("\n  ✗ Some tests failed. See details above.")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
