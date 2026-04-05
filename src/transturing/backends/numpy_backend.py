"""
NumPy-based executor backend for the compiled transformer stack machine.

Pure-NumPy reference implementation with full 55-opcode ISA.
Uses parabolic key addressing for stack, locals, heap, and call-stack memory spaces.
"""

from typing import ClassVar

import numpy as np

from transturing.core.abc import ExecutorBackend
from transturing.core.isa import (
    MASK32,
    OP_ABS,
    OP_ADD,
    OP_AND,
    OP_CALL,
    OP_CLZ,
    OP_CTZ,
    OP_DIV_S,
    OP_DIV_U,
    OP_DUP,
    OP_EQ,
    OP_EQZ,
    OP_GE_S,
    OP_GE_U,
    OP_GT_S,
    OP_GT_U,
    OP_HALT,
    OP_I32_LOAD,
    OP_I32_LOAD8_S,
    OP_I32_LOAD8_U,
    OP_I32_LOAD16_S,
    OP_I32_LOAD16_U,
    OP_I32_STORE,
    OP_I32_STORE8,
    OP_I32_STORE16,
    OP_JNZ,
    OP_JZ,
    OP_LE_S,
    OP_LE_U,
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_LOCAL_TEE,
    OP_LT_S,
    OP_LT_U,
    OP_MUL,
    OP_NE,
    OP_NEG,
    OP_NOP,
    OP_OR,
    OP_OVER,
    OP_POP,
    OP_POPCNT,
    OP_PUSH,
    OP_REM_S,
    OP_REM_U,
    OP_RETURN,
    OP_ROT,
    OP_ROTL,
    OP_ROTR,
    OP_SELECT,
    OP_SHL,
    OP_SHR_S,
    OP_SHR_U,
    OP_SUB,
    OP_SWAP,
    OP_TRAP,
    OP_XOR,
    Instruction,
    Trace,
    TraceStep,
    clz32,
    ctz32,
    popcnt32,
    rotl32,
    rotr32,
    shr_s,
    shr_u,
    sign_extend_8,
    sign_extend_16,
    to_i32,
    trunc_div,
    trunc_rem,
)
from transturing.core.registry import register_backend


class _ParabolicStore:
    """
    Key-value store using parabolic encoding for position-based addressing.

    Each write stores (2*addr, -addr² + eps*write_count) as the key.
    Read computes dot-product of all keys with query [addr, 1.0] and
    returns the value at the argmax, verifying the address matches.
    """

    __slots__ = ("_eps", "_keys", "_vals", "_write_count")

    def __init__(self, eps: float = 1e-10) -> None:
        self._keys: list[tuple[float, float]] = []
        self._vals: list[int] = []
        self._write_count = 0
        self._eps = eps

    def write(self, addr: int, val: int) -> None:
        self._keys.append(
            (2.0 * addr, -float(addr * addr) + self._eps * self._write_count)
        )
        self._vals.append(val)
        self._write_count += 1

    def read(self, addr: int) -> int:
        if not self._keys:
            return 0
        keys = np.array(self._keys)
        q = np.array([addr, 1.0])
        scores = keys @ q
        best = np.argmax(scores)
        stored_addr = round(keys[best, 0] / 2.0)
        return self._vals[best] if stored_addr == addr else 0

    def __len__(self) -> int:
        return len(self._keys)


class _ExecCtx:
    """Mutable execution state for a single program run."""

    __slots__ = (
        "call_stack",
        "heap",
        "ip",
        "locals_base",
        "locals_store",
        "next_ip",
        "sp",
        "stack",
        "top",
        "trace",
    )

    def __init__(self, trace: Trace) -> None:
        self.sp = 0
        self.ip = 0
        self.next_ip = 0
        self.top = 0
        self.locals_base = 0
        self.call_stack: list[tuple[int, int, int]] = []
        self.stack = _ParabolicStore()
        self.locals_store = _ParabolicStore()
        self.heap = _ParabolicStore()
        self.trace = trace


@register_backend
class NumPyExecutor(ExecutorBackend):
    """
    Compiled numpy executor with full 55-opcode ISA.

    Flattened from Phase14Executor <- Phase13Executor <- ExtendedExecutor
    <- CompiledExecutorNumpy. Uses parabolic key addressing (2D dot-product
    + argmax) for all memory reads. Float64 precision via numpy arrays;
    local eps=1e-10 for recency bias.
    """

    name: str = "numpy"

    _DISPATCH: ClassVar[dict[int, str]] = {
        OP_PUSH: "_handle_push",
        OP_POP: "_handle_pop",
        OP_DUP: "_handle_dup",
        OP_ADD: "_handle_add_sub",
        OP_SUB: "_handle_add_sub",
        OP_SWAP: "_handle_stack_manip",
        OP_OVER: "_handle_stack_manip",
        OP_ROT: "_handle_stack_manip",
        OP_MUL: "_handle_arithmetic",
        OP_DIV_S: "_handle_arithmetic",
        OP_DIV_U: "_handle_arithmetic",
        OP_REM_S: "_handle_arithmetic",
        OP_REM_U: "_handle_arithmetic",
        OP_EQZ: "_handle_comparison",
        OP_EQ: "_handle_comparison",
        OP_NE: "_handle_comparison",
        OP_LT_S: "_handle_comparison",
        OP_LT_U: "_handle_comparison",
        OP_GT_S: "_handle_comparison",
        OP_GT_U: "_handle_comparison",
        OP_LE_S: "_handle_comparison",
        OP_LE_U: "_handle_comparison",
        OP_GE_S: "_handle_comparison",
        OP_GE_U: "_handle_comparison",
        OP_AND: "_handle_bitwise",
        OP_OR: "_handle_bitwise",
        OP_XOR: "_handle_bitwise",
        OP_SHL: "_handle_bitwise",
        OP_SHR_S: "_handle_bitwise",
        OP_SHR_U: "_handle_bitwise",
        OP_ROTL: "_handle_bitwise",
        OP_ROTR: "_handle_bitwise",
        OP_CLZ: "_handle_unary",
        OP_CTZ: "_handle_unary",
        OP_POPCNT: "_handle_unary",
        OP_ABS: "_handle_unary",
        OP_NEG: "_handle_unary",
        OP_SELECT: "_handle_select",
        OP_LOCAL_GET: "_handle_local_ops",
        OP_LOCAL_SET: "_handle_local_ops",
        OP_LOCAL_TEE: "_handle_local_ops",
        OP_I32_LOAD: "_handle_memory_ops",
        OP_I32_STORE: "_handle_memory_ops",
        OP_I32_LOAD8_U: "_handle_memory_ops",
        OP_I32_LOAD8_S: "_handle_memory_ops",
        OP_I32_LOAD16_U: "_handle_memory_ops",
        OP_I32_LOAD16_S: "_handle_memory_ops",
        OP_I32_STORE8: "_handle_memory_ops",
        OP_I32_STORE16: "_handle_memory_ops",
        OP_CALL: "_handle_call",
        OP_RETURN: "_handle_return",
        OP_JZ: "_handle_branch",
        OP_JNZ: "_handle_branch",
        OP_NOP: "_handle_nop",
        OP_HALT: "_handle_halt",
    }

    def execute(
        self,
        prog: list[Instruction],
        max_steps: int = 50000,
    ) -> Trace:
        """Execute a program and return the execution trace."""
        ctx = _ExecCtx(Trace(program=prog))

        for _step in range(max_steps):
            if ctx.ip >= len(prog):
                break

            op = prog[ctx.ip].op
            arg = prog[ctx.ip].arg
            ctx.next_ip = ctx.ip + 1
            ctx.top = 0

            should_break = self._dispatch(ctx, op, arg)

            if should_break:
                break

            ctx.ip = ctx.next_ip

        return ctx.trace

    def _dispatch(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        """Route opcode to handler. Returns True to halt execution."""
        handler_name = self._DISPATCH.get(op)
        if handler_name is not None:
            return getattr(self, handler_name)(ctx, op, arg)
        # Unknown opcode — treat as NOP
        ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Phase 4 base ops ──

    def _handle_push(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        ctx.sp += 1
        ctx.stack.write(ctx.sp, arg)
        ctx.top = arg
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    def _handle_pop(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        ctx.sp -= 1
        ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    def _handle_dup(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        val = ctx.stack.read(ctx.sp)
        ctx.sp += 1
        ctx.stack.write(ctx.sp, val)
        ctx.top = val
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    def _handle_add_sub(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        val_a = ctx.stack.read(ctx.sp)
        val_b = ctx.stack.read(ctx.sp - 1)
        result = (val_a + val_b) & MASK32 if op == OP_ADD else (val_b - val_a) & MASK32
        ctx.sp -= 1
        ctx.stack.write(ctx.sp, result)
        ctx.top = result
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Phase 13 stack manipulation ──

    def _handle_stack_manip(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        if op == OP_SWAP:
            val_a = ctx.stack.read(ctx.sp)
            val_b = ctx.stack.read(ctx.sp - 1)
            ctx.stack.write(ctx.sp, val_b)
            ctx.stack.write(ctx.sp - 1, val_a)
            ctx.top = val_b
        elif op == OP_OVER:
            val_b = ctx.stack.read(ctx.sp - 1)
            ctx.sp += 1
            ctx.stack.write(ctx.sp, val_b)
            ctx.top = val_b
        else:  # OP_ROT
            val_top = ctx.stack.read(ctx.sp)
            val_second = ctx.stack.read(ctx.sp - 1)
            val_third = ctx.stack.read(ctx.sp - 2)
            ctx.stack.write(ctx.sp, val_third)
            ctx.stack.write(ctx.sp - 1, val_top)
            ctx.stack.write(ctx.sp - 2, val_second)
            ctx.top = val_third
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Phase 14 arithmetic ──

    def _handle_arithmetic(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        if op == OP_MUL:
            val_a = ctx.stack.read(ctx.sp)
            val_b = ctx.stack.read(ctx.sp - 1)
            result = (val_a * val_b) & MASK32
            ctx.sp -= 1
            ctx.stack.write(ctx.sp, result)
            ctx.top = result
            ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
            return False

        # DIV_S, DIV_U, REM_S, REM_U — may trap on division by zero
        val_a = ctx.stack.read(ctx.sp)
        if val_a == 0:
            ctx.trace.steps.append(TraceStep(OP_TRAP, 0, ctx.sp, 0))
            return True
        val_b = ctx.stack.read(ctx.sp - 1)
        if op in (OP_DIV_S, OP_DIV_U):
            result = trunc_div(val_b, val_a) & MASK32
        else:
            result = trunc_rem(val_b, val_a) & MASK32
        ctx.sp -= 1
        ctx.stack.write(ctx.sp, result)
        ctx.top = result
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Phase 14 comparison ──

    def _handle_comparison(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        if op == OP_EQZ:
            val_a = ctx.stack.read(ctx.sp)
            result = 1 if val_a == 0 else 0
            ctx.stack.write(ctx.sp, result)
            ctx.top = result
        else:
            val_a = ctx.stack.read(ctx.sp)
            val_b = ctx.stack.read(ctx.sp - 1)
            result = self._compare(op, val_a, val_b)
            ctx.sp -= 1
            ctx.stack.write(ctx.sp, result)
            ctx.top = result
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    @staticmethod
    def _compare(op: int, val_a: int, val_b: int) -> int:
        if op == OP_EQ:
            result = 1 if val_a == val_b else 0
        elif op == OP_NE:
            result = 1 if val_a != val_b else 0
        elif op in (OP_LT_S, OP_LT_U):
            result = 1 if val_b < val_a else 0
        elif op in (OP_GT_S, OP_GT_U):
            result = 1 if val_b > val_a else 0
        elif op in (OP_LE_S, OP_LE_U):
            result = 1 if val_b <= val_a else 0
        elif op in (OP_GE_S, OP_GE_U):
            result = 1 if val_b >= val_a else 0
        else:
            result = 0
        return result

    # ── Phase 14 bitwise ──

    def _handle_bitwise(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        val_a = ctx.stack.read(ctx.sp)
        val_b = ctx.stack.read(ctx.sp - 1)
        result = self._bitwise_result(op, val_a, val_b)
        ctx.sp -= 1
        ctx.stack.write(ctx.sp, result)
        ctx.top = result
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    @staticmethod
    def _bitwise_result(op: int, val_a: int, val_b: int) -> int:
        if op == OP_AND:
            result = to_i32(val_a) & to_i32(val_b)
        elif op == OP_OR:
            result = to_i32(val_a) | to_i32(val_b)
        elif op == OP_XOR:
            result = to_i32(val_a) ^ to_i32(val_b)
        elif op == OP_SHL:
            result = (to_i32(val_b) << (int(val_a) & 31)) & MASK32
        elif op == OP_SHR_S:
            result = shr_s(val_b, val_a)
        elif op == OP_SHR_U:
            result = shr_u(val_b, val_a)
        elif op == OP_ROTL:
            result = rotl32(val_b, val_a)
        else:
            result = rotr32(val_b, val_a)  # OP_ROTR
        return result

    # ── Phase 14 unary ──

    def _handle_unary(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        val_a = ctx.stack.read(ctx.sp)
        result = self._unary_result(op, val_a)
        ctx.stack.write(ctx.sp, result)
        ctx.top = result
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    @staticmethod
    def _unary_result(op: int, val_a: int) -> int:
        if op == OP_CLZ:
            return clz32(val_a)
        if op == OP_CTZ:
            return ctz32(val_a)
        if op == OP_POPCNT:
            return popcnt32(val_a)
        if op == OP_ABS:
            return abs(int(val_a))
        return (-int(val_a)) & MASK32  # OP_NEG

    def _handle_select(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        val_a = ctx.stack.read(ctx.sp)  # c (condition)
        val_b = ctx.stack.read(ctx.sp - 1)  # b (false value)
        val_c = ctx.stack.read(ctx.sp - 2)  # a (true value)
        result = val_c if val_a != 0 else val_b
        ctx.sp -= 2
        ctx.stack.write(ctx.sp, result)
        ctx.top = result
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Phase 15: local variables ──

    def _handle_local_ops(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        if op == OP_LOCAL_GET:
            val = ctx.locals_store.read(ctx.locals_base + arg)
            ctx.sp += 1
            ctx.stack.write(ctx.sp, val)
            ctx.top = val
        elif op == OP_LOCAL_SET:
            val = ctx.stack.read(ctx.sp)
            ctx.sp -= 1
            ctx.locals_store.write(ctx.locals_base + arg, val)
            ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        else:  # OP_LOCAL_TEE
            val = ctx.stack.read(ctx.sp)
            ctx.locals_store.write(ctx.locals_base + arg, val)
            ctx.top = val
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Phase 16: linear memory ──

    def _handle_memory_ops(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        if op == OP_I32_LOAD:
            addr = ctx.stack.read(ctx.sp)
            val = ctx.heap.read(int(addr))
            ctx.stack.write(ctx.sp, val)
            ctx.top = val
        elif op == OP_I32_LOAD8_U:
            addr = ctx.stack.read(ctx.sp)
            val = ctx.heap.read(int(addr))
            result = int(val) & 0xFF
            ctx.stack.write(ctx.sp, result)
            ctx.top = result
        elif op == OP_I32_LOAD8_S:
            addr = ctx.stack.read(ctx.sp)
            val = ctx.heap.read(int(addr))
            result = sign_extend_8(val)
            ctx.stack.write(ctx.sp, result)
            ctx.top = result
        elif op == OP_I32_LOAD16_U:
            addr = ctx.stack.read(ctx.sp)
            val = ctx.heap.read(int(addr))
            result = int(val) & 0xFFFF
            ctx.stack.write(ctx.sp, result)
            ctx.top = result
        elif op == OP_I32_LOAD16_S:
            addr = ctx.stack.read(ctx.sp)
            val = ctx.heap.read(int(addr))
            result = sign_extend_16(val)
            ctx.stack.write(ctx.sp, result)
            ctx.top = result
        elif op == OP_I32_STORE:
            val = ctx.stack.read(ctx.sp)
            addr = ctx.stack.read(ctx.sp - 1)
            ctx.heap.write(int(addr), val)
            ctx.sp -= 2
            ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        elif op == OP_I32_STORE8:
            val = ctx.stack.read(ctx.sp)
            addr = ctx.stack.read(ctx.sp - 1)
            ctx.heap.write(int(addr), int(val) & 0xFF)
            ctx.sp -= 2
            ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        else:  # OP_I32_STORE16
            val = ctx.stack.read(ctx.sp)
            addr = ctx.stack.read(ctx.sp - 1)
            ctx.heap.write(int(addr), int(val) & 0xFFFF)
            ctx.sp -= 2
            ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Phase 17: function calls ──

    def _handle_call(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        ctx.call_stack.append((ctx.ip + 1, ctx.sp, ctx.locals_base))
        ctx.locals_base = len(ctx.locals_store)
        ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        ctx.next_ip = arg
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    def _handle_return(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        if not ctx.call_stack:
            ctx.trace.steps.append(TraceStep(OP_TRAP, 0, ctx.sp, 0))
            return True
        ret_val = ctx.stack.read(ctx.sp)
        ret_addr, saved_sp, saved_locals_base = ctx.call_stack.pop()
        ctx.sp = saved_sp + 1
        ctx.stack.write(ctx.sp, ret_val)
        ctx.locals_base = saved_locals_base
        ctx.top = ret_val
        ctx.next_ip = ret_addr
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    # ── Control flow ──

    def _handle_branch(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        cond = ctx.stack.read(ctx.sp)
        ctx.sp -= 1
        ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        if (op == OP_JZ and cond == 0) or (op == OP_JNZ and cond != 0):
            ctx.next_ip = arg
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    def _handle_nop(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return False

    def _handle_halt(self, ctx: _ExecCtx, op: int, arg: int) -> bool:
        ctx.top = ctx.stack.read(ctx.sp) if ctx.sp > 0 else 0
        ctx.trace.steps.append(TraceStep(op, arg, ctx.sp, ctx.top))
        return True
