"""NumPy-based executor backend for the compiled transformer stack machine.

Pure-NumPy reference implementation with full 55-opcode ISA.
Uses parabolic key addressing for stack, locals, heap, and call-stack memory spaces.
"""

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
    Trace,
    TraceStep,
    _clz32,
    _ctz32,
    _popcnt32,
    _rotl32,
    _rotr32,
    _shr_s,
    _shr_u,
    _sign_extend_8,
    _sign_extend_16,
    _to_i32,
    _trunc_div,
    _trunc_rem,
)
from transturing.core.registry import register_backend


@register_backend
class NumPyExecutor(ExecutorBackend):
    """Compiled numpy executor with full 55-opcode ISA.

    Flattened from Phase14Executor <- Phase13Executor <- ExtendedExecutor <- CompiledExecutorNumpy.
    Uses parabolic key addressing (2D dot-product + argmax) for all memory reads.
    Float64 precision via numpy arrays; local eps=1e-10 for recency bias.
    """

    name = "numpy"

    def execute(self, prog, max_steps=50000):
        trace = Trace(program=prog)

        stack_keys = []
        stack_vals = []
        write_count = 0
        eps = 1e-10

        # Local variables address space (separate from stack)
        locals_keys = []
        locals_vals = []
        local_write_count = 0

        # Heap (linear memory) address space
        heap_keys = []
        heap_vals = []
        heap_write_count = 0

        # Call stack
        call_stack = []  # list of (ret_addr, saved_sp, saved_locals_base)
        locals_base = 0

        ip = 0
        sp = 0

        def stack_write(addr, val):
            nonlocal write_count
            stack_keys.append((2.0 * addr, -float(addr * addr) + eps * write_count))
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

        def local_write(local_idx, val):
            nonlocal local_write_count
            actual_idx = locals_base + local_idx
            locals_keys.append(
                (
                    2.0 * actual_idx,
                    -float(actual_idx * actual_idx) + eps * local_write_count,
                ),
            )
            locals_vals.append(val)
            local_write_count += 1

        def local_read(local_idx):
            if not locals_keys:
                return 0
            actual_idx = locals_base + local_idx
            keys = np.array(locals_keys)
            q = np.array([actual_idx, 1.0])
            scores = keys @ q
            best = np.argmax(scores)
            stored_idx = round(keys[best, 0] / 2.0)
            return locals_vals[best] if stored_idx == actual_idx else 0

        def heap_write(addr, val):
            nonlocal heap_write_count
            heap_keys.append((2.0 * addr, -float(addr * addr) + eps * heap_write_count))
            heap_vals.append(val)
            heap_write_count += 1

        def heap_read(addr):
            if not heap_keys:
                return 0
            keys = np.array(heap_keys)
            q = np.array([addr, 1.0])
            scores = keys @ q
            best = np.argmax(scores)
            stored_addr = round(keys[best, 0] / 2.0)
            return heap_vals[best] if stored_addr == addr else 0

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
                result = (val_a + val_b) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SUB:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = (val_b - val_a) & MASK32
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
                val_top = stack_read(sp)
                val_second = stack_read(sp - 1)
                val_third = stack_read(sp - 2)
                stack_write(sp, val_third)
                stack_write(sp - 1, val_top)
                stack_write(sp - 2, val_second)
                top = val_third

            # ── Phase 14 arithmetic ──
            elif op == OP_MUL:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = (val_a * val_b) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op in (OP_DIV_S, OP_DIV_U):
                val_a = stack_read(sp)
                if val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                val_b = stack_read(sp - 1)
                result = _trunc_div(val_b, val_a) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op in (OP_REM_S, OP_REM_U):
                val_a = stack_read(sp)
                if val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                val_b = stack_read(sp - 1)
                result = _trunc_rem(val_b, val_a) & MASK32
                sp -= 1
                stack_write(sp, result)
                top = result

            # ── Phase 14 Chunk 2: comparison ops ──
            elif op == OP_EQZ:
                val_a = stack_read(sp)
                result = 1 if val_a == 0 else 0
                stack_write(sp, result)
                top = result
            elif op in (
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
            ):
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
                stack_write(sp, result)
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
                val_a = stack_read(sp)  # c (condition)
                val_b = stack_read(sp - 1)  # b (false value)
                val_c = stack_read(sp - 2)  # a (true value)
                result = val_c if val_a != 0 else val_b
                sp -= 2
                stack_write(sp, result)
                top = result

            # ── Phase 15: local variables ──
            elif op == OP_LOCAL_GET:
                val = local_read(arg)
                sp += 1
                stack_write(sp, val)
                top = val
            elif op == OP_LOCAL_SET:
                val = stack_read(sp)
                sp -= 1
                local_write(arg, val)
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_LOCAL_TEE:
                val = stack_read(sp)
                local_write(arg, val)
                top = val

            # ── Phase 16: linear memory ──
            elif op == OP_I32_LOAD:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                stack_write(sp, val)
                top = val
            elif op == OP_I32_STORE:
                val = stack_read(sp)
                addr = stack_read(sp - 1)
                heap_write(int(addr), val)
                sp -= 2
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_I32_LOAD8_U:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = int(val) & 0xFF
                stack_write(sp, result)
                top = result
            elif op == OP_I32_LOAD8_S:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = _sign_extend_8(val)
                stack_write(sp, result)
                top = result
            elif op == OP_I32_LOAD16_U:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = int(val) & 0xFFFF
                stack_write(sp, result)
                top = result
            elif op == OP_I32_LOAD16_S:
                addr = stack_read(sp)
                val = heap_read(int(addr))
                result = _sign_extend_16(val)
                stack_write(sp, result)
                top = result
            elif op == OP_I32_STORE8:
                val = stack_read(sp)
                addr = stack_read(sp - 1)
                heap_write(int(addr), int(val) & 0xFF)
                sp -= 2
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_I32_STORE16:
                val = stack_read(sp)
                addr = stack_read(sp - 1)
                heap_write(int(addr), int(val) & 0xFFFF)
                sp -= 2
                top = stack_read(sp) if sp > 0 else 0

            # ── Phase 17: function calls ──
            elif op == OP_CALL:
                call_stack.append((ip + 1, sp, locals_base))
                locals_base = len(locals_keys)
                top = stack_read(sp) if sp > 0 else 0
                next_ip = arg
            elif op == OP_RETURN:
                if not call_stack:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break
                ret_val = stack_read(sp)
                ret_addr, saved_sp, saved_locals_base = call_stack.pop()
                sp = saved_sp + 1
                stack_write(sp, ret_val)
                locals_base = saved_locals_base
                top = ret_val
                next_ip = ret_addr

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
