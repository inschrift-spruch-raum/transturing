"""Mojo port of NumPyExecutor: full 55-opcode stack machine.

Stage 1 (issue #40): Naive structural translation — no SIMD, no pre-allocation.
Correctness first; optimization is Stage 2.

I/O contract (normal mode):
  Input:  program as space-separated "op arg op arg ..." via argv or stdin
  Output: one "op arg sp top" line per step, then "RESULT: <top>"

Timing mode (--repeat N):
  Runs the program N times silently, reports median execution time.
  Output: "TIMING_NS: <median_ns>"  (no trace lines)
  Used by the benchmark harness to measure µs/step without subprocess overhead.
"""

from std.sys import argv
from std.time import perf_counter_ns

# ─── Opcode constants ─────────────────────────────────────────────

comptime OP_PUSH = 1
comptime OP_POP = 2
comptime OP_ADD = 3
comptime OP_DUP = 4
comptime OP_HALT = 5
comptime OP_SUB = 6
comptime OP_JZ = 7
comptime OP_JNZ = 8
comptime OP_NOP = 9
comptime OP_SWAP = 10
comptime OP_OVER = 11
comptime OP_ROT = 12
comptime OP_MUL = 13
comptime OP_DIV_S = 14
comptime OP_DIV_U = 15
comptime OP_REM_S = 16
comptime OP_REM_U = 17
comptime OP_EQZ = 18
comptime OP_EQ = 19
comptime OP_NE = 20
comptime OP_LT_S = 21
comptime OP_LT_U = 22
comptime OP_GT_S = 23
comptime OP_GT_U = 24
comptime OP_LE_S = 25
comptime OP_LE_U = 26
comptime OP_GE_S = 27
comptime OP_GE_U = 28
comptime OP_AND = 29
comptime OP_OR = 30
comptime OP_XOR = 31
comptime OP_SHL = 32
comptime OP_SHR_S = 33
comptime OP_SHR_U = 34
comptime OP_ROTL = 35
comptime OP_ROTR = 36
comptime OP_CLZ = 37
comptime OP_CTZ = 38
comptime OP_POPCNT = 39
comptime OP_ABS = 40
comptime OP_NEG = 41
comptime OP_SELECT = 42
comptime OP_LOCAL_GET = 43
comptime OP_LOCAL_SET = 44
comptime OP_LOCAL_TEE = 45
comptime OP_I32_LOAD = 46
comptime OP_I32_STORE = 47
comptime OP_I32_LOAD8_U = 48
comptime OP_I32_LOAD8_S = 49
comptime OP_I32_LOAD16_U = 50
comptime OP_I32_LOAD16_S = 51
comptime OP_I32_STORE8 = 52
comptime OP_I32_STORE16 = 53
comptime OP_CALL = 54
comptime OP_RETURN = 55
comptime OP_TRAP = 99

comptime MASK32 = 0xFFFFFFFF
comptime EPS    = Float64(1e-10)


# ─── Data structures ──────────────────────────────────────────────

# Parabolic key-value entry: key = (2*addr, -addr^2 + eps*write_count)
@fieldwise_init
struct KV(Copyable, Movable):
    var k0: Float64
    var k1: Float64
    var val: Int


# Call-stack frame
@fieldwise_init
struct CallFrame(Copyable, Movable):
    var ret_addr: Int
    var saved_sp: Int
    var saved_locals_base: Int


# ─── Parabolic memory primitives ─────────────────────────────────

def mem_write(mut keys: List[KV], addr: Int, val: Int, write_count: Int):
    var a = Float64(addr)
    keys.append(KV(2.0 * a, -(a * a) + EPS * Float64(write_count), val))


def mem_read(keys: List[KV], addr: Int) -> Int:
    if len(keys) == 0:
        return 0
    var q0 = Float64(addr)
    var q1 = Float64(1.0)
    var best_idx = 0
    var best_score = keys[0].k0 * q0 + keys[0].k1 * q1
    for i in range(1, len(keys)):
        var score = keys[i].k0 * q0 + keys[i].k1 * q1
        if score > best_score:
            best_score = score
            best_idx = i
    var stored_addr = Int(keys[best_idx].k0 / 2.0 + 0.5)
    if stored_addr == addr:
        return keys[best_idx].val
    return 0


# ─── Math helpers ────────────────────────────────────────────────

def mask32(v: Int) -> Int:
    return v & MASK32


def trunc_div(b: Int, a: Int) -> Int:
    """Division truncating toward zero (WASM i32 semantics)."""
    # Python-compatible: int(b / a) truncates toward zero
    if a == 0:
        return 0  # caller handles trap
    var fb = Float64(b)
    var fa = Float64(a)
    var q = fb / fa
    if q >= 0.0:
        return Int(q)
    else:
        # truncate toward zero = ceil for negative quotient
        var qi = Int(q)
        # if there's a remainder, qi is already truncated by Float64→Int
        return qi


def trunc_rem(b: Int, a: Int) -> Int:
    """Remainder matching truncated division."""
    return b - trunc_div(b, a) * a


def to_i32(val: Int) -> Int:
    return val & MASK32


def shr_u(b: Int, a: Int) -> Int:
    """Logical (unsigned) right shift."""
    return to_i32(b) >> (a & 31)


def shr_s(b: Int, a: Int) -> Int:
    """Arithmetic (signed) right shift."""
    var v = to_i32(b)
    var shift = a & 31
    if v >= 0x80000000:
        v -= 0x100000000
    var result = v >> shift
    if result < 0:
        return result & MASK32
    return result


def rotl32(b: Int, a: Int) -> Int:
    var v = to_i32(b)
    var shift = a & 31
    if shift == 0:
        return v
    return ((v << shift) | (v >> (32 - shift))) & MASK32


def rotr32(b: Int, a: Int) -> Int:
    var v = to_i32(b)
    var shift = a & 31
    if shift == 0:
        return v
    return ((v >> shift) | (v << (32 - shift))) & MASK32


def clz32(val: Int) -> Int:
    var v = to_i32(val)
    if v == 0:
        return 32
    var n = 0
    if v <= 0x0000FFFF:
        n += 16
        v <<= 16
    if v <= 0x00FFFFFF:
        n += 8
        v <<= 8
    if v <= 0x0FFFFFFF:
        n += 4
        v <<= 4
    if v <= 0x3FFFFFFF:
        n += 2
        v <<= 2
    if v <= 0x7FFFFFFF:
        n += 1
    return n


def ctz32(val: Int) -> Int:
    var v = to_i32(val)
    if v == 0:
        return 32
    var n = 0
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


def popcnt32(val: Int) -> Int:
    var v = to_i32(val)
    v = v - ((v >> 1) & 0x55555555)
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333)
    v = (v + (v >> 4)) & 0x0F0F0F0F
    return (v * 0x01010101) & MASK32
    # Note: >> 24 done by caller; just return popcount
    # actually this formula gives popcount in low byte after shift
    # fix: return the standard result


def _popcnt32(val: Int) -> Int:
    """Correct popcount for 32-bit value."""
    var v = to_i32(val)
    var count = 0
    while v != 0:
        count += v & 1
        v >>= 1
    return count


def sign_extend_8(val: Int) -> Int:
    var v = val & 0xFF
    if v >= 0x80:
        return v - 0x100
    return v


def sign_extend_16(val: Int) -> Int:
    var v = val & 0xFFFF
    if v >= 0x8000:
        return v - 0x10000
    return v


# ─── Main executor ───────────────────────────────────────────────

def execute(prog_ops: List[Int], prog_args: List[Int], verbose: Bool = True) raises -> Int:
    """Execute program; optionally print trace; return final top-of-stack.

    verbose=True  → emit one "op arg sp top" line per step (normal mode)
    verbose=False → silent execution for timing loops
    """

    var stack_keys  = List[KV]()
    var locals_keys = List[KV]()
    var heap_keys   = List[KV]()
    var call_stack  = List[CallFrame]()

    var stack_wc  = 0
    var local_wc  = 0
    var heap_wc   = 0

    var locals_base = 0
    var ip = 0
    var sp = 0

    var prog_len = len(prog_ops)
    var max_steps = 50000

    for _step in range(max_steps):
        if ip >= prog_len:
            break

        var op  = prog_ops[ip]
        var arg = prog_args[ip]
        var next_ip = ip + 1
        var top = 0

        # ── Stack basics ──────────────────────────────────────────
        if op == OP_PUSH:
            sp += 1
            mem_write(stack_keys, sp, arg, stack_wc)
            stack_wc += 1
            top = arg

        elif op == OP_POP:
            sp -= 1
            top = mem_read(stack_keys, sp) if sp > 0 else 0

        elif op == OP_DUP:
            var v = mem_read(stack_keys, sp)
            sp += 1
            mem_write(stack_keys, sp, v, stack_wc)
            stack_wc += 1
            top = v

        elif op == OP_SWAP:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            mem_write(stack_keys, sp,     vb, stack_wc); stack_wc += 1
            mem_write(stack_keys, sp - 1, va, stack_wc); stack_wc += 1
            top = vb

        elif op == OP_OVER:
            var vb = mem_read(stack_keys, sp - 1)
            sp += 1
            mem_write(stack_keys, sp, vb, stack_wc)
            stack_wc += 1
            top = vb

        elif op == OP_ROT:
            var v_top    = mem_read(stack_keys, sp)
            var v_second = mem_read(stack_keys, sp - 1)
            var v_third  = mem_read(stack_keys, sp - 2)
            mem_write(stack_keys, sp,     v_third,  stack_wc); stack_wc += 1
            mem_write(stack_keys, sp - 1, v_top,    stack_wc); stack_wc += 1
            mem_write(stack_keys, sp - 2, v_second, stack_wc); stack_wc += 1
            top = v_third

        elif op == OP_NOP:
            top = mem_read(stack_keys, sp) if sp > 0 else 0

        elif op == OP_HALT:
            top = mem_read(stack_keys, sp) if sp > 0 else 0
            if verbose:
                print(op, arg, sp, top)
            return top

        # ── Arithmetic ───────────────────────────────────────────
        elif op == OP_ADD:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = mask32(va + vb)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_SUB:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = mask32(vb - va)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_MUL:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = mask32(va * vb)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_DIV_S or op == OP_DIV_U:
            var va = mem_read(stack_keys, sp)
            if va == 0:
                if verbose:
                    print(OP_TRAP, 0, sp, 0)
                return 0
            var vb = mem_read(stack_keys, sp - 1)
            var res = mask32(trunc_div(vb, va))
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_REM_S or op == OP_REM_U:
            var va = mem_read(stack_keys, sp)
            if va == 0:
                if verbose:
                    print(OP_TRAP, 0, sp, 0)
                return 0
            var vb = mem_read(stack_keys, sp - 1)
            var res = mask32(trunc_rem(vb, va))
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        # ── Comparisons ──────────────────────────────────────────
        elif op == OP_EQZ:
            var va = mem_read(stack_keys, sp)
            var res = 1 if va == 0 else 0
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_EQ:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = 1 if va == vb else 0
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_NE:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = 1 if va != vb else 0
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_LT_S or op == OP_LT_U:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = 1 if vb < va else 0
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_GT_S or op == OP_GT_U:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = 1 if vb > va else 0
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_LE_S or op == OP_LE_U:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = 1 if vb <= va else 0
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_GE_S or op == OP_GE_U:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = 1 if vb >= va else 0
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        # ── Bitwise ──────────────────────────────────────────────
        elif op == OP_AND:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = to_i32(va) & to_i32(vb)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_OR:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = to_i32(va) | to_i32(vb)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_XOR:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = to_i32(va) ^ to_i32(vb)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_SHL:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = mask32(to_i32(vb) << (va & 31))
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_SHR_S:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = shr_s(vb, va)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_SHR_U:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = shr_u(vb, va)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_ROTL:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = rotl32(vb, va)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_ROTR:
            var va = mem_read(stack_keys, sp)
            var vb = mem_read(stack_keys, sp - 1)
            var res = rotr32(vb, va)
            sp -= 1
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        # ── Unary + parametric ───────────────────────────────────
        elif op == OP_CLZ:
            var va = mem_read(stack_keys, sp)
            var res = clz32(va)
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_CTZ:
            var va = mem_read(stack_keys, sp)
            var res = ctz32(va)
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_POPCNT:
            var va = mem_read(stack_keys, sp)
            var res = _popcnt32(va)
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_ABS:
            var va = mem_read(stack_keys, sp)
            var res = -va if va < 0 else va
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_NEG:
            var va = mem_read(stack_keys, sp)
            var res = mask32(-va)
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        elif op == OP_SELECT:
            var va = mem_read(stack_keys, sp)      # c (condition)
            var vb = mem_read(stack_keys, sp - 1)  # b (false value)
            var vc = mem_read(stack_keys, sp - 2)  # a (true value)
            var res = vc if va != 0 else vb
            sp -= 2
            mem_write(stack_keys, sp, res, stack_wc); stack_wc += 1
            top = res

        # ── Locals ───────────────────────────────────────────────
        elif op == OP_LOCAL_GET:
            var actual_idx = locals_base + arg
            var v = mem_read(locals_keys, actual_idx)
            sp += 1
            mem_write(stack_keys, sp, v, stack_wc); stack_wc += 1
            top = v

        elif op == OP_LOCAL_SET:
            var v = mem_read(stack_keys, sp)
            sp -= 1
            var actual_idx = locals_base + arg
            mem_write(locals_keys, actual_idx, v, local_wc); local_wc += 1
            top = mem_read(stack_keys, sp) if sp > 0 else 0

        elif op == OP_LOCAL_TEE:
            var v = mem_read(stack_keys, sp)
            var actual_idx = locals_base + arg
            mem_write(locals_keys, actual_idx, v, local_wc); local_wc += 1
            top = v

        # ── Linear memory ────────────────────────────────────────
        elif op == OP_I32_LOAD:
            var addr = mem_read(stack_keys, sp)
            var v = mem_read(heap_keys, addr)
            mem_write(stack_keys, sp, v, stack_wc); stack_wc += 1
            top = v

        elif op == OP_I32_STORE:
            var v    = mem_read(stack_keys, sp)
            var addr = mem_read(stack_keys, sp - 1)
            mem_write(heap_keys, addr, v, heap_wc); heap_wc += 1
            sp -= 2
            top = mem_read(stack_keys, sp) if sp > 0 else 0

        elif op == OP_I32_LOAD8_U:
            var addr = mem_read(stack_keys, sp)
            var v = mem_read(heap_keys, addr) & 0xFF
            mem_write(stack_keys, sp, v, stack_wc); stack_wc += 1
            top = v

        elif op == OP_I32_LOAD8_S:
            var addr = mem_read(stack_keys, sp)
            var v = sign_extend_8(mem_read(heap_keys, addr))
            mem_write(stack_keys, sp, v, stack_wc); stack_wc += 1
            top = v

        elif op == OP_I32_LOAD16_U:
            var addr = mem_read(stack_keys, sp)
            var v = mem_read(heap_keys, addr) & 0xFFFF
            mem_write(stack_keys, sp, v, stack_wc); stack_wc += 1
            top = v

        elif op == OP_I32_LOAD16_S:
            var addr = mem_read(stack_keys, sp)
            var v = sign_extend_16(mem_read(heap_keys, addr))
            mem_write(stack_keys, sp, v, stack_wc); stack_wc += 1
            top = v

        elif op == OP_I32_STORE8:
            var v    = mem_read(stack_keys, sp) & 0xFF
            var addr = mem_read(stack_keys, sp - 1)
            mem_write(heap_keys, addr, v, heap_wc); heap_wc += 1
            sp -= 2
            top = mem_read(stack_keys, sp) if sp > 0 else 0

        elif op == OP_I32_STORE16:
            var v    = mem_read(stack_keys, sp) & 0xFFFF
            var addr = mem_read(stack_keys, sp - 1)
            mem_write(heap_keys, addr, v, heap_wc); heap_wc += 1
            sp -= 2
            top = mem_read(stack_keys, sp) if sp > 0 else 0

        # ── Function calls ───────────────────────────────────────
        elif op == OP_CALL:
            call_stack.append(CallFrame(ip + 1, sp, locals_base))
            locals_base = len(locals_keys)
            top = mem_read(stack_keys, sp) if sp > 0 else 0
            next_ip = arg

        elif op == OP_RETURN:
            if len(call_stack) == 0:
                if verbose:
                    print(OP_TRAP, 0, sp, 0)
                return 0
            var ret_val = mem_read(stack_keys, sp)
            var frame   = call_stack.pop()
            sp = frame.saved_sp + 1
            mem_write(stack_keys, sp, ret_val, stack_wc); stack_wc += 1
            locals_base = frame.saved_locals_base
            top = ret_val
            next_ip = frame.ret_addr

        # ── Control flow ─────────────────────────────────────────
        elif op == OP_JZ:
            var cond = mem_read(stack_keys, sp)
            sp -= 1
            top = mem_read(stack_keys, sp) if sp > 0 else 0
            if cond == 0:
                next_ip = arg

        elif op == OP_JNZ:
            var cond = mem_read(stack_keys, sp)
            sp -= 1
            top = mem_read(stack_keys, sp) if sp > 0 else 0
            if cond != 0:
                next_ip = arg

        else:
            # Unknown opcode — treat as NOP (matches NumPyExecutor)
            top = mem_read(stack_keys, sp) if sp > 0 else 0

        if verbose:
            print(op, arg, sp, top)
        ip = next_ip

    return mem_read(stack_keys, sp) if sp > 0 else 0


# ─── Helpers ─────────────────────────────────────────────────────

def sort_list(mut lst: List[Int]):
    """In-place insertion sort for timing samples (small N)."""
    for i in range(1, len(lst)):
        var key = lst[i]
        var j = i - 1
        while j >= 0 and lst[j] > key:
            lst[j + 1] = lst[j]
            j -= 1
        lst[j + 1] = key


# ─── Entry point ─────────────────────────────────────────────────

def main() raises:
    var args = argv()

    # Check for --repeat N flag (timing mode)
    var repeat = 0
    var arg_start = 1
    if len(args) > 2 and args[1] == "--repeat":
        repeat = atol(args[2])
        arg_start = 3

    # Build program string from remaining args or stdin
    var prog_str: String
    if len(args) > arg_start:
        prog_str = String()
        for i in range(arg_start, len(args)):
            if i > arg_start:
                prog_str += " "
            prog_str += args[i]
    else:
        prog_str = input()

    # Parse "op arg op arg ..." into parallel lists
    var tokens = prog_str.split(" ")
    var prog_ops  = List[Int]()
    var prog_args = List[Int]()
    var i = 0
    while i < len(tokens):
        var tok = tokens[i]
        if len(tok) == 0:
            i += 1
            continue
        var op = atol(tok)
        i += 1
        var arg = 0
        if i < len(tokens) and len(tokens[i]) > 0:
            arg = atol(tokens[i])
            i += 1
        prog_ops.append(op)
        prog_args.append(arg)

    if repeat > 0:
        # ── Timing mode: run N times silently, report median ns ──
        var samples = List[Int]()
        for _ in range(repeat):
            var t0 = Int(perf_counter_ns())
            var _ = execute(prog_ops, prog_args, verbose=False)
            samples.append(Int(perf_counter_ns()) - t0)
        sort_list(samples)
        var median = samples[repeat // 2]
        print("TIMING_NS:", median)
    else:
        # ── Normal mode: print trace + result ──
        var result = execute(prog_ops, prog_args, verbose=True)
        print("RESULT:", result)
