"""Benchmark programs for the stack machine ISA.

Three substantial, non-academic benchmark programs that exercise heap memory,
local variables, nested loops, and bitwise/arithmetic operations.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isa import (
    Instruction,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT,
    OP_SUB, OP_JZ, OP_JNZ, OP_NOP,
    OP_SWAP, OP_OVER, OP_ROT,
    OP_MUL, OP_DIV_S, OP_REM_S,
    OP_GT_S, OP_XOR, OP_AND,
    OP_LOCAL_GET, OP_LOCAL_SET, OP_LOCAL_TEE,
    OP_I32_LOAD, OP_I32_STORE,
)


# ─── FNV-1a Hash ─────────────────────────────────────────────────

def make_fnv1a(data: list) -> tuple:
    """FNV-1a 32-bit hash of a byte sequence.

    Algorithm:
        hash = 2166136261  # FNV offset basis
        for each byte:
            hash = hash XOR byte
            hash = (hash * 16777619) & 0xFFFFFFFF

    Locals:
        local[0] = hash accumulator
        local[1] = remaining count (counts down from len(data) to 0)

    Heap:
        heap[0..n-1] = input bytes
    """
    n = len(data)
    FNV_PRIME  = 16777619
    FNV_OFFSET = 2166136261
    MASK32     = 0xFFFFFFFF

    # Compute expected result in Python
    expected = FNV_OFFSET
    for byte in data:
        expected ^= byte
        expected = (expected * FNV_PRIME) & MASK32

    # --- Build program ---
    prog = []

    # Phase 1: initialise heap[0..n-1] = data
    for i, byte in enumerate(data):
        prog.append(Instruction(OP_PUSH, i))       # addr
        prog.append(Instruction(OP_PUSH, byte))    # val
        prog.append(Instruction(OP_I32_STORE))     # heap[i] = data[i]
    # ip = 3*n  (= 96 for n=32)

    heap_init_end = len(prog)  # first instruction after heap init

    # Phase 2: initialise locals
    #   local[0] = hash = FNV_OFFSET
    #   local[1] = remaining = n
    prog.append(Instruction(OP_PUSH, FNV_OFFSET))
    prog.append(Instruction(OP_LOCAL_SET, 0))      # local[0] = hash
    prog.append(Instruction(OP_PUSH, n))
    prog.append(Instruction(OP_LOCAL_SET, 1))      # local[1] = remaining count

    # Phase 3: loop body
    # loop_start: process next byte (index = n - remaining)
    loop_start = len(prog)

    prog.append(Instruction(OP_LOCAL_GET, 1))      # remaining
    prog.append(Instruction(OP_JZ, -1))            # placeholder: if remaining==0 -> done
    jz_idx = len(prog) - 1                         # index of JZ instruction to patch

    # addr = n - remaining
    prog.append(Instruction(OP_PUSH, n))
    prog.append(Instruction(OP_LOCAL_GET, 1))
    prog.append(Instruction(OP_SUB))               # n - remaining = current addr
    prog.append(Instruction(OP_I32_LOAD))          # byte = heap[addr]

    # hash = hash XOR byte
    prog.append(Instruction(OP_LOCAL_GET, 0))      # hash
    prog.append(Instruction(OP_XOR))               # hash ^ byte

    # hash = (hash * FNV_PRIME) & MASK32
    prog.append(Instruction(OP_PUSH, FNV_PRIME))
    prog.append(Instruction(OP_MUL))               # hash * FNV_PRIME
    prog.append(Instruction(OP_PUSH, MASK32))
    prog.append(Instruction(OP_AND))               # & 0xFFFFFFFF
    prog.append(Instruction(OP_LOCAL_SET, 0))      # local[0] = new hash

    # remaining -= 1
    prog.append(Instruction(OP_LOCAL_GET, 1))
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_SUB))
    prog.append(Instruction(OP_LOCAL_SET, 1))      # local[1] = remaining - 1

    # unconditional jump back to loop_start
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_JNZ, loop_start))

    # Phase 4: done
    done_addr = len(prog)
    prog[jz_idx] = Instruction(OP_JZ, done_addr)  # patch the JZ

    prog.append(Instruction(OP_LOCAL_GET, 0))      # push hash
    prog.append(Instruction(OP_HALT))

    return prog, expected


# ─── Bubble Sort ─────────────────────────────────────────────────

def make_bubble_sort(arr: list) -> tuple:
    """Bubble sort arr stored in heap memory. Returns sum of sorted array.

    Locals:
        local[0] = i  (outer loop, n-1 down to 1)
        local[1] = j  (inner loop, 0 up to i-1)
        local[2] = temp (for swap)

    Heap:
        heap[0..n-1] = array elements (initialised then sorted in-place)

    Result: sum of sorted array (invariant = sum of input).
    """
    n = len(arr)
    expected = sum(arr)   # sum is invariant under sorting

    prog = []

    # Phase 1: initialise heap
    for i, val in enumerate(arr):
        prog.append(Instruction(OP_PUSH, i))
        prog.append(Instruction(OP_PUSH, val))
        prog.append(Instruction(OP_I32_STORE))
    # ip = 3*n

    # Phase 2: outer loop init  i = n-1
    prog.append(Instruction(OP_PUSH, n - 1))
    prog.append(Instruction(OP_LOCAL_SET, 0))      # local[0] = i = n-1

    # OUTER_LOOP_START: reset j = 0
    outer_loop_start = len(prog)
    prog.append(Instruction(OP_PUSH, 0))
    prog.append(Instruction(OP_LOCAL_SET, 1))      # local[1] = j = 0

    # INNER_LOOP_START: if j == i -> goto inner_done
    inner_loop_start = len(prog)
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_LOCAL_GET, 0))      # i
    prog.append(Instruction(OP_SUB))               # j - i
    prog.append(Instruction(OP_JZ, -1))            # if j == i -> inner_done
    jz_inner_done = len(prog) - 1

    # load heap[j] and heap[j+1]
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_I32_LOAD))          # heap[j]
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_ADD))               # j+1
    prog.append(Instruction(OP_I32_LOAD))          # heap[j+1]
    # stack: [heap[j], heap[j+1]]
    # GT_S: second > top => heap[j] > heap[j+1]
    prog.append(Instruction(OP_GT_S))              # heap[j] > heap[j+1] ?
    prog.append(Instruction(OP_JZ, -1))            # if NOT (heap[j] > heap[j+1]) -> no_swap
    jz_no_swap = len(prog) - 1

    # SWAP: heap[j] <-> heap[j+1]
    # temp = heap[j]
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_I32_LOAD))          # heap[j]
    prog.append(Instruction(OP_LOCAL_SET, 2))      # temp = heap[j]
    # heap[j] = heap[j+1]
    # I32_STORE expects stack: [addr, val] (addr=second, val=top)
    prog.append(Instruction(OP_LOCAL_GET, 1))      # addr = j
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_ADD))               # j+1
    prog.append(Instruction(OP_I32_LOAD))          # val = heap[j+1]
    prog.append(Instruction(OP_I32_STORE))         # heap[j] = heap[j+1]
    # heap[j+1] = temp
    prog.append(Instruction(OP_LOCAL_GET, 1))      # addr = j
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_ADD))               # j+1
    prog.append(Instruction(OP_LOCAL_GET, 2))      # val = temp
    prog.append(Instruction(OP_I32_STORE))         # heap[j+1] = temp

    # NO_SWAP (fallthrough):
    no_swap = len(prog)
    prog[jz_no_swap] = Instruction(OP_JZ, no_swap)  # patch JZ

    # j += 1
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_ADD))               # j+1
    prog.append(Instruction(OP_LOCAL_SET, 1))      # j = j+1
    # unconditional jump to inner_loop_start
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_JNZ, inner_loop_start))

    # INNER_DONE:
    inner_done = len(prog)
    prog[jz_inner_done] = Instruction(OP_JZ, inner_done)  # patch JZ

    # i -= 1
    prog.append(Instruction(OP_LOCAL_GET, 0))      # i
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_SUB))               # i-1
    prog.append(Instruction(OP_LOCAL_TEE, 0))      # i = i-1, keep on stack
    # if i != 0 -> outer_loop_start
    prog.append(Instruction(OP_JNZ, outer_loop_start))

    # SUM: sum all elements (j re-used as index, local[1] = 0)
    prog.append(Instruction(OP_PUSH, 0))
    prog.append(Instruction(OP_LOCAL_SET, 1))      # j = 0
    prog.append(Instruction(OP_PUSH, 0))
    prog.append(Instruction(OP_LOCAL_SET, 2))      # sum = 0

    sum_loop_start = len(prog)
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_PUSH, n))
    prog.append(Instruction(OP_SUB))               # j - n
    prog.append(Instruction(OP_JZ, -1))            # if j == n -> sum_done
    jz_sum_done = len(prog) - 1

    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_I32_LOAD))          # heap[j]
    prog.append(Instruction(OP_LOCAL_GET, 2))      # sum
    prog.append(Instruction(OP_ADD))               # sum + heap[j]
    prog.append(Instruction(OP_LOCAL_SET, 2))      # sum = sum + heap[j]
    # j += 1
    prog.append(Instruction(OP_LOCAL_GET, 1))      # j
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_ADD))               # j+1
    prog.append(Instruction(OP_LOCAL_SET, 1))      # j = j+1
    # unconditional jump
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_JNZ, sum_loop_start))

    # SUM_DONE:
    sum_done = len(prog)
    prog[jz_sum_done] = Instruction(OP_JZ, sum_done)  # patch JZ

    prog.append(Instruction(OP_LOCAL_GET, 2))      # sum
    prog.append(Instruction(OP_HALT))

    return prog, expected


# ─── Sum of Primes ───────────────────────────────────────────────

def make_sum_of_primes(limit: int) -> tuple:
    """Sum all primes up to limit using trial division.

    For each candidate n from 2 to limit:
        is_prime = 1
        for d from 2 while d*d <= n:
            if n % d == 0: is_prime = 0
        if is_prime: total += n

    Locals:
        local[0] = candidate (n)
        local[1] = divisor   (d)
        local[2] = is_prime  (0 or 1)
        local[3] = total

    Note: once is_prime is set to 0 the inner loop continues iterating but
    skips the divisibility check (early-exit via is_prime flag guard).
    """
    # Expected result (Python reference)
    expected = 0
    for candidate in range(2, limit + 1):
        is_prime = True
        d = 2
        while d * d <= candidate:
            if candidate % d == 0:
                is_prime = False
                break
            d += 1
        if is_prime:
            expected += candidate

    prog = []

    # Init
    prog.append(Instruction(OP_PUSH, 0))
    prog.append(Instruction(OP_LOCAL_SET, 3))      # total = 0
    prog.append(Instruction(OP_PUSH, 2))
    prog.append(Instruction(OP_LOCAL_SET, 0))      # candidate = 2

    # OUTER_LOOP_START (ip=4):
    outer_loop_start = len(prog)                   # = 4

    # if candidate > limit: goto done
    prog.append(Instruction(OP_LOCAL_GET, 0))      # candidate
    prog.append(Instruction(OP_PUSH, limit))       # limit
    prog.append(Instruction(OP_GT_S))              # candidate > limit ?
    prog.append(Instruction(OP_JNZ, -1))           # -> done
    jnz_done = len(prog) - 1

    # is_prime = 1; divisor = 2
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_LOCAL_SET, 2))      # is_prime = 1
    prog.append(Instruction(OP_PUSH, 2))
    prog.append(Instruction(OP_LOCAL_SET, 1))      # divisor = 2

    # INNER_LOOP_START (ip = 4+8 = 12):
    inner_loop_start = len(prog)                   # = 12

    # if d*d > candidate: goto inner_done
    prog.append(Instruction(OP_LOCAL_GET, 1))      # d
    prog.append(Instruction(OP_DUP))               # d, d
    prog.append(Instruction(OP_MUL))               # d*d
    prog.append(Instruction(OP_LOCAL_GET, 0))      # candidate
    prog.append(Instruction(OP_GT_S))              # d*d > candidate ?
    prog.append(Instruction(OP_JNZ, -1))           # -> inner_done
    jnz_inner_done = len(prog) - 1

    # if is_prime == 0: skip divisibility check (goto inner_loop_next)
    prog.append(Instruction(OP_LOCAL_GET, 2))      # is_prime
    prog.append(Instruction(OP_JZ, -1))            # if is_prime == 0 -> inner_loop_next
    jz_inner_loop_next = len(prog) - 1

    # if candidate % divisor == 0: is_prime = 0
    prog.append(Instruction(OP_LOCAL_GET, 0))      # candidate
    prog.append(Instruction(OP_LOCAL_GET, 1))      # divisor
    prog.append(Instruction(OP_REM_S))             # candidate % divisor
    prog.append(Instruction(OP_JNZ, -1))           # if remainder != 0 -> no_div
    jnz_no_div = len(prog) - 1

    prog.append(Instruction(OP_PUSH, 0))
    prog.append(Instruction(OP_LOCAL_SET, 2))      # is_prime = 0

    # INNER_LOOP_NEXT (= NO_DIV):
    inner_loop_next = len(prog)
    prog[jz_inner_loop_next] = Instruction(OP_JZ, inner_loop_next)
    prog[jnz_no_div]         = Instruction(OP_JNZ, inner_loop_next)

    # divisor += 1
    prog.append(Instruction(OP_LOCAL_GET, 1))      # d
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_ADD))               # d+1
    prog.append(Instruction(OP_LOCAL_SET, 1))      # d = d+1
    # unconditional jump to inner_loop_start
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_JNZ, inner_loop_start))

    # INNER_DONE:
    inner_done = len(prog)
    prog[jnz_inner_done] = Instruction(OP_JNZ, inner_done)

    # if is_prime == 0: goto outer_loop_next
    prog.append(Instruction(OP_LOCAL_GET, 2))      # is_prime
    prog.append(Instruction(OP_JZ, -1))            # if is_prime == 0 -> outer_loop_next
    jz_outer_loop_next = len(prog) - 1

    # total += candidate
    prog.append(Instruction(OP_LOCAL_GET, 3))      # total
    prog.append(Instruction(OP_LOCAL_GET, 0))      # candidate
    prog.append(Instruction(OP_ADD))               # total + candidate
    prog.append(Instruction(OP_LOCAL_SET, 3))      # total = total + candidate

    # OUTER_LOOP_NEXT:
    outer_loop_next = len(prog)
    prog[jz_outer_loop_next] = Instruction(OP_JZ, outer_loop_next)

    # candidate += 1
    prog.append(Instruction(OP_LOCAL_GET, 0))      # candidate
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_ADD))               # candidate+1
    prog.append(Instruction(OP_LOCAL_SET, 0))      # candidate = candidate+1
    # unconditional jump to outer_loop_start
    prog.append(Instruction(OP_PUSH, 1))
    prog.append(Instruction(OP_JNZ, outer_loop_start))

    # DONE:
    done = len(prog)
    prog[jnz_done] = Instruction(OP_JNZ, done)

    prog.append(Instruction(OP_LOCAL_GET, 3))      # total
    prog.append(Instruction(OP_HALT))

    return prog, expected


# ─── Benchmark Registry ──────────────────────────────────────────

_fnv1a_prog, _fnv1a_exp   = make_fnv1a(list(range(32)))
_bubble_prog, _bubble_exp = make_bubble_sort(
    [15, 3, 9, 1, 7, 12, 5, 18, 2, 11, 8, 16, 4, 14, 6, 19, 0, 13, 17, 10]
)
_primes_prog, _primes_exp = make_sum_of_primes(100)

# Benchmark registry: (name, prog, expected, description)
BENCHMARKS = [
    ("fnv1a_32",   _fnv1a_prog,  _fnv1a_exp,
     "FNV-1a hash of bytes 0..31"),
    ("bubble_20",  _bubble_prog, _bubble_exp,
     "bubble sort 20-element array"),
    ("primes_100", _primes_prog, _primes_exp,
     "sum of primes up to 100"),
]
