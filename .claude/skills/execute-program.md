# Skill: Execute Program on Compiled Transformer Stack Machine

## Description
Execute programs on the LLM-as-Computer compiled transformer stack machine. Takes programs in assembly format, runs them via the NumPyExecutor (or Mojo binary if available), and returns formatted execution traces.

## When to Use
- User asks to run/execute a program on the stack machine
- User wants to test an algorithm (fibonacci, factorial, GCD, etc.)
- User asks "what does this program do?" with stack machine instructions
- User says `/execute` or asks to trace a program

## Instructions

### Step 1: Parse the Program

Programs can be specified as:

**Assembly tuples** (most common):
```python
program(("PUSH", 5), ("PUSH", 3), ("ADD",), ("HALT",))
```

**Named algorithms** (use generators from `programs.py`):
- `make_fibonacci(n)` — compute fib(n)
- `make_multiply(a, b)` — compute a*b via repeated addition
- `make_factorial(n)` — compute n!
- `make_gcd(a, b)` — compute GCD via Euclidean algorithm
- `make_power_of_2(n)` — compute 2^n
- `make_sum_1_to_n(n)` — compute 1+2+...+n
- `make_is_even(n)` — check parity
- `make_native_multiply(a, b)` — single MUL instruction
- `make_native_divmod(a, b)` — division + remainder
- `make_compare_binary(op, a, b)` — comparisons (op: "eq","ne","lt_s","gt_s","le_s","ge_s")
- `make_bitwise_binary(op, a, b)` — bitwise (op: "and","or","xor","shl","shr_u","rotl","rotr")
- `make_select(a, b, c)` — conditional select

**Natural language** — translate to assembly:
- "add 5 and 3" → PUSH 5, PUSH 3, ADD, HALT
- "fibonacci of 10" → make_fibonacci(10)
- "is 7 even?" → make_is_even(7)

### Step 2: Execute

Run this Python snippet (adapt program as needed):

```bash
cd /home/user/llm-as-computer && python3 -c "
import sys; sys.path.insert(0, '.')
from isa import Instruction, OP_NAMES, program
from executor import NumPyExecutor

# Build program (EDIT THIS)
prog = program(('PUSH', 5), ('PUSH', 3), ('ADD',), ('HALT',))

# Execute
ex = NumPyExecutor()
trace = ex.execute(prog, max_steps=50000)

# Format output
print(f'Program: {len(prog)} instructions')
for i, instr in enumerate(prog):
    print(f'  [{i}] {instr}')
print(f'\nExecution: {len(trace.steps)} steps')
print(f'{"Step":>5} {"Op":>12} {"Arg":>5} {"SP":>4} {"Top":>8}')
print('-' * 40)
for i, s in enumerate(trace.steps):
    name = OP_NAMES.get(s.op, f'?{s.op}')
    print(f'{i:5d} {name:>12} {s.arg:5d} {s.sp:4d} {s.top:8d}')
print(f'\nResult: {trace.steps[-1].top}')
"
```

For named algorithms, replace the program line:
```python
from programs import make_fibonacci
prog, expected = make_fibonacci(10)
# ... rest same ... then add at end:
print(f'Expected: {expected}')
```

### Step 3: Format Results

Present results as:
1. **Program listing** — numbered instructions
2. **Execution trace** — table with step, opcode name, arg, stack pointer, top-of-stack
3. **Result** — final top-of-stack value
4. **Stats** — step count, whether result matches expected (if known)

For long traces (>30 steps), show first 10 + last 10 steps with "... (N steps omitted)" in between.

## Available Opcodes (55 total)

| Category | Opcodes |
|----------|---------|
| Stack | PUSH, POP, DUP, SWAP, OVER, ROT, DROP (=POP), SELECT |
| Arithmetic | ADD, SUB, MUL, DIV_S, DIV_U, REM_S, REM_U |
| Comparison | EQZ, EQ, NE, LT_S, LT_U, GT_S, GT_U, LE_S, LE_U, GE_S, GE_U |
| Bitwise | AND, OR, XOR, SHL, SHR_S, SHR_U, ROTL, ROTR |
| Unary | CLZ, CTZ, POPCNT, ABS, NEG |
| Control | HALT, JZ addr, JNZ addr, NOP, CALL addr, RETURN |
| Locals | LOCAL_GET idx, LOCAL_SET idx, LOCAL_TEE idx |
| Memory | I32_LOAD, I32_STORE, I32_LOAD8_U/S, I32_LOAD16_U/S, I32_STORE8, I32_STORE16 |

## Notes
- Ensure numpy and torch are installed: `uv pip install numpy torch --system`
- The executor uses parabolic key addressing (same as the compiled transformer)
- Max 50,000 steps by default (prevents infinite loops)
- Division by zero produces OP_TRAP
- All arithmetic is 32-bit signed integer (i32) with wrapping
