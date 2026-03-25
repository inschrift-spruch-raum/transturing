# llm-as-computer

A compiled transformer executor that runs programs inside a transformer's own inference loop. Each instruction fetch and memory read is a parabolic attention head — no external interpreter, no tool use. The transformer *is* the computer.

Built to independently validate [Percepta's claim](https://percepta.ai/blog/can-llms-be-computers) that transformers can execute arbitrary programs via 2D convex hull attention with O(log t) per-step decoding.

## Blog Posts

- **[Yes, LLMs Can Be Computers. Now What?](https://muninn.austegard.com/blog/yes-llms-can-be-computers-now-what)** — Full narrative of the 13-phase validation, including a productive wrong turn through training.
- **[The Free Computer: Why Offloading to CPU Is a Win for Everyone](https://muninn.austegard.com/blog/the-free-computer-why-offloading-to-cpu-is-a-win-for-everyone)** — The economic argument for compiled CPU execution.

## Benchmark Results

Million-step benchmarks validating the executor at scale: [Issue #52](https://github.com/oaustegard/llm-as-computer/issues/52#issuecomment-2752773503). Key numbers: Mojo backend at **67–126M steps/sec**, Python at **2.1–3.1M steps/sec**, 1.2M steps in 17ms (Mojo) or 561ms (Python).

## ISA Reference

The executor implements a 55-opcode stack machine ISA, modeled on WebAssembly's i32 instruction subset. All operations are compiled into transformer weight matrices — the feed-forward layers dispatch opcodes, and attention heads handle memory addressing via parabolic key encoding.

### Stack Operations

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 1 | PUSH *n* | Push immediate value *n* onto stack | Loading constants and literals |
| 2 | POP | Discard top of stack | Cleaning up temporary values |
| 4 | DUP | Duplicate top of stack | Reusing a value without recomputing it |
| 10 | SWAP | Swap top two stack elements | Reordering operands for non-commutative ops |
| 11 | OVER | Copy second element to top | Accessing a value below top without destructive pop |
| 12 | ROT | Rotate top three elements (a b c → b c a) | Three-value shuffling, e.g. Fibonacci iteration |
| 42 | SELECT | Pop three; push b if c≠0, else a | Branchless conditional — like C's ternary `?:` |

### Arithmetic

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 3 | ADD | a + b | Integer addition |
| 6 | SUB | a − b | Integer subtraction |
| 13 | MUL | a × b | Integer multiplication |
| 14 | DIV_S | Signed division (truncated toward zero) | C-style signed `/` |
| 15 | DIV_U | Unsigned division | Unsigned `/` — treats operands as non-negative |
| 16 | REM_S | Signed remainder | C-style signed `%` |
| 17 | REM_U | Unsigned remainder | Unsigned `%` |
| 40 | ABS | Absolute value | `abs(x)` — flips sign if negative |
| 41 | NEG | Negate | Unary minus: `0 - x` |

### Comparison

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 18 | EQZ | Push 1 if top == 0, else 0 | Zero-test for conditionals and loop termination |
| 19 | EQ | Push 1 if a == b | Equality comparison |
| 20 | NE | Push 1 if a ≠ b | Inequality comparison |
| 21 | LT_S | Signed less-than | Signed `<` — respects two's complement sign |
| 22 | LT_U | Unsigned less-than | Unsigned `<` — treats values as non-negative |
| 23 | GT_S | Signed greater-than | Signed `>` |
| 24 | GT_U | Unsigned greater-than | Unsigned `>` |
| 25 | LE_S | Signed less-or-equal | Signed `<=` |
| 26 | LE_U | Unsigned less-or-equal | Unsigned `<=` |
| 27 | GE_S | Signed greater-or-equal | Signed `>=` |
| 28 | GE_U | Unsigned greater-or-equal | Unsigned `>=` |

### Bitwise

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 29 | AND | Bitwise AND | Masking bits, flag testing |
| 30 | OR | Bitwise OR | Setting bits, combining flags |
| 31 | XOR | Bitwise exclusive OR | Toggling bits, simple checksums |
| 32 | SHL | Shift left | Multiply by power of 2, bit packing |
| 33 | SHR_S | Arithmetic shift right | Signed divide by power of 2 (preserves sign) |
| 34 | SHR_U | Logical shift right | Unsigned divide by power of 2 |
| 35 | ROTL | Rotate left | Circular bit rotation — used in hash functions |
| 36 | ROTR | Rotate right | Circular bit rotation |
| 37 | CLZ | Count leading zeros | Fast log₂, priority encoding |
| 38 | CTZ | Count trailing zeros | Finding lowest set bit |
| 39 | POPCNT | Population count | Counting set bits — Hamming weight |

### Control Flow

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 5 | HALT | Stop execution | Program termination |
| 7 | JZ *addr* | Jump if top == 0 | Conditional branch — `if (!x) goto addr` |
| 8 | JNZ *addr* | Jump if top ≠ 0 | Conditional branch — `if (x) goto addr` |
| 9 | NOP | No operation | Alignment padding, jump targets |
| 54 | CALL *addr* | Push return address, jump to addr | Function call |
| 55 | RETURN | Pop return address, jump back | Function return |
| 99 | TRAP | Illegal operation (runtime error) | Error signaling — division by zero, stack underflow |

### Local Variables

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 43 | LOCAL.GET *i* | Push local variable *i* onto stack | Reading a named variable |
| 44 | LOCAL.SET *i* | Pop stack into local variable *i* | Writing a named variable |
| 45 | LOCAL.TEE *i* | Copy top of stack into local *i* (no pop) | Write + keep — avoids DUP before SET |

### Linear Memory

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 46 | I32.LOAD | Load 32-bit value from heap address | Reading from heap memory (arrays, structs) |
| 47 | I32.STORE | Store 32-bit value to heap address | Writing to heap memory |
| 48 | I32.LOAD8_U | Load byte, zero-extend to 32 bits | Reading unsigned bytes from memory |
| 49 | I32.LOAD8_S | Load byte, sign-extend to 32 bits | Reading signed bytes |
| 50 | I32.LOAD16_U | Load 16-bit value, zero-extend | Reading unsigned shorts |
| 51 | I32.LOAD16_S | Load 16-bit value, sign-extend | Reading signed shorts |
| 52 | I32.STORE8 | Truncate to byte and store | Writing bytes to memory |
| 53 | I32.STORE16 | Truncate to 16 bits and store | Writing shorts to memory |

## Files

### Core
```
isa.py                  ISA definition: 55 opcodes, types, embedding layout
executor.py             NumPy + PyTorch compiled transformer executors
programs.py             Test programs and algorithm generators
assembler.py            WASM-style structured control flow → flat ISA compiler
wat_parser.py           WebAssembly text format parser
c_pipeline.py           C → WAT → ISA compilation pipeline
test_consolidated.py    Integration tests (NumPy/PyTorch equivalence)
test_wat_parser.py      WAT parser test suite
```

### Mojo Backend (`src/`)
```
executor.mojo           Mojo port of the full 55-opcode executor
benchmark.py            Mojo vs NumPy micro-benchmarks
benchmarks.py           Substantial benchmark programs (FNV-1a, bubble sort, primes)
llm_vs_native.py        Honest comparison: LLM-executor vs native Python
run_mojo_tests.py       Mojo executor test runner
```

### Development (`dev/`)
```
phases/                 Phase exploration scripts (1–20) and result JSON
FINDINGS.md             Detailed per-phase findings
RD-PLAN.md              Original R&D plan and evolution
benchmark_scaling.py    Million-step scaling benchmarks (Issue #52)
```

### Other
```
WRITEUP.md              → Links to published blog posts
CLAUDE.md               Project instructions for Claude Code
viz/                    React visualizations
```
