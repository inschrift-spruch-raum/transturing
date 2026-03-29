# ISA 参考手册

55 条指令的完整参考文档。该指令集架构 (ISA) 是一个基于栈的虚拟机，以 WebAssembly 的 i32 指令子集为蓝本。

## 概述

这套 ISA 驱动着整个"transformer 即计算机"系统。所有指令都被**编译进 transformer 的权重矩阵**，而不是由外部解释器逐条执行。前馈层负责指令分发，注意力头则通过抛物线键编码 (parabolic key encoding) 完成内存寻址。每一步执行本质上就是一次 dot product 加一次 argmax，跟标准 transformer 推理中的注意力机制完全一致。

### 栈机器架构

这套 ISA 采用经典的栈机器 (stack machine) 设计。大多数指令从栈顶弹出操作数，计算后将结果压回栈顶。没有寄存器，没有显式的操作数编码。好处是编译器前端实现简单，坏处是指令序列会比寄存器机器更长。

### 五个内存空间

指令通过不同的注意力头访问五个独立的内存空间：

1. **程序内存**：取指令和参数
2. **栈内存**：读取 SP、SP-1、SP-2 位置的值
3. **局部变量**：LOCAL.GET / LOCAL.SET / LOCAL.TEE
4. **堆 / 线性内存**：I32.LOAD / STORE 及字节/短整数变体
5. **调用帧**：CALL / RETURN 保存返回地址和栈指针

### 关键语义

- **i32 溢出处理**：所有算术运算结果执行 `result & 0xFFFFFFFF`，遵循 WebAssembly 标准。例如 `PUSH 0xFFFFFFFF; PUSH 1; ADD` 结果为 `0`。
- **TRAP 运行时错误**：除以零、栈下溢等错误不会抛出 Python 异常，而是发出 OP_TRAP（操作码 99）。
- **有符号/无符号区分**：i32 的 32 位值可以同时表示有符号和无符号数。DIV_S / REM_S 按二进制补码解释，DIV_U / REM_U 将操作数视为非负整数。

## 指令表

以下 7 个分类覆盖全部 55 条指令。

### 1. Stack Operations

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 1 | PUSH *n* | Push immediate value *n* onto stack | Loading constants and literals |
| 2 | POP | Discard top of stack | Cleaning up temporary values |
| 4 | DUP | Duplicate top of stack | Reusing a value without recomputing it |
| 10 | SWAP | Swap top two stack elements | Reordering operands for non-commutative ops |
| 11 | OVER | Copy second element to top | Accessing a value below top without destructive pop |
| 12 | ROT | Rotate top three elements (a b c → b c a) | Three-value shuffling, e.g. Fibonacci iteration |
| 42 | SELECT | Pop three; push b if c≠0, else a | Branchless conditional — like C's ternary `?:` |

### 2. Arithmetic

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

### 3. Comparison

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

### 4. Bitwise

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

### 5. Control Flow

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 5 | HALT | Stop execution | Program termination |
| 7 | JZ *addr* | Jump if top == 0 | Conditional branch — `if (!x) goto addr` |
| 8 | JNZ *addr* | Jump if top ≠ 0 | Conditional branch — `if (x) goto addr` |
| 9 | NOP | No operation | Alignment padding, jump targets |
| 54 | CALL *addr* | Push return address, jump to addr | Function call |
| 55 | RETURN | Pop return address, jump back | Function return |
| 99 | TRAP | Illegal operation (runtime error) | Error signaling — division by zero, stack underflow |

### 6. Local Variables

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 43 | LOCAL.GET *i* | Push local variable *i* onto stack | Reading a named variable |
| 44 | LOCAL.SET *i* | Pop stack into local variable *i* | Writing a named variable |
| 45 | LOCAL.TEE *i* | Copy top of stack into local *i* (no pop) | Write + keep — avoids DUP before SET |

### 7. Linear Memory

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

## 相关文档

- [opcodes.md](opcodes.md) — 各指令的详细说明与使用示例
- [../architecture/overview.md](../architecture/overview.md) — 系统架构与核心概念
