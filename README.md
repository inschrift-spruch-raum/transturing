# transturing

一个**编译型 transformer 执行器**：程序直接运行在 transformer 自身的推理循环中。每一次取指和每一次内存读取，都是一个抛物线注意力头——没有外部解释器，也没有工具调用。transformer **就是** 这台计算机。

本项目用于独立验证 [Percepta 的论断](https://percepta.ai/blog/can-llms-be-computers)：transformer 能否通过 2D 凸包注意力，以每步 `O(log t)` 的复杂度执行任意程序。

## 相关博文

- **[Yes, LLMs Can Be Computers. Now What?](https://muninn.austegard.com/blog/yes-llms-can-be-computers-now-what)** —— 13 个阶段验证过程的完整叙述，包括一次虽走弯路但很有价值的训练尝试。
- **[The Free Computer: Why Offloading to CPU Is a Win for Everyone](https://muninn.austegard.com/blog/the-free-computer-why-offloading-to-cpu-is-a-win-for-everyone)** —— 关于编译型 CPU 执行路径的经济学论证。

## 工作原理

如果你是第一次接触这个项目，建议先看 **[How It Works](docs/guides/how-it-works.md)**。它会逐步演示一个 4 指令程序的执行过程，说明为什么这不是“套了壳的普通解释器”。

简短版本是：每一次内存读取，本质上都是一次点积加一次 `argmax`——也就是 transformer 在日常推理中本来就在做的那套注意力机制。

## 文档

完整文档位于 [docs/](docs/) 目录：

- **[Quick Start](docs/quickstart.md)** —— 5 分钟跑起来
- **[How It Works](docs/guides/how-it-works.md)** —— 4 指令程序的逐步执行演示
- **[Architecture](docs/architecture/overview.md)** —— 系统设计与核心概念
- **[ISA Reference](docs/isa/index.md)** —— 完整的 55 操作码参考
- **[Development](docs/development/findings-summary.md)** —— 研究发现与研发路线

## 基准结果

百万步级别的执行基准见：[Issue #52](https://github.com/oaustegard/transturing/issues/52#issuecomment-2752773503)。

关键数据：Python 执行速度 **2.1–3.1M steps/sec**，120 万步执行时间约 561ms。

## ISA 参考

该执行器实现了一套 55 操作码的栈机 ISA，以 WebAssembly 的 i32 指令子集为蓝本。

更准确地说：**通用执行器**的分发与寻址机制通过解析方式固定到模型结构中；而**具体程序**会被编译为 ISA 指令序列，再由这个执行器运行。前馈层负责操作码分发，注意力头则通过抛物线键编码完成内存寻址。

当前主文档里的程序导入路径是 `.wasm -> compile_wasm() -> 既有 lowering -> ISA`。支持范围以当前已验证的 i32 子集为准；对外记录的 WebAssembly 工作流只描述这个二进制 `.wasm` 主路径。

### 栈操作

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 1 | PUSH *n* | Push immediate value *n* onto stack | Loading constants and literals |
| 2 | POP | Discard top of stack | Cleaning up temporary values |
| 4 | DUP | Duplicate top of stack | Reusing a value without recomputing it |
| 10 | SWAP | Swap top two stack elements | Reordering operands for non-commutative ops |
| 11 | OVER | Copy second element to top | Accessing a value below top without destructive pop |
| 12 | ROT | Rotate top three elements (a b c → b c a) | Three-value shuffling, e.g. Fibonacci iteration |
| 42 | SELECT | Pop three; push b if c≠0, else a | Branchless conditional — like C's ternary `?:` |

### 算术

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

### 比较

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

### 位运算

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

### 控制流

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 5 | HALT | Stop execution | Program termination |
| 7 | JZ *addr* | Jump if top == 0 | Conditional branch — `if (!x) goto addr` |
| 8 | JNZ *addr* | Jump if top ≠ 0 | Conditional branch — `if (x) goto addr` |
| 9 | NOP | No operation | Alignment padding, jump targets |
| 54 | CALL *addr* | Push return address, jump to addr | Function call |
| 55 | RETURN | Pop return address, jump back | Function return |
| 99 | TRAP | Illegal operation (runtime error) | Error signaling — division by zero, stack underflow |

### 局部变量

| Opcode | Name | Description | Computing context |
|--------|------|-------------|-------------------|
| 43 | LOCAL.GET *i* | Push local variable *i* onto stack | Reading a named variable |
| 44 | LOCAL.SET *i* | Pop stack into local variable *i* | Writing a named variable |
| 45 | LOCAL.TEE *i* | Copy top of stack into local *i* (no pop) | Write + keep — avoids DUP before SET |

### 线性内存

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

## 文件结构

### Core (`src/transturing/core/`)
```text
core/isa.py            ISA 定义：55 个操作码、DIM 常量、数学辅助函数、Trace 类型
core/abc.py            ExecutorBackend 抽象基类
core/registry.py       后端发现（get_executor, list_backends）
core/programs.py       测试程序与算法生成器
core/assembler.py      WASM 风格结构化控制流 → 扁平 ISA 编译器
core/c_pipeline.py     C → .wasm → ISA 主编译流程
```

### Backends (`src/transturing/backends/`)
```text
backends/numpy_backend.py  NumPyExecutor（参考/演示实现）
backends/torch_backend.py  CompiledAttentionHead、TokenVocab、CompiledModel、TorchExecutor
```

### 用法
```python
# 顶层便捷导入（推荐）：
from transturing import program, get_executor, Instruction

# 直接模块导入：
from transturing.core.isa import OP_PUSH, OP_ADD, OP_HALT

# 后端专用：
from transturing.backends.torch_backend import TorchExecutor, CompiledModel

# 通过注册表使用（推荐）：
exec_np = get_executor('numpy')
exec_pt = get_executor('torch')
```

### 测试
```text
tests/test_consolidated.py    执行器正确性 + 双后端一致性测试
```

### 其他
```text
docs/                   完整文档（quickstart、architecture、ISA 参考等）
AGENTS.md               面向 OpenCode 的项目级代理说明
```
