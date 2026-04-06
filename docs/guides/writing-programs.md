# 程序编写指南

本文介绍如何为 transturing 执行器编写程序。涵盖三种方式: 直接构造指令列表、使用结构化汇编器、以及导入二进制 `.wasm` 模块。

如果你的起点是 WebAssembly 文件, 当前主路径是直接走 `.wasm -> compile_wasm() -> 既有 lowering -> ISA`。这里提到的 WebAssembly 支持范围只限于当前已验证的 i32 子集。

关于执行器内部如何运行这些程序 (注意力头、抛物线编码等), 请参考 [工作原理](how-it-works.md)。完整的 55 个操作码定义见 [ISA 参考](../isa/index.md)。

## 基本概念

执行器是一个栈机 (stack machine), 指令序列由 `Instruction` 对象组成。每个 `Instruction` 有操作码和参数两个字段:

```python
from transturing.core.isa import Instruction, OP_PUSH, OP_ADD, OP_HALT

instr = Instruction(OP_PUSH, 42)   # 操作码=1, 参数=42
```

程序执行时维护以下状态:

- **栈** (stack): PUSH 压入, ADD/SUB 等弹出操作数并压入结果
- **局部变量** (locals): LOCAL.GET/SET/TEE 按索引读写
- **堆内存** (heap): I32.LOAD/STORE 按地址读写
- **调用栈** (call stack): CALL/RETURN 管理函数调用

程序必须以 `HALT` 结束。执行到 HALT 时, 栈顶值即为程序结果。

---

## 方式一: 直接构造指令

最直接的方式是手写 `Instruction` 列表。适合简单程序和学习 ISA。

### 示例 1: 基本算术 (3 + 5)

```python
from transturing.core.isa import Instruction, OP_PUSH, OP_ADD, OP_HALT
from transturing.backends.numpy_backend import NumPyExecutor

prog = [
    Instruction(OP_PUSH, 3),    # 栈: [3]
    Instruction(OP_PUSH, 5),    # 栈: [3, 5]
    Instruction(OP_ADD),        # 弹出 5 和 3, 压入 8; 栈: [8]
    Instruction(OP_HALT),       # 结果: 8
]

trace = NumPyExecutor().execute(prog)
print(trace.steps[-1].top)      # 输出: 8
```

执行过程: PUSH 3 压入 3, PUSH 5 压入 5, ADD 弹出两个值相加得 8, HALT 返回栈顶 8。

### 示例 2: 循环倒数 (用 JNZ 反复减 1)

```python
from transturing.core.isa import Instruction, OP_PUSH, OP_DUP, OP_SUB, OP_JNZ, OP_HALT
from transturing.backends.numpy_backend import NumPyExecutor

# 从 3 倒数到 0: 3 -> 2 -> 1 -> 0
prog = [
    Instruction(OP_PUSH, 3),    # 0: 栈: [3]
    # -- 循环入口 (addr 1) --
    Instruction(OP_DUP),        # 1: 复制栈顶; 栈: [3, 3]
    Instruction(OP_PUSH, 1),    # 2: 栈: [3, 3, 1]
    Instruction(OP_SUB),        # 3: 3 - 1 = 2; 栈: [3, 2]
    Instruction(OP_DUP),        # 4: 栈: [3, 2, 2]
    Instruction(OP_JNZ, 1),     # 5: 2 != 0 → 跳回 addr 1
    Instruction(OP_HALT),       # 6: 栈顶 = 0
]

trace = NumPyExecutor().execute(prog)
print(trace.steps[-1].top)      # 输出: 0
```

循环逻辑: DUP 复制当前值用于减法, SUB 减 1 后 DUP 再次复制用于 JNZ 判断。JNZ 弹出条件值, 非 0 则跳转到指定地址。当值减到 0 时, JNZ 不跳转, 落到 HALT。

也可以用 `isa.program()` 辅助函数代替手动创建 `Instruction` 对象:

```python
from transturing.core.isa import program

prog = program(
    ("PUSH", 3),
    ("PUSH", 5),
    ("ADD",),
    ("HALT",),
)
```

`program()` 接受可变参数, 每个参数是一个元组, 元组第一个元素是操作码名称字符串。它会自动查找对应的操作码常量。

---

## 方式二: 结构化汇编器

直接用地址跳转容易出错。`assembler.py` 提供了 WASM 风格的结构化控制流, 编译器自动计算跳转目标:

```python
from transturing.core.assembler import compile_structured
from transturing.backends.numpy_backend import NumPyExecutor
```

支持的结构化指令:

| 指令 | 作用 |
|------|------|
| `('BLOCK',)` | 代码块, BR 退出到 END |
| `('LOOP',)` | 循环块, BR 跳回开头 |
| `('IF',)` / `('ELSE',)` / `('END',)` | 条件分支 |
| `('BR', n)` | 无条件跳出 n 层 |
| `('BR_IF', n)` | 条件跳出 n 层 |
| `('BR_TABLE', labels, default)` | switch 分支 |

非控制流指令直接透传给底层, 格式与 `program()` 相同。

### 示例 3: 用汇编器写倒数循环

```python
from transturing.core.assembler import compile_structured
from transturing.backends.numpy_backend import NumPyExecutor

prog = compile_structured([
    ('PUSH', 5),          # 初始值
    ('LOOP',),            # 循环起点
      ('PUSH', 1),        #
      ('SUB',),           # 减 1
      ('DUP',),           # 复制用于判断
      ('BR_IF', 0),       # 栈顶非 0 → 跳回 LOOP 起点
    ('END',),
    ('HALT',),
])

trace = NumPyExecutor().execute(prog)
print(trace.steps[-1].top)  # 输出: 0
```

关键区别: `BR_IF 0` 表示跳出 0 层 (即最内层) 循环块。对于 LOOP, BR/BR_IF 跳回循环开头; 对于 BLOCK, BR/BR_IF 跳到块末尾。

### 示例 4: 带函数调用的阶乘

CALL/RETURN 支持递归函数调用。下面是使用 LOCAL 变量和 CALL 的阶乘:

```python
from transturing.core.isa import Instruction, OP_PUSH, OP_DUP, OP_JZ, OP_SUB, OP_MUL
from transturing.core.isa import OP_LOCAL_SET, OP_LOCAL_GET, OP_CALL, OP_RETURN, OP_HALT
from transturing.backends.numpy_backend import NumPyExecutor

# 阶乘: fact(5) = 120
# 使用迭代循环 + LOCAL 变量
prog = [
    Instruction(OP_PUSH, 1),        # 0: result = 1
    Instruction(OP_PUSH, 5),        # 1: counter = 5
    # -- 循环 (addr 2) --
    Instruction(OP_DUP),            # 2: 复制 counter
    Instruction(OP_JZ, 12),         # 3: counter == 0 → 结束
    Instruction(OP_DUP),            # 4
    Instruction(OP_PUSH, 1),        # 5
    Instruction(OP_SUB),            # 6: counter - 1
    # result *= (counter - 1 + 1) = counter
    # 用 ROT 把 result 拿到栈顶
    # 交换位置: [result, counter] → 计算 result * counter
    # 这里简化为: ROT; MUL; SWAP; SUB; 循环
    # 实际 programs.py 中的 make_factorial 有完整实现
    Instruction(OP_HALT),           # 7 (简化版)
]

# 推荐直接使用 make_factorial 生成器:
from transturing.core.programs import make_factorial
prog, expected = make_factorial(5)  # expected = 120

trace = NumPyExecutor().execute(prog)
print(trace.steps[-1].top)          # 输出: 120
```

对于递归阶乘, 可以用 CALL/RETURN 实现:

```python
from transturing.core.isa import Instruction
from transturing.core.isa import OP_PUSH, OP_CALL, OP_RETURN, OP_HALT, OP_LOCAL_GET
from transturing.core.isa import OP_LOCAL_SET, OP_SUB, OP_MUL, OP_DUP, OP_NOP
from transturing.backends.numpy_backend import NumPyExecutor

# 递归 fact(n): n <= 1 返回 1, 否则 n * fact(n-1)
prog = [
    # -- main --
    Instruction(OP_PUSH, 5),        # 0: 参数 n=5
    Instruction(OP_CALL, 3),        # 1: 调用 fact
    Instruction(OP_HALT),           # 2

    # -- fact(n) 从 addr 3 开始 --
    # 参数 n 通过栈传入, 保存到 local[0]
    Instruction(OP_LOCAL_SET, 0),   # 3: local[0] = n
    Instruction(OP_LOCAL_GET, 0),   # 4: push n
    Instruction(OP_PUSH, 1),        # 5
    Instruction(OP_SUB),            # 6: n - 1
    Instruction(OP_DUP),            # 7: 复制用于判断
    # 这里简化展示; 完整实现需要 LT_S 或条件跳转
    # 实际完整版见 programs.py 中的 make_factorial()
    Instruction(OP_RETURN),         # 8
]

trace = NumPyExecutor().execute(prog)
```

完整可运行的递归实现较长, 因为需要处理基本情况 (n<=1) 和两次递归调用。项目中有现成的 `make_factorial()` 生成器, 使用 MUL + 循环实现, 推荐优先使用。

### 示例 5: 冒泡排序 (堆内存)

I32.LOAD 和 I32.STORE 操作堆内存。下面的程序对数组 `[3, 1, 2]` 排序, 结果 `[1, 2, 3]`:

```python
from transturing.core.isa import Instruction
from transturing.core.isa import (OP_PUSH, OP_HALT, OP_ADD, OP_SUB, OP_DUP, OP_JZ, OP_JNZ,
                 OP_LOCAL_SET, OP_LOCAL_GET,
                 OP_I32_LOAD, OP_I32_STORE, OP_GT_S)
from transturing.backends.numpy_backend import NumPyExecutor

# 对 [3, 1, 2] 冒泡排序, 读 mem[0] 应得到 1
prog = [
    # 初始化内存: addr=0→3, addr=1→1, addr=2→2
    Instruction(OP_PUSH, 0), Instruction(OP_PUSH, 3), Instruction(OP_I32_STORE),
    Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 1), Instruction(OP_I32_STORE),
    Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 2), Instruction(OP_I32_STORE),

    # LOCAL[0] = pass (外层), LOCAL[1] = j (内层)
    # LOCAL[2] = mem[j], LOCAL[3] = mem[j+1]
    Instruction(OP_PUSH, 0),
    Instruction(OP_LOCAL_SET, 0),      # pass = 0
    # ... 外层和内层循环省略 (完整版 62 条指令)
    # 完整版见 programs.py 中的 make_bubble_sort()

    # 结果: 读 mem[0]
    Instruction(OP_PUSH, 0),
    Instruction(OP_I32_LOAD),
    Instruction(OP_HALT),
]

# 推荐直接用测试中的完整版本
```

冒泡排序完整版约 62 条指令。核心思路:

1. `I32.STORE` 写入: 栈顶是地址, 下面是值。`PUSH addr; PUSH val; I32.STORE`
2. `I32.LOAD` 读取: 栈顶是地址。`PUSH addr; I32.LOAD` 把值压栈
3. 用 LOCAL 变量暂存 mem[j] 和 mem[j+1], 避免栈操作混乱

---

## 方式三: 导入二进制 `.wasm`

如果你的程序已经来自 C 编译器或其他 WebAssembly 工具链, 推荐直接导入二进制 `.wasm` 模块。当前公开 API 暴露了二进制解析和编译入口:

```python
from pathlib import Path

from transturing import compile_wasm, parse_wasm_binary, parse_wasm_file
from transturing.backends.numpy_backend import NumPyExecutor
```

### 基本用法

```python
wasm_bytes = Path("add.wasm").read_bytes()
prog = compile_wasm(wasm_bytes, func_name="add")

trace = NumPyExecutor().execute(prog)
print(trace.steps[-1].top)
```

### 检查模块结构

如果你需要先查看导出的函数、内存或函数体结构, 可以先解析模块再选择入口函数:

```python
module = parse_wasm_file("add.wasm")
print([export.name for export in module.exports])

same_module = parse_wasm_binary(Path("add.wasm").read_bytes())
prog = compile_wasm(same_module, func_name="add")
```

这些入口都面向当前已验证的 i32 子集。binary frontend 会把 `.wasm` 模块解码为内部 Wasm 指令表示, 然后复用同一套结构化控制流映射与 ISA lowering 语义。

---

## 使用程序生成器

`programs.py` 提供了一系列 `make_*` 函数, 自动生成常见算法的指令序列:

```python
from transturing.core.programs import make_fibonacci, make_factorial, make_gcd, make_multiply
from transturing.backends.numpy_backend import NumPyExecutor

# Fibonacci: fib(10) = 55
prog, expected = make_fibonacci(10)
trace = NumPyExecutor().execute(prog)
assert trace.steps[-1].top == expected

# 阶乘: 5! = 120
prog, expected = make_factorial(5)

# 最大公约数: gcd(12, 8) = 4
prog, expected = make_gcd(12, 8)

# 乘法 (重复加法): 7 * 8 = 56
prog, expected = make_multiply(7, 8)
```

所有生成器返回 `(prog, expected_value)` 元组, `expected_value` 是预期的栈顶结果。

---

## 测试程序

### 用 compare_traces() 验证

`isa.py` 提供了 `compare_traces()` 函数, 对比两次执行的每一步是否完全一致。这是项目中最核心的验证方式: NumPy 执行器和 PyTorch 执行器必须产生完全相同的 trace。

```python
from transturing.core.isa import compare_traces
from transturing.backends.numpy_backend import NumPyExecutor
from transturing.backends.torch_backend import TorchExecutor

prog, expected = make_fibonacci(10)

np_trace = NumPyExecutor().execute(prog)
pt_trace = TorchExecutor().execute(prog)

# 对比每一步的状态 (IP、栈、局部变量等)
match, detail = compare_traces(np_trace, pt_trace)
assert match, f"Traces differ: {detail}"

# 验证结果
assert np_trace.steps[-1].top == expected
assert pt_trace.steps[-1].top == expected
```

### 查看执行跟踪

`trace.format_trace()` 输出每一步的详细状态:

```python
trace = NumPyExecutor().execute(prog)
print(trace.format_trace())
```

输出包含每条指令执行后的 IP、操作码、栈内容、局部变量等信息。

### 常用测试模式

项目中的测试遵循固定模式:

```python
from transturing.core.isa import Instruction, compare_traces
from transturing.backends.numpy_backend import NumPyExecutor
from transturing.backends.torch_backend import TorchExecutor
from transturing.core.programs import make_factorial

def test_my_program():
    prog = [
        Instruction(OP_PUSH, 42),
        Instruction(OP_HALT),
    ]
    expected = 42

    np_exec = NumPyExecutor()
    pt_exec = TorchExecutor()

    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)

    # 1. 两个执行器 trace 必须完全一致
    match, detail = compare_traces(np_trace, pt_trace)
    assert match, detail

    # 2. 结果必须符合预期
    assert np_trace.steps[-1].top == expected
    assert pt_trace.steps[-1].top == expected

    print("PASS")
```

---

## i32 溢出语义

所有算术运算遵循 WASM 的 i32 溢出规则: 结果会与 `0xFFFFFFFF` 做按位与。例如 `0xFFFFFFFF + 1 = 0`。除以零会触发 TRAP (操作码 99), 不是 Python 异常。

```python
# 溢出示例
from transturing.core.isa import Instruction, OP_PUSH, OP_ADD, OP_HALT, MASK32
from transturing.backends.numpy_backend import NumPyExecutor

prog = [
    Instruction(OP_PUSH, 0xFFFFFFFF),
    Instruction(OP_PUSH, 1),
    Instruction(OP_ADD),
    Instruction(OP_HALT),
]

trace = NumPyExecutor().execute(prog)
print(trace.steps[-1].top)    # 输出: 0 (不是 0x100000000)
```

---

## 进一步阅读

- [工作原理](how-it-works.md): 追踪一个 4 指令程序的完整执行过程
- [ISA 参考](../isa/index.md): 完整的 55 个操作码分类索引
- [操作码详解](../isa/opcodes.md): 每个操作码的语义和参数说明
- [架构概览](../architecture/overview.md): 系统设计和关键概念
- [项目 README](../../README.md): 项目总览和文件结构
