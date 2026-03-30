# 快速开始

5 分钟从零到运行你的第一个 transformer 执行程序。

## 前置条件

- Python 3.12+
- Git

## 安装

```bash
git clone https://github.com/oaustegard/llm-as-computer.git
cd llm-as-computer
uv sync
```

> 鉤完倉後執行 `uv sync` 安裝所有依賴到 `.venv`。

## 运行第一个程序

这段程序把 3 和 5 压入栈, 相加, 然后停机。每一行都是一条指令, 每一条指令的获取都是一次注意力操作。

```python
from isa import Instruction, OP_PUSH, OP_ADD, OP_HALT
from executor import NumPyExecutor

prog = [
    Instruction(OP_PUSH, 3),
    Instruction(OP_PUSH, 5),
    Instruction(OP_ADD),
    Instruction(OP_HALT),
]

trace = NumPyExecutor().execute(prog)
print(trace.format_trace())
```

把上面的代码保存为 `first.py`, 然后在项目根目录运行:

```bash
uv run first.py
```

## 预期输出

执行跟踪会显示 4 个步骤。栈的最终状态应该只有 `8`, 即 3 + 5 的结果:

```
Step 0: OP_PUSH 3    stack=[3]
Step 1: OP_PUSH 5    stack=[3, 5]
Step 2: OP_ADD        stack=[8]
Step 3: OP_HALT       stack=[8]
```

每一步背后, transformer 都在用抛物线注意力头完成指令获取和内存读取。这不是模拟, 是真的在跑注意力计算。

## 发生了什么

简单说:

1. `OP_PUSH 3` 把 3 放到栈上
2. `OP_PUSH 5` 把 5 放到栈上
3. `OP_ADD` 弹出栈顶两个值, 相加, 结果压回栈
4. `OP_HALT` 停机

栈机执行, 55 个操作码, 编译进 transformer 权重。没有解释器, 没有 JIT, 全是矩阵运算。

## 下一步

- **[工作原理](guides/how-it-works.md)**, 看这 4 条指令在注意力层里到底发生了什么
- **[架构概览](architecture/overview.md)**, 了解抛物线编码和五大内存空间
- **[程序编写](guides/writing-programs.md)**, 学着写更复杂的程序 (循环、函数调用、堆操作)
- **[完整 ISA 参考](isa/index.md)**, 查阅全部 55 个操作码

[返回文档首页](README.md)
