# 贡献指南

感谢你对 **LLM-as-Computer** 项目的兴趣。这是一个编译型 transformer 执行器的研究项目，有一些不太常见的约定。开始之前请先阅读本指南。

完整的项目介绍见 [README.md](../README.md)。

## 开发环境设置

### 基本要求

- **Python 3.12+**（`pyproject.toml` 中指定 `>=3.12`）
- NumPy >= 1.24
- PyTorch >= 2.0

### 安装

```bash
# 克隆仓库后
uv pip install -e ".[dev]" --system

# 或者手动安装核心依赖
uv pip install numpy torch --system
```

### 可选依赖

- **C 编译管道**（`c_pipeline.py`）需要 `clang`（带 wasm32 目标支持）和 `wasm2wat`。缺少这些工具时该模块会抛出 `EnvironmentError`，不影响其他功能。
- **Mojo 后端**（`src/`）需要 Mojo 编译器。

### 验证安装

```bash
# 运行集成测试，确认一切正常
python test_consolidated.py
python test_wat_parser.py
```

## 代码规范

### 项目结构

这是一个扁平模块项目。没有包，没有 `__init__.py`。所有核心模块平铺在根目录：

```
isa.py          ← 55 个操作码、TokenVocab、嵌入函数
executor.py     ← NumPyExecutor、CompiledModel、TorchExecutor
programs.py     ← 测试程序生成器（fib、mul、gcd 等）
assembler.py    ← WASM 风格结构化控制流编译器
wat_parser.py   ← WebAssembly 文本格式解析器
```

导入链：`isa.py` ← `executor.py` ← `programs.py` ← `assembler.py` ← `wat_parser.py` ← `c_pipeline.py`

修改某个模块时注意下游依赖。`test_consolidated.py` 直接导入 `phase14_extended_isa`，改动 phase 文件可能破坏集成测试。

### 浮点精度：强制 Float64

所有编译模型必须使用 **float64**。不允许 float32。

原因：抛物线编码的 score 值按 `addr²` 增长。float32 的精度只能支持约 4000 个索引，float64 可以扩展到 2500 万以上。

```python
# 正确
dtype = torch.float64

# 错误，会导致地址定位失败
dtype = torch.float32
```

### 注意力机制：只允许 Hard-max

永远不要使用 softmax。所有注意力操作使用 argmax（hard-max）。

原因：当 key 值相同时，softmax 会给出均匀权重，无法精确定位目标地址。

### i32 溢出语义

所有算术运算遵循 WebAssembly 标准，结果需做 32 位截断：

```python
result = (a + b) & 0xFFFFFFFF
```

例如 `PUSH 0xFFFFFFFF; PUSH 1; ADD` 的结果是 `0`，不是 `0x100000000`。

### 运行时错误：使用 TRAP

除零、栈下溢等运行时错误应该发出 `OP_TRAP`（操作码 99），而不是抛出 Python 异常。不要用 try/except 包装执行逻辑。

### EPS 精度差异

NumPy 执行器使用 `eps=1e-10`，PyTorch 使用 `EPS=1e-6`（来自 `isa.py`）。这是有意为之的设计，对应不同精度上下文。不要统一这两个值。

### 地址写入的时效偏置

`eps * write_count` 这一项确保后写入同一地址的值会胜出。这是架构特性，不是 bug。不要删除或"修复"它。

### 依赖版本

这是研究仓库，不要锁定精确版本号。使用 `>=` 指定下界：

```toml
# 正确
numpy>=1.24

# 不要这样
numpy==1.24.3
```

### 不要抑制类型错误

不使用 `as any`、`@ts-ignore`、`# type: ignore`。如果有类型问题，修好它。

## 测试要求

### 手写测试框架

本项目不使用 pytest、unittest 或任何测试框架。所有测试都是手写的 `main()` 函数，使用 pass/fail 计数器。

```python
def main():
    passed = 0
    failed = 0
    
    # 你的测试
    if some_condition:
        print(f"  PASS: test description")
        passed += 1
    else:
        print(f"  FAIL: test description")
        failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    main()
```

### 双执行器验证

每条测试必须同时验证 **NumPyExecutor** 和 **TorchExecutor** 的输出一致。使用 `isa.py` 中的 `compare_traces()` 函数进行逐步对比。

```python
from isa import compare_traces

np_trace = np_executor.run(program)
torch_trace = torch_executor.run(program)

if not compare_traces(np_trace, torch_trace):
    print("FAIL: NumPy and PyTorch traces differ")
```

### 运行测试

```bash
# 完整集成测试
python test_consolidated.py
python test_wat_parser.py

# 单个阶段测试（每个文件独立运行）
python dev/phases/phase14_extended_isa.py
python dev/phases/phase17_function_calls.py
```

阶段文件（`phaseN_*.py`）是自包含的，各自带测试工具。可以直接运行。

### 添加新操作码

需要同时更新两个文件：

1. `isa.py`：操作码定义 + TokenVocab
2. `executor.py`：NumPyExecutor 的分派逻辑 + CompiledModel 的编译逻辑

### 添加测试程序

在 `programs.py` 中遵循 `make_*` 命名模式，然后加入测试运行器。

## 提交规范

### 提交信息

简洁描述你做了什么和为什么。优先说"为什么"而不是"做了什么"。

```
添加 SHR_U 操作码以支持无符号右移

扩展 ISA 至 55 个操作码，补全 WASM i32 位运算子集。
```

### 频繁提交

每次写入文件后立即提交。会话随时可能中断，不要积攒改动。

```bash
git add isa.py
git commit -m "添加 POPCNT 操作码"
```

## 禁止事项

| 禁止 | 原因 |
|------|------|
| 使用 softmax | 无法精确定位目标，key 相同时退化为均匀分布 |
| 训练已编译模型 | 所有权重通过 `_compile_weights()` 解析设定。Phase 5-10 已证明梯度下降无法学到真正的加法 |
| 使用 float32 | 抛物线地址编码在 float32 下只支持约 4K 索引 |
| 抑制类型错误 | 不允许 `# type: ignore`、`@ts-ignore` |
| 添加 `__init__.py` | 扁平模块结构，不是 Python 包 |
| 批量攒改动后提交 | 每次写入后提交，会话可能随时中断 |
| 盲读大文件 | 先看 `_MAP.md` 获取行号索引，再定向读取。`executor.py` 约 1070 行，不要从头读到尾 |
| 锁定精确依赖版本 | 研究仓库，用 `>=` 指定下界 |

## 常见任务速查

| 任务 | 入口文件 | 备注 |
|------|----------|------|
| 添加操作码 | `isa.py` + `executor.py` | 两个执行器都要更新 |
| 编写测试程序 | `programs.py` | 遵循 `make_*` 命名模式 |
| 理解嵌入编码 | `isa.py` 第 733 行起 | `embed_*` 函数 |
| 调试执行轨迹 | `isa.py` → `compare_traces()` | 逐步对比 |
| 添加结构化控制流 | `assembler.py` | WASM 风格 block/loop/if/br |
| 解析 WAT 文本 | `wat_parser.py` | 712 行，完整 WAT 语法支持 |
| 运行性能基准 | `dev/benchmark_scaling.py` 或 `src/benchmarks.py` | Mojo vs Python |

## 相关文档

- [README.md](../README.md) 项目概述和 ISA 参考
- [Quick Start](quickstart.md) 五分钟上手
- [Architecture](architecture/overview.md) 系统架构设计
- [ISA Reference](isa/index.md) 完整 55 操作码参考
- [Development Findings](development/findings-summary.md) 研究发现和 R&D 计划
