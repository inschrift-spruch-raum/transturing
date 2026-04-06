# 贡献指南

感谢你对 **LLM-as-Computer** 项目的兴趣。这是一个编译型 transformer 执行器的研究项目，有一些不太常见的约定。开始之前请先阅读本指南。

完整的项目介绍见 [README.md](../README.md)。

## 开发环境设置

### 基本要求

- **Python 3.14**（`pyproject.toml` 中指定 `>=3.14`）
- NumPy >= 1.24
- PyTorch >= 2.0

### 安装

```bash
# 克隆仓库后
uv sync --group dev

# 或者手动安装核心依赖
uv sync
```

### 可选依赖

- **C 编译管道**（`c_pipeline.py`）主路径需要 `clang`（带 wasm32 目标支持）。这已经足够支持 `compile_c()` / `compile_and_run()` 的二进制 `.wasm` 主路径。

### 验证安装

```bash
# 运行测试，确认一切正常
uv run pytest tests/ -v
```

## 代码规范

### 项目结构

这是一个 `src/` 布局的 Python 包项目。核心代码位于 `src/transturing/`：

```
src/transturing/
├── __init__.py           # Re-exports core API + registry
├── core/
│   ├── __init__.py       # Re-exports core symbols
│   ├── isa.py            # 55 个操作码、DIM 常量、数学工具、Trace 类型
│   ├── abc.py            # ExecutorBackend 抽象基类
│   ├── registry.py       # 后端发现（get_executor, list_backends）
│   ├── programs.py       # 测试程序生成器（fib、mul、gcd 等 30+ 个 make_* 函数）
│   ├── assembler.py      # WASM 风格结构化控制流编译器（block/loop/if/br → 扁平 ISA）
│   └── c_pipeline.py     # C → .wasm → ISA 主编译管线
└── backends/
    ├── __init__.py       # 仅文档字符串（动态导入保护）
    ├── numpy_backend.py  # NumPyExecutor（参考/演示后端）
    └── torch_backend.py  # CompiledAttentionHead、TokenVocab、CompiledModel、TorchExecutor（PyTorch 主后端）
```

导入图：`core/isa.py` 是共享根模块。`programs.py` 和 `assembler.py` 均从 `isa.py` 导入。`wasm_binary.py` 复用既有 lowering 语义。`c_pipeline.py` 通过 `compile_wasm()` 走主路径。后端通过 `from transturing.core.isa import ...` 导入。外部使用 `from transturing.core.isa import ...` 或 `from transturing.backends.numpy_backend import ...`。

修改某个模块时注意下游依赖。

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

NumPy 执行器使用 `eps=1e-10`，PyTorch 使用 `EPS=1e-6`（来自 `torch_backend.py`）。这是有意为之的设计，对应不同精度上下文。不要统一这两个值。

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

### 测试框架

使用 pytest，配合参数化测试和 fixture。测试位于 `tests/` 目录。

```bash
# 运行全部测试
uv run pytest tests/ -v
```

### 双执行器验证

一致性测试通过 `compare_traces()` 验证 **NumPyExecutor** 和 **TorchExecutor** 产生完全一致的执行轨迹。PyTorch 是主后端，NumPy 作为参考/演示实现。

### 添加新操作码

需要同时更新两个文件：

1. `src/transturing/core/isa.py`：操作码定义 + DIM 常量
2. `src/transturing/backends/numpy_backend.py` + `src/transturing/backends/torch_backend.py`：两个后端的分派逻辑和编译逻辑

### 添加测试程序

在 `src/transturing/core/programs.py` 中遵循 `make_*` 命名模式，然后加入测试。

## 提交规范

### 提交信息

简洁描述你做了什么和为什么。优先说"为什么"而不是"做了什么"。

```
添加 SHR_U 操作码以支持无符号右移

扩展 ISA 至 55 个操作码，补全 WASM i32 位运算子集。
```

### 频繁提交

每次写入文件后立即提交。会话随时可能中断，不要积攒改动。

## 禁止事项

| 禁止 | 原因 |
|------|------|
| 使用 softmax | 无法精确定位目标，key 相同时退化为均匀分布 |
| 训练已编译模型 | 所有权重通过 `_compile_weights()` 解析设定。Phase 5-10 已证明梯度下降无法学到真正的加法 |
| 使用 float32 | 抛物线地址编码在 float32 下只支持约 4K 索引 |
| 抑制类型错误 | 不允许 `# type: ignore`、`@ts-ignore` |
| 批量攒改动后提交 | 每次写入后提交，会话可能随时中断 |
| 盲读大文件 | 先看 `docs/reference/api.md` 获取函数索引，再定向读取。`torch_backend.py` ~1212 行，不要从头读到尾 |
| 锁定精确依赖版本 | 研究仓库，用 `>=` 指定下界 |

## 常见任务速查

| 任务 | 入口文件 | 备注 |
|------|----------|------|
| 添加操作码 | `src/transturing/core/isa.py` + 两个 backends | 两个执行器都要更新 |
| 编写测试程序 | `src/transturing/core/programs.py` | 遵循 `make_*` 命名模式 |
| 理解嵌入编码 | `src/transturing/backends/torch_backend.py` | `embed_*` 函数 |
| 调试执行轨迹 | `src/transturing/core/isa.py` → `compare_traces()` | 逐步对比 |
| 添加结构化控制流 | `src/transturing/core/assembler.py` | WASM 风格 block/loop/if/br |


当前仓库记录的 WebAssembly 支持范围只覆盖已验证的 i32 子集。贡献涉及导入路径时，请把二进制 `.wasm` 主路径描述清楚，不要为当前仓库补写不存在的文本入口或调试工作流。

## 相关文档

- [README.md](../README.md) 项目概述和 ISA 参考
- [Quick Start](quickstart.md) 五分钟上手
- [Architecture](architecture/overview.md) 系统架构设计
- [ISA Reference](isa/index.md) 完整 55 操作码参考
- [Development Findings](development/findings-summary.md) 研究发现和 R&D 计划
