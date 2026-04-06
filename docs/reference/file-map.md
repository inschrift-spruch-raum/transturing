# 仓库文件地图

按功能区域组织的完整文件索引。每个文件附一行说明和行数。项目采用 `src/` 布局，核心代码在 `src/transturing/core/`（零依赖 ISA、类型、后端抽象），后端实现在 `src/transturing/backends/`（隔离的 NumPy 和 PyTorch 实现）。

---

## 包入口

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/transturing/__init__.py` | 39 | 顶层便捷导出：D_MODEL、Instruction、Trace、program、get_executor、list_backends 等 |

从 `transturing` 直接导入即可获得常用符号，无需关心内部包结构。所有导出来自 `core/isa.py` 和 `core/registry.py`。

---

## 核心子包 (`src/transturing/core/`)

零依赖的 ISA 定义、类型系统、后端抽象和公共 API。

| 文件 | 行数 | 说明 |
|------|------|------|
| `core/__init__.py` | 37 | 重导出核心符号：ExecutorBackend、TestConfig、Instruction、program、get_executor 等 |
| `core/isa.py` | 741 | 55 操作码定义、DIM 维度常量（51 维）、数学辅助函数、Trace/TraceStep/TestConfig 类型、program() 构建器、compare_traces/test_algorithm 测试工具 |
| `core/abc.py` | 20 | `ExecutorBackend` 抽象基类，定义 `execute(prog, max_steps) -> Trace` 接口 |
| `core/registry.py` | 61 | 后端注册表：`get_executor()` 工厂函数、`list_backends()` 发现函数、`register_backend()` 装饰器 |
| `core/programs.py` | 706 | 测试程序生成器：30+ 个 `make_*` 函数（fib、multiply、gcd、factorial、位运算、select 等） |
| `core/assembler.py` | 276 | WASM 风格结构化控制流编译器（block/loop/if/br/br_table → 扁平 ISA）。入口函数 `compile_structured()`，内部类 `_Assembler` |
| `core/c_pipeline.py` | 172 | C → `.wasm` → ISA 主编译管线。入口函数 `compile_c()`、`compile_c_to_wasm()`、`compile_and_run()` |
| `core/wasm_binary.py` | 1014 | 最小 WebAssembly 二进制前端。解码当前已验证的 i32 子集, 并复用既有 lowering 语义。入口函数 `compile_wasm()`、`compile_wasm_module()`、`compile_wasm_function()`、`parse_wasm_binary()`、`parse_wasm_file()` |

依赖方向：`isa.py` 是共享根模块。`programs.py` 和 `assembler.py` 均从 `isa.py` 导入。`wasm_binary.py` 复用 `assembler.py` 的结构化 lowering 语义。`c_pipeline.py` 通过 `compile_wasm()` 走主路径。

---

## 后端子包 (`src/transturing/backends/`)

隔离的后端实现，通过 `core/abc.py` 的 `ExecutorBackend` 接口统一。

| 文件 | 行数 | 说明 |
|------|------|------|
| `backends/__init__.py` | 1 | 仅含 docstring，防止直接导入。后端通过 `get_executor()` 按需加载 |
| `backends/numpy_backend.py` | 568 | `NumPyExecutor`：纯 NumPy 编译执行器。内部类 `_ParabolicStore`（抛物线键值存储，eps=1e-10）和 `_ExecCtx`（执行上下文） |
| `backends/torch_backend.py` | 1212 | `TorchExecutor`、`CompiledModel`（nn.Module，10 个注意力头，2656 参数）、`CompiledAttentionHead`、`TokenVocab`、6 个 `embed_*` 嵌入函数、`DTYPE`/`EPS` 常量。内部类 `_ForwardResult`、`_ExecState`、`_MemoryEmbs` |

后端通过 `from transturing.core.isa import ...` 引用核心模块，彼此完全隔离。

---

## 测试 (`tests/`)

| 文件 | 行数 | 说明 |
|------|------|------|
| `tests/conftest.py` | 28 | pytest 配置：后端未安装时自动跳过测试 |
| `tests/test_consolidated.py` | 621 | 集成测试：NumPy 执行器等价性、PyTorch 执行器等价性、双执行器交叉验证 |

每个测试都通过 `compare_traces()` 验证 NumPyExecutor 和 TorchExecutor 产生完全一致的执行轨迹。

---

## 编译工具链使用流程

```
C 源码 → clang 编译为 .wasm → wasm_binary 最小前端 → 既有 lowering → 扁平 ISA → executor 执行
```

编译工具链模块位于 `src/transturing/core/` 中。`c_pipeline.py` 的主路径依赖 `wasm_binary.py` 和既有 lowering 逻辑。

---

## 文档 (`docs/`)

| 文件 | 说明 |
|------|------|
| **架构** | |
| `docs/architecture/overview.md` | 架构概览：抛物线编码、注意力头、内存空间 |
| `docs/architecture/memory-model.md` | 五大内存空间的寻址机制 |
| `docs/architecture/compilation.md` | 编译流程：执行器权重如何固定，程序如何降到 ISA |
| **指南** | |
| `docs/guides/how-it-works.md` | 逐步跟踪一个 4 指令程序的执行过程 |
| `docs/guides/writing-programs.md` | 程序编写指南：指令、汇编器和二进制 `.wasm` 导入 |
| **ISA 参考** | |
| `docs/isa/index.md` | 55 操作码分类索引 |
| `docs/isa/opcodes.md` | 操作码详解：语义、参数、执行行为 |
| **开发** | |
| `docs/development/findings-summary.md` | 20 个研究阶段的核心结论摘要 |
| `docs/development/rd-plan-summary.md` | R&D 路线图与阶段演进 |
| **参考** | |
| [docs/reference/api.md](api.md) | API 参考：执行器、编译模型、词表、嵌入函数 |
| `docs/reference/file-map.md` | 本文件，仓库文件结构与职责 |

---

## 项目元文件

| 文件 | 说明 |
|------|------|
| `pyproject.toml` | 项目配置（src/ 布局、hatchling 构建、可选依赖） |
| `uv.lock` | 可复现的依赖锁定文件 |
| `.python-version` | Python 版本要求（3.14） |
| `AGENTS.md` | 项目级 AI 代理指令 |
| `README.md` | 项目主页和 ISA 参考表 |

---

## 导航

- [API 参考](api.md)
- [文档主页](../README.md)
- [项目主页](../../README.md)
