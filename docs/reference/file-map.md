# 仓库文件地图

按功能区域组织的完整文件索引。每个文件附一行说明和行数。

---

## 核心执行引擎

| 文件 | 行数 | 说明 |
|------|------|------|
| `isa.py` | 869 | 55 操作码定义、TokenVocab 词表、抛物线嵌入函数、CompiledAttentionHead、compare_traces 对比工具 |
| `executor.py` | 1070 | NumPyExecutor（NumPy 后端）、CompiledModel（PyTorch nn.Module）、TorchExecutor（PyTorch 后端） |

这两个文件构成了整个系统的核心：`isa.py` 定义指令集和编码方案，`executor.py` 实现两个等价的执行器。所有操作码的权重通过 `_compile_weights()` 解析计算，不依赖训练。

---

## 程序与测试

| 文件 | 行数 | 说明 |
|------|------|------|
| `programs.py` | 625 | 测试程序生成器：fib、multiply、gcd、factorial、位运算等 30+ 个 `make_*` 函数 |
| `test_consolidated.py` | 468 | 集成测试：NumPy 执行器等价性、PyTorch 执行器等价性、双执行器交叉验证 |
| `test_wat_parser.py` | 500 | WAT 解析器测试套件：解析、编译、执行全链路验证 |

每个测试都必须通过 `compare_traces()` 验证 NumPyExecutor 和 TorchExecutor 产生完全一致的执行轨迹。

---

## 编译工具链

| 文件 | 行数 | 说明 |
|------|------|------|
| `assembler.py` | 225 | WASM 风格结构化控制流编译器（block/loop/if/br/br_table → 扁平 ISA） |
| `wat_parser.py` | 712 | WebAssembly 文本格式（WAT）解析器，完整支持 WAT 语法 |
| `c_pipeline.py` | 635 | C → WAT → ISA 编译管线（需要 clang + wasm2wat） |

编译方向：C 源码 → clang 编译为 WASM → wasm2wat 转 WAT 文本 → wat_parser 解析 → assembler 编译为扁平 ISA → executor 执行。

---

## Mojo 后端 (`src/`)

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/executor.mojo` | 882 | Mojo 语言移植的完整 55 操作码执行器 |
| `src/benchmark.py` | 137 | Mojo vs NumPy 微基准测试 |
| `src/benchmarks.py` | 408 | 重量级基准程序：FNV-1a 哈希、冒泡排序、素数筛 |
| `src/llm_vs_native.py` | 221 | LLM 执行器 vs 原生 Python 的性能对比 |
| `src/run_mojo_tests.py` | 450 | Mojo 执行器测试运行器 |

Mojo 后端执行速度达 67-126M steps/sec，约为 Python 后端（2.1-3.1M steps/sec）的 30-60 倍。

---

## 研究探索 (`dev/`)

### 阶段脚本 (`dev/phases/`)

每个阶段文件自包含，可直接 `python dev/phases/phaseN_*.py` 运行。

| 文件 | 行数 | 阶段 | 说明 |
|------|------|------|------|
| `phase1_hull_cache.py` | 538 | 1 | O(log t) 查找：凸包上的三元搜索 |
| `phase2_parabolic.py` | 216 | 2 | 抛物线编码作为精确内存寻址 |
| `phase2b_address_limits.py` | 431 | 2b | 残差寻址扩展到 25M+ 范围 |
| `phase3_cumsum.py` | 240 | 3 | 累积和追踪 IP/SP |
| `phase4_stack_machine.py` | 678 | 4 | 手工布线的 transformer 执行 PUSH/POP/ADD/DUP/HALT |
| `phase5_training.py` | 722 | 5 | 训练实验：56% 准确率，0/50 轨迹正确 |
| `phase6_curriculum.py` | 513 | 6 | 课程学习：56%→85%，39/50 轨迹 |
| `phase7_percepta_arch.py` | 500 | 7 | Percepta 架构 (d=36,h=18,L=7)，与阶段 6 天花板相同 |
| `phase8_microop_traces.py` | 744 | 8 | 检索 100% 解决，算术是唯一瓶颈 |
| `phase9_weighted_arithmetic.py` | 550 | 9 | 加权损失完善了倍增运算，DIFF+ADD 仍 0% |
| `phase10_digit_decomposition.py` | 802 | 10 | 数位分解（探索性） |
| `phase11_compile_executor.py` | 936 | 11 | 编译执行：100% 正确，编译优于训练 |
| `phase12_percepta_model.py` | 1389 | 12 | 真正的 PyTorch nn.Module，编译权重 |
| `phase13_isa_completeness.py` | 1153 | 13 | SWAP/OVER/ROT + Fibonacci、乘法、奇偶校验 |
| `phase14_extended_isa.py` | 2885 | 14 | 完整 55 操作码 ISA 扩展 |
| `phase15_local_variables.py` | 460 | 15 | LOCAL.GET/SET/TEE 局部变量作用域 |
| `phase16_linear_memory.py` | 580 | 16 | 堆内存：I32.LOAD/STORE + 字节/短整数变体 |
| `phase17_function_calls.py` | 495 | 17 | CALL/RETURN，递归阶乘通过 |
| `phase18_integration_tests.py` | 606 | 18 | 冒泡排序、递归 fib、多函数程序集成测试 |
| `phase19_structured_assembler.py` | 520 | 19 | block/loop/if/br/br_table 结构化控制流 |
| `phase20_type_masking_tests.py` | 378 | 20 | i32 溢出掩码（WASM 语义） |

### 研究文档

| 文件 | 行数 | 说明 |
|------|------|------|
| `dev/FINDINGS.md` | 632 | 按阶段详细记录的研究发现（632 行） |
| `dev/RD-PLAN.md` | 142 | 原始 R&D 计划及演进 |
| `dev/benchmark_scaling.py` | 248 | 百万步规模基准测试（Issue #52） |

### 阶段结果 JSON

| 文件 | 说明 |
|------|------|
| `dev/phases/phase6_results.json` | 阶段 6 课程学习结果 |
| `dev/phases/phase6b_results.json` | 阶段 6b 扩展实验结果 |
| `dev/phases/phase7_results.json` | 阶段 7 Percepta 架构结果 |
| `dev/phases/phase8_results.json` | 阶段 8 微操作追踪结果 |
| `dev/phases/phase9_results.json` | 阶段 9 加权算术结果 |
| `dev/phases/phase10_results.json` | 阶段 10 数位分解结果 |

---

## 文档 (`docs/`)

| 文件 | 行数 | 说明 |
|------|------|------|
| `docs/README.md` | 108 | 文档导航主页，含阅读路线推荐 |
| `docs/quickstart.md` | 76 | 快速开始：环境搭建与第一个程序 |
| **架构** | | |
| `docs/architecture/overview.md` | 182 | 架构概览：抛物线编码、注意力头、内存空间 |
| `docs/architecture/memory-model.md` | 295 | 五大内存空间的寻址机制 |
| `docs/architecture/compilation.md` | 171 | 编译流程：从程序到 transformer 权重 |
| **指南** | | |
| `docs/guides/how-it-works.md` | 170 | 逐步跟踪一个 4 指令程序的执行过程 |
| `docs/guides/writing-programs.md` | 451 | 程序编写指南：汇编器和 WAT 用法 |
| **ISA 参考** | | |
| `docs/isa/index.md` | 127 | 55 操作码分类索引 |
| `docs/isa/opcodes.md` | 297 | 操作码详解：语义、参数、执行行为 |
| **开发** | | |
| `docs/development/findings-summary.md` | 104 | 20 个研究阶段的核心结论摘要 |
| `docs/development/rd-plan-summary.md` | 58 | R&D 路线图与阶段演进 |
| **参考** | | |
| [docs/reference/api.md](api.md) | | API 参考：NumPyExecutor、TorchExecutor 等核心接口 |
| `docs/reference/file-map.md` | | 本文件 — 仓库文件结构与职责 |

---

## 示例程序 (`examples/`)

| 文件 | 行数 | 说明 |
|------|------|------|
| `examples/hungarian.py` | 225 | 匈牙利算法实现 |
| `examples/sudoku.py` | 141 | 数独求解器 |

---

## 可视化 (`viz/`)

| 文件 | 行数 | 说明 |
|------|------|------|
| `viz/phase1-results.jsx` | 296 | 阶段 1 实验结果的 React 可视化组件 |

---

## 项目元文件

| 文件 | 说明 |
|------|------|
| `AGENTS.md` | 项目级 AI 代理指令 |
| `README.md` | 项目主页和 ISA 参考表 |

---

## 导航

- [API 参考](api.md)
- [文档主页](../README.md)
- [项目主页](../../README.md)
