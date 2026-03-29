# API 参考

核心 Python 类和函数的接口文档。所有类分布在两个模块中：`isa.py`（ISA 定义、词表、注意力头、嵌入函数、测试工具）和 `executor.py`（两种执行器和编译模型）。

> **模块依赖链：** `isa.py` ← `executor.py` ← `programs.py` ← `assembler.py` ← `wat_parser.py` ← `c_pipeline.py`

## 导入示例

```python
from isa import (
    program, Instruction, Trace, TraceStep,
    TokenVocab, CompiledAttentionHead,
    embed_program_token, embed_stack_entry, embed_state,
    compare_traces, test_algorithm, test_trap_algorithm,
    D_MODEL, DTYPE, EPS,
)

from executor import NumPyExecutor, TorchExecutor, CompiledModel
```

---

## 执行器

### NumPyExecutor

纯 NumPy 实现的编译执行器。直接操作浮点数组，不依赖 PyTorch。所有操作码的分发逻辑内联在一个 `execute` 方法中。

**所在文件：** `executor.py`

```python
from executor import NumPyExecutor

exec_np = NumPyExecutor()
trace = exec_np.execute(prog, max_steps=50000)
```

#### execute(prog, max_steps=50000)

执行一段程序，返回完整的执行轨迹。

| 参数 | 类型 | 说明 |
|------|------|------|
| `prog` | `List[Instruction]` | 待执行的指令列表 |
| `max_steps` | `int` | 最大执行步数，防止无限循环。默认 50000 |

**返回：** `Trace`，包含所有执行步骤的轨迹对象。

执行过程中维护五个独立的内存空间，各自使用抛物线编码进行寻址：

- **栈内存**（stack）：栈顶 SP 及偏移读写
- **局部变量**（locals）：LOCAL.GET/SET/TEE 的变量空间
- **堆内存**（heap）：I32.LOAD/STORE 线性内存
- **调用栈**（call stack）：CALL/RETURN 保存返回地址和 SP
- **程序内存**（program）：指令取指

---

### TorchExecutor

基于 PyTorch 的执行器，使用 `CompiledModel`（`nn.Module`）执行每一步。内部将程序、栈、局部变量、堆、调用栈编码为嵌入向量，调用模型的 `forward` 方法完成单步计算。

**所在文件：** `executor.py`

```python
from executor import TorchExecutor

exec_pt = TorchExecutor()
trace = exec_pt.execute(prog, max_steps=50000)
```

#### \_\_init\_\_(model=None)

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `CompiledModel \| None` | 可选的自定义模型实例。默认创建新的 `CompiledModel()` |

#### execute(prog, max_steps=50000)

参数和返回值与 `NumPyExecutor.execute` 完全一致。

执行流程：先将程序编译为嵌入矩阵，然后在 `torch.no_grad()` 上下文中循环调用 `CompiledModel.forward()`，直到遇到 HALT 或达到步数上限。

---

### 双执行器验证

项目中所有测试都要求 NumPy 和 PyTorch 两个执行器产生**完全相同**的轨迹。用 `compare_traces()` 进行逐令牌比对。

```python
from isa import program, compare_traces, test_algorithm
from executor import NumPyExecutor, TorchExecutor

np_exec = NumPyExecutor()
pt_exec = TorchExecutor()

prog = program(("PUSH", 7), ("PUSH", 3), ("ADD",), ("HALT",))

np_trace = np_exec.execute(prog)
pt_trace = pt_exec.execute(prog)

match, detail = compare_traces(np_trace, pt_trace)
print(f"一致: {match}, 详情: {detail}")
# → 一致: True, 详情: match
```

---

## 编译模型

### CompiledModel

继承 `torch.nn.Module` 的编译型 transformer。所有权重通过 `_compile_weights()` 解析设定，不经过训练。10 个注意力头分工明确，前馈层由线性路由矩阵加非线性覆盖组成。

**所在文件：** `executor.py`

```python
from executor import CompiledModel

model = CompiledModel(d_model=36)
model.eval()
```

#### \_\_init\_\_(d_model=36)

| 参数 | 类型 | 说明 |
|------|------|------|
| `d_model` | `int` | 模型维度，默认 36（由 `D_MODEL` 常量定义） |

初始化时立即调用 `_compile_weights()`，完成全部权重的解析赋值。模型包含以下组件：

**10 个注意力头：**

| 编号 | 名称 | 功能 | v_dim |
|------|------|------|-------|
| 0 | `head_prog_op` | 取指令操作码 | 1 |
| 1 | `head_prog_arg` | 取指令参数 | 1 |
| 2 | `head_stack_a` | 读栈顶 SP | 1 |
| 3 | `head_stack_b` | 读栈 SP-1 | 1 |
| 4 | `head_stack_c` | 读栈 SP-2 | 1 |
| 5 | `head_local_val` | 取局部变量值 | 1 |
| 6 | `head_local_addr` | 验证局部变量地址 | 1 |
| 7 | `head_heap_val` | 取堆内存值 | 1 |
| 8 | `head_heap_addr` | 验证堆内存地址 | 1 |
| 9 | `head_call_stack` | 读取调用栈帧 | 3 |

**前馈分发：**
- `M_top`：线性路由矩阵，形状 `(N_OPCODES, 6)`
- `sp_deltas`：每个操作码对应的栈指针偏移量

#### forward(query_emb, prog_embs, stack_embs, local_embs=None, heap_embs=None, call_embs=None, locals_base=0)

执行单步推理。

| 参数 | 类型 | 说明 |
|------|------|------|
| `query_emb` | `Tensor (D,)` | 当前状态的查询嵌入（由 `embed_state` 生成） |
| `prog_embs` | `Tensor (N_prog, D)` | 程序指令嵌入矩阵 |
| `stack_embs` | `Tensor (N_stack, D)` | 栈写入记录嵌入 |
| `local_embs` | `Tensor (N_local, D) \| None` | 局部变量嵌入 |
| `heap_embs` | `Tensor (N_heap, D) \| None` | 堆内存嵌入 |
| `call_embs` | `Tensor (N_call, D) \| None` | 调用栈嵌入 |
| `locals_base` | `int` | 当前局部变量的基地址偏移 |

**返回：** 10 元组 `(opcode, arg, sp_delta, top, opcode_one_hot, val_a, val_b, val_c, local_val, heap_val)`

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `opcode` | `int` | 当前指令操作码 |
| `arg` | `int` | 当前指令参数 |
| `sp_delta` | `int` | 栈指针变化量 |
| `top` | `int` | 执行后的栈顶值 |
| `opcode_one_hot` | `Tensor` | 操作码的 one-hot 编码 |
| `val_a` | `int` | 栈[SP] 的值 |
| `val_b` | `int` | 栈[SP-1] 的值 |
| `val_c` | `int` | 栈[SP-2] 的值 |
| `local_val` | `int` | 局部变量值 |
| `heap_val` | `int` | 堆内存值 |

---

## 词表与注意力

### TokenVocab

固定词表类。覆盖执行轨迹中出现的所有令牌类型，提供编码/解码和编译嵌入表的功能。

**所在文件：** `isa.py`

```python
from isa import TokenVocab

vocab = TokenVocab()
print(vocab)
# → TokenVocab(vocab_size=323, opcodes=56, values=256, sp_deltas=7, specials=4)
```

词表布局：

| 范围 | 数量 | 内容 |
|------|------|------|
| 0-3 | 4 | 特殊令牌：PAD, COMMIT, BRANCH_TAKEN, BRANCH_NOT_TAKEN |
| 4-59 | 56 | 操作码令牌：55 个 ISA 操作码 + TRAP |
| 60-315 | 256 | 字节值令牌：0-255 |
| 316-322 | 7 | SP 增量令牌：-3 到 +3 |

**总量：323 个令牌**

#### encode(token)

将结构化令牌编码为词表 ID。

```python
vocab.encode(("op", 1))         # → 5  (PUSH 操作码)
vocab.encode(("val", 42))       # → 102
vocab.encode(("sp_delta", -1))  # → 318
vocab.encode("COMMIT")          # → 1
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `token` | `tuple \| str` | `("op", 操作码)`, `("val", 0-255)`, `("sp_delta", -3~3)`, `("special", 名称)`, 或特殊令牌字符串 |

**返回：** `int`，令牌 ID。

#### decode(tid)

将词表 ID 解码回结构化表示。

```python
vocab.decode(5)    # → ("op", 1)
vocab.decode(102)  # → ("val", 42)
vocab.decode(0)    # → ("special", "PAD")
```

**返回：** `tuple(tag, value)`

#### compile_embedding(d_model=None)

构建 `nn.Embedding` 嵌入表，权重解析设定并冻结。每个令牌的嵌入向量在指定维度上编码其语义：操作码令牌设置 `DIM_OPCODE`，值令牌设置 `DIM_VALUE`，等等。

**返回：** `nn.Embedding(vocab_size, d_model)`，`requires_grad=False`

#### compile_unembedding(embedding=None, d_model=None)

构建 `nn.Linear` 反嵌入层，使得 `argmax(unembed(embed(tok))) == tok`。使用嵌入矩阵的转置作为权重，并附加偏置校正以处理嵌入范数差异。

**返回：** `nn.Linear(d_model, vocab_size)`，`requires_grad=False`

#### opcode_name(op_code) / token_name(tid)

将操作码或令牌 ID 转为可读名称。

```python
vocab.opcode_name(1)    # → "PUSH"
vocab.token_name(102)   # → "V42"
```

---

### CompiledAttentionHead

硬最大值（hard-max）注意力头，权重解析设定。这是整个系统的核心计算单元，用于所有内存空间的寻址。

**所在文件：** `isa.py`

计算过程：

```
q = W_Q @ query_embedding       → (head_dim,)
K = W_K @ memory_embeddings      → (N, head_dim)
V = W_V @ memory_embeddings      → (N, v_dim)
scores = K @ q                   → (N,)
output = V[argmax(scores)]       → (v_dim,)
```

使用 2D 抛物线键空间：`k = (2j, -j²)`，点积注意力在目标位置产生尖锐峰值。

#### \_\_init\_\_(d_model=36, head_dim=2, v_dim=1, use_bias_q=False)

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `d_model` | `int` | 36 | 模型维度 |
| `head_dim` | `int` | 2 | 注意力头维度（抛物线键空间需要 2 维） |
| `v_dim` | `int` | 1 | 值维度 |
| `use_bias_q` | `bool` | False | 查询投影是否使用偏置 |

模型内部自动转为 `float64`（`self.double()`）。

#### forward(query_emb, memory_embs)

执行一次硬最大值注意力查找。

| 参数 | 类型 | 说明 |
|------|------|------|
| `query_emb` | `Tensor (D,)` | 单个查询嵌入 |
| `memory_embs` | `Tensor (N, D)` | 内存条目嵌入矩阵 |

**返回：** 三元组 `(value, score, idx)`

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `value` | `Tensor (v_dim,)` | 最佳匹配条目提取的值 |
| `score` | `Tensor ()` | 获胜注意力分数 |
| `idx` | `int` | 被选中条目的索引 |

当 `memory_embs` 为空时，返回零向量、`-inf` 分数和索引 -1。

---

## 辅助函数

### program(*instrs)

快速构建指令列表的便捷函数。接受元组形式的指令描述，返回 `List[Instruction]`。

**所在文件：** `isa.py`

```python
from isa import program

# 计算 7 + 3
prog = program(("PUSH", 7), ("PUSH", 3), ("ADD",), ("HALT",))
# → [PUSH 7, PUSH 3, ADD, HALT]
```

支持的指令格式：

| 格式 | 示例 | 说明 |
|------|------|------|
| `("OPCODE", arg)` | `("PUSH", 42)` | 带参数的操作码 |
| `("OPCODE",)` | `("ADD",)` | 无参数的操作码 |

所有操作码名称（不区分大小写）：PUSH, POP, ADD, DUP, HALT, SUB, JZ, JNZ, NOP, SWAP, OVER, ROT, MUL, DIV_S, DIV_U, REM_S, REM_U, EQZ, EQ, NE, LT_S, LT_U, GT_S, GT_U, LE_S, LE_U, GE_S, GE_U, AND, OR, XOR, SHL, SHR_S, SHR_U, ROTL, ROTR, CLZ, CTZ, POPCNT, ABS, NEG, SELECT, LOCAL_GET, LOCAL_SET, LOCAL_TEE, I32_LOAD, I32_STORE, I32_LOAD8_U, I32_LOAD8_S, I32_LOAD16_U, I32_LOAD16_S, I32_STORE8, I32_STORE16, CALL, RETURN。

---

### compare_traces(trace_a, trace_b)

逐令牌比较两条执行轨迹。

**所在文件：** `isa.py`

**返回：** `(match: bool, detail: str)`

- 轨迹长度不同时返回 `(False, "length mismatch: ...")`
- 某步令牌不匹配时返回 `(False, "step N: ...")`
- 完全一致时返回 `(True, "match")`

---

### test_algorithm(name, prog, expected, np_exec, pt_exec, verbose=False)

在两个执行器上运行同一程序，验证结果和轨迹一致性。

**所在文件：** `isa.py`

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str` | 测试名称，用于输出显示 |
| `prog` | `List[Instruction]` | 待测程序 |
| `expected` | `int` | 期望的最终栈顶值 |
| `np_exec` | `NumPyExecutor` | NumPy 执行器实例 |
| `pt_exec` | `TorchExecutor` | PyTorch 执行器实例 |
| `verbose` | `bool` | 失败时打印详细诊断信息 |

**返回：** `(all_ok: bool, steps: int)`

验证三个条件：NumPy 结果正确、PyTorch 结果正确、两条轨迹完全一致。

### test_trap_algorithm(name, prog, np_exec, pt_exec, verbose=False)

测试预期会触发 TRAP 的程序（如除零、栈下溢）。

**所在文件：** `isa.py`

**返回：** `bool`，两个执行器都正确触发 TRAP 且轨迹一致时为 `True`。

---

## 嵌入函数

这组函数将执行状态编码为 `D_MODEL` 维的 `float64` 向量。每个函数设置不同的"类型标记"维度，使注意力头能区分不同内存空间。

**所在文件：** `isa.py`

| 函数 | 签名 | 用途 |
|------|------|------|
| `embed_program_token` | `(pos: int, instr: Instruction) → Tensor(D,)` | 编码程序指令，使用抛物线位置键 |
| `embed_stack_entry` | `(addr: int, value: int, write_order: int) → Tensor(D,)` | 编码栈写入记录 |
| `embed_local_entry` | `(local_idx: int, value: int, write_order: int) → Tensor(D,)` | 编码局部变量写入 |
| `embed_heap_entry` | `(addr: int, value: int, write_order: int) → Tensor(D,)` | 编码堆内存写入 |
| `embed_call_frame` | `(depth, ret_addr, saved_sp, locals_base, write_order) → Tensor(D,)` | 编码调用栈帧 |
| `embed_state` | `(ip: int, sp: int) → Tensor(D,)` | 编码当前执行状态（IP + SP），用作查询向量 |

`write_order` 参数实现"近因偏好"：相同地址的后续写入会获得略高的注意力分数（通过 `EPS * write_order` 项）。

---

## 数据类型

### Instruction

```python
@dataclass
class Instruction:
    op: int       # 操作码 (1-55, 或 99 表示 TRAP)
    arg: int = 0  # 参数（如 PUSH 的值、JZ 的跳转地址）
```

### Trace

```python
@dataclass
class Trace:
    program: List[Instruction]
    steps: List[TraceStep] = field(default_factory=list)
```

### TraceStep

```python
@dataclass
class TraceStep:
    op: int       # 本步执行的操作码
    arg: int      # 本步的参数
    sp: int       # 执行后的栈指针
    top: int      # 执行后的栈顶值
```

---

## 常量

| 常量 | 值 | 说明 |
|------|-----|------|
| `D_MODEL` | 36 | 模型维度 |
| `DTYPE` | `torch.float64` | 数值精度（强制 float64） |
| `EPS` | `1e-6` | PyTorch 精度常量（近次编码用） |
| `N_OPCODES` | 55 | ISA 操作码数量（不含 TRAP） |
| `OP_TRAP` | 99 | 运行时错误操作码 |

---

## 相关文档

- [文件地图](file-map.md)，仓库结构和每个文件的职责
- [项目主页](../README.md)，文档导航和阅读指南
- [ISA 参考](../isa/index.md)，55 个操作码的完整定义
- [架构概览](../architecture/overview.md)，系统设计和技术背景
