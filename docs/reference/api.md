# API 参考

核心 Python 类和函数的接口文档。项目分为两层子包：`src/transturing/core/`（零依赖核心：ISA 定义、类型、后端抽象、测试工具）和 `src/transturing/backends/`（隔离的后端实现：NumPy 和 PyTorch）。

> **模块依赖图：** `core/isa.py` 是共享根模块。`programs.py` 和 `assembler.py` 均从 `isa.py` 导入。`wasm_binary.py` 复用既有 lowering 语义。`c_pipeline.py` 通过 `compile_wasm()` 走主路径。后端通过 `from transturing.core.isa import ...` 引用核心模块。

## 导入示例

```python
# 顶层便捷导入（推荐）:
from transturing import (
    D_MODEL, MASK32, N_OPCODES, TOKENS_PER_STEP,
    Instruction, Trace, TraceStep,
    compare_traces, compile_wasm, compile_wasm_function, compile_wasm_module,
    get_executor, list_backends, parse_wasm_binary, parse_wasm_file,
    program, test_algorithm, test_trap_algorithm,
)

# 后端注册表（推荐用于用户代码）:
exec_np = get_executor('numpy')   # 返回 NumPyExecutor
exec_pt = get_executor('torch')   # 返回 TorchExecutor

# 直接后端导入:
from transturing.backends.numpy_backend import NumPyExecutor
from transturing.backends.torch_backend import (
    TorchExecutor, CompiledModel, CompiledAttentionHead,
    TokenVocab, DTYPE, EPS,
    embed_program_token, embed_stack_entry, embed_state,
    embed_local_entry, embed_heap_entry, embed_call_frame,
)

# 核心模块导入:
from transturing.core.isa import OP_PUSH, OP_ADD, OP_HALT, MASK32
from transturing.core.assembler import compile_structured
from transturing.core.wasm_binary import (
    compile_wasm, compile_wasm_function, compile_wasm_module,
    parse_wasm_binary, parse_wasm_file,
)
from transturing.core.programs import make_fibonacci, make_factorial
from transturing.core.abc import ExecutorBackend
from transturing.core.registry import get_executor, list_backends, register_backend
```

对当前程序导入 API 来说, 主路径是二进制 `.wasm` 模块经由 `compile_wasm()` / `compile_wasm_module()` / `compile_wasm_function()` 以及 `parse_wasm_binary()` / `parse_wasm_file()` 进入既有 lowering 语义。当前记录的支持范围只覆盖已验证的 i32 子集。

---

## 后端抽象与注册表

### ExecutorBackend

所有执行器后端的抽象基类，定义在 `core/abc.py` 中。`NumPyExecutor` 和 `TorchExecutor` 都实现此接口。

**所在文件：** `src/transturing/core/abc.py`

```python
from transturing.core.abc import ExecutorBackend
```

```python
class ExecutorBackend(ABC):
    name: str  # 类级常量，标识后端（如 'numpy'、'torch'）

    @abstractmethod
    def execute(self, prog: list[Instruction], max_steps: int = 50000) -> Trace:
        """执行程序并返回完整轨迹。"""
```

### get_executor(name=None)

获取执行器实例的工厂函数。

**所在文件：** `src/transturing/core/registry.py`

```python
from transturing import get_executor

# 自动选择（优先 torch > numpy）:
exec = get_executor()

# 指定后端:
exec_np = get_executor('numpy')
exec_pt = get_executor('torch')
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | `str \| None` | 后端名称（`'numpy'` 或 `'torch'`）。`None` 时按 torch > numpy 优先级自动选择 |

**返回：** `ExecutorBackend` 实例。

不可用时抛出 `RuntimeError` 或 `ValueError`。

### list_backends()

列出当前可用的后端名称。

**所在文件：** `src/transturing/core/registry.py`

```python
from transturing import list_backends

print(list_backends())
# → ['torch', 'numpy']
```

**返回：** `list[str]`

### register_backend(cls)

装饰器，用于注册新的后端类。

**所在文件：** `src/transturing/core/registry.py`

```python
from transturing.core.registry import register_backend

@register_backend
class MyExecutor(ExecutorBackend):
    name = "my_backend"
    ...
```

---

## 执行器

### NumPyExecutor

纯 NumPy 实现的编译执行器。直接操作浮点数组，不依赖 PyTorch。所有操作码的分发逻辑内联在一个 `execute` 方法中，通过 `_ParabolicStore` 和 `_ExecCtx` 管理执行状态。

**所在文件：** `src/transturing/backends/numpy_backend.py`

```python
from transturing.backends.numpy_backend import NumPyExecutor

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

- **程序内存**（program）：指令取指
- **栈内存**（stack）：栈顶 SP 及偏移读写
- **局部变量**（locals）：LOCAL.GET/SET/TEE 的变量空间
- **堆内存**（heap）：I32.LOAD/STORE 线性内存
- **调用栈**（call stack）：CALL/RETURN 保存返回地址和 SP

内部实现细节：

- **`_ParabolicStore`**：抛物线键值存储，支持 `write(addr, val)` 和 `read(addr)` 操作，使用 `eps=1e-10` 实现近因偏好
- **`_ExecCtx`**：执行上下文，封装栈、局部变量、堆、调用栈、IP、SP 等状态

---

### TorchExecutor

基于 PyTorch 的执行器，使用 `CompiledModel`（`nn.Module`）执行每一步。内部将程序、栈、局部变量、堆、调用栈编码为嵌入向量，调用模型的 `forward` 方法完成单步计算。

**所在文件：** `src/transturing/backends/torch_backend.py`

```python
from transturing.backends.torch_backend import TorchExecutor

exec_pt = TorchExecutor()
trace = exec_pt.execute(prog, max_steps=50000)
```

#### \_\_init\_\_(model=None)

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `CompiledModel \| None` | 可选的自定义模型实例。默认创建新的 `CompiledModel()` |

#### execute(prog, max_steps=50000)

参数和返回值与 `NumPyExecutor.execute` 完全一致。

执行流程：先将程序**编码**为嵌入矩阵，然后在 `torch.no_grad()` 上下文中循环调用 `CompiledModel.forward()`，直到遇到 HALT 或达到步数上限。

内部实现细节：

- **`_ForwardResult`**：`forward()` 返回的命名元组，包含 opcode、arg、sp_delta、top、val_a/b/c 等字段
- **`_ExecState`**：运行时执行状态，追踪栈/局部变量/堆/调用栈的嵌入矩阵和写计数

---

### 双执行器验证

项目中所有测试都要求 NumPy 和 PyTorch 两个执行器产生**完全相同**的轨迹。用 `compare_traces()` 进行逐令牌比对。

```python
from transturing import program, compare_traces
from transturing.backends.numpy_backend import NumPyExecutor
from transturing.backends.torch_backend import TorchExecutor

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

继承 `torch.nn.Module` 的编译型 transformer。所有权重通过 `_compile_weights()` 解析设定，不经过训练。10 个注意力头分工明确，前馈层由线性路由矩阵加运行时非线性语义计算组成。这里编译的是**通用执行器**，而不是每个具体程序各自的一套模型参数。

**所在文件：** `src/transturing/backends/torch_backend.py`

```python
from transturing.backends.torch_backend import CompiledModel

model = CompiledModel(d_model=51)
model.eval()
```

#### \_\_init\_\_(d_model=51)

| 参数 | 类型 | 说明 |
|------|------|------|
| `d_model` | `int` | 模型维度，默认 51（由 `D_MODEL` 常量定义） |

初始化时立即调用 `_compile_weights()`，完成全部权重的解析赋值。模型包含以下组件：

**10 个注意力头：**

| 编号 | 名称 | 功能 | v_dim | 备注 |
|------|------|------|-------|------|
| 0 | `head_prog_op` | 取指令操作码 | 1 | |
| 1 | `head_prog_arg` | 取指令参数 | 1 | |
| 2 | `head_stack_a` | 读栈顶 SP | 1 | |
| 3 | `head_stack_b` | 读栈 SP-1 | 1 | W_Q 使用偏置 |
| 4 | `head_stack_c` | 读栈 SP-2 | 1 | W_Q 使用偏置 |
| 5 | `head_local_val` | 取局部变量值 | 1 | |
| 6 | `head_local_addr` | 验证局部变量地址 | 1 | |
| 7 | `head_heap_val` | 取堆内存值 | 1 | |
| 8 | `head_heap_addr` | 验证堆内存地址 | 1 | |
| 9 | `head_call_stack` | 读取调用栈帧 | 3 | v_dim=3（ret_addr, saved_sp, locals_base） |

**前馈分发：**
- `M_top`：线性路由矩阵，形状 `(55, 6)`，注册为 buffer
- `sp_deltas`：每个操作码对应的栈指针偏移量，形状 `(55,)`，注册为 buffer

**参数统计：**
- `nn.Parameter` 总量：2656
- 含 buffer 总量：3041

#### forward(query_emb, prog_embs, mem, locals_base=0)

执行单步推理。

| 参数 | 类型 | 说明 |
|------|------|------|
| `query_emb` | `Tensor (D,)` | 当前状态的查询嵌入（由 `embed_state` 生成） |
| `prog_embs` | `Tensor (N_prog, D)` | 程序指令嵌入矩阵 |
| `mem` | `_MemoryEmbs` | 命名元组，包含 stack、local、heap、call 嵌入 |
| `locals_base` | `int` | 当前局部变量的基地址偏移 |

**返回：** `_ForwardResult`，包含以下字段：

| 返回值 | 类型 | 说明 |
|--------|------|------|
| `opcode` | `int` | 当前指令操作码 |
| `arg` | `int` | 当前指令参数 |
| `sp_delta` | `int` | 栈指针变化量 |
| `top` | `int` | 执行后的栈顶值 |
| `val_a` | `int` | 栈[SP] 的值 |
| `val_b` | `int` | 栈[SP-1] 的值 |
| `val_c` | `int` | 栈[SP-2] 的值 |

---

## 词表与注意力

### TokenVocab

固定词表类。覆盖执行轨迹中出现的所有令牌类型，提供编码/解码和编译嵌入表的功能。

**所在文件：** `src/transturing/backends/torch_backend.py`

```python
from transturing.backends.torch_backend import TokenVocab

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

**所在文件：** `src/transturing/backends/torch_backend.py`

计算过程：

```
q = W_Q @ query_embedding       → (head_dim,)
K = W_K @ memory_embeddings      → (N, head_dim)
V = W_V @ memory_embeddings      → (N, v_dim)
scores = K @ q                   → (N,)
output = V[argmax(scores)]       → (v_dim,)
```

使用 2D 抛物线键空间：`k = (2j, -j²)`，点积注意力在目标位置产生尖锐峰值。

#### \_\_init\_\_(d_model=51, head_dim=2, v_dim=1, use_bias_q=False)

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `d_model` | `int` | 51 | 模型维度 |
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

**所在文件：** `src/transturing/core/isa.py`

```python
from transturing import program

# 计算 7 + 3
prog = program(("PUSH", 7), ("PUSH", 3), ("ADD",), ("HALT",))
# → [PUSH 7, PUSH 3, ADD, HALT]
```

支持的指令格式：

| 格式 | 示例 | 说明 |
|------|------|------|
| `("OPCODE", arg)` | `("PUSH", 42)` | 带参数的操作码 |
| `("OPCODE",)` | `("ADD",)` | 无参数的操作码 |

所有操作码名称（不区分大小写）：PUSH, POP, ADD, DUP, HALT, SUB, JZ, JNZ, NOP, SWAP, OVER, ROT, MUL, DIV_S, DIV_U, REM_S, REM_U, EQZ, EQ, NE, LT_S, LT_U, GT_S, GT_U, LE_S, LE_U, GE_S, GE_U, AND, OR, XOR, SHL, SHR_S, SHR_U, ROTL, ROTR, CLZ, CTZ, POPCNT, ABS, NEG, SELECT, LOCAL.GET, LOCAL.SET, LOCAL.TEE, I32.LOAD, I32.STORE, I32.LOAD8_U, I32.LOAD8_S, I32.LOAD16_U, I32.LOAD16_S, I32.STORE8, I32.STORE16, CALL, RETURN。

---

### compare_traces(trace_a, trace_b)

逐令牌比较两条执行轨迹。

**所在文件：** `src/transturing/core/isa.py`

**返回：** `(match: bool, detail: str)`

- 轨迹长度不同时返回 `(False, "length mismatch: ...")`
- 某步令牌不匹配时返回 `(False, "step N: ...")`
- 完全一致时返回 `(True, "match")`

---

### test_algorithm(cfg)

在两个执行器上运行同一程序，验证结果和轨迹一致性。参数通过 `TestConfig` 打包传入。

**所在文件：** `src/transturing/core/isa.py`

```python
def test_algorithm(cfg: TestConfig) -> tuple[bool, int]:
```

**返回：** `(all_ok: bool, steps: int)`

验证三个条件：NumPy 结果正确、PyTorch 结果正确、两条轨迹完全一致。

### test_trap_algorithm(cfg)

测试预期会触发 TRAP 的程序（如除零、栈下溢）。参数通过 `TestConfig` 打包传入。

**所在文件：** `src/transturing/core/isa.py`

```python
def test_trap_algorithm(cfg: TestConfig) -> bool:
```

**返回：** `bool`，两个执行器都正确触发 TRAP 且轨迹一致时为 `True`。

---

## 嵌入函数

这组函数将执行状态编码为 `D_MODEL`（51）维的 `float64` 向量。每个函数设置不同的"类型标记"维度，使注意力头能区分不同内存空间。

**所在文件：** `src/transturing/backends/torch_backend.py`

```python
from transturing.backends.torch_backend import (
    embed_program_token, embed_stack_entry, embed_state,
    embed_local_entry, embed_heap_entry, embed_call_frame,
)
```

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

## 数学辅助函数

**所在文件：** `src/transturing/core/isa.py`

```python
from transturing.core.isa import trunc_div, to_i32, clz32, ctz32, popcnt32
```

| 函数 | 说明 |
|------|------|
| `trunc_div(a, b)` | 截断除法（向零取整） |
| `trunc_rem(a, b)` | 截断余数 |
| `to_i32(val)` | 将值转换为有符号 i32（32 位整数溢出掩码） |
| `shr_u(a, b)` | 无符号右移 |
| `shr_s(a, b)` | 有符号右移（保留符号位） |
| `rotl32(a, b)` | 32 位循环左移 |
| `rotr32(a, b)` | 32 位循环右移 |
| `clz32(val)` | 统计前导零 |
| `ctz32(val)` | 统计后缀零 |
| `popcnt32(val)` | 统计置位位数（汉明重量） |
| `sign_extend_8(val)` | 8 位符号扩展到 32 位 |
| `sign_extend_16(val)` | 16 位符号扩展到 32 位 |

---

## 数据类型

### Instruction

**所在文件：** `src/transturing/core/isa.py`

```python
@dataclass
class Instruction:
    op: int       # 操作码 (1-55, 或 99 表示 TRAP)
    arg: int = 0  # 参数（如 PUSH 的值、JZ 的跳转地址）
```

### Trace

**所在文件：** `src/transturing/core/isa.py`

```python
@dataclass
class Trace:
    program: List[Instruction]
    steps: List[TraceStep] = field(default_factory=list)
```

### TraceStep

**所在文件：** `src/transturing/core/isa.py`

```python
@dataclass
class TraceStep:
    op: int       # 本步执行的操作码
    arg: int      # 本步的参数
    sp: int       # 执行后的栈指针
    top: int      # 执行后的栈顶值
```

### TestConfig

**所在文件：** `src/transturing/core/isa.py`

```python
@dataclass
class TestConfig:
    name: str
    prog: list[Instruction]
    expected: int | None   # None 表示不验证结果值
    np_exec: ExecutorBackend
    pt_exec: ExecutorBackend
    verbose: bool = False
```

`test_algorithm` 和 `test_trap_algorithm` 的参数打包类。使用 `ExecutorBackend` 抽象类型，因此可以接受任何注册的后端实例（NumPyExecutor 或 TorchExecutor）。

---

## 常量

### 核心常量（isa.py）

| 常量 | 值 | 说明 |
|------|-----|------|
| `D_MODEL` | 51 | 模型维度 |
| `N_OPCODES` | 55 | ISA 操作码数量（不含 TRAP） |
| `OP_TRAP` | 99 | 运行时错误操作码 |
| `MASK32` | `0xFFFFFFFF` | i32 溢出掩码 |
| `TOKENS_PER_STEP` | 4 | 每步轨迹的令牌数量 |

51 个维度布局：3 个类型标记 + 2×5 抛物线键对（prog/stack/local/heap/call）+ 24 个操作码标志 + 4 个状态维度（IP/SP/OPCODE/VALUE）+ DIM_ONE + local/heap/call 标志 + 调用值维度（ret_addr/saved_sp/locals_base）。

### PyTorch 后端常量（torch_backend.py）

| 常量 | 值 | 说明 |
|------|-----|------|
| `DTYPE` | `torch.float64` | 数值精度（强制 float64） |
| `EPS` | `1e-6` | PyTorch 精度常量（近次编码用） |

### NumPy 后端常量

NumPy 执行器使用 `eps=1e-10`，定义在 `numpy_backend.py` 的 `_ParabolicStore` 类中。与 PyTorch 的 `EPS=1e-6` 不同，这是设计上的差异（不同的精度上下文）。

---

## 相关文档

- [文件地图](file-map.md)，仓库结构和每个文件的职责
- [项目主页](../../README.md)，文档导航和阅读指南
- [ISA 参考](../isa/index.md)，55 个操作码的完整定义
- [架构概览](../architecture/overview.md)，系统设计和技术背景
