# 操作码详解

本页对 ISA 中**复杂且常用**的操作码进行深入说明，包含栈效应图、Python 代码示例和典型用途。简单操作码见底部[快速参考](#快速参考)部分。

> 完整操作码列表请参阅 [index.md](index.md)。程序编写指南见 [../guides/writing-programs.md](../guides/writing-programs.md)。

---

## SELECT — 无分支条件选择 (Opcode 42)

`SELECT` 是一条**参数化指令**，它在一条指令内完成三元条件选择，无需跳转。等价于 C 语言的 `c ? a : b`。

### 栈效应

```
执行前:  ... | a | b | c |    (c 在栈顶)
执行后:  ... | result |      (result = a if c≠0, else b)
```

- 弹出三个值：`c`（条件，栈顶）、`b`（假值，中间）、`a`（真值，最先进栈）
- 若 `c ≠ 0`，压入 `a`；否则压入 `b`
- 栈深度减少 2

### 代码示例

**基本用法 — 三元选择：**

```python
from isa import Instruction, OP_PUSH, OP_SELECT, OP_HALT

# 等价于: c ? a : b
a, b, c = 10, 20, 1
prog = [
    Instruction(OP_PUSH, a),    # 真值
    Instruction(OP_PUSH, b),    # 假值
    Instruction(OP_PUSH, c),    # 条件
    Instruction(OP_SELECT),     # c≠0 → 推入 a; c==0 → 推入 b
    Instruction(OP_HALT),
]
# 结果: 10 (因为 c=1 ≠ 0，选择 a)
```

**实际应用 — 求 max(a, b)：**

来自 `programs.py` 的 `make_select_max`：

```python
from isa import Instruction, OP_PUSH, OP_GT_S, OP_SELECT, OP_HALT

def make_select_max(a, b):
    """用 GT_S + SELECT 计算 max(a, b)，无需条件跳转。"""
    prog = [
        Instruction(OP_PUSH, a),    # 0
        Instruction(OP_PUSH, b),    # 1
        Instruction(OP_PUSH, a),    # 2: 真值 (a 较大时选它)
        Instruction(OP_PUSH, b),    # 3: 用于比较
        Instruction(OP_GT_S),       # 4: a > b ? 1 : 0
        Instruction(OP_SELECT),     # 5: 条件=1 → 选 a；条件=0 → 选 b
        Instruction(OP_HALT),       # 6
    ]
    return prog, max(a, b)
```

### 典型用途

| 场景 | 说明 |
|------|------|
| 无分支求极值 | `max`/`min` 不需要 `JZ`/`JNZ` 跳转链 |
| 条件赋值 | 根据 flag 选择两个值之一 |
| 避免流水线气泡 | 在编译型执行器中，SELECT 是单步完成，没有跳转开销 |

---

## CALL / RETURN — 函数调用与返回 (Opcode 54/55)

`CALL addr` 保存当前执行上下文并跳转到目标地址，`RETURN` 恢复上下文并返回。支持递归调用。

### 栈效应

**CALL addr：**
```
执行前:  ... | arg |          (栈上可以有参数)
执行后:  ... | arg |          (栈不变，但内部保存了返回地址)
         → IP 跳转到 addr
```

**RETURN：**
```
执行前:  ... | ret_val |      (返回值在栈顶)
执行后:  ... | ret_val |      (返回值保留，SP 恢复到调用点 +1)
         → IP 跳回调用点下一条指令
```

### 内部机制

CALL 执行时：
1. 将 `(返回地址, 当前SP, 当前locals_base)` 压入**调用帧栈**
2. 重置 `locals_base`（新函数有独立的局部变量空间）
3. `IP` 跳转到 `addr`

RETURN 执行时：
1. 读取栈顶作为返回值 `ret_val`
2. 从调用帧栈弹出 `(返回地址, 保存的SP, 保存的locals_base)`
3. `SP` 恢复为 `saved_sp + 1`
4. 将 `ret_val` 写入新的栈顶
5. `IP` 跳回返回地址

若调用帧栈为空时执行 RETURN，产生 **TRAP**（运行时错误）。

### 代码示例

来自 `programs.py` 的 `make_factorial` 展示了循环实现的阶乘（与 CALL/RETURN 机制相同的乘法循环）：

```python
from isa import (Instruction, OP_PUSH, OP_DUP, OP_ROT, OP_MUL, OP_SWAP,
                 OP_SUB, OP_JZ, OP_JNZ, OP_POP, OP_HALT)

def make_factorial(n):
    """计算 n! 使用循环乘法。"""
    prog = [
        Instruction(OP_PUSH, 1),      # 0: result = 1
        Instruction(OP_PUSH, n),      # 1: counter = n
        # ── 循环体 (addr 2) ──
        Instruction(OP_DUP),          # 2
        Instruction(OP_JZ, 12),       # 3: counter == 0 → 完成
        Instruction(OP_DUP),          # 4
        Instruction(OP_ROT),          # 5: [result, counter, counter] → [counter, counter, result]
        Instruction(OP_MUL),          # 6: counter * result
        Instruction(OP_SWAP),         # 7
        Instruction(OP_PUSH, 1),      # 8
        Instruction(OP_SUB),          # 9: counter - 1
        Instruction(OP_PUSH, 1),      # 10: 永真条件
        Instruction(OP_JNZ, 2),       # 11: 无条件跳回循环头
        # ── 结束 (addr 12) ──
        Instruction(OP_POP),          # 12: 丢弃 counter=0
        Instruction(OP_HALT),         # 13: 栈顶 = n!
    ]
    return prog
```

递归函数调用模式（使用 CALL/RETURN）：

```python
# 递归阶乘的典型布局：
# 地址 0: CALL factorial_addr    — 调用阶乘函数
# 地址 1: HALT                   — 主程序结束
# 地址 factorial_addr:
#   ... 函数体 ...
#   RETURN                       — 返回调用者
```

### 典型用途

| 场景 | 说明 |
|------|------|
| 递归函数 | 阶乘、斐波那契等递归算法 |
| 代码复用 | 将常用操作封装为函数 |
| 模块化 | 大程序分解为独立函数 |

---

## LOCAL.GET / LOCAL.SET / LOCAL.TEE — 局部变量 (Opcode 43–45)

局部变量提供**命名变量作用域**，避免复杂的栈操作。每个函数调用有独立的局部变量空间。

### 栈效应

**LOCAL.GET *i*：**
```
执行前:  ... |
执行后:  ... | local[i] |     (local[i] 压入栈顶，栈深度 +1)
```

**LOCAL.SET *i*：**
```
执行前:  ... | val |
执行后:  ... |                  (val 弹出并写入 local[i]，栈深度 -1)
```

**LOCAL.TEE *i*：**
```
执行前:  ... | val |
执行后:  ... | val |           (val 复制到 local[i]，但不弹出！栈深度不变)
```

### 三者区别

| 指令 | 栈变化 | 用途 |
|------|--------|------|
| LOCAL.GET *i* | +1（压入） | 读取变量 |
| LOCAL.SET *i* | -1（弹出） | 写入变量，值被消耗 |
| LOCAL.TEE *i* | 0（不变） | 写入变量，**值保留在栈上** |

`LOCAL.TEE` 等价于 `DUP + LOCAL.SET`，但只需一条指令。当你需要把一个值同时存入变量并在后续计算中继续使用时，TEE 是最佳选择。

### 代码示例

```python
from isa import (Instruction, OP_PUSH, OP_LOCAL_GET, OP_LOCAL_SET,
                 OP_LOCAL_TEE, OP_ADD, OP_MUL, OP_HALT)

# 计算 x² + x，使用局部变量保存中间结果
prog = [
    Instruction(OP_PUSH, 5),            # 0: 压入 x = 5
    Instruction(OP_LOCAL_TEE, 0),       # 1: local[0] = 5（栈上仍保留 5）
    Instruction(OP_LOCAL_GET, 0),       # 2: 压入 local[0] = 5 → 栈: [5, 5]
    Instruction(OP_MUL),                # 3: 5 * 5 = 25 → 栈: [25]
    Instruction(OP_LOCAL_GET, 0),       # 4: 压入 local[0] = 5 → 栈: [25, 5]
    Instruction(OP_ADD),                # 5: 25 + 5 = 30
    Instruction(OP_HALT),               # 6: 结果 = 30
]
```

### 典型用途

| 场景 | 说明 |
|------|------|
| 保存循环计数器 | `LOCAL.SET` 保存，`LOCAL.GET` 读取 |
| 缓存计算结果 | `LOCAL.TEE` 同时存变量和传递给后续操作 |
| 函数参数 | 调用前将参数存入局部变量 |

---

## I32.LOAD / I32.STORE — 堆内存访问 (Opcode 46–53)

堆内存（线性内存）提供基于地址的随机读写，支持数组、结构体等数据结构。8 条指令覆盖 32 位、16 位、8 位三种访问宽度。

### 栈效应

**加载类（读取）：**
```
I32.LOAD:      ... | addr | → ... | value |        (addr 处的 32 位值)
I32.LOAD8_U:   ... | addr | → ... | byte_val |     (零扩展字节)
I32.LOAD8_S:   ... | addr | → ... | byte_val |     (符号扩展字节)
I32.LOAD16_U:  ... | addr | → ... | short_val |    (零扩展 16 位)
I32.LOAD16_S:  ... | addr | → ... | short_val |    (符号扩展 16 位)
```
栈深度不变（原地替换）。

**存储类（写入）：**
```
I32.STORE:     ... | addr | val | → ... |           (写 32 位)
I32.STORE8:    ... | addr | val | → ... |           (写低 8 位)
I32.STORE16:   ... | addr | val | → ... |           (写低 16 位)
```
栈深度减少 2（弹出 addr 和 val）。

### 变体说明

| 指令 | 宽度 | 扩展方式 | 适用场景 |
|------|------|----------|----------|
| I32.LOAD | 32 位 | — | 读写完整整数 |
| I32.LOAD8_U | 8 位 | 零扩展 (高位填 0) | 读取无符号字节 |
| I32.LOAD8_S | 8 位 | 符号扩展 (复制符号位) | 读取有符号字节 (-128~127) |
| I32.LOAD16_U | 16 位 | 零扩展 | 读取无符号短整数 |
| I32.LOAD16_S | 16 位 | 符号扩展 | 读取有符号短整数 (-32768~32767) |
| I32.STORE | 32 位 | — | 写入完整整数 |
| I32.STORE8 | 8 位 | 截断至低 8 位 | 写入单字节 |
| I32.STORE16 | 16 位 | 截断至低 16 位 | 写入双字节 |

### 代码示例

**数组读写 — 给地址 10 的值加上 100 后写回：**

```python
from isa import Instruction, OP_PUSH, OP_I32_LOAD, OP_I32_STORE, OP_ADD, OP_HALT

prog = [
    Instruction(OP_PUSH, 10),           # 0: addr = 10
    Instruction(OP_PUSH, 10),           # 1: addr（再压一次，用于后面的 STORE）
    Instruction(OP_I32_LOAD),           # 2: 读取 heap[10] → 栈: [addr, value]
    Instruction(OP_PUSH, 100),          # 3: 栈: [addr, value, 100]
    Instruction(OP_ADD),                # 4: 栈: [addr, value+100]
    Instruction(OP_I32_STORE),          # 5: heap[addr] = value+100
    Instruction(OP_HALT),
]
```

**字节级读写：**

```python
from isa import Instruction, OP_PUSH, OP_I32_STORE8, OP_I32_LOAD8_U, OP_HALT

# 将 0xAB 写入地址 4，再作为无符号字节读回
prog = [
    Instruction(OP_PUSH, 4),            # addr
    Instruction(OP_PUSH, 0xAB),         # value
    Instruction(OP_I32_STORE8),         # heap[4] 的低字节 = 0xAB
    Instruction(OP_PUSH, 4),            # addr
    Instruction(OP_I32_LOAD8_U),        # 读取 heap[4] 低字节 = 0xAB
    Instruction(OP_HALT),
]
```

### 典型用途

| 场景 | 说明 |
|------|------|
| 数组操作 | `base_addr + index * elem_size` 寻址 |
| 结构体字段 | 固定偏移量读写 |
| 字符串处理 | LOAD8_U 逐字节读取 |
| 内存映射 I/O | 通过固定地址访问设备寄存器 |

---

## ROT / SWAP / OVER — 栈操作 (Opcode 10–12)

这三条指令是 Forth 风格栈编程的核心。在栈机器中，操作数顺序经常不匹配，这些指令用来**重新排列栈上的值**。

### 栈效应

**SWAP：**
```
执行前:  ... | a | b |
执行后:  ... | b | a |      (交换栈顶两个元素)
```
栈深度不变。

**OVER：**
```
执行前:  ... | a | b |
执行后:  ... | a | b | a |  (复制第二个元素到栈顶)
```
栈深度 +1。相当于 `DUP` 但复制的是次顶层而非顶层。

**ROT：**
```
执行前:  ... | a | b | c |
执行后:  ... | b | c | a |  (将第三个元素旋转到栈顶)
```
栈深度不变。a 被绕过 b、c 移到栈顶。

### 代码示例

**Fibonacci 迭代 — ROT + SWAP + OVER 的经典组合：**

来自 `programs.py` 的 `make_fibonacci`：

```python
from isa import (Instruction, OP_PUSH, OP_SWAP, OP_OVER, OP_ADD, OP_ROT,
                 OP_SUB, OP_DUP, OP_JNZ, OP_POP, OP_HALT)

def make_fibonacci(n):
    """计算 fib(n)，迭代算法。

    栈布局: [counter, a, b]
    每次迭代: [counter, b, a] → [counter, b, a, b] → [counter, b, a+b]
    然后 ROT 将 counter 转回栈顶进行递减。
    """
    prog = [
        Instruction(OP_PUSH, 0),      # a = fib(0)
        Instruction(OP_PUSH, 1),      # b = fib(1)
        Instruction(OP_PUSH, n - 1),  # counter = n-1
        Instruction(OP_ROT),          # [1, n-1, 0]
        Instruction(OP_ROT),          # [n-1, 0, 1] = [counter, a, b]
        # ── 循环体 (addr 5) ──
        Instruction(OP_SWAP),         # 5: [counter, b, a]
        Instruction(OP_OVER),         # 6: [counter, b, a, b]
        Instruction(OP_ADD),          # 7: [counter, b, a+b]
        Instruction(OP_ROT),          # 8: [b, a+b, counter]
        Instruction(OP_PUSH, 1),      # 9
        Instruction(OP_SUB),          # 10: [b, a+b, counter-1]
        Instruction(OP_DUP),          # 11: [..., counter-1, counter-1]
        Instruction(OP_JNZ, 15),      # 12: counter-1 ≠ 0 → 继续
        Instruction(OP_POP),          # 13: 丢弃 counter=0
        Instruction(OP_HALT),         # 14: 栈顶 = fib(n)
        # ── 继续循环 (addr 15) ──
        Instruction(OP_ROT),          # 15: [new_b, counter-1, new_a]
        Instruction(OP_ROT),          # 16: [counter-1, new_a, new_b]
        Instruction(OP_PUSH, 1),      # 17
        Instruction(OP_JNZ, 5),       # 18: 永真跳转
    ]
    return prog
```

**OVER — 欧几里得 GCD 算法中的复用：**

来自 `programs.py` 的 `make_gcd`：

```python
from isa import (Instruction, OP_PUSH, OP_DUP, OP_JZ, OP_SWAP,
                 OP_OVER, OP_REM_S, OP_JNZ, OP_NOP, OP_POP, OP_HALT)

def make_gcd(a, b):
    """GCD(a, b) 欧几里得算法。"""
    prog = [
        Instruction(OP_PUSH, a),      # 0: a
        Instruction(OP_PUSH, b),      # 1: b
        # ── 循环 (addr 2) ──
        Instruction(OP_DUP),          # 2: [a, b, b]
        Instruction(OP_JZ, 10),       # 3: b == 0 → 结束
        Instruction(OP_SWAP),         # 4: [b, a]
        Instruction(OP_OVER),         # 5: [b, a, b] — OVER 复制新 b 到栈顶
        Instruction(OP_REM_S),        # 6: [b, a % b] — 求余
        Instruction(OP_PUSH, 1),      # 7: 永真条件
        Instruction(OP_JNZ, 2),       # 8: 继续循环
        Instruction(OP_NOP),          # 9: 占位
        # ── 结束 (addr 10) ──
        Instruction(OP_POP),          # 10: 丢弃 b=0
        Instruction(OP_HALT),         # 11: 栈顶 = GCD
    ]
    return prog
```

### 三条指令对比

| 指令 | 效果 | 栈深度变化 | 何时使用 |
|------|------|-----------|----------|
| SWAP | 交换栈顶两个 | 0 | 操作数顺序不对时 |
| OVER | 复制次顶层 | +1 | 需要复用下方的值 |
| ROT | 第三元素→栈顶 | 0 | 三值循环（如 Fibonacci） |

---

## 快速参考

以下操作码语义简单，按类别列出要点。

### 基础栈操作

| 指令 | 栈效应 | 备注 |
|------|--------|------|
| `PUSH n` | `... → ... \| n` | 唯一带立即数参数的栈操作 |
| `POP` | `... \| val → ...` | 丢弃栈顶 |
| `DUP` | `... \| val → ... \| val \| val` | 复制栈顶，相当于 `OVER` 作用于同一个值 |
| `NOP` | 无变化 | 占位，用作跳转目标或对齐 |
| `HALT` | 无变化 | 终止执行，栈顶为程序返回值 |

### 算术运算

所有算术结果执行 `result & 0xFFFFFFFF`（i32 溢出截断）。

| 指令 | 栈效应 | 备注 |
|------|--------|------|
| `ADD` | `a b → (a+b) & 0xFFFFFFFF` | 交换律，操作数顺序无关 |
| `SUB` | `a b → (a-b) & 0xFFFFFFFF` | **非交换**：次顶层 - 栈顶 |
| `MUL` | `a b → (a×b) & 0xFFFFFFFF` | 交换律 |
| `DIV_S` | `a b → trunc(a/b)` | 有符号除法，向零截断。b=0 → TRAP |
| `DIV_U` | `a b → trunc(a/b)` | 无符号除法。b=0 → TRAP |
| `REM_S` | `a b → a mod b` | 有符号取余，符号跟随被除数。b=0 → TRAP |
| `REM_U` | `a b → a mod b` | 无符号取余。b=0 → TRAP |
| `ABS` | `a → \|a\|` | 绝对值（一元） |
| `NEG` | `a → (-a) & 0xFFFFFFFF` | 取反（一元），结果经 i32 截断 |

### 比较运算

所有比较返回 `1`（真）或 `0`（假）。操作数顺序：`次顶层 OP 栈顶`。

```python
# GT_S:  PUSH 3, PUSH 5, GT_S → 0  (因为 3 不大于 5)
# GT_S:  PUSH 5, PUSH 3, GT_S → 1  (因为 5 大于 3)
```

| 指令 | 语义 | 备注 |
|------|------|------|
| `EQZ` | `a → (a == 0 ? 1 : 0)` | 一元零测试，常用于循环终止条件 |
| `EQ` | `a b → (a == b ? 1 : 0)` | 相等比较 |
| `NE` | `a b → (a ≠ b ? 1 : 0)` | 不等比较 |
| `LT_S` / `LT_U` | `a b → (a < b ? 1 : 0)` | 小于 |
| `GT_S` / `GT_U` | `a b → (a > b ? 1 : 0)` | 大于 |
| `LE_S` / `LE_U` | `a b → (a ≤ b ? 1 : 0)` | 小于等于 |
| `GE_S` / `GE_U` | `a b → (a ≥ b ? 1 : 0)` | 大于等于 |

> `_S` 后缀按二进制补码解释有符号数，`_U` 后缀将操作数视为非负整数。

### 位运算

| 指令 | 栈效应 | 备注 |
|------|--------|------|
| `AND` | `a b → a & b` | 位与，常用于掩码 |
| `OR` | `a b → a \| b` | 位或，常用于设置标志位 |
| `XOR` | `a b → a ^ b` | 异或，常用于翻转位 |
| `SHL` | `a b → (b << (a & 31)) & 0xFFFFFFFF` | 左移，移位量取低 5 位 |
| `SHR_S` | `a b → b >>_s (a & 31)` | 算术右移（保留符号位） |
| `SHR_U` | `a b → b >>_u (a & 31)` | 逻辑右移（高位填零） |
| `ROTL` | `a b → rotate_left(b, a & 31)` | 循环左移 |
| `ROTR` | `a b → rotate_right(b, a & 31)` | 循环右移 |
| `CLZ` | `a → count_leading_zeros(a)` | 一元，常用于快速求 log₂ |
| `CTZ` | `a → count_trailing_zeros(a)` | 一元，常用于找最低有效位 |
| `POPCNT` | `a → popcount(a)` | 一元，统计 1 的位数（汉明重量） |

### 控制流

| 指令 | 栈效应 | 备注 |
|------|--------|------|
| `JZ addr` | `... \| cond → ...` | 若 `cond == 0` 则跳转到 addr。弹出条件值 |
| `JNZ addr` | `... \| cond → ...` | 若 `cond ≠ 0` 则跳转到 addr。弹出条件值 |
| `NOP` | 无变化 | 占位指令 |
| `HALT` | 无变化 | 停机，栈顶为返回值 |
| `TRAP` (99) | — | 运行时错误信号（除零、栈下溢等），不由程序员显式使用 |

> 永真跳转技巧：`PUSH 1; JNZ target` 实现无条件跳转。

---

## 相关文档

- [index.md](index.md) — 55 个操作码的完整分类索引表
- [../guides/writing-programs.md](../guides/writing-programs.md) — 使用汇编器和 WAT 格式编写程序
