# How It Works: Not Just an Abstraction

A natural first reaction: *isn't this just a regular interpreter with extra steps?*

No. And tracing through a tiny example shows why.

## The program

```
PUSH 3
PUSH 5
ADD
HALT
```

Result: 8 on the stack. Four instructions. Let's watch both a normal interpreter and the compiled transformer executor run this.

## How a normal interpreter does it

```python
stack = []
ip = 0

# PUSH 3
stack.append(3)       # stack = [3]
ip += 1

# PUSH 5
stack.append(5)       # stack = [3, 5]
ip += 1

# ADD
a = stack.pop()       # 5
b = stack.pop()       # 3
stack.append(a + b)   # stack = [8]
ip += 1

# HALT
```

Direct array access. `stack[i]` returns the value at index `i` because that's what arrays do. The CPU's memory hardware handles addressing. Nothing interesting happens.

## How the compiled transformer does it

There is no array. There are no indices. There is only a growing sequence of **embedding vectors** — like tokens in a transformer's context window — and **attention** is the only mechanism for reading from them.

### Writing to memory: parabolic keys

When PUSH 3 executes and writes value 3 to stack address 1, it doesn't store `stack[1] = 3`. It appends an embedding vector to the context with a **parabolic key**:

```
key = (2·addr, −addr²) = (2·1, −1²) = (2, −1)
value = 3
```

When PUSH 5 writes value 5 to stack address 2:

```
key = (2·2, −2²) = (4, −4)
value = 5
```

Now memory looks like this — not an array, but a set of 2D key-value pairs in the context:

| Entry | Key (2D) | Value |
|-------|----------|-------|
| stack@1 | (2, −1) | 3 |
| stack@2 | (4, −4) | 5 |

### Reading from memory: attention

ADD needs the values at stack addresses 2 and 1. Here's how it reads address 2.

It forms a **query vector**:

```
query = (addr, 1) = (2, 1)
```

Then computes a **dot product** against every key in memory:

```
score(stack@1) = (2, −1) · (2, 1) = 2·2 + (−1)·1 = 3
score(stack@2) = (4, −4) · (2, 1) = 4·2 + (−4)·1 = 4  ← winner
```

The **argmax** selects stack@2. Its value is 5. That's the read.

For address 1:

```
query = (1, 1)
score(stack@1) = (2, −1) · (1, 1) = 2·1 + (−1)·1 = 1  ← winner
score(stack@2) = (4, −4) · (1, 1) = 4·1 + (−4)·1 = 0
```

Argmax selects stack@1. Value is 3.

The ADD attention head then writes the result:

```
key = (2·1, −1²) = (2, −1)    ← same address 1, but with a recency tiebreaker
value = 8
```

### Why this works mathematically

The score function for querying address `i` against a key stored at address `j` is:

```
score(j; i) = (2j)·i + (−j²)·1 = 2ij − j² = −(j − i)² + i²
```

This is a **downward parabola centered at j = i**. It peaks exactly at the target address and falls off quadratically in both directions. Argmax always picks the right entry. It's not approximate — it's exact, by construction.

This is the same dot-product-then-argmax operation that happens in every transformer attention head. The only difference is that the keys are engineered (compiled into the weight matrices) rather than learned.

## What the code actually does

The `NumPyExecutor` does this with literal numpy dot products:

```python
def stack_read(addr):
    keys = np.array(stack_keys)       # all 2D keys in memory
    q = np.array([addr, 1.0])         # query vector
    scores = keys @ q                 # dot products
    best = np.argmax(scores)          # hard attention
    return stack_vals[best]
```

The `TorchExecutor` goes further — it uses real `nn.Linear` weight matrices (`W_Q`, `W_K`, `W_V`) set to implement the parabolic encoding. The program counter, stack reads, argument fetches, and opcode dispatch all happen through PyTorch `matmul` and `argmax` operations. When you call `model.forward()`, you're running a standard transformer forward pass. The weights just happen to be analytically set rather than gradient-trained.

## The instruction fetch works the same way

The program is also stored as parabolic key-value pairs in the context. Instruction 0 gets key `(0, 0)`, instruction 1 gets key `(2, −1)`, instruction 2 gets key `(4, −4)`, etc.

To fetch the instruction at the current program counter, the executor forms a query and runs attention over the program tokens. The same mechanism that reads the stack also reads the program.

## So what *is* different?

In a normal interpreter, memory addressing is free — the CPU's hardware does it. Here, every memory read is a dot product over the entire context. That's expensive. It's `O(t)` per step where `t` is the context length (the parabolic structure enables `O(log t)` via ternary search, but the point stands).

So why would anyone want this? Because:

1. **It's differentiable.** Every step is a matrix multiply and an argmax. If you replace hard argmax with softmax, the entire execution trace becomes a differentiable computation graph. You can backpropagate through program execution.

2. **It runs inside the model.** No tool calls, no sandboxed interpreter, no context switch. The transformer plans a computation and executes it in the same forward pass.

3. **Programs become weights.** The 55-opcode ISA compiles into 964 parameters. A program isn't text that gets interpreted — it's weight matrices that get multiplied. Deployment is `model.load_state_dict()`.

The naive version is slower than a regular interpreter. The structured version (with convex hull search) scales to millions of steps on a CPU. The important thing isn't speed — it's that the execution lives inside the attention mechanism, making it composable with everything else transformers do.

## Try it yourself

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

Every `stack_read` in that execution is a dot product. Every instruction fetch is a dot product. That's the whole idea.
