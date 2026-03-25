"""
Phase 13: General-Purpose Stack Computer — ISA Completeness

Extends the compiled transformer's ISA with standard stack manipulation opcodes
(SWAP, OVER, ROT), then demonstrates general-purpose computation by executing
a diverse algorithm suite: Fibonacci, multiply, power-of-2, sum-of-1-to-N,
and parity test.

New opcodes:
  SWAP (10)  — exchange top two stack elements
  OVER (11)  — copy second-to-top element to top
  ROT  (12)  — rotate top 3: [a, b, c] → [b, c, a]

Architecture change:
  Head 4: Stack read at SP-2 (for ROT). Uses reserved head slot with Q bias (-2, 0).

Key result: The compiled transformer executes all algorithms correctly, proving
it is a general-purpose stack computer — not just a special-case executor.
"""

import numpy as np
import torch
import torch.nn as nn
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_stack_machine import (
    program, Instruction, ReferenceExecutor, Trace, TraceStep,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT, OP_NAMES,
    TOKENS_PER_STEP, ALL_TESTS,
)

# Import Phase 12 model components
from phase12_percepta_model import (
    PerceptaModel, PerceptaExtendedExecutor, CompiledAttentionHead,
    embed_program_token, embed_stack_entry, embed_state,
    D_MODEL, DTYPE, EPS, N_OPCODES as N_OPCODES_BASE,
    DIM_IS_PROG, DIM_IS_STACK, DIM_IS_STATE,
    DIM_PROG_KEY_0, DIM_PROG_KEY_1,
    DIM_STACK_KEY_0, DIM_STACK_KEY_1,
    DIM_OPCODE, DIM_VALUE, DIM_IP, DIM_SP, DIM_ONE,
    DIM_IS_PUSH, DIM_IS_POP, DIM_IS_ADD, DIM_IS_DUP, DIM_IS_HALT,
    DIM_IS_SUB, DIM_IS_JZ, DIM_IS_JNZ, DIM_IS_NOP,
    OPCODE_DIM_MAP as OPCODE_DIM_MAP_BASE,
    OPCODE_IDX as OPCODE_IDX_BASE,
    OP_SUB, OP_JZ, OP_JNZ, OP_NOP,
)

# Import Phase 11 numpy executor
from phase11_compile_executor import ExtendedExecutor


# ─── New Opcodes ──────────────────────────────────────────────────

OP_SWAP = 10
OP_OVER = 11
OP_ROT  = 12

OP_NAMES_P13 = {
    **OP_NAMES,
    OP_SUB: "SUB", OP_JZ: "JZ", OP_JNZ: "JNZ", OP_NOP: "NOP",
    OP_SWAP: "SWAP", OP_OVER: "OVER", OP_ROT: "ROT",
}

# New one-hot dimension assignments (using reserved dims 21-23)
DIM_IS_SWAP = 21
DIM_IS_OVER = 22
DIM_IS_ROT  = 23

# Extended opcode maps
OPCODE_DIM_MAP = {
    **OPCODE_DIM_MAP_BASE,
    OP_SWAP: DIM_IS_SWAP,
    OP_OVER: DIM_IS_OVER,
    OP_ROT:  DIM_IS_ROT,
}

OPCODE_IDX = {
    **OPCODE_IDX_BASE,
    OP_SWAP: 9,
    OP_OVER: 10,
    OP_ROT:  11,
}

N_OPCODES = 12  # 9 base + 3 new


# ─── Extended Embedding ───────────────────────────────────────────

def embed_program_token_ext(pos, instr):
    """Create embedding for a program instruction (extended ISA)."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_PROG]     = 1.0
    emb[DIM_PROG_KEY_0]  = 2.0 * pos
    emb[DIM_PROG_KEY_1]  = -float(pos * pos)
    emb[DIM_OPCODE]      = float(instr.op)
    emb[DIM_VALUE]       = float(instr.arg)
    emb[DIM_ONE]         = 1.0
    dim = OPCODE_DIM_MAP.get(instr.op)
    if dim is not None:
        emb[dim] = 1.0
    return emb


# ─── NumPy Executor with SWAP/OVER/ROT ───────────────────────────

class Phase13Executor(ExtendedExecutor):
    """Compiled numpy executor with full Phase 13 ISA.

    Adds SWAP, OVER, ROT to Phase 11's ExtendedExecutor.
    """

    def execute(self, prog, max_steps=5000):
        trace = Trace(program=prog)

        prog_keys = np.array([(2.0*j, -float(j*j)) for j in range(len(prog))])
        prog_ops = np.array([instr.op for instr in prog])
        prog_args = np.array([instr.arg for instr in prog])

        stack_keys = []
        stack_vals = []
        write_count = 0
        eps = 1e-10

        ip = 0
        sp = 0

        def stack_write(addr, val):
            nonlocal write_count
            stack_keys.append((2.0*addr, -float(addr*addr) + eps*write_count))
            stack_vals.append(val)
            write_count += 1

        def stack_read(addr):
            if not stack_keys:
                return 0
            keys = np.array(stack_keys)
            q = np.array([addr, 1.0])
            scores = keys @ q
            best = np.argmax(scores)
            stored_addr = round(keys[best, 0] / 2.0)
            return stack_vals[best] if stored_addr == addr else 0

        for step in range(max_steps):
            if ip >= len(prog):
                break

            op = prog[ip].op
            arg = prog[ip].arg
            next_ip = ip + 1

            if op == OP_PUSH:
                sp += 1
                stack_write(sp, arg)
                top = arg
            elif op == OP_POP:
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_ADD:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = val_a + val_b
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_SUB:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                result = val_b - val_a
                sp -= 1
                stack_write(sp, result)
                top = result
            elif op == OP_DUP:
                val = stack_read(sp)
                sp += 1
                stack_write(sp, val)
                top = val
            elif op == OP_SWAP:
                val_a = stack_read(sp)
                val_b = stack_read(sp - 1)
                stack_write(sp, val_b)
                stack_write(sp - 1, val_a)
                top = val_b
            elif op == OP_OVER:
                val_b = stack_read(sp - 1)
                sp += 1
                stack_write(sp, val_b)
                top = val_b
            elif op == OP_ROT:
                val_top    = stack_read(sp)
                val_second = stack_read(sp - 1)
                val_third  = stack_read(sp - 2)
                # [a, b, c] → [b, c, a]: third→top, top→second, second→third
                stack_write(sp, val_third)
                stack_write(sp - 1, val_top)
                stack_write(sp - 2, val_second)
                top = val_third
            elif op == OP_JZ:
                val = stack_read(sp)
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
                if val == 0:
                    next_ip = arg
            elif op == OP_JNZ:
                val = stack_read(sp)
                sp -= 1
                top = stack_read(sp) if sp > 0 else 0
                if val != 0:
                    next_ip = arg
            elif op == OP_NOP:
                top = stack_read(sp) if sp > 0 else 0
            elif op == OP_HALT:
                top = stack_read(sp) if sp > 0 else 0
                trace.steps.append(TraceStep(op, arg, sp, top))
                break
            else:
                raise RuntimeError(f"Unknown opcode {op}")

            trace.steps.append(TraceStep(op, arg, sp, top))
            ip = next_ip

        return trace


# ─── PyTorch Model with SP-2 Head ────────────────────────────────

class Phase13Model(PerceptaModel):
    """Compiled transformer extended with SWAP, OVER, ROT.

    Adds Head 4 (stack read at SP-2) for ROT support.
    Extends FF dispatch to 12 opcodes with 4-input value routing.
    """

    def __init__(self, d_model=D_MODEL):
        # Skip PerceptaModel.__init__(), do nn.Module init + build manually
        nn.Module.__init__(self)
        self.d_model = d_model

        # Heads 0-3: same as PerceptaModel
        self.head_prog_op  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_prog_arg = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_a  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_b  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1, use_bias_q=True)

        # Head 4: Stack read at SP-2 (for ROT)
        self.head_stack_c  = CompiledAttentionHead(d_model, head_dim=2, v_dim=1, use_bias_q=True)

        # FF dispatch: extended to 12 opcodes, 4 value inputs
        # M_top maps [arg, val_a, val_b, val_c] to candidate top values
        self.register_buffer('M_top', torch.zeros(N_OPCODES, 4, dtype=DTYPE))
        self.register_buffer('sp_deltas', torch.zeros(N_OPCODES, dtype=DTYPE))

        self._compile_weights()

    def _compile_weights(self):
        """Set all weight matrices analytically."""
        with torch.no_grad():
            # ── Head 0: Program opcode fetch ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_IP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_prog_op.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_PROG_KEY_0] = 1.0
            W[1, DIM_PROG_KEY_1] = 1.0
            self.head_prog_op.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_OPCODE] = 1.0
            self.head_prog_op.W_V.weight.copy_(W)

            # ── Head 1: Program argument fetch ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_IP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_prog_arg.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_PROG_KEY_0] = 1.0
            W[1, DIM_PROG_KEY_1] = 1.0
            self.head_prog_arg.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_prog_arg.W_V.weight.copy_(W)

            # ── Head 2: Stack read at SP ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_a.W_Q.weight.copy_(W)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_a.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_a.W_V.weight.copy_(W)

            # ── Head 3: Stack read at SP-1 ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_b.W_Q.weight.copy_(W)
            b = torch.zeros(2)
            b[0] = -1.0
            self.head_stack_b.W_Q.bias.copy_(b)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_b.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_b.W_V.weight.copy_(W)

            # ── Head 4: Stack read at SP-2 (NEW for ROT) ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP]  = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_c.W_Q.weight.copy_(W)
            b = torch.zeros(2)
            b[0] = -2.0  # offset: query becomes (sp-2, 1)
            self.head_stack_c.W_Q.bias.copy_(b)

            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_c.W_K.weight.copy_(W)

            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_c.W_V.weight.copy_(W)

            # ── FF dispatch: extended routing ──
            # M_top maps [arg, val_a, val_b, val_c] → candidate top per opcode
            #                         arg  va   vb   vc
            self.M_top[0]  = torch.tensor([ 1.,  0.,  0.,  0.])  # PUSH: top = arg
            self.M_top[1]  = torch.tensor([ 0.,  0.,  1.,  0.])  # POP:  top = val_b
            self.M_top[2]  = torch.tensor([ 0.,  1.,  1.,  0.])  # ADD:  top = va + vb
            self.M_top[3]  = torch.tensor([ 0.,  1.,  0.,  0.])  # DUP:  top = va
            self.M_top[4]  = torch.tensor([ 0.,  1.,  0.,  0.])  # HALT: top = va
            self.M_top[5]  = torch.tensor([ 0., -1.,  1.,  0.])  # SUB:  top = vb - va
            self.M_top[6]  = torch.tensor([ 0.,  0.,  1.,  0.])  # JZ:   top = vb
            self.M_top[7]  = torch.tensor([ 0.,  0.,  1.,  0.])  # JNZ:  top = vb
            self.M_top[8]  = torch.tensor([ 0.,  1.,  0.,  0.])  # NOP:  top = va
            self.M_top[9]  = torch.tensor([ 0.,  0.,  1.,  0.])  # SWAP: top = vb
            self.M_top[10] = torch.tensor([ 0.,  0.,  1.,  0.])  # OVER: top = vb
            self.M_top[11] = torch.tensor([ 0.,  0.,  0.,  1.])  # ROT:  top = vc

            # SP deltas
            #  PUSH POP  ADD  DUP HALT SUB  JZ  JNZ  NOP SWAP OVER ROT
            self.sp_deltas.copy_(torch.tensor(
                [1., -1., -1., 1., 0., -1., -1., -1., 0., 0., 1., 0.]))

    def forward(self, query_emb, prog_embs, stack_embs):
        """Execute one step. Returns (opcode, arg, sp_delta, top, opcode_one_hot, val_a, val_b, val_c)."""
        # Head 0: Fetch opcode
        opcode_val, _, _ = self.head_prog_op(query_emb, prog_embs)
        # Head 1: Fetch argument
        arg_val, _, _ = self.head_prog_arg(query_emb, prog_embs)

        # Head 2: Read stack[SP]
        if stack_embs.shape[0] > 0:
            val_a_raw, _, idx_a = self.head_stack_a(query_emb, stack_embs)
            stored_addr_a = round(stack_embs[idx_a, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp = round(query_emb[DIM_SP].item())
            val_a = val_a_raw[0] if stored_addr_a == queried_sp else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_a = torch.tensor(0.0, dtype=DTYPE)

        # Head 3: Read stack[SP-1]
        if stack_embs.shape[0] > 0:
            val_b_raw, _, idx_b = self.head_stack_b(query_emb, stack_embs)
            stored_addr_b = round(stack_embs[idx_b, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m1 = round(query_emb[DIM_SP].item()) - 1
            val_b = val_b_raw[0] if stored_addr_b == queried_sp_m1 else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_b = torch.tensor(0.0, dtype=DTYPE)

        # Head 4: Read stack[SP-2] (NEW)
        if stack_embs.shape[0] > 0:
            val_c_raw, _, idx_c = self.head_stack_c(query_emb, stack_embs)
            stored_addr_c = round(stack_embs[idx_c, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m2 = round(query_emb[DIM_SP].item()) - 2
            val_c = val_c_raw[0] if stored_addr_c == queried_sp_m2 else torch.tensor(0.0, dtype=DTYPE)
        else:
            val_c = torch.tensor(0.0, dtype=DTYPE)

        # Decode
        opcode = round(opcode_val[0].item())
        arg = round(arg_val[0].item())

        # FF Dispatch
        opcode_one_hot = torch.zeros(N_OPCODES, dtype=DTYPE)
        idx = OPCODE_IDX.get(opcode, -1)
        if idx >= 0:
            opcode_one_hot[idx] = 1.0

        values = torch.stack([
            torch.tensor(float(arg), dtype=DTYPE),
            val_a, val_b, val_c
        ])
        candidates = self.M_top @ values
        top = (opcode_one_hot * candidates).sum()
        sp_delta = (opcode_one_hot * self.sp_deltas).sum()

        return (opcode, arg, int(sp_delta.item()), round(top.item()),
                opcode_one_hot, round(val_a.item()), round(val_b.item()), round(val_c.item()))


# ─── PyTorch Executor ─────────────────────────────────────────────

class Phase13PyTorchExecutor:
    """Executes programs using Phase13Model with full ISA support."""

    def __init__(self, model=None):
        self.model = model or Phase13Model()
        self.model.eval()

    def execute(self, prog, max_steps=5000):
        trace = Trace(program=prog)

        prog_embs = torch.stack([
            embed_program_token_ext(i, instr)
            for i, instr in enumerate(prog)
        ])

        stack_embs_list = []
        write_count = 0
        ip = 0
        sp = 0

        with torch.no_grad():
            for step in range(max_steps):
                if ip >= len(prog):
                    break

                query = embed_state(ip, sp)
                stack_embs = (torch.stack(stack_embs_list)
                              if stack_embs_list
                              else torch.zeros(0, D_MODEL, dtype=DTYPE))

                opcode, arg, sp_delta, top, _, val_a, val_b, val_c = \
                    self.model.forward(query, prog_embs, stack_embs)

                if opcode == OP_HALT:
                    trace.steps.append(TraceStep(opcode, arg, sp, top))
                    break

                # For JZ/JNZ: read condition BEFORE updating SP
                cond_val = None
                if opcode in (OP_JZ, OP_JNZ):
                    cond_val = val_a  # stack[sp] before pop

                new_sp = sp + sp_delta

                # Stack writes per opcode
                if opcode in (OP_PUSH, OP_DUP, OP_OVER):
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode in (OP_ADD, OP_SUB):
                    stack_embs_list.append(
                        embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_SWAP:
                    # Write val_b at sp, val_a at sp-1
                    stack_embs_list.append(
                        embed_stack_entry(sp, val_b, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count))
                    write_count += 1
                elif opcode == OP_ROT:
                    # [a, b, c] → [b, c, a]: write a at sp, c at sp-1, b at sp-2
                    stack_embs_list.append(
                        embed_stack_entry(sp, val_c, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 2, val_b, write_count))
                    write_count += 1

                trace.steps.append(TraceStep(opcode, arg, new_sp, top))
                sp = new_sp

                # IP update
                if opcode == OP_JZ:
                    ip = arg if cond_val == 0 else ip + 1
                elif opcode == OP_JNZ:
                    ip = arg if cond_val != 0 else ip + 1
                else:
                    ip += 1

        return trace


# ─── Algorithm Suite: Program Generators ──────────────────────────

def fib(n):
    """Reference Fibonacci."""
    if n <= 0: return 0
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def make_fibonacci(n):
    """Generate a program that computes fib(n).

    Algorithm: iterative [counter, a, b] → SWAP, OVER, ADD → [counter, b, a+b]
    with ROT to cycle the counter.

    Returns (program, expected_result).
    """
    if n == 0:
        return [Instruction(OP_PUSH, 0), Instruction(OP_HALT)], 0
    if n == 1:
        return [Instruction(OP_PUSH, 1), Instruction(OP_HALT)], 1

    # n >= 2: need n-1 iterations
    prog = [
        Instruction(OP_PUSH, 0),      # 0: a = fib(0)
        Instruction(OP_PUSH, 1),      # 1: b = fib(1)
        Instruction(OP_PUSH, n - 1),  # 2: counter = n-1
        # Rearrange [0, 1, n-1] → [n-1, 0, 1]
        Instruction(OP_ROT),          # 3: [1, n-1, 0]
        Instruction(OP_ROT),          # 4: [n-1, 0, 1] = [counter, a, b]
        # ── Loop body (addr 5) ──
        # [counter, a, b] → [counter, b, a+b]
        Instruction(OP_SWAP),         # 5: [counter, b, a]
        Instruction(OP_OVER),         # 6: [counter, b, a, b]
        Instruction(OP_ADD),          # 7: [counter, b, a+b]
        # Cycle counter to top
        Instruction(OP_ROT),          # 8: [b, a+b, counter]
        # Decrement + test
        Instruction(OP_PUSH, 1),      # 9
        Instruction(OP_SUB),          # 10: [b, a+b, counter-1]
        Instruction(OP_DUP),          # 11: [..., counter-1, counter-1]
        Instruction(OP_JNZ, 15),      # 12: if counter-1 != 0 → continue
        # Exit: counter exhausted
        Instruction(OP_POP),          # 13: drop counter=0
        Instruction(OP_HALT),         # 14: top = fib(n)
        # ── Continue loop (addr 15) ──
        # Stack: [new_a, new_b, counter-1]
        # Rearrange to [counter-1, new_a, new_b]
        Instruction(OP_ROT),          # 15: [new_b, counter-1, new_a]
        Instruction(OP_ROT),          # 16: [counter-1, new_a, new_b]
        # Unconditional jump back
        Instruction(OP_PUSH, 1),      # 17
        Instruction(OP_JNZ, 5),       # 18: always taken
    ]
    return prog, fib(n)


def make_multiply(a, b):
    """Generate a program that computes a * b via repeated addition.

    Algorithm: accumulator starts at 0, add 'a' to it 'b' times.
    Stack layout: [a, result, counter]

    Returns (program, expected_result).
    """
    if b == 0 or a == 0:
        return [Instruction(OP_PUSH, 0), Instruction(OP_HALT)], 0

    prog = [
        Instruction(OP_PUSH, a),      # 0: a
        Instruction(OP_PUSH, 0),      # 1: result = 0
        Instruction(OP_PUSH, b),      # 2: counter = b
        # ── Loop (addr 3) ──
        Instruction(OP_DUP),          # 3: [a, result, counter, counter]
        Instruction(OP_JZ, 14),       # 4: if counter == 0 → done
        # Decrement counter
        Instruction(OP_PUSH, 1),      # 5
        Instruction(OP_SUB),          # 6: [a, result, counter-1]
        # Cycle: [a, result, counter-1] → [counter-1, a, result]
        Instruction(OP_ROT),          # 7: [result, counter-1, a]
        Instruction(OP_ROT),          # 8: [counter-1, a, result]
        # Add a to result: OVER copies a, ADD adds to result
        Instruction(OP_OVER),         # 9: [counter-1, a, result, a]
        Instruction(OP_ADD),          # 10: [counter-1, a, result+a]
        # Cycle back: [counter-1, a, result+a] → [a, result+a, counter-1]
        Instruction(OP_ROT),          # 11: [a, result+a, counter-1]
        # Unconditional jump
        Instruction(OP_PUSH, 1),      # 12
        Instruction(OP_JNZ, 3),       # 13: always taken
        # ── Done (addr 14) ──
        # Stack: [a, result, 0] (after JZ popped counter copy)
        Instruction(OP_POP),          # 14: [a, result]
        Instruction(OP_SWAP),         # 15: [result, a]
        Instruction(OP_POP),          # 16: [result]
        Instruction(OP_HALT),         # 17
    ]
    return prog, a * b


def make_power_of_2(n):
    """Generate a program that computes 2^n via repeated doubling.

    Algorithm: start with 1, DUP+ADD (doubles) n times.
    Stack layout: [value, counter]

    Returns (program, expected_result).
    """
    if n == 0:
        return [Instruction(OP_PUSH, 1), Instruction(OP_HALT)], 1

    prog = [
        Instruction(OP_PUSH, 1),      # 0: value = 1
        Instruction(OP_PUSH, n),      # 1: counter = n
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [value, counter, counter]
        Instruction(OP_JZ, 10),       # 3: if counter == 0 → done
        # Decrement counter
        Instruction(OP_PUSH, 1),      # 4
        Instruction(OP_SUB),          # 5: [value, counter-1]
        # Double the value: swap to top, dup+add, swap back
        Instruction(OP_SWAP),         # 6: [counter-1, value]
        Instruction(OP_DUP),          # 7: [counter-1, value, value]
        Instruction(OP_ADD),          # 8: [counter-1, 2*value]
        Instruction(OP_SWAP),         # 9: [2*value, counter-1]
        # Unconditional jump (counter-1 is on top; it may be 0, so use PUSH 1)
        # Actually, just loop back to the DUP/JZ test at addr 2
        Instruction(OP_PUSH, 1),      # 10 (label target adjusted below)
        Instruction(OP_JNZ, 2),       # 11
        # ── Done ──
        # Stack: [value, 0] (after JZ popped 0 copy)
        Instruction(OP_POP),          # 12 (label target adjusted below)
        Instruction(OP_HALT),         # 13
    ]
    # Fix: JZ target should be 12 (DONE = POP), unconditional jump at 10-11
    prog[3] = Instruction(OP_JZ, 12)
    return prog, 2 ** n


def make_sum_1_to_n(n):
    """Generate a program that computes 1 + 2 + ... + n.

    Algorithm: accumulator += counter, decrement counter, loop.
    Stack layout: [accumulator, counter]

    Returns (program, expected_result).
    """
    if n == 0:
        return [Instruction(OP_PUSH, 0), Instruction(OP_HALT)], 0

    prog = [
        Instruction(OP_PUSH, 0),      # 0: accumulator = 0
        Instruction(OP_PUSH, n),      # 1: counter = n
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [acc, counter, counter]
        Instruction(OP_JZ, 10),       # 3: if counter == 0 → done
        # Add counter to accumulator
        Instruction(OP_OVER),         # 4: [acc, counter, acc]
        Instruction(OP_ADD),          # 5: [acc, counter+acc]  wait...
        # Hmm, ADD pops counter and acc_copy, pushes sum.
        # Stack: [acc, counter+acc]. But we want [acc+counter, counter-1].
        # Rethink: use SWAP to get counter to accessible position first.
    ]
    # Let me redesign. Stack: [acc, counter]
    # Each iteration: acc += counter, counter -= 1
    prog = [
        Instruction(OP_PUSH, 0),      # 0: acc = 0
        Instruction(OP_PUSH, n),      # 1: counter = n
        # ── Loop (addr 2) ──
        Instruction(OP_DUP),          # 2: [acc, counter, counter]
        Instruction(OP_JZ, 12),       # 3: if counter == 0 → done
        # Save counter, add to acc
        Instruction(OP_DUP),          # 4: [acc, counter, counter]
        Instruction(OP_ROT),          # 5: [counter, counter, acc]
        Instruction(OP_ADD),          # 6: [counter, counter+acc]
        Instruction(OP_SWAP),         # 7: [counter+acc, counter]
        # Decrement counter
        Instruction(OP_PUSH, 1),      # 8
        Instruction(OP_SUB),          # 9: [counter+acc, counter-1]
        # Loop back
        Instruction(OP_PUSH, 1),      # 10
        Instruction(OP_JNZ, 2),       # 11: always jump
        # ── Done (addr 12) ──
        # Stack: [acc, 0] (after JZ popped 0 copy)
        Instruction(OP_POP),          # 12
        Instruction(OP_HALT),         # 13
    ]
    return prog, n * (n + 1) // 2


def make_is_even(n):
    """Generate a program that returns 1 if n is even, 0 if odd.

    Algorithm: subtract 2 repeatedly until reaching 0 (even) or 1 (odd).

    Returns (program, expected_result).
    """
    prog = [
        Instruction(OP_PUSH, n),      # 0: n
        # ── Loop (addr 1) ──
        Instruction(OP_DUP),          # 1: [n, n]
        Instruction(OP_JZ, 11),       # 2: if n == 0 → even
        # Check if n == 1
        Instruction(OP_PUSH, 1),      # 3: [n, 1]
        Instruction(OP_SUB),          # 4: [n-1]
        Instruction(OP_DUP),          # 5: [n-1, n-1]
        Instruction(OP_JZ, 14),       # 6: if n-1 == 0 → odd (n was 1)
        # Subtract 1 more (total -2)
        Instruction(OP_PUSH, 1),      # 7
        Instruction(OP_SUB),          # 8: [n-2]
        # Loop back
        Instruction(OP_PUSH, 1),      # 9
        Instruction(OP_JNZ, 1),       # 10: always jump
        # ── Even (addr 11) ──
        Instruction(OP_POP),          # 11: drop 0
        Instruction(OP_PUSH, 1),      # 12: result = 1
        Instruction(OP_HALT),         # 13
        # ── Odd (addr 14) ──
        Instruction(OP_POP),          # 14: drop 0
        Instruction(OP_PUSH, 0),      # 15: result = 0
        Instruction(OP_HALT),         # 16
    ]
    return prog, 1 if n % 2 == 0 else 0


# ─── Test Suite ───────────────────────────────────────────────────

def compare_traces(trace_a, trace_b):
    """Compare two traces token by token. Returns (match: bool, detail: str)."""
    if len(trace_a.steps) != len(trace_b.steps):
        return False, f"length mismatch: {len(trace_a.steps)} vs {len(trace_b.steps)}"
    for i, (a, b) in enumerate(zip(trace_a.steps, trace_b.steps)):
        if a.tokens() != b.tokens():
            return False, f"step {i}: {a.tokens()} vs {b.tokens()}"
    return True, "match"


def test_new_opcodes():
    """Test SWAP, OVER, ROT individually on both executors."""
    print("=" * 60)
    print("Test 1: New Opcode Unit Tests")
    print("=" * 60)

    np_exec = Phase13Executor()
    pt_exec = Phase13PyTorchExecutor()

    tests = [
        # (name, program, expected_top)
        ("swap_basic",
         [Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 5),
          Instruction(OP_SWAP), Instruction(OP_HALT)],
         3),
        ("swap_pop_verify",
         [Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 5),
          Instruction(OP_SWAP), Instruction(OP_POP), Instruction(OP_HALT)],
         5),
        ("over_basic",
         [Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 5),
          Instruction(OP_OVER), Instruction(OP_HALT)],
         3),
        ("over_preserves",
         [Instruction(OP_PUSH, 3), Instruction(OP_PUSH, 5),
          Instruction(OP_OVER), Instruction(OP_POP), Instruction(OP_HALT)],
         5),
        ("rot_basic",
         [Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 3),
          Instruction(OP_ROT), Instruction(OP_HALT)],
         1),  # [1,2,3] → ROT → [2,3,1], top=1
        ("rot_second",
         [Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 3),
          Instruction(OP_ROT), Instruction(OP_POP), Instruction(OP_HALT)],
         3),  # [2,3,1] → POP → [2,3], top=3
        ("rot_third",
         [Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 3),
          Instruction(OP_ROT), Instruction(OP_POP), Instruction(OP_POP),
          Instruction(OP_HALT)],
         2),  # [2,3] → POP → [2], top=2
        ("rot_twice_identity_shift",
         [Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 3),
          Instruction(OP_ROT), Instruction(OP_ROT), Instruction(OP_HALT)],
         2),  # ROT×2: [1,2,3]→[2,3,1]→[3,1,2], top=2
        ("rot_thrice_identity",
         [Instruction(OP_PUSH, 1), Instruction(OP_PUSH, 2), Instruction(OP_PUSH, 3),
          Instruction(OP_ROT), Instruction(OP_ROT), Instruction(OP_ROT),
          Instruction(OP_HALT)],
         3),  # ROT×3 = identity: [1,2,3], top=3
    ]

    passed = 0
    total = len(tests) * 2  # numpy + pytorch

    for name, prog, expected in tests:
        for label, executor in [("numpy", np_exec), ("torch", pt_exec)]:
            trace = executor.execute(prog)
            top = trace.steps[-1].top if trace.steps else None
            ok = (top == expected)
            if ok:
                passed += 1
            status = "PASS" if ok else "FAIL"
            print(f"  {status}  {label:5s}  {name:30s}  expected={expected:>5}  got={top}")

    # Also verify numpy/pytorch traces match
    trace_match = 0
    for name, prog, _ in tests:
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        match, _ = compare_traces(np_trace, pt_trace)
        if match:
            trace_match += 1

    print(f"\n  Opcode tests: {passed}/{total} passed")
    print(f"  Trace match:  {trace_match}/{len(tests)} numpy==pytorch")
    return passed == total and trace_match == len(tests)


def test_head_sp2():
    """Verify Head 4 (SP-2 read) works correctly."""
    print("\n" + "=" * 60)
    print("Test 2: Head 4 (SP-2 Read) Verification")
    print("=" * 60)

    model = Phase13Model()
    model.eval()

    # Stack: addr=1→42, addr=2→7, addr=3→99
    stack_entries = [
        embed_stack_entry(1, 42, 0),
        embed_stack_entry(2, 7, 1),
        embed_stack_entry(3, 99, 2),
    ]
    stack_embs = torch.stack(stack_entries)

    passed = 0
    total = 0

    with torch.no_grad():
        # SP=3: head_stack_c should read addr=1 (SP-2=1) → 42
        query = embed_state(0, 3)
        val, _, idx = model.head_stack_c(query, stack_embs)
        stored_addr = round(stack_embs[idx, DIM_STACK_KEY_0].item() / 2.0)
        fetched = round(val[0].item()) if stored_addr == 1 else 0
        ok = (fetched == 42)
        total += 1
        if ok: passed += 1
        print(f"  {'PASS' if ok else 'FAIL'}  SP=3 → stack[1] = {fetched} (expected 42)")

        # SP=4: head_stack_c should read addr=2 (SP-2=2) → 7
        stack_entries.append(embed_stack_entry(4, 100, 3))
        stack_embs = torch.stack(stack_entries)
        query = embed_state(0, 4)
        val, _, idx = model.head_stack_c(query, stack_embs)
        stored_addr = round(stack_embs[idx, DIM_STACK_KEY_0].item() / 2.0)
        fetched = round(val[0].item()) if stored_addr == 2 else 0
        ok = (fetched == 7)
        total += 1
        if ok: passed += 1
        print(f"  {'PASS' if ok else 'FAIL'}  SP=4 → stack[2] = {fetched} (expected 7)")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def test_algorithm(name, prog, expected, np_exec, pt_exec, verbose=False):
    """Run an algorithm on both executors and verify."""
    np_trace = np_exec.execute(prog)
    pt_trace = pt_exec.execute(prog)

    np_top = np_trace.steps[-1].top if np_trace.steps else None
    pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
    match, detail = compare_traces(np_trace, pt_trace)

    np_ok = (np_top == expected)
    pt_ok = (pt_top == expected)
    all_ok = np_ok and pt_ok and match

    status = "PASS" if all_ok else "FAIL"
    print(f"  {status}  {name:30s}  expected={expected:>6}  "
          f"numpy={np_top:>6}  torch={pt_top:>6}  "
          f"steps={len(np_trace.steps):>4}  trace_match={'Y' if match else 'N'}")

    if not all_ok and verbose:
        if not match:
            print(f"         Trace mismatch: {detail}")
        if not np_ok:
            print(f"         NumPy wrong: got {np_top}, expected {expected}")
        if not pt_ok:
            print(f"         PyTorch wrong: got {pt_top}, expected {expected}")

    return all_ok, len(np_trace.steps)


def test_fibonacci():
    """Test Fibonacci on multiple inputs."""
    print("\n" + "=" * 60)
    print("Test 3: Fibonacci")
    print("=" * 60)

    np_exec = Phase13Executor()
    pt_exec = Phase13PyTorchExecutor()

    cases = [0, 1, 2, 3, 5, 7, 10]
    passed = 0
    for n in cases:
        prog, expected = make_fibonacci(n)
        ok, steps = test_algorithm(f"fib({n})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_multiply():
    """Test multiplication via repeated addition."""
    print("\n" + "=" * 60)
    print("Test 4: Multiply (repeated addition)")
    print("=" * 60)

    np_exec = Phase13Executor()
    pt_exec = Phase13PyTorchExecutor()

    cases = [(0, 5), (5, 0), (1, 7), (3, 4), (5, 3), (7, 7), (12, 10)]
    passed = 0
    for a, b in cases:
        prog, expected = make_multiply(a, b)
        ok, steps = test_algorithm(f"mul({a},{b})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_power_of_2():
    """Test 2^n via repeated doubling."""
    print("\n" + "=" * 60)
    print("Test 5: Power of 2 (repeated doubling)")
    print("=" * 60)

    np_exec = Phase13Executor()
    pt_exec = Phase13PyTorchExecutor()

    cases = [0, 1, 2, 3, 4, 5, 7]
    passed = 0
    for n in cases:
        prog, expected = make_power_of_2(n)
        ok, steps = test_algorithm(f"2^{n}", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_sum_1_to_n():
    """Test sum of 1..n."""
    print("\n" + "=" * 60)
    print("Test 6: Sum of 1..N")
    print("=" * 60)

    np_exec = Phase13Executor()
    pt_exec = Phase13PyTorchExecutor()

    cases = [0, 1, 2, 5, 10, 15]
    passed = 0
    for n in cases:
        prog, expected = make_sum_1_to_n(n)
        ok, steps = test_algorithm(f"sum(1..{n})", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_is_even():
    """Test parity via repeated subtraction."""
    print("\n" + "=" * 60)
    print("Test 7: Parity Test (is_even)")
    print("=" * 60)

    np_exec = Phase13Executor()
    pt_exec = Phase13PyTorchExecutor()

    cases = [0, 1, 2, 3, 4, 7, 10, 15, 20]
    passed = 0
    for n in cases:
        prog, expected = make_is_even(n)
        label = "even" if expected else "odd"
        ok, steps = test_algorithm(f"is_even({n})→{label}", prog, expected, np_exec, pt_exec, verbose=True)
        if ok: passed += 1

    print(f"\n  Result: {passed}/{len(cases)} passed")
    return passed == len(cases)


def test_regression():
    """Verify existing Phase 4/11 test programs still work."""
    print("\n" + "=" * 60)
    print("Test 8: Regression (Phase 4 + Phase 11 tests)")
    print("=" * 60)

    np_exec = Phase13Executor()
    pt_exec = Phase13PyTorchExecutor()

    passed = 0
    total = len(ALL_TESTS)

    for name, test_fn in ALL_TESTS:
        prog, expected_top = test_fn()
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
        match, _ = compare_traces(np_trace, pt_trace)
        ok = (np_top == expected_top and pt_top == expected_top and match)
        if ok: passed += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:20s}  expected={expected_top:>5}  "
              f"numpy={np_top}  torch={pt_top}  match={'Y' if match else 'N'}")

    # Extended ISA regression (Phase 11 tests)
    ext_tests = [
        ("sub_basic",
         [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 3),
          Instruction(OP_SUB), Instruction(OP_HALT)], 7),
        ("loop_countdown",
         [Instruction(OP_PUSH, 3), Instruction(OP_DUP),
          Instruction(OP_PUSH, 1), Instruction(OP_SUB),
          Instruction(OP_DUP), Instruction(OP_JNZ, 1),
          Instruction(OP_HALT)], 0),
        ("jz_taken",
         [Instruction(OP_PUSH, 0), Instruction(OP_JZ, 3),
          Instruction(OP_HALT),
          Instruction(OP_PUSH, 42), Instruction(OP_HALT)], 42),
    ]

    for name, prog, expected in ext_tests:
        np_trace = np_exec.execute(prog)
        pt_trace = pt_exec.execute(prog)
        np_top = np_trace.steps[-1].top if np_trace.steps else None
        pt_top = pt_trace.steps[-1].top if pt_trace.steps else None
        match, _ = compare_traces(np_trace, pt_trace)
        ok = (np_top == expected and pt_top == expected and match)
        if ok: passed += 1
        total += 1
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name:20s}  expected={expected:>5}  "
              f"numpy={np_top}  torch={pt_top}  match={'Y' if match else 'N'}")

    print(f"\n  Result: {passed}/{total} passed")
    return passed == total


def test_model_summary():
    """Report model architecture and parameter counts."""
    print("\n" + "=" * 60)
    print("Model Architecture Summary")
    print("=" * 60)

    model = Phase13Model()
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())

    print(f"  d_model:          {D_MODEL}")
    print(f"  head_dim:         2")
    print(f"  n_active_heads:   5 (was 4 in Phase 12)")
    print(f"  head_slots:       18 (13 still reserved)")
    print(f"  ISA opcodes:      {N_OPCODES} (was {N_OPCODES_BASE} in Phase 12)")
    print(f"  trainable params: {total_params}")
    print(f"  buffer params:    {total_buffers}")
    print(f"  total compiled:   {total_params + total_buffers}")
    print()

    # Head breakdown
    heads = [
        ("Head 0: Prog opcode", model.head_prog_op),
        ("Head 1: Prog arg",    model.head_prog_arg),
        ("Head 2: Stack SP",    model.head_stack_a),
        ("Head 3: Stack SP-1",  model.head_stack_b),
        ("Head 4: Stack SP-2",  model.head_stack_c),
    ]
    head_params = 0
    for name, head in heads:
        hp = sum(p.numel() for p in head.parameters())
        head_params += hp
        print(f"  {name:25s}  params={hp}")
    print(f"  {'FF dispatch (buffers)':25s}  params={total_buffers}")
    print(f"  {'Total':25s}  params={total_params + total_buffers}")

    return True


# ─── Main ─────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 13: General-Purpose Stack Computer — ISA Completeness")
    print("=" * 60)
    print(f"  Extended ISA: {', '.join(OP_NAMES_P13[op] for op in sorted(OP_NAMES_P13))}")
    print(f"  New opcodes:  SWAP, OVER, ROT")
    print(f"  New head:     Head 4 (stack SP-2 read)")
    print()

    t0 = time.time()
    results = []

    results.append(("New opcodes",    test_new_opcodes()))
    results.append(("Head SP-2",      test_head_sp2()))
    results.append(("Fibonacci",      test_fibonacci()))
    results.append(("Multiply",       test_multiply()))
    results.append(("Power of 2",     test_power_of_2()))
    results.append(("Sum 1..N",       test_sum_1_to_n()))
    results.append(("Parity",         test_is_even()))
    results.append(("Regression",     test_regression()))
    results.append(("Model summary",  test_model_summary()))

    elapsed = time.time() - t0

    # ── Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok: all_pass = False
        print(f"  {status}  {name}")

    print(f"\n  Time: {elapsed:.2f}s")

    if all_pass:
        print("\n  ✓ The compiled transformer is a GENERAL-PURPOSE STACK COMPUTER.")
        print("    All algorithms execute correctly on both numpy and PyTorch executors.")
        print("    The extended ISA (12 opcodes, 5 attention heads) supports:")
        print("      - Iteration with accumulation (Fibonacci, Sum)")
        print("      - Nested-loop-equivalent computation (Multiply)")
        print("      - Repeated doubling (Power of 2)")
        print("      - Conditional branching (Parity test)")
        print("      - 3-value stack juggling (ROT, SWAP, OVER)")
    else:
        print("\n  ✗ Some tests failed. See details above.")

    return all_pass


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
