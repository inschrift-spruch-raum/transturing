"""PyTorch-based executor backend for the compiled transformer stack machine.

Contains:
  - DTYPE, EPS: torch-specific constants
  - CompiledAttentionHead: hard-max attention head (nn.Module)
  - TokenVocab: token vocabulary with compiled embeddings
  - Embedding functions: embed_program_token, embed_stack_entry, etc.
  - CompiledModel: full compiled transformer (nn.Module)
  - TorchExecutor: executes programs via CompiledModel
"""

import torch
from torch import nn

from transturing.core.abc import ExecutorBackend
from transturing.core.isa import (
    D_MODEL,
    DIM_CALL_KEY_0,
    DIM_CALL_KEY_1,
    DIM_CALL_LOCALS_BASE,
    DIM_CALL_RET_ADDR,
    DIM_CALL_SAVED_SP,
    DIM_HEAP_KEY_0,
    DIM_HEAP_KEY_1,
    DIM_IP,
    DIM_IS_CALL_STACK,
    DIM_IS_HEAP,
    DIM_IS_LOCAL,
    DIM_IS_PROG,
    DIM_IS_STACK,
    DIM_IS_STATE,
    DIM_LOCAL_KEY_0,
    DIM_LOCAL_KEY_1,
    DIM_ONE,
    DIM_OPCODE,
    DIM_PROG_KEY_0,
    DIM_PROG_KEY_1,
    DIM_SP,
    DIM_STACK_KEY_0,
    DIM_STACK_KEY_1,
    DIM_VALUE,
    MASK32,
    N_OPCODES,
    OP_ABS,
    OP_ADD,
    OP_AND,
    OP_CALL,
    OP_CLZ,
    OP_CTZ,
    OP_DIV_S,
    OP_DIV_U,
    OP_DUP,
    OP_EQ,
    OP_EQZ,
    OP_GE_S,
    OP_GE_U,
    OP_GT_S,
    OP_GT_U,
    OP_HALT,
    OP_I32_LOAD,
    OP_I32_LOAD8_S,
    OP_I32_LOAD8_U,
    OP_I32_LOAD16_S,
    OP_I32_LOAD16_U,
    OP_I32_STORE,
    OP_I32_STORE8,
    OP_I32_STORE16,
    OP_JNZ,
    OP_JZ,
    OP_LE_S,
    OP_LE_U,
    OP_LOCAL_GET,
    OP_LOCAL_SET,
    OP_LOCAL_TEE,
    OP_LT_S,
    OP_LT_U,
    OP_MUL,
    OP_NAMES,
    OP_NE,
    OP_NEG,
    OP_OR,
    OP_OVER,
    OP_POPCNT,
    OP_PUSH,
    OP_REM_S,
    OP_REM_U,
    OP_RETURN,
    OP_ROT,
    OP_ROTL,
    OP_ROTR,
    OP_SELECT,
    OP_SHL,
    OP_SHR_S,
    OP_SHR_U,
    OP_SUB,
    OP_SWAP,
    OP_TRAP,
    OP_XOR,
    OPCODE_DIM_MAP,
    OPCODE_IDX,
    Trace,
    TraceStep,
    _clz32,
    _ctz32,
    _popcnt32,
    _rotl32,
    _rotr32,
    _shr_s,
    _shr_u,
    _sign_extend_8,
    _sign_extend_16,
    _to_i32,
    _trunc_div,
    _trunc_rem,
)
from transturing.core.registry import register_backend

# ─── Torch-specific constants ────────────────────────────────────

DTYPE = torch.float64
EPS = 1e-6


# ─── Compiled Attention Head ──────────────────────────────────────


class CompiledAttentionHead(nn.Module):
    """Hard-max attention head with analytically set W_Q, W_K, W_V."""

    def __init__(self, d_model=D_MODEL, head_dim=2, v_dim=1, use_bias_q=False):
        super().__init__()
        self.W_Q = nn.Linear(d_model, head_dim, bias=use_bias_q)
        self.W_K = nn.Linear(d_model, head_dim, bias=False)
        self.W_V = nn.Linear(d_model, v_dim, bias=False)
        self.double()

    def forward(self, query_emb, memory_embs):
        """Hard-max attention lookup."""
        if memory_embs.shape[0] == 0:
            return (
                torch.zeros(self.W_V.out_features, dtype=DTYPE),
                torch.tensor(-float("inf"), dtype=DTYPE),
                -1,
            )

        q = self.W_Q(query_emb)
        K = self.W_K(memory_embs)
        V = self.W_V(memory_embs)

        scores = K @ q
        best = scores.argmax().item()

        return V[best], scores[best], best


# ─── Token Vocabulary ──────────────────────────────────────────────


class TokenVocab:
    """Fixed token vocabulary for the compiled transformer."""

    OPCODE = "op"
    VALUE = "val"
    SP_DELTA = "sp_delta"
    SPECIAL = "special"

    PAD = "PAD"
    COMMIT = "COMMIT"
    BRANCH_TAKEN = "BRANCH_TAKEN"
    BRANCH_NOT_TAKEN = "BRANCH_NOT_TAKEN"

    _SPECIAL_BASE = 0
    _SPECIAL_COUNT = 4
    _OPCODE_BASE = _SPECIAL_BASE + _SPECIAL_COUNT
    _OPCODE_COUNT = N_OPCODES + 1
    _VALUE_BASE = _OPCODE_BASE + _OPCODE_COUNT
    _VALUE_COUNT = 256
    _SP_DELTA_BASE = _VALUE_BASE + _VALUE_COUNT
    _SP_DELTA_MIN = -3
    _SP_DELTA_MAX = 3
    _SP_DELTA_COUNT = _SP_DELTA_MAX - _SP_DELTA_MIN + 1

    VOCAB_SIZE = _SP_DELTA_BASE + _SP_DELTA_COUNT

    PAD_ID = 0
    COMMIT_ID = 1
    BRANCH_TAKEN_ID = 2
    BRANCH_NOT_TAKEN_ID = 3

    _SPECIAL_NAMES = {
        PAD_ID: PAD,
        COMMIT_ID: COMMIT,
        BRANCH_TAKEN_ID: BRANCH_TAKEN,
        BRANCH_NOT_TAKEN_ID: BRANCH_NOT_TAKEN,
    }
    _SPECIAL_IDS = {v: k for k, v in _SPECIAL_NAMES.items()}

    def __init__(self):
        self._opcode_to_tid = {}
        self._tid_to_opcode = {}
        for op_code, idx in OPCODE_IDX.items():
            tid = self._OPCODE_BASE + idx
            self._opcode_to_tid[op_code] = tid
            self._tid_to_opcode[tid] = op_code
        trap_tid = self._OPCODE_BASE + N_OPCODES
        self._opcode_to_tid[OP_TRAP] = trap_tid
        self._tid_to_opcode[trap_tid] = OP_TRAP
        self.vocab_size = self.VOCAB_SIZE

    def encode(self, token):
        """Encode a token to its vocabulary ID."""
        if isinstance(token, str):
            if token in self._SPECIAL_IDS:
                return self._SPECIAL_IDS[token]
            raise ValueError(f"Unknown special token: {token!r}")
        tag, val = token
        if tag == self.OPCODE:
            if val not in self._opcode_to_tid:
                raise ValueError(f"Unknown opcode: {val}")
            return self._opcode_to_tid[val]
        if tag == self.VALUE:
            if not (0 <= val <= 255):
                raise ValueError(f"Value out of byte range: {val}")
            return self._VALUE_BASE + val
        if tag == self.SP_DELTA:
            if not (self._SP_DELTA_MIN <= val <= self._SP_DELTA_MAX):
                raise ValueError(f"SP delta {val} out of range")
            return self._SP_DELTA_BASE + (val - self._SP_DELTA_MIN)
        if tag == self.SPECIAL:
            if val in self._SPECIAL_IDS:
                return self._SPECIAL_IDS[val]
            raise ValueError(f"Unknown special token: {val!r}")
        raise ValueError(f"Unknown token tag: {tag!r}")

    def decode(self, tid):
        """Decode a token ID back to its structured representation."""
        if not (0 <= tid < self.vocab_size):
            raise ValueError(f"Token ID {tid} out of range")
        if tid < self._OPCODE_BASE:
            return (self.SPECIAL, self._SPECIAL_NAMES[tid])
        if tid < self._VALUE_BASE:
            return (self.OPCODE, self._tid_to_opcode[tid])
        if tid < self._SP_DELTA_BASE:
            return (self.VALUE, tid - self._VALUE_BASE)
        return (self.SP_DELTA, (tid - self._SP_DELTA_BASE) + self._SP_DELTA_MIN)

    def compile_embedding(self, d_model=None):
        """Build nn.Embedding with analytically set weights."""
        if d_model is None:
            d_model = D_MODEL
        emb = nn.Embedding(self.vocab_size, d_model)
        W = torch.zeros(self.vocab_size, d_model, dtype=DTYPE)

        W[self.COMMIT_ID, DIM_ONE] = 1.0
        W[self.COMMIT_ID, DIM_OPCODE] = -1.0
        W[self.BRANCH_TAKEN_ID, DIM_ONE] = 1.0
        W[self.BRANCH_TAKEN_ID, DIM_OPCODE] = -2.0
        W[self.BRANCH_NOT_TAKEN_ID, DIM_ONE] = 1.0
        W[self.BRANCH_NOT_TAKEN_ID, DIM_OPCODE] = -3.0

        for op_code, tid in self._opcode_to_tid.items():
            W[tid, DIM_IS_PROG] = 1.0
            W[tid, DIM_OPCODE] = float(op_code)
            W[tid, DIM_ONE] = 1.0
            dim = OPCODE_DIM_MAP.get(op_code)
            if dim is not None and dim < d_model:
                W[tid, dim] = 1.0

        for v in range(256):
            tid = self._VALUE_BASE + v
            W[tid, DIM_IS_STACK] = 1.0
            W[tid, DIM_VALUE] = float(v)
            W[tid, DIM_ONE] = 1.0

        for delta in range(self._SP_DELTA_MIN, self._SP_DELTA_MAX + 1):
            tid = self._SP_DELTA_BASE + (delta - self._SP_DELTA_MIN)
            W[tid, DIM_IS_STATE] = 1.0
            W[tid, DIM_SP] = float(delta)
            W[tid, DIM_ONE] = 1.0

        emb.weight = nn.Parameter(W, requires_grad=False)
        return emb

    def compile_unembedding(self, embedding=None, d_model=None):
        """Build nn.Linear unembedding head."""
        if d_model is None:
            d_model = D_MODEL
        if embedding is None:
            embedding = self.compile_embedding(d_model)
        E = embedding.weight.data
        unembed = nn.Linear(d_model, self.vocab_size, bias=True)
        norms_sq = (E * E).sum(dim=1)
        unembed.weight = nn.Parameter(E.clone(), requires_grad=False)
        unembed.bias = nn.Parameter(-0.5 * norms_sq, requires_grad=False)
        return unembed

    def opcode_name(self, op_code):
        return OP_NAMES.get(op_code, f"?{op_code}")

    def token_name(self, tid):
        tag, val = self.decode(tid)
        if tag == self.SPECIAL:
            return val
        if tag == self.OPCODE:
            return self.opcode_name(val)
        if tag == self.VALUE:
            return f"V{val}"
        if tag == self.SP_DELTA:
            return f"SP{'+' if val >= 0 else ''}{val}"
        return f"?{tid}"

    def __repr__(self):
        return (
            f"TokenVocab(vocab_size={self.vocab_size}, "
            f"opcodes={len(self._opcode_to_tid)}, values=256, "
            f"sp_deltas={self._SP_DELTA_COUNT}, specials={self._SPECIAL_COUNT})"
        )


# ─── Embedding Functions ──────────────────────────────────────────


def embed_program_token(pos, instr):
    """Create d_model-dim embedding for a program instruction."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_PROG] = 1.0
    emb[DIM_PROG_KEY_0] = 2.0 * pos
    emb[DIM_PROG_KEY_1] = -float(pos * pos)
    emb[DIM_OPCODE] = float(instr.op)
    emb[DIM_VALUE] = float(instr.arg)
    emb[DIM_ONE] = 1.0
    dim = OPCODE_DIM_MAP.get(instr.op)
    if dim is not None:
        emb[dim] = 1.0
    return emb


def embed_stack_entry(addr, value, write_order):
    """Create d_model-dim embedding for a stack write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_STACK] = 1.0
    emb[DIM_STACK_KEY_0] = 2.0 * addr
    emb[DIM_STACK_KEY_1] = -float(addr * addr) + EPS * write_order
    emb[DIM_VALUE] = float(value)
    emb[DIM_ONE] = 1.0
    return emb


def embed_local_entry(local_idx, value, write_order):
    """Create embedding for a local variable write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_LOCAL] = 1.0
    emb[DIM_LOCAL_KEY_0] = 2.0 * local_idx
    emb[DIM_LOCAL_KEY_1] = -float(local_idx * local_idx) + EPS * write_order
    emb[DIM_VALUE] = float(value)
    emb[DIM_ONE] = 1.0
    return emb


def embed_heap_entry(addr, value, write_order):
    """Create embedding for a heap memory write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_HEAP] = 1.0
    emb[DIM_HEAP_KEY_0] = 2.0 * addr
    emb[DIM_HEAP_KEY_1] = -float(addr * addr) + EPS * write_order
    emb[DIM_VALUE] = float(value)
    emb[DIM_ONE] = 1.0
    return emb


def embed_call_frame(depth, ret_addr, saved_sp, locals_base, write_order):
    """Create embedding for a call stack frame."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_CALL_STACK] = 1.0
    emb[DIM_CALL_KEY_0] = 2.0 * depth
    emb[DIM_CALL_KEY_1] = -float(depth * depth) + EPS * write_order
    emb[DIM_CALL_RET_ADDR] = float(ret_addr)
    emb[DIM_CALL_SAVED_SP] = float(saved_sp)
    emb[DIM_CALL_LOCALS_BASE] = float(locals_base)
    emb[DIM_ONE] = 1.0
    return emb


def embed_state(ip, sp):
    """Create d_model-dim query embedding encoding current execution state."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_STATE] = 1.0
    emb[DIM_IP] = float(ip)
    emb[DIM_SP] = float(sp)
    emb[DIM_ONE] = 1.0
    return emb


# ─── Compiled PyTorch Model ─────────────────────────────────────────


class CompiledModel(nn.Module):
    """Compiled transformer with 10 attention heads and linear+nonlinear FF dispatch."""

    def __init__(self, d_model=D_MODEL):
        nn.Module.__init__(self)
        self.d_model = d_model

        self.head_prog_op = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_prog_arg = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_a = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_stack_b = CompiledAttentionHead(
            d_model,
            head_dim=2,
            v_dim=1,
            use_bias_q=True,
        )
        self.head_stack_c = CompiledAttentionHead(
            d_model,
            head_dim=2,
            v_dim=1,
            use_bias_q=True,
        )
        self.head_local_val = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_local_addr = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_heap_val = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_heap_addr = CompiledAttentionHead(d_model, head_dim=2, v_dim=1)
        self.head_call_stack = CompiledAttentionHead(d_model, head_dim=2, v_dim=3)

        self.register_buffer("M_top", torch.zeros(N_OPCODES, 6, dtype=DTYPE))
        self.register_buffer("sp_deltas", torch.zeros(N_OPCODES, dtype=DTYPE))

        self._compile_weights()

    def _compile_weights(self):
        """Set all weight matrices analytically."""
        with torch.no_grad():
            # ── Head 0: Program opcode fetch ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_IP] = 1.0
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
            W[0, DIM_IP] = 1.0
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
            W[0, DIM_SP] = 1.0
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
            W[0, DIM_SP] = 1.0
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

            # ── Head 4: Stack read at SP-2 ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_SP] = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_stack_c.W_Q.weight.copy_(W)
            b = torch.zeros(2)
            b[0] = -2.0
            self.head_stack_c.W_Q.bias.copy_(b)
            W = torch.zeros(2, self.d_model)
            W[0, DIM_STACK_KEY_0] = 1.0
            W[1, DIM_STACK_KEY_1] = 1.0
            self.head_stack_c.W_K.weight.copy_(W)
            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_stack_c.W_V.weight.copy_(W)

            # ── Head 5: Local value fetch ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE] = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_local_val.W_Q.weight.copy_(W)
            W = torch.zeros(2, self.d_model)
            W[0, DIM_LOCAL_KEY_0] = 1.0
            W[1, DIM_LOCAL_KEY_1] = 1.0
            self.head_local_val.W_K.weight.copy_(W)
            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_local_val.W_V.weight.copy_(W)

            # ── Head 6: Local address verify ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE] = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_local_addr.W_Q.weight.copy_(W)
            W = torch.zeros(2, self.d_model)
            W[0, DIM_LOCAL_KEY_0] = 1.0
            W[1, DIM_LOCAL_KEY_1] = 1.0
            self.head_local_addr.W_K.weight.copy_(W)
            W = torch.zeros(1, self.d_model)
            W[0, DIM_LOCAL_KEY_0] = 0.5
            self.head_local_addr.W_V.weight.copy_(W)

            # ── Head 7: Heap value fetch ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE] = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_heap_val.W_Q.weight.copy_(W)
            W = torch.zeros(2, self.d_model)
            W[0, DIM_HEAP_KEY_0] = 1.0
            W[1, DIM_HEAP_KEY_1] = 1.0
            self.head_heap_val.W_K.weight.copy_(W)
            W = torch.zeros(1, self.d_model)
            W[0, DIM_VALUE] = 1.0
            self.head_heap_val.W_V.weight.copy_(W)

            # ── Head 8: Heap address verify ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE] = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_heap_addr.W_Q.weight.copy_(W)
            W = torch.zeros(2, self.d_model)
            W[0, DIM_HEAP_KEY_0] = 1.0
            W[1, DIM_HEAP_KEY_1] = 1.0
            self.head_heap_addr.W_K.weight.copy_(W)
            W = torch.zeros(1, self.d_model)
            W[0, DIM_HEAP_KEY_0] = 0.5
            self.head_heap_addr.W_V.weight.copy_(W)

            # ── Head 9: Call stack read ──
            W = torch.zeros(2, self.d_model)
            W[0, DIM_VALUE] = 1.0
            W[1, DIM_ONE] = 1.0
            self.head_call_stack.W_Q.weight.copy_(W)
            W = torch.zeros(2, self.d_model)
            W[0, DIM_CALL_KEY_0] = 1.0
            W[1, DIM_CALL_KEY_1] = 1.0
            self.head_call_stack.W_K.weight.copy_(W)
            W = torch.zeros(3, self.d_model)
            W[0, DIM_CALL_RET_ADDR] = 1.0
            W[1, DIM_CALL_SAVED_SP] = 1.0
            W[2, DIM_CALL_LOCALS_BASE] = 1.0
            self.head_call_stack.W_V.weight.copy_(W)

            # ── FF dispatch: linear routing ──
            self.M_top[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.M_top[1] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.M_top[2] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.M_top[3] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            self.M_top[4] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            self.M_top[5] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            self.M_top[6] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.M_top[7] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.M_top[8] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            self.M_top[9] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.M_top[10] = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.M_top[11] = torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
            self.M_top[12] = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

            self.M_top[OPCODE_IDX[OP_LOCAL_GET]] = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            )
            self.M_top[OPCODE_IDX[OP_LOCAL_SET]] = torch.tensor(
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            )
            self.M_top[OPCODE_IDX[OP_LOCAL_TEE]] = torch.tensor(
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            )
            self.M_top[OPCODE_IDX[OP_I32_LOAD]] = torch.tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )
            self.M_top[OPCODE_IDX[OP_I32_STORE]] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            self.M_top[OPCODE_IDX[OP_I32_STORE8]] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            self.M_top[OPCODE_IDX[OP_I32_STORE16]] = torch.tensor(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            )
            self.M_top[OPCODE_IDX[OP_CALL]] = torch.tensor(
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            )
            self.M_top[OPCODE_IDX[OP_RETURN]] = torch.tensor(
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            )

            self.sp_deltas.copy_(
                torch.tensor(
                    [
                        1.0,
                        -1.0,
                        -1.0,
                        1.0,
                        0.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        0.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -2.0,
                        1.0,
                        -1.0,
                        0.0,
                        0.0,
                        -2.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        -2.0,
                        -2.0,
                        0.0,
                        0.0,
                    ],
                ),
            )

    def forward(
        self,
        query_emb,
        prog_embs,
        stack_embs=None,
        local_embs=None,
        heap_embs=None,
        call_embs=None,
        locals_base=0,
    ):
        """Execute one step."""
        if stack_embs is None:
            stack_embs = torch.zeros(0, self.d_model, dtype=DTYPE)
        if local_embs is None:
            local_embs = torch.zeros(0, self.d_model, dtype=DTYPE)
        if heap_embs is None:
            heap_embs = torch.zeros(0, self.d_model, dtype=DTYPE)
        if call_embs is None:
            call_embs = torch.zeros(0, self.d_model, dtype=DTYPE)

        opcode_val, _, _ = self.head_prog_op(query_emb, prog_embs)
        arg_val, _, _ = self.head_prog_arg(query_emb, prog_embs)

        # Head 2: Read stack[SP]
        if stack_embs.shape[0] > 0:
            val_a_raw, _, idx_a = self.head_stack_a(query_emb, stack_embs)
            stored_addr_a = round(stack_embs[idx_a, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp = round(query_emb[DIM_SP].item())
            val_a = (
                val_a_raw[0]
                if stored_addr_a == queried_sp
                else torch.tensor(0.0, dtype=DTYPE)
            )
        else:
            val_a = torch.tensor(0.0, dtype=DTYPE)

        # Head 3: Read stack[SP-1]
        if stack_embs.shape[0] > 0:
            val_b_raw, _, idx_b = self.head_stack_b(query_emb, stack_embs)
            stored_addr_b = round(stack_embs[idx_b, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m1 = round(query_emb[DIM_SP].item()) - 1
            val_b = (
                val_b_raw[0]
                if stored_addr_b == queried_sp_m1
                else torch.tensor(0.0, dtype=DTYPE)
            )
        else:
            val_b = torch.tensor(0.0, dtype=DTYPE)

        # Head 4: Read stack[SP-2]
        if stack_embs.shape[0] > 0:
            val_c_raw, _, idx_c = self.head_stack_c(query_emb, stack_embs)
            stored_addr_c = round(stack_embs[idx_c, DIM_STACK_KEY_0].item() / 2.0)
            queried_sp_m2 = round(query_emb[DIM_SP].item()) - 2
            val_c = (
                val_c_raw[0]
                if stored_addr_c == queried_sp_m2
                else torch.tensor(0.0, dtype=DTYPE)
            )
        else:
            val_c = torch.tensor(0.0, dtype=DTYPE)

        # Heads 5-6: Read local variable
        arg_raw = round(arg_val[0].item())
        actual_local_idx = locals_base + arg_raw
        if local_embs.shape[0] > 0:
            local_query = torch.zeros(self.d_model, dtype=DTYPE)
            local_query[DIM_VALUE] = float(actual_local_idx)
            local_query[DIM_ONE] = 1.0
            local_val_raw, _, idx_l = self.head_local_val(local_query, local_embs)
            local_addr_raw, _, _ = self.head_local_addr(local_query, local_embs)
            stored_local_addr = round(local_addr_raw[0].item())
            local_val = (
                local_val_raw[0]
                if stored_local_addr == actual_local_idx
                else torch.tensor(0.0, dtype=DTYPE)
            )
        else:
            local_val = torch.tensor(0.0, dtype=DTYPE)

        opcode = round(opcode_val[0].item())
        arg = arg_raw

        # Heads 7-8: Read heap memory
        heap_query_addr = round(val_a.item())
        if heap_embs.shape[0] > 0:
            heap_query = torch.zeros(self.d_model, dtype=DTYPE)
            heap_query[DIM_VALUE] = float(heap_query_addr)
            heap_query[DIM_ONE] = 1.0
            heap_val_raw, _, idx_h = self.head_heap_val(heap_query, heap_embs)
            heap_addr_raw, _, _ = self.head_heap_addr(heap_query, heap_embs)
            stored_heap_addr = round(heap_addr_raw[0].item())
            heap_val = (
                heap_val_raw[0]
                if stored_heap_addr == heap_query_addr
                else torch.tensor(0.0, dtype=DTYPE)
            )
        else:
            heap_val = torch.tensor(0.0, dtype=DTYPE)

        # FF Dispatch — linear path
        opcode_one_hot = torch.zeros(N_OPCODES, dtype=DTYPE)
        idx = OPCODE_IDX.get(opcode, -1)
        if idx >= 0:
            opcode_one_hot[idx] = 1.0

        values = torch.stack(
            [
                torch.tensor(float(arg), dtype=DTYPE),
                val_a,
                val_b,
                val_c,
                local_val,
                heap_val,
            ],
        )
        candidates = self.M_top @ values
        top_linear = (opcode_one_hot * candidates).sum()

        # FF Dispatch — nonlinear path
        va = round(val_a.item())
        vb = round(val_b.item())
        nonlinear = torch.zeros(N_OPCODES, dtype=DTYPE)

        nonlinear[OPCODE_IDX[OP_ADD]] = float((va + vb) & MASK32)
        nonlinear[OPCODE_IDX[OP_SUB]] = float((vb - va) & MASK32)
        nonlinear[OPCODE_IDX[OP_MUL]] = float((va * vb) & MASK32)
        if va != 0:
            nonlinear[OPCODE_IDX[OP_DIV_S]] = float(_trunc_div(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_DIV_U]] = float(_trunc_div(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_REM_S]] = float(_trunc_rem(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_REM_U]] = float(_trunc_rem(vb, va) & MASK32)

        nonlinear[OPCODE_IDX[OP_EQZ]] = 1.0 if va == 0 else 0.0
        nonlinear[OPCODE_IDX[OP_EQ]] = 1.0 if va == vb else 0.0
        nonlinear[OPCODE_IDX[OP_NE]] = 1.0 if va != vb else 0.0
        nonlinear[OPCODE_IDX[OP_LT_S]] = 1.0 if vb < va else 0.0
        nonlinear[OPCODE_IDX[OP_LT_U]] = 1.0 if vb < va else 0.0
        nonlinear[OPCODE_IDX[OP_GT_S]] = 1.0 if vb > va else 0.0
        nonlinear[OPCODE_IDX[OP_GT_U]] = 1.0 if vb > va else 0.0
        nonlinear[OPCODE_IDX[OP_LE_S]] = 1.0 if vb <= va else 0.0
        nonlinear[OPCODE_IDX[OP_LE_U]] = 1.0 if vb <= va else 0.0
        nonlinear[OPCODE_IDX[OP_GE_S]] = 1.0 if vb >= va else 0.0
        nonlinear[OPCODE_IDX[OP_GE_U]] = 1.0 if vb >= va else 0.0
        nonlinear[OPCODE_IDX[OP_AND]] = float(_to_i32(va) & _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_OR]] = float(_to_i32(va) | _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_XOR]] = float(_to_i32(va) ^ _to_i32(vb))
        nonlinear[OPCODE_IDX[OP_SHL]] = float((_to_i32(vb) << (int(va) & 31)) & MASK32)
        nonlinear[OPCODE_IDX[OP_SHR_S]] = float(_shr_s(vb, va))
        nonlinear[OPCODE_IDX[OP_SHR_U]] = float(_shr_u(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTL]] = float(_rotl32(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTR]] = float(_rotr32(vb, va))
        nonlinear[OPCODE_IDX[OP_CLZ]] = float(_clz32(va))
        nonlinear[OPCODE_IDX[OP_CTZ]] = float(_ctz32(va))
        nonlinear[OPCODE_IDX[OP_POPCNT]] = float(_popcnt32(va))
        nonlinear[OPCODE_IDX[OP_ABS]] = float(abs(int(va)))
        nonlinear[OPCODE_IDX[OP_NEG]] = float((-int(va)) & MASK32)

        vc = round(val_c.item())
        nonlinear[OPCODE_IDX[OP_SELECT]] = float(vc if va != 0 else vb)

        hv = round(heap_val.item())
        nonlinear[OPCODE_IDX[OP_I32_LOAD8_U]] = float(int(hv) & 0xFF)
        nonlinear[OPCODE_IDX[OP_I32_LOAD8_S]] = float(_sign_extend_8(hv))
        nonlinear[OPCODE_IDX[OP_I32_LOAD16_U]] = float(int(hv) & 0xFFFF)
        nonlinear[OPCODE_IDX[OP_I32_LOAD16_S]] = float(_sign_extend_16(hv))

        top_nonlinear = (opcode_one_hot * nonlinear).sum()
        top = top_linear + top_nonlinear
        sp_delta = (opcode_one_hot * self.sp_deltas).sum()

        return (
            opcode,
            arg,
            int(sp_delta.item()),
            round(top.item()),
            opcode_one_hot,
            round(val_a.item()),
            round(val_b.item()),
            round(val_c.item()),
            round(local_val.item()),
            round(heap_val.item()),
        )


# ─── PyTorch Executor ────────────────────────────────────────────


@register_backend
class TorchExecutor(ExecutorBackend):
    """Executes programs using CompiledModel."""

    name = "torch"

    def __init__(self, model=None):
        self.model = model or CompiledModel()
        self.model.eval()

    def execute(self, prog, max_steps=50000):
        trace = Trace(program=prog)

        prog_embs = torch.stack(
            [embed_program_token(i, instr) for i, instr in enumerate(prog)],
        )

        stack_embs_list = []
        local_embs_list = []
        heap_embs_list = []
        call_embs_list = []
        write_count = 0
        local_write_count = 0
        heap_write_count = 0
        call_write_count = 0
        call_stack = []
        locals_base = 0
        ip = 0
        sp = 0

        with torch.no_grad():
            for step in range(max_steps):
                if ip >= len(prog):
                    break

                query = embed_state(ip, sp)
                stack_embs = (
                    torch.stack(stack_embs_list)
                    if stack_embs_list
                    else torch.zeros(0, D_MODEL, dtype=DTYPE)
                )
                local_embs = (
                    torch.stack(local_embs_list)
                    if local_embs_list
                    else torch.zeros(0, D_MODEL, dtype=DTYPE)
                )
                heap_embs = (
                    torch.stack(heap_embs_list)
                    if heap_embs_list
                    else torch.zeros(0, D_MODEL, dtype=DTYPE)
                )
                call_embs = (
                    torch.stack(call_embs_list)
                    if call_embs_list
                    else torch.zeros(0, D_MODEL, dtype=DTYPE)
                )

                (
                    opcode,
                    arg,
                    sp_delta,
                    top,
                    _,
                    val_a,
                    val_b,
                    val_c,
                    local_val,
                    heap_val,
                ) = self.model.forward(
                    query,
                    prog_embs,
                    stack_embs,
                    local_embs,
                    heap_embs,
                    call_embs,
                    locals_base,
                )

                if opcode == OP_HALT:
                    trace.steps.append(TraceStep(opcode, arg, sp, top))
                    break

                if opcode in (OP_DIV_S, OP_DIV_U, OP_REM_S, OP_REM_U) and val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                    break

                cond_val = None
                if opcode in (OP_JZ, OP_JNZ):
                    cond_val = val_a

                new_sp = sp + sp_delta

                if (
                    opcode in (OP_PUSH, OP_DUP, OP_OVER)
                    or opcode
                    in (
                        OP_ADD,
                        OP_SUB,
                        OP_MUL,
                        OP_DIV_S,
                        OP_DIV_U,
                        OP_REM_S,
                        OP_REM_U,
                        OP_EQ,
                        OP_NE,
                        OP_LT_S,
                        OP_LT_U,
                        OP_GT_S,
                        OP_GT_U,
                        OP_LE_S,
                        OP_LE_U,
                        OP_GE_S,
                        OP_GE_U,
                        OP_AND,
                        OP_OR,
                        OP_XOR,
                        OP_SHL,
                        OP_SHR_S,
                        OP_SHR_U,
                        OP_ROTL,
                        OP_ROTR,
                    )
                    or opcode in (OP_EQZ, OP_CLZ, OP_CTZ, OP_POPCNT, OP_ABS, OP_NEG)
                    or opcode == OP_SELECT
                ):
                    stack_embs_list.append(embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_SWAP:
                    stack_embs_list.append(embed_stack_entry(sp, val_b, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count),
                    )
                    write_count += 1
                elif opcode == OP_ROT:
                    stack_embs_list.append(embed_stack_entry(sp, val_c, write_count))
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 1, val_a, write_count),
                    )
                    write_count += 1
                    stack_embs_list.append(
                        embed_stack_entry(sp - 2, val_b, write_count),
                    )
                    write_count += 1
                elif opcode == OP_LOCAL_GET:
                    stack_embs_list.append(embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_LOCAL_SET or opcode == OP_LOCAL_TEE:
                    local_embs_list.append(
                        embed_local_entry(locals_base + arg, val_a, local_write_count),
                    )
                    local_write_count += 1
                elif opcode == OP_I32_LOAD or opcode in (
                    OP_I32_LOAD8_U,
                    OP_I32_LOAD8_S,
                    OP_I32_LOAD16_U,
                    OP_I32_LOAD16_S,
                ):
                    stack_embs_list.append(embed_stack_entry(new_sp, top, write_count))
                    write_count += 1
                elif opcode == OP_I32_STORE:
                    heap_embs_list.append(
                        embed_heap_entry(int(val_b), val_a, heap_write_count),
                    )
                    heap_write_count += 1
                elif opcode == OP_I32_STORE8:
                    heap_embs_list.append(
                        embed_heap_entry(
                            int(val_b),
                            int(val_a) & 0xFF,
                            heap_write_count,
                        ),
                    )
                    heap_write_count += 1
                elif opcode == OP_I32_STORE16:
                    heap_embs_list.append(
                        embed_heap_entry(
                            int(val_b),
                            int(val_a) & 0xFFFF,
                            heap_write_count,
                        ),
                    )
                    heap_write_count += 1
                elif opcode == OP_CALL:
                    call_stack.append((ip + 1, sp, locals_base))
                    call_embs_list.append(
                        embed_call_frame(
                            len(call_stack) - 1,
                            ip + 1,
                            sp,
                            locals_base,
                            call_write_count,
                        ),
                    )
                    call_write_count += 1
                    locals_base = len(local_embs_list)
                    trace.steps.append(TraceStep(opcode, arg, sp, top))
                    ip = arg
                    continue
                elif opcode == OP_RETURN:
                    if not call_stack:
                        trace.steps.append(TraceStep(OP_TRAP, 0, sp, 0))
                        break
                    ret_val = val_a
                    ret_addr, saved_sp, saved_locals_base = call_stack.pop()
                    sp = saved_sp + 1
                    stack_embs_list.append(embed_stack_entry(sp, ret_val, write_count))
                    write_count += 1
                    locals_base = saved_locals_base
                    trace.steps.append(TraceStep(opcode, arg, sp, int(ret_val)))
                    ip = ret_addr
                    continue

                trace.steps.append(TraceStep(opcode, arg, new_sp, top))
                sp = new_sp

                if opcode == OP_JZ:
                    ip = arg if cond_val == 0 else ip + 1
                elif opcode == OP_JNZ:
                    ip = arg if cond_val != 0 else ip + 1
                else:
                    ip += 1

        return trace
