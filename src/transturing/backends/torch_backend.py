"""
PyTorch-based executor backend for the compiled transformer stack machine.

Contains:
  - DTYPE, EPS: torch-specific constants
  - CompiledAttentionHead: hard-max attention head (nn.Module)
  - TokenVocab: token vocabulary with compiled embeddings
  - Embedding functions: embed_program_token, embed_stack_entry, etc.
  - CompiledModel: full compiled transformer (nn.Module)
  - TorchExecutor: executes programs via CompiledModel
"""

from typing import ClassVar, NamedTuple

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
    Instruction,
    Trace,
    TraceStep,
    clz32,
    ctz32,
    popcnt32,
    rotl32,
    rotr32,
    shr_s,
    shr_u,
    sign_extend_8,
    sign_extend_16,
    to_i32,
    trunc_div,
    trunc_rem,
)
from transturing.core.registry import register_backend

# ─── Torch-specific constants ────────────────────────────────────

DTYPE = torch.float64
EPS = 1e-6

# ─── Memory embedding container ──────────────────────────────────

_MemoryEmbs = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
"""(stack, local, heap, call) embedding tensors."""


# ─── Opcode dispatch sets ────────────────────────────────────────

_PUSH_RESULT_OPS: frozenset[int] = frozenset(
    {
        OP_PUSH,
        OP_DUP,
        OP_OVER,
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
        OP_EQZ,
        OP_CLZ,
        OP_CTZ,
        OP_POPCNT,
        OP_ABS,
        OP_NEG,
        OP_SELECT,
        OP_LOCAL_GET,
        OP_I32_LOAD,
        OP_I32_LOAD8_U,
        OP_I32_LOAD8_S,
        OP_I32_LOAD16_U,
        OP_I32_LOAD16_S,
    }
)

_LOCAL_WRITE_OPS: frozenset[int] = frozenset({OP_LOCAL_SET, OP_LOCAL_TEE})

_I32_STORE_MASKS: dict[int, int] = {
    OP_I32_STORE: MASK32,
    OP_I32_STORE8: 0xFF,
    OP_I32_STORE16: 0xFFFF,
}

_DIV_OPS: frozenset[int] = frozenset({OP_DIV_S, OP_DIV_U, OP_REM_S, OP_REM_U})


# ─── Compiled Attention Head ──────────────────────────────────────


class CompiledAttentionHead(nn.Module):
    """Hard-max attention head with analytically set W_Q, W_K, W_V."""

    def __init__(
        self,
        d_model: int = D_MODEL,
        head_dim: int = 2,
        v_dim: int = 1,
        *,
        use_bias_q: bool = False,
    ) -> None:
        """Initialize attention head with given dimensions."""
        super().__init__()
        self.W_Q = nn.Linear(d_model, head_dim, bias=use_bias_q)
        self.W_K = nn.Linear(d_model, head_dim, bias=False)
        self.W_V = nn.Linear(d_model, v_dim, bias=False)
        self.double()

    def forward(
        self,
        query_emb: torch.Tensor,
        memory_embs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Hard-max attention lookup."""
        if memory_embs.shape[0] == 0:
            return (
                torch.zeros(self.W_V.out_features, dtype=DTYPE),
                torch.tensor(-float("inf"), dtype=DTYPE),
                -1,
            )

        q = self.W_Q(query_emb)
        k = self.W_K(memory_embs)
        v = self.W_V(memory_embs)

        scores = k @ q
        best = scores.argmax().item()

        return v[best], scores[best], best


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

    _SPECIAL_NAMES: ClassVar[dict[int, str]] = {
        PAD_ID: PAD,
        COMMIT_ID: COMMIT,
        BRANCH_TAKEN_ID: BRANCH_TAKEN,
        BRANCH_NOT_TAKEN_ID: BRANCH_NOT_TAKEN,
    }
    _SPECIAL_IDS: ClassVar[dict[str, int]] = {v: k for k, v in _SPECIAL_NAMES.items()}

    def __init__(self) -> None:
        """Initialize token vocabulary mapping."""
        self._opcode_to_tid: dict[int, int] = {}
        self._tid_to_opcode: dict[int, int] = {}
        for op_code, idx in OPCODE_IDX.items():
            tid = self._OPCODE_BASE + idx
            self._opcode_to_tid[op_code] = tid
            self._tid_to_opcode[tid] = op_code
        trap_tid = self._OPCODE_BASE + N_OPCODES
        self._opcode_to_tid[OP_TRAP] = trap_tid
        self._tid_to_opcode[trap_tid] = OP_TRAP
        self.vocab_size = self.VOCAB_SIZE

    _MAX_BYTE_VALUE: ClassVar[int] = 255

    def encode(self, token: str | tuple[str, int | str]) -> int:
        """Encode a token to its vocabulary ID."""
        if isinstance(token, str):
            return self._encode_special_str(token)
        tag, val = token
        return self._encode_tagged(tag, val)

    def _encode_special_str(self, token: str) -> int:
        """Encode a bare string token (special token name)."""
        if token in self._SPECIAL_IDS:
            return self._SPECIAL_IDS[token]
        msg = f"Unknown special token: {token!r}"
        raise ValueError(msg)

    def _encode_tagged(self, tag: str, val: int | str) -> int:
        """Dispatch a tagged token to the appropriate encoder."""
        if tag == self.OPCODE:
            return self._encode_opcode(val)
        if tag == self.VALUE:
            return self._encode_value(val)
        if tag == self.SP_DELTA:
            return self._encode_sp_delta(val)
        if tag == self.SPECIAL:
            return self._encode_special_val(val)
        msg = f"Unknown token tag: {tag!r}"
        raise ValueError(msg)

    def _encode_opcode(self, val: int | str) -> int:
        """Encode an opcode token to its vocabulary ID."""
        if not isinstance(val, int):
            msg = f"Opcode value must be int, got {type(val).__name__}"
            raise TypeError(msg)
        if val not in self._opcode_to_tid:
            msg = f"Unknown opcode: {val}"
            raise ValueError(msg)
        return self._opcode_to_tid[val]

    def _encode_value(self, val: int | str) -> int:
        """Encode a byte value token to its vocabulary ID."""
        if not isinstance(val, int):
            msg = f"Value must be int, got {type(val).__name__}"
            raise TypeError(msg)
        if not (0 <= val <= self._MAX_BYTE_VALUE):
            msg = f"Value out of byte range: {val}"
            raise ValueError(msg)
        return self._VALUE_BASE + val

    def _encode_sp_delta(self, val: int | str) -> int:
        """Encode an SP delta token to its vocabulary ID."""
        if not isinstance(val, int):
            msg = f"SP delta must be int, got {type(val).__name__}"
            raise TypeError(msg)
        if not (self._SP_DELTA_MIN <= val <= self._SP_DELTA_MAX):
            msg = f"SP delta {val} out of range"
            raise ValueError(msg)
        return self._SP_DELTA_BASE + (val - self._SP_DELTA_MIN)

    def _encode_special_val(self, val: int | str) -> int:
        """Encode a special token passed as tuple ('special', name)."""
        if isinstance(val, str) and val in self._SPECIAL_IDS:
            return self._SPECIAL_IDS[val]
        msg = f"Unknown special token: {val!r}"
        raise ValueError(msg)

    def decode(self, tid: int) -> tuple[str, int | str]:
        """Decode a token ID back to its structured representation."""
        if not (0 <= tid < self.vocab_size):
            msg = f"Token ID {tid} out of range"
            raise ValueError(msg)
        if tid < self._OPCODE_BASE:
            return (self.SPECIAL, self._SPECIAL_NAMES[tid])
        if tid < self._VALUE_BASE:
            return (self.OPCODE, self._tid_to_opcode[tid])
        if tid < self._SP_DELTA_BASE:
            return (self.VALUE, tid - self._VALUE_BASE)
        return (self.SP_DELTA, (tid - self._SP_DELTA_BASE) + self._SP_DELTA_MIN)

    def compile_embedding(self, d_model: int | None = None) -> nn.Embedding:
        """Build nn.Embedding with analytically set weights."""
        if d_model is None:
            d_model = D_MODEL
        emb = nn.Embedding(self.vocab_size, d_model)
        w = torch.zeros(self.vocab_size, d_model, dtype=DTYPE)

        w[self.COMMIT_ID, DIM_ONE] = 1.0
        w[self.COMMIT_ID, DIM_OPCODE] = -1.0
        w[self.BRANCH_TAKEN_ID, DIM_ONE] = 1.0
        w[self.BRANCH_TAKEN_ID, DIM_OPCODE] = -2.0
        w[self.BRANCH_NOT_TAKEN_ID, DIM_ONE] = 1.0
        w[self.BRANCH_NOT_TAKEN_ID, DIM_OPCODE] = -3.0

        for op_code, tid in self._opcode_to_tid.items():
            w[tid, DIM_IS_PROG] = 1.0
            w[tid, DIM_OPCODE] = float(op_code)
            w[tid, DIM_ONE] = 1.0
            dim = OPCODE_DIM_MAP.get(op_code)
            if dim is not None and dim < d_model:
                w[tid, dim] = 1.0

        for v in range(256):
            tid = self._VALUE_BASE + v
            w[tid, DIM_IS_STACK] = 1.0
            w[tid, DIM_VALUE] = float(v)
            w[tid, DIM_ONE] = 1.0

        for delta in range(self._SP_DELTA_MIN, self._SP_DELTA_MAX + 1):
            tid = self._SP_DELTA_BASE + (delta - self._SP_DELTA_MIN)
            w[tid, DIM_IS_STATE] = 1.0
            w[tid, DIM_SP] = float(delta)
            w[tid, DIM_ONE] = 1.0

        emb.weight = nn.Parameter(w, requires_grad=False)
        return emb

    def compile_unembedding(
        self,
        embedding: nn.Embedding | None = None,
        d_model: int | None = None,
    ) -> nn.Linear:
        """Build nn.Linear unembedding head."""
        if d_model is None:
            d_model = D_MODEL
        if embedding is None:
            embedding = self.compile_embedding(d_model)
        e = embedding.weight.data
        unembed = nn.Linear(d_model, self.vocab_size, bias=True)
        norms_sq = (e * e).sum(dim=1)
        unembed.weight = nn.Parameter(e.clone(), requires_grad=False)
        unembed.bias = nn.Parameter(-0.5 * norms_sq, requires_grad=False)
        return unembed

    def opcode_name(self, op_code: int) -> str:
        """Return human-readable name for an opcode."""
        return OP_NAMES.get(op_code, f"?{op_code}")

    def token_name(self, tid: int) -> str:
        """Return human-readable name for a token ID."""
        tag, val = self.decode(tid)
        if tag == self.SPECIAL:
            return str(val)
        if not isinstance(val, int):
            msg = f"Expected int value for tag {tag!r}"
            raise TypeError(msg)
        if tag == self.OPCODE:
            return self.opcode_name(val)
        if tag == self.VALUE:
            return f"V{val}"
        if tag == self.SP_DELTA:
            return f"SP{'+' if val >= 0 else ''}{val}"
        return f"?{tid}"

    def __repr__(self) -> str:
        """Return string representation of vocabulary."""
        return (
            f"TokenVocab(vocab_size={self.vocab_size}, "
            f"opcodes={len(self._opcode_to_tid)}, values=256, "
            f"sp_deltas={self._SP_DELTA_COUNT}, specials={self._SPECIAL_COUNT})"
        )


# ─── Embedding Functions ──────────────────────────────────────────


def embed_program_token(pos: int, instr: Instruction) -> torch.Tensor:
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


def embed_stack_entry(addr: int, value: int, write_order: int) -> torch.Tensor:
    """Create d_model-dim embedding for a stack write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_STACK] = 1.0
    emb[DIM_STACK_KEY_0] = 2.0 * addr
    emb[DIM_STACK_KEY_1] = -float(addr * addr) + EPS * write_order
    emb[DIM_VALUE] = float(value)
    emb[DIM_ONE] = 1.0
    return emb


def embed_local_entry(local_idx: int, value: int, write_order: int) -> torch.Tensor:
    """Create embedding for a local variable write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_LOCAL] = 1.0
    emb[DIM_LOCAL_KEY_0] = 2.0 * local_idx
    emb[DIM_LOCAL_KEY_1] = -float(local_idx * local_idx) + EPS * write_order
    emb[DIM_VALUE] = float(value)
    emb[DIM_ONE] = 1.0
    return emb


def embed_heap_entry(addr: int, value: int, write_order: int) -> torch.Tensor:
    """Create embedding for a heap memory write record."""
    emb = torch.zeros(D_MODEL, dtype=DTYPE)
    emb[DIM_IS_HEAP] = 1.0
    emb[DIM_HEAP_KEY_0] = 2.0 * addr
    emb[DIM_HEAP_KEY_1] = -float(addr * addr) + EPS * write_order
    emb[DIM_VALUE] = float(value)
    emb[DIM_ONE] = 1.0
    return emb


def embed_call_frame(
    depth: int,
    ret_addr: int,
    saved_sp: int,
    locals_base: int,
    write_order: int,
) -> torch.Tensor:
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


def embed_state(ip: int, sp: int) -> torch.Tensor:
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

    M_top: torch.Tensor
    sp_deltas: torch.Tensor

    def __init__(self, d_model: int = D_MODEL) -> None:
        """Initialize compiled transformer model."""
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

    def _compile_head(
        self,
        head: CompiledAttentionHead,
        qkv: tuple[
            list[list[tuple[int, float]]],
            list[list[tuple[int, float]]],
            list[list[tuple[int, float]]],
        ],
        q_bias: list[float] | None = None,
    ) -> None:
        """Set attention head weights from per-row (dim, value) specifications."""
        q_rows, k_rows, v_rows = qkv
        d_model = self.d_model
        w_q = torch.zeros(len(q_rows), d_model)
        for row, entries in enumerate(q_rows):
            for dim, val in entries:
                w_q[row, dim] = val
        head.W_Q.weight.copy_(w_q)

        w_k = torch.zeros(len(k_rows), d_model)
        for row, entries in enumerate(k_rows):
            for dim, val in entries:
                w_k[row, dim] = val
        head.W_K.weight.copy_(w_k)

        w_v = torch.zeros(len(v_rows), d_model)
        for row, entries in enumerate(v_rows):
            for dim, val in entries:
                w_v[row, dim] = val
        head.W_V.weight.copy_(w_v)

        if q_bias is not None:
            b = torch.zeros(len(q_bias))
            for i, val in enumerate(q_bias):
                b[i] = val
            head.W_Q.bias.copy_(b)

    def _compile_heads(self) -> None:
        """Compile all 10 attention head weights."""
        # Heads 0-1: Program memory (opcode + argument fetch)
        for head in (self.head_prog_op, self.head_prog_arg):
            self._compile_head(
                head,
                (
                    [[(DIM_IP, 1.0)], [(DIM_ONE, 1.0)]],
                    [[(DIM_PROG_KEY_0, 1.0)], [(DIM_PROG_KEY_1, 1.0)]],
                    [[(DIM_OPCODE if head is self.head_prog_op else DIM_VALUE, 1.0)]],
                ),
            )

        # Head 2: Stack read at SP
        self._compile_head(
            self.head_stack_a,
            (
                [[(DIM_SP, 1.0)], [(DIM_ONE, 1.0)]],
                [[(DIM_STACK_KEY_0, 1.0)], [(DIM_STACK_KEY_1, 1.0)]],
                [[(DIM_VALUE, 1.0)]],
            ),
        )
        # Head 3: Stack read at SP-1 (bias shifts query by -1)
        self._compile_head(
            self.head_stack_b,
            (
                [[(DIM_SP, 1.0)], [(DIM_ONE, 1.0)]],
                [[(DIM_STACK_KEY_0, 1.0)], [(DIM_STACK_KEY_1, 1.0)]],
                [[(DIM_VALUE, 1.0)]],
            ),
            q_bias=[-1.0, 0.0],
        )
        # Head 4: Stack read at SP-2 (bias shifts query by -2)
        self._compile_head(
            self.head_stack_c,
            (
                [[(DIM_SP, 1.0)], [(DIM_ONE, 1.0)]],
                [[(DIM_STACK_KEY_0, 1.0)], [(DIM_STACK_KEY_1, 1.0)]],
                [[(DIM_VALUE, 1.0)]],
            ),
            q_bias=[-2.0, 0.0],
        )

        # Heads 5-6: Local variable (value + address verify)
        for head, v_spec in (
            (self.head_local_val, [[(DIM_VALUE, 1.0)]]),
            (self.head_local_addr, [[(DIM_LOCAL_KEY_0, 0.5)]]),
        ):
            self._compile_head(
                head,
                (
                    [[(DIM_VALUE, 1.0)], [(DIM_ONE, 1.0)]],
                    [[(DIM_LOCAL_KEY_0, 1.0)], [(DIM_LOCAL_KEY_1, 1.0)]],
                    v_spec,
                ),
            )

        # Heads 7-8: Heap memory (value + address verify)
        for head, v_spec in (
            (self.head_heap_val, [[(DIM_VALUE, 1.0)]]),
            (self.head_heap_addr, [[(DIM_HEAP_KEY_0, 0.5)]]),
        ):
            self._compile_head(
                head,
                (
                    [[(DIM_VALUE, 1.0)], [(DIM_ONE, 1.0)]],
                    [[(DIM_HEAP_KEY_0, 1.0)], [(DIM_HEAP_KEY_1, 1.0)]],
                    v_spec,
                ),
            )

        # Head 9: Call stack read (3-dim value output)
        self._compile_head(
            self.head_call_stack,
            (
                [[(DIM_VALUE, 1.0)], [(DIM_ONE, 1.0)]],
                [[(DIM_CALL_KEY_0, 1.0)], [(DIM_CALL_KEY_1, 1.0)]],
                [
                    [(DIM_CALL_RET_ADDR, 1.0)],
                    [(DIM_CALL_SAVED_SP, 1.0)],
                    [(DIM_CALL_LOCALS_BASE, 1.0)],
                ],
            ),
        )

    def _compile_weights(self) -> None:
        """Set all weight matrices analytically."""
        with torch.no_grad():
            self._compile_heads()

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

    def _read_stack_at(
        self,
        query_emb: torch.Tensor,
        stack_embs: torch.Tensor,
        head: CompiledAttentionHead,
        sp_offset: int,
    ) -> torch.Tensor:
        """Read stack value at SP + sp_offset using the given attention head."""
        if stack_embs.shape[0] == 0:
            return torch.tensor(0.0, dtype=DTYPE)
        val_raw, _, idx = head(query_emb, stack_embs)
        stored_addr = round(stack_embs[idx, DIM_STACK_KEY_0].item() / 2.0)
        queried_addr = round(query_emb[DIM_SP].item()) + sp_offset
        return (
            val_raw[0]
            if stored_addr == queried_addr
            else torch.tensor(0.0, dtype=DTYPE)
        )

    def _read_local(
        self,
        local_embs: torch.Tensor,
        actual_local_idx: int,
    ) -> torch.Tensor:
        """Read local variable value at the given index."""
        if local_embs.shape[0] == 0:
            return torch.tensor(0.0, dtype=DTYPE)
        local_query = torch.zeros(self.d_model, dtype=DTYPE)
        local_query[DIM_VALUE] = float(actual_local_idx)
        local_query[DIM_ONE] = 1.0
        local_val_raw, _, _ = self.head_local_val(local_query, local_embs)
        local_addr_raw, _, _ = self.head_local_addr(local_query, local_embs)
        stored_addr = round(local_addr_raw[0].item())
        return (
            local_val_raw[0]
            if stored_addr == actual_local_idx
            else torch.tensor(0.0, dtype=DTYPE)
        )

    def _read_heap(self, val_a: torch.Tensor, heap_embs: torch.Tensor) -> torch.Tensor:
        """Read heap value at the address encoded in val_a."""
        heap_query_addr = round(val_a.item())
        if heap_embs.shape[0] == 0:
            return torch.tensor(0.0, dtype=DTYPE)
        heap_query = torch.zeros(self.d_model, dtype=DTYPE)
        heap_query[DIM_VALUE] = float(heap_query_addr)
        heap_query[DIM_ONE] = 1.0
        heap_val_raw, _, _ = self.head_heap_val(heap_query, heap_embs)
        heap_addr_raw, _, _ = self.head_heap_addr(heap_query, heap_embs)
        stored_addr = round(heap_addr_raw[0].item())
        return (
            heap_val_raw[0]
            if stored_addr == heap_query_addr
            else torch.tensor(0.0, dtype=DTYPE)
        )

    @staticmethod
    def _compute_nonlinear(
        va: int,
        vb: int,
        vc: int,
        hv: int,
    ) -> torch.Tensor:
        """Compute nonlinear dispatch results for all opcodes."""
        nonlinear = torch.zeros(N_OPCODES, dtype=DTYPE)

        nonlinear[OPCODE_IDX[OP_ADD]] = float((va + vb) & MASK32)
        nonlinear[OPCODE_IDX[OP_SUB]] = float((vb - va) & MASK32)
        nonlinear[OPCODE_IDX[OP_MUL]] = float((va * vb) & MASK32)
        if va != 0:
            nonlinear[OPCODE_IDX[OP_DIV_S]] = float(trunc_div(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_DIV_U]] = float(trunc_div(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_REM_S]] = float(trunc_rem(vb, va) & MASK32)
            nonlinear[OPCODE_IDX[OP_REM_U]] = float(trunc_rem(vb, va) & MASK32)

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
        nonlinear[OPCODE_IDX[OP_AND]] = float(to_i32(va) & to_i32(vb))
        nonlinear[OPCODE_IDX[OP_OR]] = float(to_i32(va) | to_i32(vb))
        nonlinear[OPCODE_IDX[OP_XOR]] = float(to_i32(va) ^ to_i32(vb))
        nonlinear[OPCODE_IDX[OP_SHL]] = float((to_i32(vb) << (int(va) & 31)) & MASK32)
        nonlinear[OPCODE_IDX[OP_SHR_S]] = float(shr_s(vb, va))
        nonlinear[OPCODE_IDX[OP_SHR_U]] = float(shr_u(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTL]] = float(rotl32(vb, va))
        nonlinear[OPCODE_IDX[OP_ROTR]] = float(rotr32(vb, va))
        nonlinear[OPCODE_IDX[OP_CLZ]] = float(clz32(va))
        nonlinear[OPCODE_IDX[OP_CTZ]] = float(ctz32(va))
        nonlinear[OPCODE_IDX[OP_POPCNT]] = float(popcnt32(va))
        nonlinear[OPCODE_IDX[OP_ABS]] = float(abs(int(va)))
        nonlinear[OPCODE_IDX[OP_NEG]] = float((-int(va)) & MASK32)
        nonlinear[OPCODE_IDX[OP_SELECT]] = float(vc if va != 0 else vb)

        nonlinear[OPCODE_IDX[OP_I32_LOAD8_U]] = float(int(hv) & 0xFF)
        nonlinear[OPCODE_IDX[OP_I32_LOAD8_S]] = float(sign_extend_8(hv))
        nonlinear[OPCODE_IDX[OP_I32_LOAD16_U]] = float(int(hv) & 0xFFFF)
        nonlinear[OPCODE_IDX[OP_I32_LOAD16_S]] = float(sign_extend_16(hv))

        return nonlinear

    def forward(
        self,
        query_emb: torch.Tensor,
        prog_embs: torch.Tensor,
        mem: _MemoryEmbs,
        locals_base: int = 0,
    ) -> tuple[int, int, int, int, torch.Tensor, int, int, int, int, int]:
        """Execute one step."""
        stack_embs, local_embs, heap_embs, _call_embs = mem

        opcode_val, _, _ = self.head_prog_op(query_emb, prog_embs)
        arg_val, _, _ = self.head_prog_arg(query_emb, prog_embs)

        val_a = self._read_stack_at(query_emb, stack_embs, self.head_stack_a, 0)
        val_b = self._read_stack_at(query_emb, stack_embs, self.head_stack_b, -1)
        val_c = self._read_stack_at(query_emb, stack_embs, self.head_stack_c, -2)

        arg_raw = round(arg_val[0].item())
        actual_local_idx = locals_base + arg_raw
        local_val = self._read_local(local_embs, actual_local_idx)

        opcode = round(opcode_val[0].item())
        heap_val = self._read_heap(val_a, heap_embs)

        # FF Dispatch — linear path
        opcode_one_hot = torch.zeros(N_OPCODES, dtype=DTYPE)
        idx = OPCODE_IDX.get(opcode, -1)
        if idx >= 0:
            opcode_one_hot[idx] = 1.0

        values = torch.stack(
            [
                torch.tensor(float(arg_raw), dtype=DTYPE),
                val_a,
                val_b,
                val_c,
                local_val,
                heap_val,
            ]
        )
        candidates = self.M_top @ values
        top_linear = (opcode_one_hot * candidates).sum()

        # FF Dispatch — nonlinear path
        va = round(val_a.item())
        vb = round(val_b.item())
        vc = round(val_c.item())
        hv = round(heap_val.item())
        nonlinear = self._compute_nonlinear(va, vb, vc, hv)

        top_nonlinear = (opcode_one_hot * nonlinear).sum()
        top = top_linear + top_nonlinear
        sp_delta = (opcode_one_hot * self.sp_deltas).sum()

        return (
            opcode,
            arg_raw,
            int(sp_delta.item()),
            round(top.item()),
            opcode_one_hot,
            va,
            vb,
            vc,
            round(local_val.item()),
            hv,
        )


# ─── Executor helpers ─────────────────────────────────────────────


def _stack_or_empty(embs: list[torch.Tensor]) -> torch.Tensor:
    """Stack a list of tensors, or return an empty tensor of correct shape."""
    return torch.stack(embs) if embs else torch.zeros(0, D_MODEL, dtype=DTYPE)


class _ForwardResult(NamedTuple):
    """Results from a single model forward pass used by the executor."""

    opcode: int
    arg: int
    sp_delta: int
    top: int
    val_a: int
    val_b: int
    val_c: int


class _ExecState:
    """Mutable execution state for the interpreter loop."""

    def __init__(self) -> None:
        self.stack_embs: list[torch.Tensor] = []
        self.local_embs: list[torch.Tensor] = []
        self.heap_embs: list[torch.Tensor] = []
        self.call_embs: list[torch.Tensor] = []
        self.write_count: int = 0
        self.local_write_count: int = 0
        self.heap_write_count: int = 0
        self.call_write_count: int = 0
        self.call_stack: list[tuple[int, int, int]] = []
        self.locals_base: int = 0
        self.ip: int = 0
        self.sp: int = 0


# ─── PyTorch Executor ────────────────────────────────────────────


@register_backend
class TorchExecutor(ExecutorBackend):
    """Executes programs using CompiledModel."""

    name: str = "torch"

    def __init__(self, model: CompiledModel | None = None) -> None:
        """Initialize with optional compiled model."""
        self.model = model or CompiledModel()
        self.model.eval()

    def execute(self, prog: list[Instruction], max_steps: int = 50000) -> Trace:
        """Execute a program and return the execution trace."""
        trace = Trace(program=prog)

        prog_embs = torch.stack(
            [embed_program_token(i, instr) for i, instr in enumerate(prog)],
        )
        state = _ExecState()

        with torch.no_grad():
            for _step in range(max_steps):
                if state.ip >= len(prog):
                    break

                query = embed_state(state.ip, state.sp)
                mem = self._build_mem(state)
                fwd = self._forward_step(query, prog_embs, mem, state.locals_base)

                if fwd.opcode == OP_HALT:
                    trace.steps.append(
                        TraceStep(fwd.opcode, fwd.arg, state.sp, fwd.top),
                    )
                    break

                if fwd.opcode in _DIV_OPS and fwd.val_a == 0:
                    trace.steps.append(TraceStep(OP_TRAP, 0, state.sp, 0))
                    break

                if fwd.opcode == OP_CALL:
                    self._handle_call(fwd, state, trace)
                    continue

                if fwd.opcode == OP_RETURN:
                    if self._handle_return(fwd, state, trace):
                        continue
                    break

                self._apply_memory_writes(fwd, state)
                state.sp += fwd.sp_delta
                trace.steps.append(
                    TraceStep(fwd.opcode, fwd.arg, state.sp, fwd.top),
                )
                state.ip = self._next_ip(fwd.opcode, fwd.arg, fwd.val_a, state.ip)

        return trace

    @staticmethod
    def _build_mem(state: _ExecState) -> _MemoryEmbs:
        """Build memory embedding tensors from accumulated lists."""
        return (
            _stack_or_empty(state.stack_embs),
            _stack_or_empty(state.local_embs),
            _stack_or_empty(state.heap_embs),
            _stack_or_empty(state.call_embs),
        )

    def _forward_step(
        self,
        query: torch.Tensor,
        prog_embs: torch.Tensor,
        mem: _MemoryEmbs,
        locals_base: int,
    ) -> _ForwardResult:
        """Run one model forward pass and return structured results."""
        (
            opcode,
            arg,
            sp_delta,
            top,
            _,
            val_a,
            val_b,
            val_c,
            _local_val,
            _heap_val,
        ) = self.model.forward(query, prog_embs, mem, locals_base)
        return _ForwardResult(opcode, arg, sp_delta, top, val_a, val_b, val_c)

    @staticmethod
    def _apply_memory_writes(fwd: _ForwardResult, state: _ExecState) -> None:
        """Apply memory mutations for non-control-flow opcodes."""
        new_sp = state.sp + fwd.sp_delta

        if fwd.opcode in _PUSH_RESULT_OPS:
            state.stack_embs.append(
                embed_stack_entry(new_sp, fwd.top, state.write_count),
            )
            state.write_count += 1
        elif fwd.opcode == OP_SWAP:
            state.stack_embs.append(
                embed_stack_entry(state.sp, fwd.val_b, state.write_count),
            )
            state.write_count += 1
            state.stack_embs.append(
                embed_stack_entry(state.sp - 1, fwd.val_a, state.write_count),
            )
            state.write_count += 1
        elif fwd.opcode == OP_ROT:
            state.stack_embs.append(
                embed_stack_entry(state.sp, fwd.val_c, state.write_count),
            )
            state.write_count += 1
            state.stack_embs.append(
                embed_stack_entry(state.sp - 1, fwd.val_a, state.write_count),
            )
            state.write_count += 1
            state.stack_embs.append(
                embed_stack_entry(state.sp - 2, fwd.val_b, state.write_count),
            )
            state.write_count += 1
        elif fwd.opcode in _LOCAL_WRITE_OPS:
            state.local_embs.append(
                embed_local_entry(
                    state.locals_base + fwd.arg,
                    fwd.val_a,
                    state.local_write_count,
                ),
            )
            state.local_write_count += 1
        elif fwd.opcode in _I32_STORE_MASKS:
            mask = _I32_STORE_MASKS[fwd.opcode]
            store_val = int(fwd.val_a) & mask if mask != MASK32 else fwd.val_a
            state.heap_embs.append(
                embed_heap_entry(int(fwd.val_b), store_val, state.heap_write_count),
            )
            state.heap_write_count += 1

    @staticmethod
    def _handle_call(
        fwd: _ForwardResult,
        state: _ExecState,
        trace: Trace,
    ) -> None:
        """Handle CALL opcode: push frame and jump."""
        state.call_stack.append((state.ip + 1, state.sp, state.locals_base))
        state.call_embs.append(
            embed_call_frame(
                len(state.call_stack) - 1,
                state.ip + 1,
                state.sp,
                state.locals_base,
                state.call_write_count,
            ),
        )
        state.call_write_count += 1
        state.locals_base = len(state.local_embs)
        trace.steps.append(TraceStep(fwd.opcode, fwd.arg, state.sp, fwd.top))
        state.ip = fwd.arg

    @staticmethod
    def _handle_return(
        fwd: _ForwardResult,
        state: _ExecState,
        trace: Trace,
    ) -> bool:
        """Handle RETURN opcode. Returns True to continue, False to break (TRAP)."""
        if not state.call_stack:
            trace.steps.append(TraceStep(OP_TRAP, 0, state.sp, 0))
            return False
        ret_addr, saved_sp, saved_locals_base = state.call_stack.pop()
        state.sp = saved_sp + 1
        state.stack_embs.append(
            embed_stack_entry(state.sp, fwd.val_a, state.write_count),
        )
        state.write_count += 1
        state.locals_base = saved_locals_base
        trace.steps.append(
            TraceStep(fwd.opcode, fwd.arg, state.sp, int(fwd.val_a)),
        )
        state.ip = ret_addr
        return True

    @staticmethod
    def _next_ip(opcode: int, arg: int, val_a: int, ip: int) -> int:
        """Compute the next instruction pointer after the given opcode."""
        if opcode == OP_JZ:
            return arg if val_a == 0 else ip + 1
        if opcode == OP_JNZ:
            return arg if val_a != 0 else ip + 1
        return ip + 1
