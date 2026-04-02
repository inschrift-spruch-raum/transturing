"""Structured control flow assembler for the stack machine.

Compiles WASM-style structured programs to flat Instruction lists
suitable for NumPyExecutor and TorchExecutor.

Supported constructs
--------------------
  ('BLOCK',)                      block; BR/BR_IF exits forward to END
  ('LOOP',)                       loop; BR/BR_IF continues backward to start
  ('IF',)                         conditional (pops condition; 0=skip, non-0=enter)
  ('ELSE',)                       else branch
  ('END',)                        close BLOCK / LOOP / IF
  ('BR', n)                       unconditional break n levels out
  ('BR_IF', n)                    conditional break (pops condition; branches if non-0)
  ('BR_TABLE', labels, default)   switch: pops index i;
                                    if i < len(labels): branch labels[i] levels out
                                    else: branch default levels out
  any other tuple                 direct flat instruction (passed to program())

Lowering rules
--------------
  BLOCK  →  (no instruction; record forward end-label)
  LOOP   →  (no instruction; record backward start-addr + forward end-label)
  IF     →  JZ <else-or-end>
  ELSE   →  PUSH 1; JNZ <end>   then resolve else-label here
  END    →  (resolve pending end-labels; resolve else-label if no ELSE was seen)
  BR n   →  LOOP target: PUSH 1; JNZ <start>   (known backward addr)
             BLOCK/IF target: PUSH 1; JNZ <end>  (forward ref, patched at END)
  BR_IF n→  LOOP target: JNZ <start>
             BLOCK/IF target: JNZ <end>           (forward ref)
  BR_TABLE [l0..lN-1] default →
             for i in 0..N-1: DUP; PUSH i; EQ; JNZ <case_i_handler>
             POP; <jump to default target>
             for i in 0..N-1: <case_i_handler>: POP; <jump to labels[i] target>

Invariant: BR_TABLE with N cases emits exactly N+1 conditional/unconditional
jumps in the comparison chain (N case-JNZ + 1 default jump).

References
----------
  Issue #36: Tier 3 Chunk 1 — Structured control flow assembler + BR_TABLE

"""

from .isa import (
    OP_DUP,
    OP_EQ,
    OP_JNZ,
    OP_JZ,
    OP_POP,
    OP_PUSH,
    Instruction,
)
from .isa import (
    program as _flat,
)


def compile_structured(wasm_instrs):
    """Compile structured WASM-style instructions to flat Instruction list.

    Args:
        wasm_instrs: list of tuples.  Structured constructs are:
            ('BLOCK',), ('LOOP',), ('IF',), ('ELSE',), ('END',),
            ('BR', n), ('BR_IF', n), ('BR_TABLE', labels, default).
            Any other tuple is treated as a direct flat instruction and
            passed through to isa.program().

    Returns:
        List[Instruction] — flat instructions accepted by NumPyExecutor /
        TorchExecutor.

    Raises:
        ValueError: on mismatched ELSE/END or out-of-range BR depth.
        TypeError:  if an element of wasm_instrs is not a tuple/list.

    Example::

        prog = compile_structured([
            ('PUSH', 5),
            ('BLOCK',),
              ('PUSH', 1),
              ('BR_IF', 0),   # exit block (always, since 1 != 0)
              ('PUSH', 99),   # unreachable
            ('END',),
            ('HALT',),
        ])
        # Flat: PUSH 5; PUSH 1; JNZ 6; PUSH 99; HALT

    """
    flat = []  # growing list of Instruction
    lbl_stack = []  # control-flow frames (dicts) pushed by BLOCK/LOOP/IF
    pending = {}  # label_id -> [flat-indices of JNZ/JZ to backpatch]
    _nxt = [0]  # mutable counter for label IDs

    # ── label helpers ────────────────────────────────────────────────

    def _alloc():
        lbl = _nxt[0]
        _nxt[0] += 1
        pending[lbl] = []
        return lbl

    def _resolve(lbl):
        """Patch every pending jump for lbl to the current flat length."""
        addr = len(flat)
        for idx in pending.pop(lbl):
            flat[idx] = Instruction(flat[idx].op, addr)

    def _jcc_fwd(op, lbl):
        """Emit a conditional jump (OP_JZ or OP_JNZ) to a forward label."""
        flat.append(Instruction(op, 0))
        pending[lbl].append(len(flat) - 1)

    def _jump_fwd(lbl):
        """Emit unconditional jump (PUSH 1; JNZ) to a forward label."""
        flat.append(Instruction(OP_PUSH, 1))
        flat.append(Instruction(OP_JNZ, 0))
        pending[lbl].append(len(flat) - 1)

    def _jump_addr(addr):
        """Emit unconditional jump (PUSH 1; JNZ) to a known address."""
        flat.append(Instruction(OP_PUSH, 1))
        flat.append(Instruction(OP_JNZ, addr))

    def _blk(n):
        """Return the n-th enclosing frame (0 = innermost)."""
        idx = len(lbl_stack) - 1 - n
        if idx < 0:
            raise ValueError(
                f"BR/BR_IF/BR_TABLE depth {n} exceeds label stack depth "
                f"{len(lbl_stack)}",
            )
        return lbl_stack[idx]

    def _jump_to_blk(blk):
        """Emit a jump (unconditional) to the natural target of blk.

        For LOOP: jump to start (backward, known addr).
        For BLOCK/IF: jump to end (forward, pending label).
        """
        if blk["kind"] == "loop":
            _jump_addr(blk["start"])
        else:
            _jump_fwd(blk["end"])

    def _jcc_to_blk(op, blk):
        """Emit a conditional jump (op) to the natural target of blk."""
        if blk["kind"] == "loop":
            flat.append(Instruction(op, blk["start"]))
        else:
            _jcc_fwd(op, blk["end"])

    # ── main compilation loop ─────────────────────────────────────────

    for raw in wasm_instrs:
        if not isinstance(raw, (list, tuple)):
            raise TypeError(
                f"Expected tuple/list instruction, got {type(raw).__name__}: {raw!r}",
            )
        name = raw[0].upper() if isinstance(raw[0], str) else raw[0]

        if name == "BLOCK":
            lbl_stack.append({"kind": "block", "end": _alloc()})

        elif name == "LOOP":
            lbl_stack.append(
                {
                    "kind": "loop",
                    "start": len(flat),  # backward target = current position
                    "end": _alloc(),
                },
            )

        elif name == "IF":
            else_lbl, end_lbl = _alloc(), _alloc()
            _jcc_fwd(OP_JZ, else_lbl)  # jump to else/end if condition == 0
            lbl_stack.append({"kind": "if", "else": else_lbl, "end": end_lbl})

        elif name == "ELSE":
            blk = lbl_stack[-1]
            if blk["kind"] != "if":
                raise ValueError(f"ELSE without matching IF (found {blk['kind']!r})")
            _jump_fwd(blk["end"])  # then-branch skips else-body
            _resolve(blk["else"])  # else-body starts here

        elif name == "END":
            blk = lbl_stack.pop()
            if blk["kind"] == "if":
                # If no ELSE was seen, the else-label still needs resolving.
                if blk["else"] in pending:
                    _resolve(blk["else"])
            _resolve(blk["end"])

        elif name == "BR":
            n = raw[1]
            _jump_to_blk(_blk(n))

        elif name == "BR_IF":
            n = raw[1]
            _jcc_to_blk(OP_JNZ, _blk(n))

        elif name == "BR_TABLE":
            labels, default_depth = raw[1], raw[2]
            n_cases = len(labels)

            # Allocate a handler label for every case.
            handler_lbls = [_alloc() for _ in range(n_cases)]

            # ── N comparison chains ──────────────────────────────────
            # Each: DUP; PUSH i; EQ; JNZ <handler_i>
            # Stack invariant: original index is preserved until handler POP.
            for i, hlbl in enumerate(handler_lbls):
                flat.append(Instruction(OP_DUP, 0))
                flat.append(Instruction(OP_PUSH, i))
                flat.append(Instruction(OP_EQ, 0))
                _jcc_fwd(OP_JNZ, hlbl)

            # ── default: POP index + jump ────────────────────────────
            flat.append(Instruction(OP_POP, 0))
            _jump_to_blk(_blk(default_depth))

            # ── case handlers: POP index + jump to case target ───────
            for depth, hlbl in zip(labels, handler_lbls):
                _resolve(hlbl)
                flat.append(Instruction(OP_POP, 0))
                _jump_to_blk(_blk(depth))

        else:
            # Pass through as a direct flat instruction.
            flat.extend(_flat(raw))

    return flat
