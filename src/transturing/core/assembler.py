"""
Structured control flow assembler for the stack machine.

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

from __future__ import annotations

from .isa import (
    OP_DUP,
    OP_EQ,
    OP_JNZ,
    OP_JZ,
    OP_POP,
    OP_PUSH,
    Instruction,
    WasmInstr,
)
from .isa import (
    program as _flat,
)

# Expected element counts for structured instructions.
_BR_ARGS_LEN = 2
_BR_TABLE_ARGS_LEN = 3


def compile_structured(wasm_instrs: list[WasmInstr]) -> list[Instruction]:
    """
    Compile structured WASM-style instructions to flat Instruction list.

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
    asm = _Assembler()
    return asm.compile(wasm_instrs)


class _Assembler:
    """Stateful assembler that compiles structured WASM to flat instructions."""

    def __init__(self) -> None:
        self.flat: list[Instruction] = []
        self.lbl_stack: list[dict[str, int | str]] = []
        self.pending: dict[int, list[int]] = {}
        self._nxt: int = 0

    # ── label helpers ────────────────────────────────────────────

    def _alloc(self) -> int:
        lbl = self._nxt
        self._nxt += 1
        self.pending[lbl] = []
        return lbl

    def _resolve(self, lbl: int) -> None:
        """Patch every pending jump for lbl to the current flat length."""
        addr = len(self.flat)
        for idx in self.pending.pop(lbl):
            self.flat[idx] = Instruction(self.flat[idx].op, addr)

    def _jcc_fwd(self, op: int, lbl: int) -> None:
        """Emit a conditional jump (OP_JZ or OP_JNZ) to a forward label."""
        self.flat.append(Instruction(op, 0))
        self.pending[lbl].append(len(self.flat) - 1)

    def _jump_fwd(self, lbl: int) -> None:
        """Emit unconditional jump (PUSH 1; JNZ) to a forward label."""
        self.flat.append(Instruction(OP_PUSH, 1))
        self.flat.append(Instruction(OP_JNZ, 0))
        self.pending[lbl].append(len(self.flat) - 1)

    def _jump_addr(self, addr: int) -> None:
        """Emit unconditional jump (PUSH 1; JNZ) to a known address."""
        self.flat.append(Instruction(OP_PUSH, 1))
        self.flat.append(Instruction(OP_JNZ, addr))

    def _blk(self, n: int) -> dict[str, int | str]:
        """Return the n-th enclosing frame (0 = innermost)."""
        idx = len(self.lbl_stack) - 1 - n
        if idx < 0:
            msg = (
                f"BR/BR_IF/BR_TABLE depth {n} exceeds label stack depth "
                f"{len(self.lbl_stack)}"
            )
            raise ValueError(
                msg,
            )
        return self.lbl_stack[idx]

    def _jump_to_blk(self, blk: dict[str, int | str]) -> None:
        """
        Emit a jump (unconditional) to the natural target of blk.

        For LOOP: jump to start (backward, known addr).
        For BLOCK/IF: jump to end (forward, pending label).
        """
        if blk["kind"] == "loop":
            self._jump_addr(int(blk["start"]))
        else:
            self._jump_fwd(int(blk["end"]))

    def _jcc_to_blk(self, op: int, blk: dict[str, int | str]) -> None:
        """Emit a conditional jump (op) to the natural target of blk."""
        if blk["kind"] == "loop":
            self.flat.append(Instruction(op, int(blk["start"])))
        else:
            self._jcc_fwd(op, int(blk["end"]))

    # ── instruction handlers ─────────────────────────────────────

    def _handle_block(self) -> None:
        self.lbl_stack.append({"kind": "block", "end": self._alloc()})

    def _handle_loop(self) -> None:
        self.lbl_stack.append(
            {
                "kind": "loop",
                "start": len(self.flat),  # backward target = current position
                "end": self._alloc(),
            },
        )

    def _handle_if(self) -> None:
        else_lbl, end_lbl = self._alloc(), self._alloc()
        self._jcc_fwd(OP_JZ, else_lbl)  # jump to else/end if condition == 0
        self.lbl_stack.append({"kind": "if", "else": else_lbl, "end": end_lbl})

    def _handle_else(self) -> None:
        blk = self.lbl_stack[-1]
        if blk["kind"] != "if":
            msg = f"ELSE without matching IF (found {blk['kind']!r})"
            raise ValueError(msg)
        self._jump_fwd(int(blk["end"]))  # then-branch skips else-body
        self._resolve(int(blk["else"]))  # else-body starts here

    def _handle_end(self) -> None:
        blk = self.lbl_stack.pop()
        if blk["kind"] == "if" and int(blk["else"]) in self.pending:
            self._resolve(int(blk["else"]))
        self._resolve(int(blk["end"]))

    def _handle_br(self, raw: WasmInstr) -> None:
        if len(raw) != _BR_ARGS_LEN:
            msg = f"BR requires (name, depth), got {len(raw)} elements"
            raise ValueError(msg)
        self._jump_to_blk(self._blk(raw[1]))

    def _handle_br_if(self, raw: WasmInstr) -> None:
        if len(raw) != _BR_ARGS_LEN:
            msg = f"BR_IF requires (name, depth), got {len(raw)} elements"
            raise ValueError(msg)
        self._jcc_to_blk(OP_JNZ, self._blk(raw[1]))

    def _handle_br_table(self, raw: WasmInstr) -> None:
        if len(raw) != _BR_TABLE_ARGS_LEN:
            msg = f"BR_TABLE requires (name, labels, default), got {len(raw)} elements"
            raise ValueError(msg)
        labels: list[int] = raw[1]
        default_depth: int = raw[2]
        n_cases = len(labels)

        # Allocate a handler label for every case.
        handler_lbls = [self._alloc() for _ in range(n_cases)]

        # ── N comparison chains ──────────────────────────────────
        # Each: DUP; PUSH i; EQ; JNZ <handler_i>
        # Stack invariant: original index is preserved until handler POP.
        for i, hlbl in enumerate(handler_lbls):
            self.flat.append(Instruction(OP_DUP, 0))
            self.flat.append(Instruction(OP_PUSH, i))
            self.flat.append(Instruction(OP_EQ, 0))
            self._jcc_fwd(OP_JNZ, hlbl)

        # ── default: POP index + jump ────────────────────────────
        self.flat.append(Instruction(OP_POP, 0))
        self._jump_to_blk(self._blk(default_depth))

        # ── case handlers: POP index + jump to case target ───────
        for depth, hlbl in zip(labels, handler_lbls, strict=False):
            self._resolve(hlbl)
            self.flat.append(Instruction(OP_POP, 0))
            self._jump_to_blk(self._blk(depth))

    # ── main compilation loop ────────────────────────────────────

    def compile(self, wasm_instrs: list[WasmInstr]) -> list[Instruction]:
        for raw in wasm_instrs:
            name = raw[0].upper()

            if name == "BLOCK":
                self._handle_block()
            elif name == "LOOP":
                self._handle_loop()
            elif name == "IF":
                self._handle_if()
            elif name == "ELSE":
                self._handle_else()
            elif name == "END":
                self._handle_end()
            elif name == "BR":
                self._handle_br(raw)
            elif name == "BR_IF":
                self._handle_br_if(raw)
            elif name == "BR_TABLE":
                self._handle_br_table(raw)
            else:
                # Pass through as a direct flat instruction.
                self.flat.extend(_flat(raw))

        return self.flat
