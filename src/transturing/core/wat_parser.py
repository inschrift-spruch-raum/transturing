"""
WAT (WebAssembly Text Format) parser for the stack machine ISA.

Parses a subset of WAT text format and converts it to List[Instruction].
Focuses on the i32 subset that maps to our existing opcodes.

This is the input side of Percepta's C -> WASM -> tokens pipeline.

Supported WAT instructions
--------------------------
  Arithmetic: i32.const, i32.add, i32.sub, i32.mul, i32.div_s, i32.div_u,
              i32.rem_s, i32.rem_u
  Comparison: i32.eqz, i32.eq, i32.ne, i32.lt_s, i32.lt_u, i32.gt_s,
              i32.gt_u, i32.le_s, i32.le_u, i32.ge_s, i32.ge_u
  Bitwise:    i32.and, i32.or, i32.xor, i32.shl, i32.shr_s, i32.shr_u,
              i32.rotl, i32.rotr, i32.clz, i32.ctz, i32.popcnt
  Locals:     local.get, local.set, local.tee
  Memory:     i32.load, i32.store, i32.load8_u, i32.load8_s,
              i32.load16_u, i32.load16_s, i32.store8, i32.store16
  Control:    block, loop, if, else, end, br, br_if, br_table
  Functions:  call, return
  Parametric: drop (-> POP), select
  Stack:      nop

References
----------
  Issue #49: WAT parser

"""

from __future__ import annotations

import dataclasses
import re
import sys
from dataclasses import field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

from .assembler import compile_structured
from .isa import (
    OP_ADD,
    OP_HALT,
    OP_LOCAL_GET,
    OP_POP,
    OP_PUSH,
    Instruction,
    WasmInstr,
)
from .isa import (
    program as _prog,
)

# ─── WAT mnemonic -> assembler tuple name mapping ─────────────────

# Simple instructions (no argument)
_SIMPLE_OPS: dict[str, str] = {
    # Arithmetic
    "i32.add": "ADD",
    "i32.sub": "SUB",
    "i32.mul": "MUL",
    "i32.div_s": "DIV_S",
    "i32.div_u": "DIV_U",
    "i32.rem_s": "REM_S",
    "i32.rem_u": "REM_U",
    # Comparison
    "i32.eqz": "EQZ",
    "i32.eq": "EQ",
    "i32.ne": "NE",
    "i32.lt_s": "LT_S",
    "i32.lt_u": "LT_U",
    "i32.gt_s": "GT_S",
    "i32.gt_u": "GT_U",
    "i32.le_s": "LE_S",
    "i32.le_u": "LE_U",
    "i32.ge_s": "GE_S",
    "i32.ge_u": "GE_U",
    # Bitwise
    "i32.and": "AND",
    "i32.or": "OR",
    "i32.xor": "XOR",
    "i32.shl": "SHL",
    "i32.shr_s": "SHR_S",
    "i32.shr_u": "SHR_U",
    "i32.rotl": "ROTL",
    "i32.rotr": "ROTR",
    "i32.clz": "CLZ",
    "i32.ctz": "CTZ",
    "i32.popcnt": "POPCNT",
    # Stack manipulation
    "drop": "POP",
    "select": "SELECT",
    "nop": "NOP",
    # Memory
    "i32.load": "I32.LOAD",
    "i32.store": "I32.STORE",
    "i32.load8_u": "I32.LOAD8_U",
    "i32.load8_s": "I32.LOAD8_S",
    "i32.load16_u": "I32.LOAD16_U",
    "i32.load16_s": "I32.LOAD16_S",
    "i32.store8": "I32.STORE8",
    "i32.store16": "I32.STORE16",
    # Functions
    "return": "RETURN",
}


# Keywords that terminate br_table label lists
_KEYWORDS: frozenset[str] = frozenset(
    {
        "block",
        "loop",
        "if",
        "else",
        "end",
        "br",
        "br_if",
        "br_table",
        "call",
        "return",
        "nop",
        "unreachable",
        "halt",
        "drop",
        "select",
        "dup",
        "swap",
        "over",
        "rot",
    }
    | set(_SIMPLE_OPS.keys())
    | {
        "i32.const",
        "local.get",
        "local.set",
        "local.tee",
    },
)


def _tokenize(text: str) -> list[str]:
    """
    Tokenize WAT text into a flat list of tokens.

    Handles:
    - Stripping ;; line comments and (; block comments ;)
    - Parentheses as separate tokens
    - Quoted strings (preserved as single tokens)
    - All other whitespace-separated tokens
    """
    # Remove block comments (; ... ;)
    text = re.sub(r"\(;.*?;\)", "", text, flags=re.DOTALL)
    # Remove line comments
    text = re.sub(r";;[^\n]*", "", text)

    tokens: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        if c in " \t\n\r":
            i += 1
        elif c == "(":
            tokens.append("(")
            i += 1
        elif c == ")":
            tokens.append(")")
            i += 1
        elif c == '"':
            # Quoted string — find closing quote (handle escaped quotes)
            j = i + 1
            while j < len(text):
                if text[j] == "\\":
                    j += 2
                elif text[j] == '"':
                    j += 1
                    break
                else:
                    j += 1
            tokens.append(text[i:j])
            i = j
        else:
            # Regular token — read until whitespace or paren
            j = i
            while j < len(text) and text[j] not in " \t\n\r()":
                j += 1
            tokens.append(text[i:j])
            i = j
    return tokens


def _parse_int(s: str) -> int:
    """Parse a WAT integer literal (decimal or hex, with optional sign)."""
    s = s.replace("_", "")  # WAT allows _ separators
    if s.startswith(("0x", "-0x", "+0x")):
        return int(s, 16)
    return int(s)


# ─── Parser state ──────────────────────────────────────────────────

# Declaration keywords to skip in s-expressions
_DECL_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "param",
        "result",
        "local",
        "export",
        "import",
        "memory",
        "table",
        "global",
        "elem",
        "data",
    }
)

# Keywords to skip inside folded s-expression operands
_SKIP_DECLS: frozenset[str] = frozenset(
    {
        "func",
        "module",
        "param",
        "result",
        "local",
        "type",
        "export",
        "import",
        "memory",
        "table",
        "global",
    }
)

# Type declaration keywords for func preamble
_TYPE_DECLS: frozenset[str] = frozenset(
    {"param", "result", "local", "type", "export", "import"}
)

# Extended simple ops (no argument, not in _SIMPLE_OPS)
_EXTENDED_SIMPLE: dict[str, str] = {
    "dup": "DUP",
    "swap": "SWAP",
    "over": "OVER",
    "rot": "ROT",
    "unreachable": "HALT",
    "halt": "HALT",
    "else": "ELSE",
}


@dataclasses.dataclass
class _ParserState:
    """Mutable state shared across parser functions."""

    tokens: list[str]
    pos: int = 0
    instrs: list[WasmInstr] = field(default_factory=list)
    label_stack: list[str | None] = field(default_factory=list)


# ─── Low-level parser helpers ──────────────────────────────────────


def _skip_sexpr(state: _ParserState) -> None:
    """Skip a complete s-expression (balanced parens)."""
    if state.tokens[state.pos] != "(":
        state.pos += 1
        return
    depth = 1
    state.pos += 1
    while state.pos < len(state.tokens) and depth > 0:
        if state.tokens[state.pos] == "(":
            depth += 1
        elif state.tokens[state.pos] == ")":
            depth -= 1
        state.pos += 1


def _resolve_br_target(state: _ParserState, target: str | int) -> int:
    """Resolve a br target: integer index or $label name."""
    if isinstance(target, int):
        return target
    # Named label — search label stack from top
    for i, name in enumerate(reversed(state.label_stack)):
        if name == target:
            return i
    msg = f"Unknown branch label: {target}"
    raise ValueError(msg)


def _parse_local_idx(state: _ParserState) -> int:
    """Parse a local variable index (integer only, named locals not supported)."""
    if state.pos < len(state.tokens) and state.tokens[state.pos] not in ("(", ")"):
        tok = state.tokens[state.pos]
        state.pos += 1
        return _parse_int(tok)
    msg = "local.get/set/tee requires an index argument"
    raise ValueError(msg)


def _parse_br_target(state: _ParserState) -> str | int:
    """Parse a br target: integer depth or $label name."""
    if state.pos < len(state.tokens) and state.tokens[state.pos] not in ("(", ")"):
        tok = state.tokens[state.pos]
        state.pos += 1
        if tok.startswith("$"):
            return tok
        return _parse_int(tok)
    msg = "br/br_if requires a target argument"
    raise ValueError(msg)


def _parse_br_target_raw(state: _ParserState) -> str | int:
    """Parse a br_table target without consuming it from the resolve context."""
    tok = state.tokens[state.pos]
    state.pos += 1
    if tok.startswith("$"):
        return tok
    return _parse_int(tok)


# ─── Instruction handlers (dispatch targets) ──────────────────────


def _handle_i32_const(state: _ParserState) -> None:
    """Handle i32.const <value>."""
    if state.pos < len(state.tokens) and state.tokens[state.pos] not in ("(", ")"):
        val = _parse_int(state.tokens[state.pos])
        state.pos += 1
    else:
        msg = "i32.const requires a value argument"
        raise ValueError(msg)
    state.instrs.append(("PUSH", val))


def _handle_local_get(state: _ParserState) -> None:
    """Handle local.get <index>."""
    idx = _parse_local_idx(state)
    state.instrs.append(("LOCAL.GET", idx))


def _handle_local_set(state: _ParserState) -> None:
    """Handle local.set <index>."""
    idx = _parse_local_idx(state)
    state.instrs.append(("LOCAL.SET", idx))


def _handle_local_tee(state: _ParserState) -> None:
    """Handle local.tee <index>."""
    idx = _parse_local_idx(state)
    state.instrs.append(("LOCAL.TEE", idx))


def _handle_call(state: _ParserState) -> None:
    """Handle call <index>."""
    if state.pos < len(state.tokens) and state.tokens[state.pos] not in ("(", ")"):
        idx = _parse_int(state.tokens[state.pos])
        state.pos += 1
    else:
        msg = "call requires a function index"
        raise ValueError(msg)
    state.instrs.append(("CALL", idx))


def _handle_block_start(
    state: _ParserState,
    instr: str,
    *,
    skip_types: tuple[str, ...] = ("result",),
) -> None:
    """Handle block/loop/if with optional label and type annotations."""
    label: str | None = None
    if state.pos < len(state.tokens) and state.tokens[state.pos].startswith("$"):
        label = state.tokens[state.pos]
        state.pos += 1
    while state.pos < len(state.tokens) and state.tokens[state.pos] == "(":
        peek = state.tokens[state.pos + 1] if state.pos + 1 < len(state.tokens) else ""
        if peek in skip_types:
            _skip_sexpr(state)
        else:
            break
    state.label_stack.append(label)
    state.instrs.append((instr,))


def _handle_block(state: _ParserState) -> None:
    _handle_block_start(state, "BLOCK")


def _handle_loop(state: _ParserState) -> None:
    _handle_block_start(state, "LOOP")


def _handle_if(state: _ParserState) -> None:
    _handle_block_start(state, "IF", skip_types=("result", "param"))


def _handle_end(state: _ParserState) -> None:
    if state.label_stack:
        state.label_stack.pop()
    state.instrs.append(("END",))


def _handle_br(state: _ParserState) -> None:
    target = _parse_br_target(state)
    depth = _resolve_br_target(state, target)
    state.instrs.append(("BR", depth))


def _handle_br_if(state: _ParserState) -> None:
    target = _parse_br_target(state)
    depth = _resolve_br_target(state, target)
    state.instrs.append(("BR_IF", depth))


def _handle_br_table(state: _ParserState) -> None:
    """Handle br_table <label>* <default_label>."""
    targets: list[str | int] = []
    while (
        state.pos < len(state.tokens)
        and state.tokens[state.pos] not in ("(", ")")
        and state.tokens[state.pos].lower() not in _KEYWORDS
    ):
        targets.append(_parse_br_target_raw(state))
    if len(targets) < 1:
        msg = "br_table requires at least a default label"
        raise ValueError(msg)
    default = _resolve_br_target(state, targets[-1])
    labels = [_resolve_br_target(state, t) for t in targets[:-1]]
    state.instrs.append(("BR_TABLE", labels, default))


# Dispatch table: instruction name -> handler function
_INSTR_HANDLERS: dict[str, Callable[[_ParserState], None]] = {
    "i32.const": _handle_i32_const,
    "local.get": _handle_local_get,
    "local.set": _handle_local_set,
    "local.tee": _handle_local_tee,
    "call": _handle_call,
    "block": _handle_block,
    "loop": _handle_loop,
    "if": _handle_if,
    "end": _handle_end,
    "br": _handle_br,
    "br_if": _handle_br_if,
    "br_table": _handle_br_table,
}


# ─── Core parser functions ────────────────────────────────────────


def _parse_instruction(state: _ParserState, tok: str) -> None:
    """Parse a single instruction token and its arguments."""
    state.pos += 1  # consume the instruction token
    tok_lower = tok.lower()

    # Simple ops (no argument)
    if tok_lower in _SIMPLE_OPS:
        state.instrs.append((_SIMPLE_OPS[tok_lower],))
        return

    # Extended simple ops (no argument)
    if tok_lower in _EXTENDED_SIMPLE:
        state.instrs.append((_EXTENDED_SIMPLE[tok_lower],))
        return

    # Complex instructions via dispatch
    handler = _INSTR_HANDLERS.get(tok_lower)
    if handler is not None:
        handler(state)
        return

    msg = f"Unknown WAT instruction: {tok!r}"
    raise ValueError(msg)


def _parse_deep_nested(state: _ParserState) -> None:
    """Parse deeply nested s-expression operands within a folded instruction."""
    while state.pos < len(state.tokens) and state.tokens[state.pos] == "(":
        state.pos += 1
        if state.pos < len(state.tokens):
            _parse_instruction(state, state.tokens[state.pos])
            while state.pos < len(state.tokens) and state.tokens[state.pos] != ")":
                state.pos += 1
            if state.pos < len(state.tokens):
                state.pos += 1


def _parse_sexpr_folded(state: _ParserState, inner: str) -> None:
    """Handle folded s-expression instruction form: (op operand1 operand2 ...)."""
    saved_instrs_len = len(state.instrs)

    # Parse the instruction (this may consume immediate args)
    _parse_instruction(state, inner)

    # Collect nested s-expression operands (evaluated before the operator)
    nested_ops: list[WasmInstr] = []
    while state.pos < len(state.tokens) and state.tokens[state.pos] == "(":
        state.pos += 1  # skip '('
        if state.pos >= len(state.tokens):
            break
        inner2 = state.tokens[state.pos]
        if inner2 in _SKIP_DECLS:
            _skip_sexpr(state)
            continue
        # Save, parse nested, collect
        nested_saved = len(state.instrs)
        _parse_instruction(state, inner2)
        # Recursively handle deeper nesting
        _parse_deep_nested(state)
        # Skip to closing ')' of this nested operand
        while state.pos < len(state.tokens) and state.tokens[state.pos] != ")":
            state.pos += 1
        if state.pos < len(state.tokens):
            state.pos += 1  # skip ')'
        nested_ops.extend(state.instrs[nested_saved:])
        del state.instrs[nested_saved:]

    # Skip any remaining tokens before closing ')'
    while state.pos < len(state.tokens) and state.tokens[state.pos] != ")":
        state.pos += 1

    if state.pos < len(state.tokens) and state.tokens[state.pos] == ")":
        state.pos += 1  # consume closing ')'

    # Reorder: nested operands first, then the operator
    if nested_ops:
        op_instrs = state.instrs[saved_instrs_len:]
        del state.instrs[saved_instrs_len:]
        state.instrs.extend(nested_ops)
        state.instrs.extend(op_instrs)


def _parse_func(state: _ParserState) -> None:
    """Parse a (func ...) s-expression."""
    state.pos += 1  # skip "func"
    # Skip optional function name
    if state.pos < len(state.tokens) and state.tokens[state.pos].startswith("$"):
        state.pos += 1
    # Skip param/result/local type declarations
    while state.pos < len(state.tokens) and state.tokens[state.pos] == "(":
        peek = state.tokens[state.pos + 1] if state.pos + 1 < len(state.tokens) else ""
        if peek in _TYPE_DECLS:
            _skip_sexpr(state)
        else:
            break
    # Parse function body
    _parse_body(state)
    if state.pos < len(state.tokens) and state.tokens[state.pos] == ")":
        state.pos += 1  # consume func closing ')'


def _parse_module(state: _ParserState) -> None:
    """Parse a (module ...) s-expression."""
    state.pos += 1  # skip "module"
    # Skip optional module name
    if state.pos < len(state.tokens) and state.tokens[state.pos].startswith("$"):
        state.pos += 1
    _parse_body(state)
    if state.pos < len(state.tokens) and state.tokens[state.pos] == ")":
        state.pos += 1


def _parse_sexpr_body(state: _ParserState, inner: str) -> None:
    """Handle an s-expression body after '(' has been consumed."""
    if inner in _DECL_KEYWORDS:
        _skip_sexpr(state)
        return
    if inner == "func":
        _parse_func(state)
        return
    if inner == "module":
        _parse_module(state)
        return
    # S-expression instruction form: (i32.add (i32.const 3) (i32.const 5))
    _parse_sexpr_folded(state, inner)


def _parse_body(state: _ParserState) -> None:
    """Parse instruction body from current position."""
    while state.pos < len(state.tokens):
        tok = state.tokens[state.pos]

        if tok == ")":
            # End of enclosing s-expression
            return

        if tok == "(":
            # S-expression form: (instr ...)
            state.pos += 1  # skip '('
            if state.pos >= len(state.tokens):
                break
            _parse_sexpr_body(state, state.tokens[state.pos])
            continue

        # Linear (non-s-expression) instruction
        _parse_instruction(state, tok)


def _tokens_to_structured(tokens: list[str]) -> list[WasmInstr]:
    """
    Convert flat token list to structured assembler tuples.

    This is the core of the parser. It walks the token stream and
    produces tuples compatible with assembler.compile_structured().

    Handles:
    - (func ...) wrappers — extracts body instructions
    - (module ...) wrappers — extracts all function bodies
    - (param ...), (result ...), (local ...) declarations — skipped
    - Named labels ($label) on block/loop/if — tracked for br resolution
    """
    state = _ParserState(tokens=tokens)
    _parse_body(state)
    return state.instrs


# ─── Public API ────────────────────────────────────────────────────


def parse_wat(text: str, *, append_halt: bool = True) -> list[Instruction]:
    """
    Parse WAT text and return a flat List[Instruction].

    Args:
        text: WAT source code (module, function, or bare instructions).
        append_halt: If True, append HALT if the program doesn't end with one.

    Returns:
        List[Instruction] ready for NumPyExecutor / TorchExecutor.

    Example::

        prog = parse_wat('''
            i32.const 3
            i32.const 5
            i32.add
        ''')
        # -> [Instruction(1,3), Instruction(1,5), Instruction(3,0), Instruction(5,0)]

    Example with function wrapper::

        prog = parse_wat('''
            (func $add (param i32 i32) (result i32)
              local.get 0
              local.get 1
              i32.add
            )
        ''')
        # -> [Instruction(43,0), Instruction(43,1), Instruction(3,0), Instruction(5,0)]

    """
    tokens = _tokenize(text)
    if not tokens:
        return []

    structured = _tokens_to_structured(tokens)
    if not structured:
        return []

    # Check if there are any structured control flow constructs
    has_structured = any(
        t[0] in ("BLOCK", "LOOP", "IF", "ELSE", "END", "BR", "BR_IF", "BR_TABLE")
        for t in structured
    )

    flat = compile_structured(structured) if has_structured else _prog(*structured)

    # Append HALT if needed
    if append_halt and (not flat or flat[-1].op != OP_HALT):
        flat.append(Instruction(OP_HALT, 0))

    return flat


# ─── Self-test ────────────────────────────────────────────────────


def _run_self_tests() -> None:
    """Run self-tests when invoked as __main__."""
    passed = 0
    failed = 0

    def check(_name: str, got: object, expected: object) -> None:
        """Assert test result matches expected value."""
        nonlocal passed, failed
        if got == expected:
            passed += 1
        else:
            failed += 1

    # Test 1: Basic arithmetic
    prog = parse_wat("i32.const 3  i32.const 5  i32.add")
    expected = _prog(("PUSH", 3), ("PUSH", 5), ("ADD",), ("HALT",))
    check("basic arithmetic", prog, expected)

    # Test 2: Function wrapper
    prog = parse_wat("""
        (func $add (param i32 i32) (result i32)
          local.get 0
          local.get 1
          i32.add
        )
    """)
    check(
        "func wrapper",
        prog[:3],
        [
            Instruction(OP_LOCAL_GET, 0),
            Instruction(OP_LOCAL_GET, 1),
            Instruction(OP_ADD, 0),
        ],
    )

    # Test 3: Comments
    prog = parse_wat("""
        ;; This is a comment
        i32.const 42  ;; inline comment
        (; block comment ;)
    """)
    check("comments", prog, [Instruction(OP_PUSH, 42), Instruction(OP_HALT, 0)])

    # Test 4: Hex literals
    prog = parse_wat("i32.const 0xff", append_halt=False)
    check("hex literal", prog, [Instruction(OP_PUSH, 255)])

    # Test 5: Control flow - block/br
    prog = parse_wat("""
        i32.const 5
        block
          i32.const 1
          br_if 0
          i32.const 99
        end
    """)
    check("block/br_if produces instructions", len(prog) > 0, expected=True)

    # Test 6: Loop
    prog = parse_wat("""
        i32.const 10
        loop $L
          i32.const 1
          i32.sub
          dup
          br_if 0
        end
    """)
    check("loop/br_if produces instructions", len(prog) > 0, expected=True)

    # Test 7: drop -> POP
    prog = parse_wat("i32.const 1 i32.const 2 drop", append_halt=False)
    check("drop -> POP", prog[2].op, OP_POP)

    # Test 8: Module wrapper
    prog = parse_wat("""
        (module
          (func $main
            i32.const 42
          )
        )
    """)
    check("module wrapper", prog[0], Instruction(OP_PUSH, 42))

    # Test 9: All comparison ops
    for op in ["i32.eq", "i32.ne", "i32.lt_s", "i32.gt_u", "i32.le_s", "i32.ge_u"]:
        prog = parse_wat(f"i32.const 1 i32.const 2 {op}", append_halt=False)
        check(f"{op} parses", len(prog), 3)

    # Test 10: All bitwise ops
    for op in [
        "i32.and",
        "i32.or",
        "i32.xor",
        "i32.shl",
        "i32.shr_s",
        "i32.shr_u",
        "i32.rotl",
        "i32.rotr",
    ]:
        prog = parse_wat(f"i32.const 1 i32.const 2 {op}", append_halt=False)
        check(f"{op} parses", len(prog), 3)

    # Test 11: Unary ops
    for op in ["i32.clz", "i32.ctz", "i32.popcnt", "i32.eqz"]:
        prog = parse_wat(f"i32.const 1 {op}", append_halt=False)
        check(f"{op} parses", len(prog), 2)

    # Test 12: Memory ops
    for op in ["i32.load", "i32.store", "i32.load8_u", "i32.store16"]:
        prog = parse_wat(f"i32.const 0 {op}", append_halt=False)
        check(f"{op} parses", len(prog), 2)

    # Test 13: Named labels
    prog = parse_wat("""
        block $outer
          block $inner
            i32.const 1
            br_if $inner
          end
        end
    """)
    check("named labels parse", len(prog) > 0, expected=True)

    # Test 14: if/else/end
    prog = parse_wat("""
        i32.const 1
        if
          i32.const 10
        else
          i32.const 20
        end
    """)
    check("if/else/end parses", len(prog) > 0, expected=True)

    # Test 15: br_table
    prog = parse_wat("""
        i32.const 0
        block $a
          block $b
            br_table 0 1 0
          end
        end
    """)
    check("br_table parses", len(prog) > 0, expected=True)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_self_tests()
