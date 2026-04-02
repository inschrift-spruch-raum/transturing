"""WAT (WebAssembly Text Format) parser for the stack machine ISA.

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

import re

from .assembler import compile_structured
from .isa import OP_HALT, Instruction

# ─── WAT mnemonic -> assembler tuple name mapping ─────────────────

# Simple instructions (no argument)
_SIMPLE_OPS = {
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
_KEYWORDS = frozenset(
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
    """Tokenize WAT text into a flat list of tokens.

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

    tokens = []
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
    if s.startswith("0x") or s.startswith("-0x") or s.startswith("+0x"):
        return int(s, 16)
    return int(s)


def _tokens_to_structured(tokens: list[str]) -> list[tuple]:
    """Convert flat token list to structured assembler tuples.

    This is the core of the parser. It walks the token stream and
    produces tuples compatible with assembler.compile_structured().

    Handles:
    - (func ...) wrappers — extracts body instructions
    - (module ...) wrappers — extracts all function bodies
    - (param ...), (result ...), (local ...) declarations — skipped
    - Named labels ($label) on block/loop/if — tracked for br resolution
    """
    instrs = []
    pos = 0
    label_stack = []  # stack of label names for br target resolution

    def _skip_sexpr():
        """Skip a complete s-expression (balanced parens)."""
        nonlocal pos
        if tokens[pos] != "(":
            pos += 1
            return
        depth = 1
        pos += 1
        while pos < len(tokens) and depth > 0:
            if tokens[pos] == "(":
                depth += 1
            elif tokens[pos] == ")":
                depth -= 1
            pos += 1

    def _resolve_br_target(target) -> int:
        """Resolve a br target: integer index or $label name."""
        if isinstance(target, int):
            return target
        # Named label — search label stack from top
        for i, name in enumerate(reversed(label_stack)):
            if name == target:
                return i
        raise ValueError(f"Unknown branch label: {target}")

    def _parse_body():
        """Parse instruction body from current position."""
        nonlocal pos

        while pos < len(tokens):
            tok = tokens[pos]

            if tok == ")":
                # End of enclosing s-expression
                return

            if tok == "(":
                # S-expression form: (instr ...)
                pos += 1  # skip '('
                if pos >= len(tokens):
                    break
                inner = tokens[pos]

                # Skip type/param/result/local declarations
                if inner in (
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
                ):
                    _skip_sexpr()
                    continue

                if inner == "func":
                    pos += 1
                    # Skip optional function name
                    if pos < len(tokens) and tokens[pos].startswith("$"):
                        pos += 1
                    # Skip param/result/local type declarations
                    while pos < len(tokens) and tokens[pos] == "(":
                        peek = tokens[pos + 1] if pos + 1 < len(tokens) else ""
                        if peek in (
                            "param",
                            "result",
                            "local",
                            "type",
                            "export",
                            "import",
                        ):
                            _skip_sexpr()
                        else:
                            break
                    # Parse function body
                    _parse_body()
                    if pos < len(tokens) and tokens[pos] == ")":
                        pos += 1  # consume func closing ')'
                    continue

                if inner == "module":
                    pos += 1
                    # Skip optional module name
                    if pos < len(tokens) and tokens[pos].startswith("$"):
                        pos += 1
                    _parse_body()
                    if pos < len(tokens) and tokens[pos] == ")":
                        pos += 1
                    continue

                # S-expression instruction form: (i32.add (i32.const 3) (i32.const 5))
                # In WAT folded form, operand sub-expressions are evaluated first,
                # then the operator is applied. So we need to:
                # 1. Collect any immediate args (like the value in i32.const 5)
                # 2. Parse nested s-expression operands BEFORE emitting the op
                # 3. Emit the operator instruction last

                # Save position to collect immediate args
                saved_instrs_len = len(instrs)

                # Parse the instruction (this may consume immediate args)
                _parse_instruction(inner)

                # Check if there are nested s-expression operands
                nested_ops = []
                while pos < len(tokens) and tokens[pos] == "(":
                    # Parse nested operand — these go BEFORE the operator
                    nested_start = len(instrs)
                    # Recurse into the nested s-expression
                    pos += 1  # skip '('
                    if pos < len(tokens):
                        inner2 = tokens[pos]
                        if inner2 in (
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
                        ):
                            _skip_sexpr()
                            continue
                        # Save, parse nested, collect
                        nested_saved = len(instrs)
                        _parse_instruction(inner2)
                        # Recursively handle deeper nesting
                        while pos < len(tokens) and tokens[pos] == "(":
                            inner_pos = pos
                            pos += 1
                            if pos < len(tokens):
                                _parse_instruction(tokens[pos])
                                while pos < len(tokens) and tokens[pos] != ")":
                                    pos += 1
                                if pos < len(tokens):
                                    pos += 1
                        # Skip to closing ')'
                        while pos < len(tokens) and tokens[pos] != ")":
                            pos += 1
                        if pos < len(tokens):
                            pos += 1  # skip ')'
                        nested_ops.extend(instrs[nested_saved:])
                        del instrs[nested_saved:]

                # Skip any remaining tokens before closing ')'
                while pos < len(tokens) and tokens[pos] != ")":
                    pos += 1

                if pos < len(tokens) and tokens[pos] == ")":
                    pos += 1  # consume closing ')'

                # If we have nested operands, reorder: operands first, then op
                if nested_ops:
                    op_instrs = instrs[saved_instrs_len:]
                    del instrs[saved_instrs_len:]
                    instrs.extend(nested_ops)
                    instrs.extend(op_instrs)

                continue

            # Linear (non-s-expression) instruction
            _parse_instruction(tok)

    def _parse_instruction(tok: str):
        """Parse a single instruction token and its arguments."""
        nonlocal pos
        pos += 1  # consume the instruction token

        tok_lower = tok.lower()

        # ── Simple ops (no argument) ──
        if tok_lower in _SIMPLE_OPS:
            instrs.append((_SIMPLE_OPS[tok_lower],))
            return

        # ── i32.const <value> ──
        if tok_lower == "i32.const":
            if pos < len(tokens) and tokens[pos] not in ("(", ")"):
                val = _parse_int(tokens[pos])
                pos += 1
            else:
                raise ValueError("i32.const requires a value argument")
            instrs.append(("PUSH", val))
            return

        # ── local.get/set/tee <index> ──
        if tok_lower == "local.get":
            idx = _parse_local_idx()
            instrs.append(("LOCAL.GET", idx))
            return
        if tok_lower == "local.set":
            idx = _parse_local_idx()
            instrs.append(("LOCAL.SET", idx))
            return
        if tok_lower == "local.tee":
            idx = _parse_local_idx()
            instrs.append(("LOCAL.TEE", idx))
            return

        # ── call <index> ──
        if tok_lower == "call":
            if pos < len(tokens) and tokens[pos] not in ("(", ")"):
                idx = _parse_int(tokens[pos])
                pos += 1
            else:
                raise ValueError("call requires a function index")
            instrs.append(("CALL", idx))
            return

        # ── Control flow ──
        if tok_lower == "block":
            label = None
            # Optional $label
            if pos < len(tokens) and tokens[pos].startswith("$"):
                label = tokens[pos]
                pos += 1
            # Skip optional (result ...) type annotation
            while pos < len(tokens) and tokens[pos] == "(":
                peek = tokens[pos + 1] if pos + 1 < len(tokens) else ""
                if peek == "result":
                    _skip_sexpr()
                else:
                    break
            label_stack.append(label)
            instrs.append(("BLOCK",))
            return

        if tok_lower == "loop":
            label = None
            if pos < len(tokens) and tokens[pos].startswith("$"):
                label = tokens[pos]
                pos += 1
            while pos < len(tokens) and tokens[pos] == "(":
                peek = tokens[pos + 1] if pos + 1 < len(tokens) else ""
                if peek == "result":
                    _skip_sexpr()
                else:
                    break
            label_stack.append(label)
            instrs.append(("LOOP",))
            return

        if tok_lower == "if":
            label = None
            if pos < len(tokens) and tokens[pos].startswith("$"):
                label = tokens[pos]
                pos += 1
            while pos < len(tokens) and tokens[pos] == "(":
                peek = tokens[pos + 1] if pos + 1 < len(tokens) else ""
                if peek in ("result", "param"):
                    _skip_sexpr()
                else:
                    break
            label_stack.append(label)
            instrs.append(("IF",))
            return

        if tok_lower == "else":
            instrs.append(("ELSE",))
            return

        if tok_lower == "end":
            if label_stack:
                label_stack.pop()
            instrs.append(("END",))
            return

        if tok_lower == "br":
            target = _parse_br_target()
            depth = _resolve_br_target(target)
            instrs.append(("BR", depth))
            return

        if tok_lower == "br_if":
            target = _parse_br_target()
            depth = _resolve_br_target(target)
            instrs.append(("BR_IF", depth))
            return

        if tok_lower == "br_table":
            # br_table <label>* <default_label>
            targets = []
            while (
                pos < len(tokens)
                and tokens[pos] not in ("(", ")")
                and tokens[pos].lower() not in _KEYWORDS
            ):
                targets.append(_parse_br_target_raw())
            if len(targets) < 1:
                raise ValueError("br_table requires at least a default label")
            default = _resolve_br_target(targets[-1])
            labels = [_resolve_br_target(t) for t in targets[:-1]]
            instrs.append(("BR_TABLE", labels, default))
            return

        # ── unreachable -> TRAP via HALT ──
        if tok_lower == "unreachable":
            instrs.append(("HALT",))
            return

        # ── HALT (our extension) ──
        if tok_lower == "halt":
            instrs.append(("HALT",))
            return

        # ── Stack manipulation (our extensions) ──
        if tok_lower == "dup":
            instrs.append(("DUP",))
            return
        if tok_lower == "swap":
            instrs.append(("SWAP",))
            return
        if tok_lower == "over":
            instrs.append(("OVER",))
            return
        if tok_lower == "rot":
            instrs.append(("ROT",))
            return

        raise ValueError(f"Unknown WAT instruction: {tok!r}")

    def _parse_local_idx() -> int:
        """Parse a local variable index (integer only, named locals not supported)."""
        nonlocal pos
        if pos < len(tokens) and tokens[pos] not in ("(", ")"):
            tok = tokens[pos]
            pos += 1
            return _parse_int(tok)
        raise ValueError("local.get/set/tee requires an index argument")

    def _parse_br_target():
        """Parse a br target: integer depth or $label name."""
        nonlocal pos
        if pos < len(tokens) and tokens[pos] not in ("(", ")"):
            tok = tokens[pos]
            pos += 1
            if tok.startswith("$"):
                return tok
            return _parse_int(tok)
        raise ValueError("br/br_if requires a target argument")

    def _parse_br_target_raw():
        """Parse a br_table target without consuming it from the resolve context."""
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok.startswith("$"):
            return tok
        return _parse_int(tok)

    _parse_body()
    return instrs


def parse_wat(text: str, *, append_halt: bool = True) -> list[Instruction]:
    """Parse WAT text and return a flat List[Instruction].

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
        # -> [Instruction(1, 3), Instruction(1, 5), Instruction(3, 0), Instruction(5, 0)]

    Example with function wrapper::

        prog = parse_wat('''
            (func $add (param i32 i32) (result i32)
              local.get 0
              local.get 1
              i32.add
            )
        ''')
        # -> [Instruction(43, 0), Instruction(43, 1), Instruction(3, 0), Instruction(5, 0)]

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

    if has_structured:
        flat = compile_structured(structured)
    else:
        # No control flow — use isa.program() directly
        from .isa import program as _prog

        flat = _prog(*structured)

    # Append HALT if needed
    if append_halt and (not flat or flat[-1].op != OP_HALT):
        flat.append(Instruction(OP_HALT, 0))

    return flat


# ─── Self-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    from .isa import OP_ADD, OP_LOCAL_GET, OP_PUSH
    from .isa import program as _prog

    print("=== WAT Parser Self-Test ===\n")
    passed = 0
    failed = 0

    def check(name, got, expected):
        global passed, failed
        if got == expected:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    expected: {expected}")
            print(f"    got:      {got}")
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
    check("block/br_if produces instructions", len(prog) > 0, True)

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
    check("loop/br_if produces instructions", len(prog) > 0, True)

    # Test 7: drop -> POP
    prog = parse_wat("i32.const 1 i32.const 2 drop", append_halt=False)
    from .isa import OP_POP

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
    check("named labels parse", len(prog) > 0, True)

    # Test 14: if/else/end
    prog = parse_wat("""
        i32.const 1
        if
          i32.const 10
        else
          i32.const 20
        end
    """)
    check("if/else/end parses", len(prog) > 0, True)

    # Test 15: br_table
    prog = parse_wat("""
        i32.const 0
        block $a
          block $b
            br_table 0 1 0
          end
        end
    """)
    check("br_table parses", len(prog) > 0, True)

    print(f"\n{passed} passed, {failed} failed")
    if failed:
        exit(1)
    print("All tests passed!")
