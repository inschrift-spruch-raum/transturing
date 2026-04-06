# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, PLR2004, TC003, TRY003, UP047
"""
Binary WebAssembly decoder for the supported i32 subset.

This module decodes a tightly scoped subset of MVP WebAssembly binaries into
structured module/function data. Function bodies are emitted as existing
``WasmInstr`` tuples so they can later flow into ``compile_structured()``.

Supported pieces:
  - module header/version
  - type, function, memory, export, code sections
  - i32-only func signatures, locals. and instruction bodies
  - structured control flow markers: BLOCK / LOOP / IF / ELSE / END
  - calls, branches, i32 arithmetic/comparison/bitwise, parametric ops
  - i32 memory ops with memarg decoding

Explicitly rejected:
  - imports, tables, globals, start, element/data segments. data_count
  - non-i32 value types/signatures/locals
  - unsupported opcodes or binary features not representable by WasmInstr
  - memory ops with non-zero offsets
  - block signatures other than the empty block type
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from .assembler import compile_structured
from .isa import OP_CALL, OP_HALT, OP_JNZ, OP_JZ, OP_RETURN, Instruction, WasmInstr
from .isa import program as _make_prog

_WASM_MAGIC = b"\x00asm"
_WASM_VERSION = 1

_SECTION_TYPE = 1
_SECTION_IMPORT = 2
_SECTION_FUNCTION = 3
_SECTION_TABLE = 4
_SECTION_MEMORY = 5
_SECTION_GLOBAL = 6
_SECTION_EXPORT = 7
_SECTION_START = 8
_SECTION_ELEMENT = 9
_SECTION_CODE = 10
_SECTION_DATA = 11
_SECTION_DATA_COUNT = 12

_VALTYPE_I32 = 0x7F
_FUNC_TYPE = 0x60
_EMPTY_BLOCK_TYPE = 0x40

_VALTYPE_NAMES: dict[int, str] = {
    0x7F: "i32",
    0x7E: "i64",
    0x7D: "f32",
    0x7C: "f64",
    0x7B: "v128",
    0x70: "funcref",
    0x6F: "externref",
}

_EXPORT_KIND_FUNC = 0x00
_EXPORT_KIND_TABLE = 0x01
_EXPORT_KIND_MEMORY = 0x02
_EXPORT_KIND_GLOBAL = 0x03

_KIND_NAMES = {
    _EXPORT_KIND_FUNC: "func",
    _EXPORT_KIND_TABLE: "table",
    _EXPORT_KIND_MEMORY: "memory",
    _EXPORT_KIND_GLOBAL: "global",
}

_CONTROL_OPS: dict[int, str] = {
    0x1A: "POP",
    0x1B: "SELECT",
    0x01: "NOP",
    0x0F: "RETURN",
}

_ARG_OPS: dict[int, str] = {
    0x20: "LOCAL.GET",
    0x21: "LOCAL.SET",
    0x22: "LOCAL.TEE",
    0x10: "CALL",
    0x0C: "BR",
    0x0D: "BR_IF",
}

_MEMORY_OPS: dict[int, str] = {
    0x28: "I32.LOAD",
    0x2C: "I32.LOAD8_S",
    0x2D: "I32.LOAD8_U",
    0x2E: "I32.LOAD16_S",
    0x2F: "I32.LOAD16_U",
    0x36: "I32.STORE",
    0x3A: "I32.STORE8",
    0x3B: "I32.STORE16",
}

_SIMPLE_BINARY_OPS: dict[int, str] = {
    0x45: "i32.eqz",
    0x46: "i32.eq",
    0x47: "i32.ne",
    0x48: "i32.lt_s",
    0x49: "i32.lt_u",
    0x4A: "i32.gt_s",
    0x4B: "i32.gt_u",
    0x4C: "i32.le_s",
    0x4D: "i32.le_u",
    0x4E: "i32.ge_s",
    0x4F: "i32.ge_u",
    0x67: "i32.clz",
    0x68: "i32.ctz",
    0x69: "i32.popcnt",
    0x6A: "i32.add",
    0x6B: "i32.sub",
    0x6C: "i32.mul",
    0x6D: "i32.div_s",
    0x6E: "i32.div_u",
    0x6F: "i32.rem_s",
    0x70: "i32.rem_u",
    0x71: "i32.and",
    0x72: "i32.or",
    0x73: "i32.xor",
    0x74: "i32.shl",
    0x75: "i32.shr_s",
    0x76: "i32.shr_u",
    0x77: "i32.rotl",
    0x78: "i32.rotr",
}

_SUPPORTED_DECODED_INSTRS = frozenset(
    {
        "PUSH",
        "POP",
        "SELECT",
        "NOP",
        "RETURN",
        "LOCAL.GET",
        "LOCAL.SET",
        "LOCAL.TEE",
        "CALL",
        "BR",
        "BR_IF",
        "BR_TABLE",
        "I32.LOAD",
        "I32.LOAD8_S",
        "I32.LOAD8_U",
        "I32.LOAD16_S",
        "I32.LOAD16_U",
        "I32.STORE",
        "I32.STORE8",
        "I32.STORE16",
        "BLOCK",
        "LOOP",
        "IF",
        "ELSE",
        "END",
        "ADD",
        "SUB",
        "MUL",
        "DIV_S",
        "DIV_U",
        "REM_S",
        "REM_U",
        "EQZ",
        "EQ",
        "NE",
        "LT_S",
        "LT_U",
        "GT_S",
        "GT_U",
        "LE_S",
        "LE_U",
        "GE_S",
        "GE_U",
        "AND",
        "OR",
        "XOR",
        "SHL",
        "SHR_S",
        "SHR_U",
        "ROTL",
        "ROTR",
        "CLZ",
        "CTZ",
        "POPCNT",
    }
)

_WAT_TO_WASM_INSTR: dict[str, str] = {
    "i32.add": "ADD",
    "i32.sub": "SUB",
    "i32.mul": "MUL",
    "i32.div_s": "DIV_S",
    "i32.div_u": "DIV_U",
    "i32.rem_s": "REM_S",
    "i32.rem_u": "REM_U",
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
}

_STRUCTURED_CF_NAMES = frozenset(
    {"BLOCK", "LOOP", "IF", "ELSE", "END", "BR", "BR_IF", "BR_TABLE"}
)

_BOILERPLATE_EXPORTS = frozenset(
    {
        "__wasm_call_ctors",
        "memory",
        "__dso_handle",
        "__data_end",
        "__stack_low",
        "__stack_high",
        "__global_base",
        "__heap_base",
        "__heap_end",
        "__memory_base",
        "__table_base",
    }
)

_ItemT = TypeVar("_ItemT")


class WasmBinaryDecodeError(ValueError):
    """Deterministic error for malformed or unsupported WASM binaries."""


@dataclass(frozen=True)
class WasmFunctionType:
    """Decoded function signature."""

    params: list[str]
    results: list[str]


@dataclass(frozen=True)
class WasmMemory:
    """Decoded linear memory metadata."""

    min_pages: int
    max_pages: int | None = None


@dataclass(frozen=True)
class WasmExport:
    """Decoded export entry."""

    name: str
    kind: str
    index: int


@dataclass(frozen=True)
class WasmFunction:
    """Decoded function body plus metadata."""

    index: int
    type_index: int
    params: list[str]
    results: list[str]
    locals: list[str]
    body: list[WasmInstr]
    export_names: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class WasmBinaryModule:
    """Structured representation of the supported subset of a WASM module."""

    types: list[WasmFunctionType]
    functions: list[WasmFunction]
    memories: list[WasmMemory]
    exports: list[WasmExport]

    def get_exported_function(self, name: str) -> WasmFunction:
        """Return a function by export name."""
        for export in self.exports:
            if export.kind == "func" and export.name == name:
                return self.functions[export.index]
        msg = f"No exported function named {name!r}"
        raise KeyError(msg)


class _Reader:
    """Little-endian binary reader with offset-aware errors."""

    def __init__(self, data: bytes, *, context: str = "module") -> None:
        self.data = data
        self.pos = 0
        self.context = context

    def _fail(self, message: str) -> WasmBinaryDecodeError:
        return WasmBinaryDecodeError(
            f"{self.context} decode error at byte {self.pos}: {message}",
        )

    def fail(self, message: str) -> WasmBinaryDecodeError:
        """Return a positioned decode error for external helpers."""
        return self._fail(message)

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def read_byte(self) -> int:
        if self.pos >= len(self.data):
            raise self._fail("unexpected end of input")
        value = self.data[self.pos]
        self.pos += 1
        return value

    def read_exact(self, size: int) -> bytes:
        if size < 0:
            raise self._fail(f"negative read size {size}")
        end = self.pos + size
        if end > len(self.data):
            raise self._fail(f"expected {size} bytes, found only {self.remaining()}")
        chunk = self.data[self.pos : end]
        self.pos = end
        return chunk

    def read_u32(self) -> int:
        result = 0
        shift = 0
        for _ in range(5):
            byte = self.read_byte()
            result |= (byte & 0x7F) << shift
            if byte & 0x80 == 0:
                if result > 0xFFFFFFFF:
                    raise self._fail(f"uleb128 value {result} exceeds u32")
                return result
            shift += 7
        raise self._fail("invalid u32 leb128: too many bytes")

    def read_i32(self) -> int:
        result = 0
        shift = 0
        byte = 0
        for _ in range(5):
            byte = self.read_byte()
            result |= (byte & 0x7F) << shift
            shift += 7
            if byte & 0x80 == 0:
                break
        else:
            raise self._fail("invalid i32 leb128: too many bytes")

        if shift < 32 and (byte & 0x40):
            result |= -1 << shift
        if result < -(1 << 31) or result > (1 << 31) - 1:
            raise self._fail(f"sleb128 value {result} exceeds i32")
        return result

    def read_name(self) -> str:
        size = self.read_u32()
        raw = self.read_exact(size)
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise self._fail("invalid UTF-8 name") from exc

    def skip_remaining(self) -> None:
        """Consume any unread bytes in the current reader scope."""
        self.pos = len(self.data)


def _expect_fully_consumed(reader: _Reader, *, where: str) -> None:
    if reader.remaining() != 0:
        raise WasmBinaryDecodeError(
            f"{where} has {reader.remaining()} trailing bytes after decode",
        )


def _read_vec(
    reader: _Reader, item_reader: Callable[[_Reader], _ItemT]
) -> list[_ItemT]:
    count = reader.read_u32()
    return [item_reader(reader) for _ in range(count)]


def _read_valtype(reader: _Reader, *, what: str) -> str:
    value = reader.read_byte()
    if value != _VALTYPE_I32:
        type_name = _VALTYPE_NAMES.get(value, f"unknown(0x{value:02x})")
        raise reader.fail(
            f"unsupported {what} value type {type_name} (0x{value:02x}); "
            "only i32 is supported"
        )
    return "i32"


def _read_func_type(reader: _Reader) -> WasmFunctionType:
    form = reader.read_byte()
    if form != _FUNC_TYPE:
        raise reader.fail(
            f"unsupported type form 0x{form:02x}; expected func type 0x60"
        )
    params = _read_vec(
        reader, lambda typed_reader: _read_valtype(typed_reader, what="parameter")
    )
    results = _read_vec(
        reader, lambda typed_reader: _read_valtype(typed_reader, what="result")
    )
    if len(results) > 1:
        raise reader.fail("multi-value function results are not supported")
    return WasmFunctionType(params=params, results=results)


def _read_limits(reader: _Reader) -> WasmMemory:
    flags = reader.read_byte()
    if flags == 0x00:
        return WasmMemory(min_pages=reader.read_u32())
    if flags == 0x01:
        minimum = reader.read_u32()
        maximum = reader.read_u32()
        return WasmMemory(min_pages=minimum, max_pages=maximum)
    raise reader.fail(f"unsupported memory limits flag 0x{flags:02x}")


def _read_export(reader: _Reader) -> WasmExport:
    name = reader.read_name()
    kind_byte = reader.read_byte()
    kind = _KIND_NAMES.get(kind_byte)
    if kind is None:
        raise reader.fail(f"unsupported export kind 0x{kind_byte:02x}")
    index = reader.read_u32()
    if kind not in {"func", "memory"}:
        msg = (
            f"unsupported export kind {kind!r}; only func and memory exports "
            "are supported"
        )
        raise reader.fail(msg)
    return WasmExport(name=name, kind=kind, index=index)


def _read_block_type(reader: _Reader, opname: str) -> None:
    block_type = reader.read_byte()
    if block_type != _EMPTY_BLOCK_TYPE:
        msg = (
            f"unsupported {opname} block type 0x{block_type:02x}; only empty "
            "block type is supported"
        )
        raise reader.fail(
            msg,
        )


def _read_memarg(reader: _Reader, opname: str) -> None:
    align = reader.read_u32()
    offset = reader.read_u32()
    if offset != 0:
        msg = (
            f"unsupported {opname} memarg offset {offset}; WasmInstr cannot "
            "represent non-zero offsets"
        )
        raise reader.fail(
            msg,
        )
    _ = align


def _opcode_family_name(opcode: int) -> str | None:
    if opcode == 0x42 or 0x50 <= opcode <= 0x5A or 0x79 <= opcode <= 0x8A:
        return "i64 instruction family"
    if opcode in {0x43, 0x44} or 0x5B <= opcode <= 0x66 or 0x8B <= opcode <= 0xBF:
        return "floating-point instruction family"
    if opcode == 0x11:
        return "indirect call instruction family"
    if opcode in {0x23, 0x24}:
        return "global instruction family"
    return None


def _raise_unsupported_opcode(reader: _Reader, opcode: int) -> None:
    family = _opcode_family_name(opcode)
    if family is not None:
        msg = (
            f"unsupported {family} opcode 0x{opcode:02x}; only the i32 subset "
            "is supported"
        )
        raise reader.fail(
            msg,
        )
    raise reader.fail(f"unsupported opcode 0x{opcode:02x}")


def _validate_i32_types(types: list[str], *, what: str, context: str) -> None:
    for type_name in types:
        if type_name != "i32":
            msg = (
                f"unsupported {what} type {type_name} in {context}; only i32 is "
                "supported"
            )
            raise WasmBinaryDecodeError(
                msg,
            )


def _validate_supported_function(func: WasmFunction) -> None:
    _validate_i32_types(func.params, what="parameter", context=f"function {func.index}")
    _validate_i32_types(func.results, what="result", context=f"function {func.index}")
    _validate_i32_types(func.locals, what="local", context=f"function {func.index}")
    if len(func.results) > 1:
        raise WasmBinaryDecodeError(
            f"unsupported multi-value function results in function {func.index}"
        )
    for instr in func.body:
        instr_name = instr[0]
        if instr_name not in _SUPPORTED_DECODED_INSTRS:
            msg = (
                f"unsupported decoded instruction {instr_name!r} in function "
                f"{func.index}"
            )
            raise WasmBinaryDecodeError(
                msg,
            )


def _validate_supported_module(module: WasmBinaryModule) -> None:
    if len(module.memories) > 1:
        raise WasmBinaryDecodeError("multiple memories are not supported")
    for type_index, signature in enumerate(module.types):
        _validate_i32_types(
            signature.params, what="parameter", context=f"type {type_index}"
        )
        _validate_i32_types(
            signature.results, what="result", context=f"type {type_index}"
        )
        if len(signature.results) > 1:
            raise WasmBinaryDecodeError(
                f"unsupported multi-value function results in type {type_index}"
            )
    for export in module.exports:
        if export.kind not in {"func", "memory"}:
            msg = (
                f"unsupported export kind {export.kind!r}; only func and memory "
                "exports are supported"
            )
            raise WasmBinaryDecodeError(
                msg,
            )
    for func in module.functions:
        _validate_supported_function(func)


def _decode_expr(reader: _Reader, *, nested: bool) -> tuple[list[WasmInstr], str]:
    instrs: list[WasmInstr] = []
    while True:
        opcode = reader.read_byte()
        if opcode == 0x0B:
            return instrs, "end"
        if opcode == 0x05:
            if not nested:
                raise reader.fail("unexpected else at function body level")
            return instrs, "else"
        if opcode == 0x41:
            instrs.append(("PUSH", reader.read_i32()))
            continue
        if opcode in _CONTROL_OPS:
            instrs.append((_CONTROL_OPS[opcode],))
            continue
        if opcode in _ARG_OPS:
            instrs.append((_ARG_OPS[opcode], reader.read_u32()))
            continue
        if opcode == 0x0E:
            labels = _read_vec(reader, lambda typed_reader: typed_reader.read_u32())
            default = reader.read_u32()
            instrs.append(("BR_TABLE", labels, default))
            continue
        if opcode in _MEMORY_OPS:
            opname = _MEMORY_OPS[opcode]
            _read_memarg(reader, opname)
            instrs.append((opname,))
            continue
        if opcode in _SIMPLE_BINARY_OPS:
            instrs.append((_WAT_TO_WASM_INSTR[_SIMPLE_BINARY_OPS[opcode]],))
            continue
        if opcode in {0x02, 0x03, 0x04}:
            opname = {0x02: "BLOCK", 0x03: "LOOP", 0x04: "IF"}[opcode]
            _read_block_type(reader, opname)
            instrs.append((opname,))
            nested_instrs, terminator = _decode_expr(reader, nested=True)
            instrs.extend(nested_instrs)
            if terminator == "else":
                if opname != "IF":
                    raise reader.fail(
                        f"unexpected else inside {opname.lower()} construct"
                    )
                instrs.append(("ELSE",))
                else_instrs, else_terminator = _decode_expr(reader, nested=True)
                instrs.extend(else_instrs)
                if else_terminator != "end":
                    raise reader.fail("if-else construct did not terminate with end")
            instrs.append(("END",))
            continue
        _raise_unsupported_opcode(reader, opcode)


def _read_code_entry(
    reader: _Reader, func_index: int
) -> tuple[list[str], list[WasmInstr]]:
    body_size = reader.read_u32()
    body_reader = _Reader(
        reader.read_exact(body_size),
        context=f"function {func_index} body",
    )
    local_groups = body_reader.read_u32()
    locals_flat: list[str] = []
    for _ in range(local_groups):
        count = body_reader.read_u32()
        local_type = _read_valtype(body_reader, what="local")
        locals_flat.extend([local_type] * count)
    body, terminator = _decode_expr(body_reader, nested=False)
    if terminator != "end":
        raise body_reader.fail("function body did not terminate with end")
    _expect_fully_consumed(body_reader, where=f"function {func_index} body")
    return locals_flat, body


def _read_custom_section(section_reader: _Reader) -> None:
    """Accept and skip custom section payload after its required name."""
    _ = section_reader.read_name()
    section_reader.skip_remaining()


def _read_code_section(
    section_reader: _Reader,
) -> list[tuple[list[str], list[WasmInstr]]]:
    """Read all function bodies from the code section."""
    code_count = section_reader.read_u32()
    return [
        _read_code_entry(section_reader, func_index) for func_index in range(code_count)
    ]


def parse_wasm_binary(data: bytes | bytearray | memoryview) -> WasmBinaryModule:
    """Decode a binary ``.wasm`` module from bytes-like input."""
    raw = bytes(data)
    reader = _Reader(raw)
    if reader.read_exact(4) != _WASM_MAGIC:
        raise reader.fail("invalid WASM magic header")

    version = int.from_bytes(reader.read_exact(4), byteorder="little", signed=False)
    if version != _WASM_VERSION:
        raise reader.fail(f"unsupported WASM version {version}; expected 1")

    types: list[WasmFunctionType] = []
    function_type_indices: list[int] = []
    memories: list[WasmMemory] = []
    exports: list[WasmExport] = []
    code_entries: list[tuple[list[str], list[WasmInstr]]] = []

    seen_sections: set[int] = set()
    last_section_id = 0

    while reader.remaining() > 0:
        section_id = reader.read_byte()
        payload_size = reader.read_u32()
        section_reader = _Reader(
            reader.read_exact(payload_size),
            context=f"section {section_id}",
        )

        if section_id != 0:
            if section_id in seen_sections:
                raise reader.fail(f"duplicate section id {section_id}")
            if section_id < last_section_id:
                msg = (
                    f"section id {section_id} is out of order after section id "
                    f"{last_section_id}"
                )
                raise reader.fail(
                    msg,
                )
            seen_sections.add(section_id)
            last_section_id = section_id

        if section_id == 0:
            _read_custom_section(section_reader)
        elif section_id == _SECTION_TYPE:
            types = _read_vec(section_reader, _read_func_type)
        elif section_id == _SECTION_IMPORT:
            raise section_reader.fail("import section is not supported")
        elif section_id == _SECTION_FUNCTION:
            function_type_indices = _read_vec(
                section_reader,
                lambda typed_reader: typed_reader.read_u32(),
            )
        elif section_id == _SECTION_TABLE:
            raise section_reader.fail("table section is not supported")
        elif section_id == _SECTION_MEMORY:
            memories = _read_vec(section_reader, _read_limits)
            if len(memories) > 1:
                raise section_reader.fail("multiple memories are not supported")
        elif section_id == _SECTION_GLOBAL:
            raise section_reader.fail("global section is not supported")
        elif section_id == _SECTION_EXPORT:
            exports = _read_vec(section_reader, _read_export)
        elif section_id == _SECTION_START:
            raise section_reader.fail("start section is not supported")
        elif section_id == _SECTION_ELEMENT:
            raise section_reader.fail("element section is not supported")
        elif section_id == _SECTION_CODE:
            code_entries = _read_code_section(section_reader)
        elif section_id == _SECTION_DATA:
            raise section_reader.fail("data section is not supported")
        elif section_id == _SECTION_DATA_COUNT:
            raise section_reader.fail("data_count section is not supported")
        else:
            raise section_reader.fail(f"unknown section id {section_id}")

        _expect_fully_consumed(section_reader, where=f"section {section_id}")

    if function_type_indices and not types:
        raise WasmBinaryDecodeError("function section present without a type section")
    if len(function_type_indices) != len(code_entries):
        raise WasmBinaryDecodeError(
            "function and code section function counts do not match",
        )

    function_export_names: dict[int, list[str]] = {}
    for export in exports:
        if export.kind == "func":
            function_export_names.setdefault(export.index, []).append(export.name)

    functions: list[WasmFunction] = []
    for func_index, (type_index, code_entry) in enumerate(
        zip(function_type_indices, code_entries, strict=True),
    ):
        if type_index >= len(types):
            raise WasmBinaryDecodeError(
                f"function {func_index} references missing type index {type_index}",
            )
        locals_flat, body = code_entry
        signature = types[type_index]
        functions.append(
            WasmFunction(
                index=func_index,
                type_index=type_index,
                params=list(signature.params),
                results=list(signature.results),
                locals=locals_flat,
                body=body,
                export_names=function_export_names.get(func_index, []),
            ),
        )

    for export in exports:
        if export.kind == "func" and export.index >= len(functions):
            msg = (
                f"export {export.name!r} references missing function index "
                f"{export.index}"
            )
            raise WasmBinaryDecodeError(
                msg,
            )
        if export.kind == "memory" and export.index >= len(memories):
            msg = (
                f"export {export.name!r} references missing memory index {export.index}"
            )
            raise WasmBinaryDecodeError(
                msg,
            )

    return WasmBinaryModule(
        types=types,
        functions=functions,
        memories=memories,
        exports=exports,
    )


def parse_wasm_file(path: str | Path) -> WasmBinaryModule:
    """Decode a binary ``.wasm`` module from a filesystem path."""
    wasm_path = Path(path)
    return parse_wasm_binary(wasm_path.read_bytes())


# ─── Binary-to-ISA adapter ─────────────────────────────────────────


def _has_structured_cf(body: list[WasmInstr]) -> bool:
    """Check whether *body* contains structured control-flow markers."""
    return any(instr[0] in _STRUCTURED_CF_NAMES for instr in body)


def _offset_jumps(
    prefix: list[Instruction],
    body: list[Instruction],
) -> list[Instruction]:
    """Prepend *prefix* to *body*, shifting every JZ/JNZ target by ``len(prefix)``."""
    offset = len(prefix)
    adjusted: list[Instruction] = []
    for instr in body:
        if instr.op in (OP_JZ, OP_JNZ):
            adjusted.append(Instruction(instr.op, instr.arg + offset))
        else:
            adjusted.append(instr)
    return prefix + adjusted


def _offset_calls(
    body: list[Instruction],
    call_targets: dict[int, int],
    *,
    offset: int,
) -> list[Instruction]:
    """Shift direct CALL targets by *offset* using the flat address map."""
    adjusted: list[Instruction] = []
    for instr in body:
        if instr.op == OP_CALL:
            adjusted.append(Instruction(instr.op, call_targets[instr.arg] + offset))
        else:
            adjusted.append(instr)
    return adjusted


def _auto_detect_function(module: WasmBinaryModule) -> WasmFunction:
    """Return the first non-boilerplate exported function, or the first function."""
    for export in module.exports:
        if export.kind == "func" and export.name not in _BOILERPLATE_EXPORTS:
            return module.functions[export.index]
    if module.functions:
        return module.functions[0]
    msg = "No user-defined functions found in WASM module"
    raise WasmBinaryDecodeError(msg)


def _lower_wasm_body(func: WasmFunction) -> list[Instruction]:
    """Lower a decoded WASM body without adding setup or termination."""
    try:
        if _has_structured_cf(func.body):
            return compile_structured(func.body)
        return _make_prog(*func.body) if func.body else []
    except KeyError as exc:
        bad_name = exc.args[0] if exc.args else "<unknown>"
        msg = f"unsupported decoded instruction {bad_name!r} in function {func.index}"
        raise WasmBinaryDecodeError(msg) from exc


def _with_param_local_setup(
    func: WasmFunction, body: list[Instruction]
) -> list[Instruction]:
    """Apply the standard parameter/local setup prefix to a lowered body."""
    n_params = len(func.params)
    n_extra_locals = len(func.locals)

    if n_params == 0 and n_extra_locals == 0:
        return body

    setup_instrs: list[WasmInstr] = []
    for i in range(n_extra_locals):
        setup_instrs.extend([("PUSH", 0), ("LOCAL.SET", n_params + i)])
    setup_instrs.extend(("LOCAL.SET", i) for i in range(n_params - 1, -1, -1))

    setup = _make_prog(*setup_instrs)
    return _offset_jumps(setup, body)


def _ensure_terminal(body: list[Instruction], *, terminal_op: int) -> list[Instruction]:
    """Append the requested terminal instruction when the body does not end."""
    if body and body[-1].op == terminal_op:
        return body
    return [*body, Instruction(terminal_op, 0)]


def _compile_wasm_function_body(
    func: WasmFunction,
    *,
    terminal_op: int,
) -> list[Instruction]:
    """Compile one decoded function with setup and the requested terminator."""
    lowered = _lower_wasm_body(func)
    with_setup = _with_param_local_setup(func, lowered)
    return _ensure_terminal(with_setup, terminal_op=terminal_op)


def _compile_module_functions(
    module: WasmBinaryModule,
    *,
    entry_index: int,
) -> list[Instruction]:
    """Link a decoded module into one runnable flat ISA program."""
    order = [
        entry_index,
        *[func.index for func in module.functions if func.index != entry_index],
    ]
    compiled: dict[int, list[Instruction]] = {
        func_index: _compile_wasm_function_body(
            module.functions[func_index],
            terminal_op=OP_RETURN,
        )
        for func_index in order
    }

    wrapper = [Instruction(OP_CALL, 0), Instruction(OP_HALT, 0)]
    call_targets: dict[int, int] = {}
    next_addr = len(wrapper)
    for func_index in order:
        call_targets[func_index] = next_addr
        next_addr += len(compiled[func_index])

    linked: list[Instruction] = [
        Instruction(OP_CALL, call_targets[entry_index]),
        *wrapper[1:],
    ]
    for func_index in order:
        linked.extend(_offset_calls(compiled[func_index], call_targets, offset=0))
    linked.append(Instruction(OP_HALT, 0))
    return linked


def compile_wasm_function(func: WasmFunction) -> list[Instruction]:
    """
    Compile a decoded WASM function to flat instructions.

    The returned list is consumable by both NumPyExecutor and TorchExecutor.
    Parameter locals and extra locals are set up following the same convention
    as the existing C pipeline:

    1. Extra locals are initialised to zero.
    2. Arguments that were pushed on the operand stack are popped into
       parameter locals in reverse order (top-of-stack = last param).

    Structured control flow (BLOCK/LOOP/IF/ELSE/END/BR/BR_IF/BR_TABLE)
    is lowered through ``compile_structured()``.  A trailing HALT is
    appended when absent.
    """
    return _compile_wasm_function_body(func, terminal_op=OP_HALT)


def compile_wasm_module(
    data: bytes | bytearray | memoryview,
    *,
    func_name: str | None = None,
) -> list[Instruction]:
    """
    Decode and compile a binary WASM module to flat instructions.

    Parses the binary via ``parse_wasm_binary()``, selects the target
    function (by export name or auto-detection), and compiles it through
    ``compile_wasm_function()``.

    Args:
        data: Raw ``.wasm`` bytes.
        func_name: Export name of the function to compile.  When *None* the
            first non-boilerplate exported function is selected automatically.

    Returns:
        ``list[Instruction]`` ready for execution.

    """
    module = parse_wasm_binary(data)
    _validate_supported_module(module)

    if func_name is not None:
        func = module.get_exported_function(func_name)
    else:
        func = _auto_detect_function(module)

    if len(module.functions) == 1:
        return compile_wasm_function(func)

    return _compile_module_functions(module, entry_index=func.index)


def compile_wasm(
    data: bytes | bytearray | memoryview,
    *,
    func_name: str | None = None,
) -> list[Instruction]:
    """Backward-compatible alias for ``compile_wasm_module()``."""
    return compile_wasm_module(data, func_name=func_name)


__all__ = [
    "WasmBinaryDecodeError",
    "WasmBinaryModule",
    "WasmExport",
    "WasmFunction",
    "WasmFunctionType",
    "WasmMemory",
    "compile_wasm",
    "compile_wasm_function",
    "compile_wasm_module",
    "parse_wasm_binary",
    "parse_wasm_file",
]
