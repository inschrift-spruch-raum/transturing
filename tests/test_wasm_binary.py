# ruff: noqa: D103, PLR2004
"""Tests for the minimal binary WASM frontend."""

from __future__ import annotations

import importlib
import sys
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest

from transturing.backends.numpy_backend import (  # pyright: ignore[reportMissingTypeStubs]
    NumPyExecutor,
)
from transturing.backends.torch_backend import (  # pyright: ignore[reportMissingTypeStubs]
    TorchExecutor,
)
from transturing.core.isa import (  # pyright: ignore[reportMissingTypeStubs]
    Instruction,
    Trace,
)

if TYPE_CHECKING:
    from pathlib import Path

_wasm_binary = importlib.import_module("transturing.core.wasm_binary")
WasmBinaryDecodeError = _wasm_binary.WasmBinaryDecodeError
parse_wasm_binary = _wasm_binary.parse_wasm_binary
parse_wasm_file = _wasm_binary.parse_wasm_file


def _uleb(value: int) -> bytes:
    out = bytearray()
    current = value
    while True:
        byte = current & 0x7F
        current >>= 7
        if current:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _sleb32(value: int) -> bytes:
    out = bytearray()
    current = value
    while True:
        byte = current & 0x7F
        current >>= 7
        sign_bit = byte & 0x40
        done = (current == 0 and sign_bit == 0) or (current == -1 and sign_bit != 0)
        if done:
            out.append(byte)
            return bytes(out)
        out.append(byte | 0x80)


def _vec(items: list[bytes]) -> bytes:
    return _uleb(len(items)) + b"".join(items)


def _name(text: str) -> bytes:
    raw = text.encode("utf-8")
    return _uleb(len(raw)) + raw


def _section(section_id: int, payload: bytes) -> bytes:
    return bytes([section_id]) + _uleb(len(payload)) + payload


def _func_type(params: list[int], results: list[int]) -> bytes:
    return (
        bytes([0x60])
        + _vec([bytes([p]) for p in params])
        + _vec([bytes([r]) for r in results])
    )


def _limits(min_pages: int, max_pages: int | None = None) -> bytes:
    if max_pages is None:
        return b"\x00" + _uleb(min_pages)
    return b"\x01" + _uleb(min_pages) + _uleb(max_pages)


def _export(name: str, kind: int, index: int) -> bytes:
    return _name(name) + bytes([kind]) + _uleb(index)


def _code_entry(local_groups: list[tuple[int, int]], instrs: list[bytes]) -> bytes:
    locals_blob = _uleb(len(local_groups)) + b"".join(
        _uleb(count) + bytes([value_type]) for count, value_type in local_groups
    )
    body = locals_blob + b"".join(instrs) + b"\x0b"
    return _uleb(len(body)) + body


def _module(*sections: bytes) -> bytes:
    return b"\x00asm" + (1).to_bytes(4, byteorder="little") + b"".join(sections)


def _sample_module() -> bytes:
    type_section = _section(
        1,
        _vec(
            [
                _func_type([], []),
                _func_type([0x7F], []),
            ]
        ),
    )
    function_section = _section(3, _vec([_uleb(0), _uleb(1)]))
    memory_section = _section(5, _vec([_limits(1, 2)]))
    export_section = _section(
        7,
        _vec(
            [
                _export("helper", 0x00, 0),
                _export("main", 0x00, 1),
                _export("memory", 0x02, 0),
            ]
        ),
    )
    helper_body = _code_entry(
        [(2, 0x7F)],
        [
            b"\x41" + _sleb32(9),
            b"\x21" + _uleb(0),
            b"\x20" + _uleb(0),
            b"\x22" + _uleb(1),
            b"\x1a",
            b"\x41" + _sleb32(0),
            b"\x41" + _sleb32(42),
            b"\x36" + _uleb(2) + _uleb(0),
            b"\x41" + _sleb32(0),
            b"\x28" + _uleb(2) + _uleb(0),
            b"\x1a",
            b"\x0f",
        ],
    )
    main_body = _code_entry(
        [],
        [
            b"\x02\x40",
            b"\x03\x40",
            b"\x41" + _sleb32(1),
            b"\x0d" + _uleb(1),
            b"\x41" + _sleb32(0),
            b"\x0e" + _vec([_uleb(1), _uleb(0)]) + _uleb(1),
            b"\x0b",
            b"\x0b",
            b"\x41" + _sleb32(3),
            b"\x41" + _sleb32(5),
            b"\x6a",
            b"\x1a",
            b"\x20" + _uleb(0),
            b"\x04\x40",
            b"\x41" + _sleb32(1),
            b"\x05",
            b"\x41" + _sleb32(2),
            b"\x0b",
            b"\x1a",
            b"\x10" + _uleb(0),
        ],
    )
    code_section = _section(10, _vec([helper_body, main_body]))
    return _module(
        type_section, function_section, memory_section, export_section, code_section
    )


def _binary_arithmetic_module() -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [],
                    [
                        b"\x41" + _sleb32(7),
                        b"\x41" + _sleb32(8),
                        b"\x6a",
                    ],
                ),
            ]
        ),
    )
    return _module(type_section, function_section, code_section)


def _binary_locals_module() -> bytes:
    type_section = _section(1, _vec([_func_type([0x7F], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    export_section = _section(7, _vec([_export("main", 0x00, 0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [(1, 0x7F)],
                    [
                        b"\x20" + _uleb(0),
                        b"\x41" + _sleb32(5),
                        b"\x6a",
                        b"\x21" + _uleb(1),
                        b"\x20" + _uleb(1),
                    ],
                ),
            ]
        ),
    )
    return _module(type_section, function_section, export_section, code_section)


def _binary_control_flow_module() -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [],
                    [
                        b"\x41" + _sleb32(1),
                        b"\x04\x40",
                        b"\x41" + _sleb32(11),
                        b"\x05",
                        b"\x41" + _sleb32(22),
                        b"\x0b",
                    ],
                ),
            ]
        ),
    )
    return _module(type_section, function_section, code_section)


def _binary_memory_module() -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    memory_section = _section(5, _vec([_limits(1)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [],
                    [
                        b"\x41" + _sleb32(0),
                        b"\x41" + _sleb32(99),
                        b"\x36" + _uleb(2) + _uleb(0),
                        b"\x41" + _sleb32(0),
                        b"\x28" + _uleb(2) + _uleb(0),
                    ],
                ),
            ]
        ),
    )
    return _module(type_section, function_section, memory_section, code_section)


def _binary_loop_sum_module() -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    export_section = _section(7, _vec([_export("main", 0x00, 0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [(2, 0x7F)],
                    [
                        b"\x41" + _sleb32(0),
                        b"\x21" + _uleb(0),
                        b"\x41" + _sleb32(10),
                        b"\x21" + _uleb(1),
                        b"\x03\x40",
                        b"\x20" + _uleb(0),
                        b"\x20" + _uleb(1),
                        b"\x6a",
                        b"\x21" + _uleb(0),
                        b"\x20" + _uleb(1),
                        b"\x41" + _sleb32(1),
                        b"\x6b",
                        b"\x22" + _uleb(1),
                        b"\x0d" + _uleb(0),
                        b"\x0b",
                        b"\x20" + _uleb(0),
                    ],
                ),
            ]
        ),
    )
    return _module(type_section, function_section, export_section, code_section)


def _binary_nested_block_branch_module() -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [],
                    [
                        b"\x02\x40",
                        b"\x02\x40",
                        b"\x41" + _sleb32(1),
                        b"\x0c" + _uleb(1),
                        b"\x41" + _sleb32(99),
                        b"\x0b",
                        b"\x41" + _sleb32(88),
                        b"\x0b",
                    ],
                ),
            ]
        ),
    )
    return _module(type_section, function_section, code_section)


def _binary_memory_width_module() -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    memory_section = _section(5, _vec([_limits(1)]))
    export_section = _section(7, _vec([_export("main", 0x00, 0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [],
                    [
                        b"\x41" + _sleb32(0),
                        b"\x41" + _sleb32(255),
                        b"\x3a" + _uleb(0) + _uleb(0),
                        b"\x41" + _sleb32(0),
                        b"\x2d" + _uleb(0) + _uleb(0),
                        b"\x1a",
                        b"\x41" + _sleb32(0),
                        b"\x2c" + _uleb(0) + _uleb(0),
                        b"\x1a",
                        b"\x41" + _sleb32(4),
                        b"\x41" + _sleb32(-2),
                        b"\x3b" + _uleb(1) + _uleb(0),
                        b"\x41" + _sleb32(4),
                        b"\x2f" + _uleb(1) + _uleb(0),
                        b"\x1a",
                        b"\x41" + _sleb32(4),
                        b"\x2e" + _uleb(1) + _uleb(0),
                    ],
                ),
            ]
        ),
    )
    return _module(
        type_section, function_section, memory_section, export_section, code_section
    )


def _binary_call_module() -> bytes:
    type_section = _section(
        1,
        _vec(
            [
                _func_type([], []),
                _func_type([], []),
            ]
        ),
    )
    function_section = _section(3, _vec([_uleb(0), _uleb(1)]))
    export_section = _section(7, _vec([_export("main", 0x00, 1)]))
    helper_body = _code_entry([], [b"\x41" + _sleb32(9)])
    main_body = _code_entry([], [b"\x10" + _uleb(0)])
    code_section = _section(10, _vec([helper_body, main_body]))
    return _module(type_section, function_section, export_section, code_section)


def _binary_multi_function_module() -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0), _uleb(0), _uleb(0)]))
    export_section = _section(7, _vec([_export("main", 0x00, 2)]))
    increment_body = _code_entry([], [b"\x41" + _sleb32(7)])
    twice_body = _code_entry([], [b"\x10" + _uleb(0), b"\x10" + _uleb(0)])
    main_body = _code_entry([], [b"\x10" + _uleb(1)])
    code_section = _section(10, _vec([increment_body, twice_body, main_body]))
    return _module(type_section, function_section, export_section, code_section)


def _binary_br_table_runtime_module(selector: int) -> bytes:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    export_section = _section(7, _vec([_export("main", 0x00, 0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [],
                    [
                        b"\x02\x40",
                        b"\x02\x40",
                        b"\x02\x40",
                        b"\x41" + _sleb32(selector),
                        b"\x0e" + _vec([_uleb(0), _uleb(1)]) + _uleb(2),
                        b"\x41" + _sleb32(11),
                        b"\x0c" + _uleb(2),
                        b"\x0b",
                        b"\x41" + _sleb32(22),
                        b"\x0c" + _uleb(1),
                        b"\x0b",
                        b"\x41" + _sleb32(33),
                        b"\x0b",
                    ],
                ),
            ]
        ),
    )
    return _module(type_section, function_section, export_section, code_section)


def _binary_param_call_module() -> bytes:
    type_section = _section(
        1,
        _vec(
            [
                _func_type([0x7F], []),
                _func_type([], []),
            ]
        ),
    )
    function_section = _section(3, _vec([_uleb(0), _uleb(0), _uleb(1)]))
    export_section = _section(7, _vec([_export("main", 0x00, 2)]))
    double_body = _code_entry(
        [],
        [
            b"\x20" + _uleb(0),
            b"\x20" + _uleb(0),
            b"\x6a",
        ],
    )
    helper_body = _code_entry(
        [(1, 0x7F)],
        [
            b"\x20" + _uleb(0),
            b"\x10" + _uleb(0),
            b"\x41" + _sleb32(3),
            b"\x6a",
            b"\x22" + _uleb(1),
            b"\x20" + _uleb(1),
            b"\x6a",
        ],
    )
    main_body = _code_entry(
        [],
        [
            b"\x41" + _sleb32(4),
            b"\x10" + _uleb(1),
        ],
    )
    code_section = _section(10, _vec([double_body, helper_body, main_body]))
    return _module(type_section, function_section, export_section, code_section)


def _binary_entry_param_call_module() -> bytes:
    type_section = _section(1, _vec([_func_type([0x7F], [])]))
    function_section = _section(3, _vec([_uleb(0), _uleb(0)]))
    export_section = _section(7, _vec([_export("main", 0x00, 1)]))
    helper_body = _code_entry(
        [],
        [
            b"\x20" + _uleb(0),
            b"\x20" + _uleb(0),
            b"\x6a",
        ],
    )
    main_body = _code_entry(
        [],
        [
            b"\x20" + _uleb(0),
            b"\x10" + _uleb(0),
            b"\x41" + _sleb32(5),
            b"\x6a",
        ],
    )
    code_section = _section(10, _vec([helper_body, main_body]))
    return _module(type_section, function_section, export_section, code_section)


def _run_binary_program(
    wasm_bytes: bytes,
    *,
    func_name: str | None = None,
    args: list[int] | None = None,
) -> tuple[Trace, Trace]:
    flat = compile_wasm_module(wasm_bytes, func_name=func_name)
    prefix = [Instruction(OP_PUSH, arg) for arg in args or []]
    prog = prefix + flat
    np_trace = NumPyExecutor().execute(prog)
    pt_trace = TorchExecutor().execute(prog)
    return np_trace, pt_trace


_RUNNABLE_BINARY_CASES: list[tuple[str, bytes, str | None, list[int], int]] = [
    ("arithmetic", _binary_arithmetic_module(), None, [], 15),
    ("locals", _binary_locals_module(), "main", [7], 12),
    ("control_flow", _binary_control_flow_module(), None, [], 11),
    ("br_table_selector_0", _binary_br_table_runtime_module(0), "main", [], 22),
    ("br_table_selector_1", _binary_br_table_runtime_module(1), "main", [], 33),
    (
        "br_table_selector_default",
        _binary_br_table_runtime_module(5),
        "main",
        [],
        0,
    ),
    ("loop_sum", _binary_loop_sum_module(), "main", [], 55),
    ("nested_branch", _binary_nested_block_branch_module(), None, [], 1),
    ("memory", _binary_memory_module(), None, [], 99),
    ("memory_widths", _binary_memory_width_module(), "main", [], -2),
    ("direct_call", _binary_call_module(), "main", [], 9),
    ("multi_function", _binary_multi_function_module(), "main", [], 7),
    ("param_call", _binary_param_call_module(), "main", [], 22),
]


def test_parse_wasm_binary_decodes_supported_subset() -> None:
    module = parse_wasm_binary(_sample_module())

    assert [memory.min_pages for memory in module.memories] == [1]
    assert module.memories[0].max_pages == 2
    assert [export.name for export in module.exports] == ["helper", "main", "memory"]

    helper = module.get_exported_function("helper")
    assert helper.locals == ["i32", "i32"]
    assert helper.body == [
        ("PUSH", 9),
        ("LOCAL.SET", 0),
        ("LOCAL.GET", 0),
        ("LOCAL.TEE", 1),
        ("POP",),
        ("PUSH", 0),
        ("PUSH", 42),
        ("I32.STORE",),
        ("PUSH", 0),
        ("I32.LOAD",),
        ("POP",),
        ("RETURN",),
    ]

    main = module.get_exported_function("main")
    assert main.params == ["i32"]
    assert main.body == [
        ("BLOCK",),
        ("LOOP",),
        ("PUSH", 1),
        ("BR_IF", 1),
        ("PUSH", 0),
        ("BR_TABLE", [1, 0], 1),
        ("END",),
        ("END",),
        ("PUSH", 3),
        ("PUSH", 5),
        ("ADD",),
        ("POP",),
        ("LOCAL.GET", 0),
        ("IF",),
        ("PUSH", 1),
        ("ELSE",),
        ("PUSH", 2),
        ("END",),
        ("POP",),
        ("CALL", 0),
    ]


def test_parse_wasm_binary_decodes_loop_locals_and_branch_structure() -> None:
    module = parse_wasm_binary(_binary_loop_sum_module())

    main = module.get_exported_function("main")
    assert main.locals == ["i32", "i32"]
    assert main.body == [
        ("PUSH", 0),
        ("LOCAL.SET", 0),
        ("PUSH", 10),
        ("LOCAL.SET", 1),
        ("LOOP",),
        ("LOCAL.GET", 0),
        ("LOCAL.GET", 1),
        ("ADD",),
        ("LOCAL.SET", 0),
        ("LOCAL.GET", 1),
        ("PUSH", 1),
        ("SUB",),
        ("LOCAL.TEE", 1),
        ("BR_IF", 0),
        ("END",),
        ("LOCAL.GET", 0),
    ]


def test_parse_wasm_binary_decodes_memory_width_variants() -> None:
    module = parse_wasm_binary(_binary_memory_width_module())

    main = module.get_exported_function("main")
    assert main.body == [
        ("PUSH", 0),
        ("PUSH", 255),
        ("I32.STORE8",),
        ("PUSH", 0),
        ("I32.LOAD8_U",),
        ("POP",),
        ("PUSH", 0),
        ("I32.LOAD8_S",),
        ("POP",),
        ("PUSH", 4),
        ("PUSH", -2),
        ("I32.STORE16",),
        ("PUSH", 4),
        ("I32.LOAD16_U",),
        ("POP",),
        ("PUSH", 4),
        ("I32.LOAD16_S",),
    ]


def test_parse_wasm_binary_decodes_multi_function_call_graph() -> None:
    module = parse_wasm_binary(_binary_multi_function_module())

    assert [func.export_names for func in module.functions] == [[], [], ["main"]]
    assert [func.params for func in module.functions] == [[], [], []]
    assert [func.results for func in module.functions] == [[], [], []]
    assert module.functions[0].body == [("PUSH", 7)]
    assert module.functions[1].body == [
        ("CALL", 0),
        ("CALL", 0),
    ]
    assert module.functions[2].body == [("CALL", 1)]


def test_parse_wasm_binary_decodes_br_table_runtime_structure() -> None:
    module = parse_wasm_binary(_binary_br_table_runtime_module(1))

    main = module.get_exported_function("main")
    assert main.body == [
        ("BLOCK",),
        ("BLOCK",),
        ("BLOCK",),
        ("PUSH", 1),
        ("BR_TABLE", [0, 1], 2),
        ("PUSH", 11),
        ("BR", 2),
        ("END",),
        ("PUSH", 22),
        ("BR", 1),
        ("END",),
        ("PUSH", 33),
        ("END",),
    ]


def test_parse_wasm_binary_decodes_param_call_graph_with_locals() -> None:
    module = parse_wasm_binary(_binary_param_call_module())

    assert [func.params for func in module.functions] == [["i32"], ["i32"], []]
    assert [func.locals for func in module.functions] == [[], ["i32"], []]
    assert module.functions[0].body == [
        ("LOCAL.GET", 0),
        ("LOCAL.GET", 0),
        ("ADD",),
    ]
    assert module.functions[1].body == [
        ("LOCAL.GET", 0),
        ("CALL", 0),
        ("PUSH", 3),
        ("ADD",),
        ("LOCAL.TEE", 1),
        ("LOCAL.GET", 1),
        ("ADD",),
    ]
    assert module.functions[2].body == [
        ("PUSH", 4),
        ("CALL", 1),
    ]


def test_parse_wasm_file_reads_path_input(tmp_path: Path) -> None:
    wasm_path = tmp_path / "sample.wasm"
    wasm_path.write_bytes(_sample_module())

    module = parse_wasm_file(wasm_path)

    assert module.get_exported_function("helper").export_names == ["helper"]


def test_parse_wasm_binary_rejects_nonzero_memarg_offset() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry([], [b"\x41" + _sleb32(0), b"\x28" + _uleb(2) + _uleb(4)]),
            ]
        ),
    )

    with pytest.raises(WasmBinaryDecodeError, match="non-zero offsets"):
        parse_wasm_binary(_module(type_section, function_section, code_section))


def test_parse_wasm_binary_rejects_unsupported_block_type() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry([], [b"\x04\x7f", b"\x41" + _sleb32(1), b"\x0b"]),
            ]
        ),
    )

    with pytest.raises(
        WasmBinaryDecodeError, match="only empty block type is supported"
    ):
        parse_wasm_binary(_module(type_section, function_section, code_section))


def test_parse_wasm_binary_rejects_unsupported_opcode() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(10, _vec([_code_entry([], [b"\x42\x00"])]))

    with pytest.raises(WasmBinaryDecodeError, match="i64 instruction family"):
        parse_wasm_binary(_module(type_section, function_section, code_section))


def test_parse_wasm_binary_rejects_floating_point_instruction_family() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(10, _vec([_code_entry([], [b"\x43\x00\x00\x00\x00"])]))

    with pytest.raises(
        WasmBinaryDecodeError, match="floating-point instruction family"
    ):
        parse_wasm_binary(_module(type_section, function_section, code_section))


@pytest.mark.parametrize(
    ("section_id", "payload", "match"),
    [
        (2, _vec([]), "import section is not supported"),
        (6, _vec([]), "global section is not supported"),
        (8, b"", "start section is not supported"),
        (11, _vec([]), "data section is not supported"),
    ],
    ids=["import", "global", "start", "data"],
)
def test_parse_wasm_binary_rejects_unsupported_sections(
    section_id: int, payload: bytes, match: str
) -> None:
    with pytest.raises(WasmBinaryDecodeError, match=match):
        parse_wasm_binary(_module(_section(section_id, payload)))


@pytest.mark.parametrize(
    ("section_id", "match"),
    [
        (4, "table section is not supported"),
        (9, "element section is not supported"),
        (12, "data_count section is not supported"),
    ],
    ids=["table", "element", "data_count"],
)
def test_parse_wasm_binary_rejects_more_unsupported_sections(
    section_id: int, match: str
) -> None:
    with pytest.raises(WasmBinaryDecodeError, match=match):
        parse_wasm_binary(_module(_section(section_id, _vec([]))))


def test_parse_wasm_binary_rejects_non_i32_local_type() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    body = _uleb(1) + _uleb(1) + b"\x7e" + b"\x0b"
    code_section = _section(10, _vec([_uleb(len(body)) + body]))

    with pytest.raises(WasmBinaryDecodeError, match="only i32 is supported"):
        parse_wasm_binary(_module(type_section, function_section, code_section))


def test_parse_wasm_binary_rejects_multi_value_results() -> None:
    type_section = _section(1, _vec([_func_type([], [0x7F, 0x7F])]))

    with pytest.raises(
        WasmBinaryDecodeError, match="multi-value function results are not supported"
    ):
        parse_wasm_binary(_module(type_section))


def test_parse_wasm_binary_rejects_bad_header() -> None:
    with pytest.raises(WasmBinaryDecodeError, match="invalid WASM magic header"):
        parse_wasm_binary(b"not wasm")


def test_parse_wasm_binary_rejects_unsupported_version() -> None:
    bad_version = b"\x00asm" + (2).to_bytes(4, byteorder="little")

    with pytest.raises(WasmBinaryDecodeError, match="unsupported WASM version 2"):
        parse_wasm_binary(bad_version)


def test_parse_wasm_binary_rejects_duplicate_sections() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))

    with pytest.raises(WasmBinaryDecodeError, match="duplicate section id 1"):
        parse_wasm_binary(_module(type_section, type_section))


def test_parse_wasm_binary_rejects_out_of_order_sections() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))
    code_section = _section(10, _vec([]))

    with pytest.raises(WasmBinaryDecodeError, match="section id 1 is out of order"):
        parse_wasm_binary(_module(code_section, type_section))


def test_parse_wasm_binary_allows_custom_sections_with_payload() -> None:
    custom_payload = _name("producers") + b"extra-custom-bytes"
    custom_section = _section(0, custom_payload)

    module = parse_wasm_binary(_module(custom_section, _section(1, _vec([]))))

    assert module.types == []
    assert module.functions == []


compile_wasm_function = _wasm_binary.compile_wasm_function
compile_wasm = _wasm_binary.compile_wasm
compile_wasm_module = _wasm_binary.compile_wasm_module
_c_pipeline = importlib.import_module("transturing.core.c_pipeline")
compile_c = _c_pipeline.compile_c
compile_and_run = _c_pipeline.compile_and_run


def test_compile_wasm_function_produces_flat_instructions() -> None:
    module = parse_wasm_binary(_sample_module())

    main = module.get_exported_function("main")
    flat = compile_wasm_function(main)

    # Should produce non-empty list of Instruction
    assert len(flat) > 0
    # Should end with HALT
    assert flat[-1].op == 5  # OP_HALT
    # Should contain JZ/JNZ (lowered from structured CF)
    assert any(i.op in (7, 8) for i in flat)  # OP_JZ=7, OP_JNZ=8

    # Should contain LOCAL.SET for param setup (main has 1 i32 param)
    assert any(i.op == 44 for i in flat)  # OP_LOCAL_SET=44


def test_compile_wasm_module_selects_function_by_name() -> None:
    flat = compile_wasm_module(_sample_module(), func_name="main")
    assert len(flat) > 0
    assert flat[-1].op == 5  # OP_HALT


def test_compile_wasm_alias_matches_compile_wasm_module() -> None:
    assert compile_wasm(_sample_module(), func_name="main") == compile_wasm_module(
        _sample_module(), func_name="main"
    )


def test_compile_wasm_returns_instruction_list_shape() -> None:
    flat: list[Instruction] = compile_wasm(_sample_module(), func_name="main")

    assert isinstance(flat, list)
    assert flat
    assert all(type(instr) is Instruction for instr in flat)
    assert flat[-1].op == OP_HALT


def test_compile_wasm_module_auto_detects_function() -> None:
    flat = compile_wasm_module(_sample_module())
    assert len(flat) > 0
    assert flat[-1].op == 5  # OP_HALT


def test_compile_wasm_module_rejects_i64_before_lowering() -> None:
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(10, _vec([_code_entry([], [b"\x42\x00"])]))
    wasm_bytes = _module(type_section, function_section, code_section)

    with (
        patch.object(_wasm_binary, "compile_wasm_function") as lower_function,
        pytest.raises(WasmBinaryDecodeError, match="i64 instruction family"),
    ):
        compile_wasm_module(wasm_bytes)

    lower_function.assert_not_called()


def test_compile_wasm_module_no_funcs_raises() -> None:
    # Module with type section but no functions/code sections
    type_section = _section(1, _vec([]))
    with pytest.raises(
        _wasm_binary.WasmBinaryDecodeError, match="No user-defined functions"
    ):
        compile_wasm_module(_module(type_section))


def test_compile_c_uses_binary_wasm_handoff() -> None:
    wasm_bytes = _binary_locals_module()

    with (
        patch.object(
            _c_pipeline, "compile_c_to_wasm", return_value=wasm_bytes
        ) as to_wasm,
        patch.object(_c_pipeline, "compile_wasm", wraps=compile_wasm) as binary_compile,
    ):
        flat = compile_c("int ignored(void) { return 0; }", func_name="main")

    to_wasm.assert_called_once()
    binary_compile.assert_called_once_with(wasm_bytes, func_name="main")
    assert flat[-1].op == OP_HALT


def test_compile_c_uses_supported_binary_path_without_extra_tools() -> None:
    wasm_bytes = _binary_locals_module()

    with (
        patch.object(_c_pipeline, "compile_c_to_wasm", return_value=wasm_bytes),
        patch.object(
            _c_pipeline,
            "_check_toolchain",
            return_value={"clang": "clang"},
        ),
    ):
        flat = compile_c("int ignored(void) { return 0; }", func_name="main")

    assert flat
    assert flat[-1].op == OP_HALT


def test_c_pipeline_imports_without_eager_deleted_text_frontend_dependency() -> None:
    removed_frontend_module = "transturing.core." + "wa" + "t_parser"

    with patch.dict(
        sys.modules,
        {removed_frontend_module: None},
    ):
        module = importlib.reload(_c_pipeline)

        try:
            with patch.object(
                module, "compile_c_to_wasm", return_value=_sample_module()
            ):
                flat = module.compile_c(
                    "int ignored(void) { return 0; }", func_name="main"
                )
        finally:
            importlib.reload(module)

    assert flat
    assert flat[-1].op == OP_HALT


def test_compile_and_run_uses_supported_binary_path() -> None:
    wasm_bytes = _binary_locals_module()

    with (
        patch.object(_c_pipeline, "compile_c_to_wasm", return_value=wasm_bytes),
        patch.object(
            _c_pipeline,
            "_check_toolchain",
            return_value={"clang": "clang"},
        ),
    ):
        result = compile_and_run(
            "int ignored(int n) { return n; }", [9], func_name="main"
        )

    assert result == 14


def test_compile_and_run_rebases_prefixed_args_for_absolute_calls() -> None:
    wasm_bytes = _binary_entry_param_call_module()

    with (
        patch.object(_c_pipeline, "compile_c_to_wasm", return_value=wasm_bytes),
        patch.object(
            _c_pipeline,
            "_check_toolchain",
            return_value={"clang": "clang"},
        ),
    ):
        result = compile_and_run(
            "int ignored(int n) { return helper(n); }", [6], func_name="main"
        )

    assert result == 17


def test_compile_wasm_function_no_params_no_locals() -> None:
    module = parse_wasm_binary(_sample_module())

    helper = module.get_exported_function("helper")
    flat = compile_wasm_function(helper)

    assert len(flat) > 0
    assert flat[-1].op == 5  # OP_HALT


# ─── Adapter parity + rejection tests ────────────────────────────────

_assembler = importlib.import_module("transturing.core.assembler")
compile_structured = _assembler.compile_structured

_isa = importlib.import_module("transturing.core.isa")
WasmFunction = _wasm_binary.WasmFunction
OP_JZ = _isa.OP_JZ
OP_JNZ = _isa.OP_JNZ
OP_PUSH = _isa.OP_PUSH
OP_POP = _isa.OP_POP
OP_HALT = _isa.OP_HALT
OP_LOCAL_SET = _isa.OP_LOCAL_SET


def test_binary_structured_ir_lowers_through_compile_structured() -> None:
    """Decoded BLOCK/LOOP/IF body lowers to flat JZ/JNZ via compile_structured."""
    module = parse_wasm_binary(_sample_module())
    main = module.get_exported_function("main")

    # The body already contains BLOCK, LOOP, IF, ELSE, END, BR_IF, BR_TABLE
    structured = main.body
    assert any(t[0] == "BLOCK" for t in structured)
    assert any(t[0] == "LOOP" for t in structured)
    assert any(t[0] == "IF" for t in structured)

    flat = compile_structured(structured)
    assert len(flat) > 0

    # Structured constructs must be lowered to JZ/JNZ (no BLOCK/LOOP/IF in flat)
    assert any(i.op in (OP_JZ, OP_JNZ) for i in flat)

    # Verify round-trip through compile_wasm_function produces equivalent result
    full = compile_wasm_function(main)
    assert full[-1].op == OP_HALT
    # Param setup + lowered body should be present
    assert any(i.op == OP_LOCAL_SET for i in full)


def test_binary_simple_body_no_structured_cf() -> None:
    """Function with no structured CF uses direct program() path."""
    # Build a module with a single-argument-free function that just pushes and adds
    type_section = _section(1, _vec([_func_type([], [])]))
    function_section = _section(3, _vec([_uleb(0)]))
    code_section = _section(
        10,
        _vec(
            [
                _code_entry(
                    [],
                    [
                        b"\x41" + _sleb32(3),
                        b"\x41" + _sleb32(5),
                        b"\x6a",  # i32.add
                    ],
                ),
            ]
        ),
    )

    module = parse_wasm_binary(_module(type_section, function_section, code_section))
    func = module.functions[0]

    assert func.body == [("PUSH", 3), ("PUSH", 5), ("ADD",)]

    flat = compile_wasm_function(func)
    assert len(flat) == 4  # PUSH 3, PUSH 5, ADD, HALT
    assert flat[0] == _isa.Instruction(OP_PUSH, 3)
    assert flat[1] == _isa.Instruction(OP_PUSH, 5)
    assert flat[2].op == _isa.OP_ADD
    assert flat[3].op == OP_HALT


def test_compile_wasm_function_rejects_unknown_instruction() -> None:
    """Adapter must not silently drop unknown instruction names."""
    bad_func = WasmFunction(
        index=0,
        type_index=0,
        params=[],
        results=[],
        locals=[],
        body=[("BOGUS_OP",)],
    )
    with pytest.raises(
        WasmBinaryDecodeError,
        match="unsupported decoded instruction 'BOGUS_OP'",
    ):
        compile_wasm_function(bad_func)


def test_adapter_preserves_br_table_structure() -> None:
    """BR_TABLE tuple (name, labels, default) survives decode → compile_structured."""
    module = parse_wasm_binary(_sample_module())
    main = module.get_exported_function("main")

    # Find the BR_TABLE instruction in the decoded body
    br_table_instrs = [t for t in main.body if t[0] == "BR_TABLE"]
    assert len(br_table_instrs) == 1
    labels, default = br_table_instrs[0][1], br_table_instrs[0][2]
    assert isinstance(labels, list)
    assert isinstance(default, int)

    # compile_structured must accept it without error
    flat = compile_structured(main.body)
    assert len(flat) > 0
    # BR_TABLE lowers to DUP/PUSH/EQ/JNZ chains — verify JNZ presence
    assert any(i.op == OP_JNZ for i in flat)


@pytest.mark.parametrize(
    ("_name", "wasm_bytes", "func_name", "args", "expected"),
    _RUNNABLE_BINARY_CASES,
    ids=[name for name, *_rest in _RUNNABLE_BINARY_CASES],
)
def test_binary_wasm_runnable_parity_across_backends(
    _name: str,
    wasm_bytes: bytes,
    func_name: str | None,
    args: list[int],
    expected: int,
) -> None:
    np_trace, pt_trace = _run_binary_program(wasm_bytes, func_name=func_name, args=args)

    match, detail = _isa.compare_traces(np_trace, pt_trace)
    assert match, f"Trace mismatch for {_name}: {detail}"
    assert np_trace.steps[-1].top == expected
    assert pt_trace.steps[-1].top == expected


def test_binary_wasm_direct_call_lowering_links_to_flat_addresses() -> None:
    flat = compile_wasm_module(_binary_call_module(), func_name="main")

    call_instrs = [instr for instr in flat if instr.op == _isa.OP_CALL]
    assert len(call_instrs) == 2
    assert all(instr.arg >= 2 for instr in call_instrs)

    np_trace = NumPyExecutor().execute(flat)
    pt_trace = TorchExecutor().execute(flat)
    match, detail = _isa.compare_traces(np_trace, pt_trace)

    assert match, detail
    assert np_trace.steps[-1].op == OP_HALT
    assert pt_trace.steps[-1].op == OP_HALT
    assert np_trace.steps[-1].top == 9
    assert pt_trace.steps[-1].top == 9


def test_binary_multi_function_lowering_rewrites_nested_call_targets() -> None:
    flat = compile_wasm_module(_binary_multi_function_module(), func_name="main")

    call_instrs = [instr for instr in flat if instr.op == _isa.OP_CALL]
    assert len(call_instrs) == 4
    assert len({instr.arg for instr in call_instrs}) >= 3
    assert all(instr.arg >= 2 for instr in call_instrs)

    np_trace = NumPyExecutor().execute(flat)
    pt_trace = TorchExecutor().execute(flat)
    match, detail = _isa.compare_traces(np_trace, pt_trace)

    assert match, detail
    assert np_trace.steps[-1].top == 7
    assert pt_trace.steps[-1].top == 7
