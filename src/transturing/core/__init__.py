"""Core package: ISA definition, types, backend abstraction, and public API."""

from .abc import ExecutorBackend
from .isa import (
    D_MODEL,
    MASK32,
    N_OPCODES,
    TOKENS_PER_STEP,
    Instruction,
    TestConfig,
    Trace,
    TraceStep,
    compare_traces,
    program,
    test_algorithm,
    test_trap_algorithm,
)
from .registry import get_executor, list_backends, register_backend
from .wasm_binary import (
    compile_wasm,
    compile_wasm_function,
    compile_wasm_module,
    parse_wasm_binary,
    parse_wasm_file,
)

__all__ = [
    "D_MODEL",
    "MASK32",
    "N_OPCODES",
    "TOKENS_PER_STEP",
    "ExecutorBackend",
    "Instruction",
    "TestConfig",
    "Trace",
    "TraceStep",
    "compare_traces",
    "compile_wasm",
    "compile_wasm_function",
    "compile_wasm_module",
    "get_executor",
    "list_backends",
    "parse_wasm_binary",
    "parse_wasm_file",
    "program",
    "register_backend",
    "test_algorithm",
    "test_trap_algorithm",
]
