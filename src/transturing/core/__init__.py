"""Core package: ISA definition, types, backend abstraction, and public API."""

from .abc import ExecutorBackend
from .isa import (
    D_MODEL,
    MASK32,
    N_OPCODES,
    TOKENS_PER_STEP,
    Instruction,
    Trace,
    TraceStep,
    compare_traces,
    program,
    test_algorithm,
    test_trap_algorithm,
)
from .registry import get_executor, list_backends, register_backend

__all__ = [
    "D_MODEL",
    "MASK32",
    "N_OPCODES",
    "TOKENS_PER_STEP",
    "ExecutorBackend",
    "Instruction",
    "Trace",
    "TraceStep",
    "compare_traces",
    "get_executor",
    "list_backends",
    "program",
    "register_backend",
    "test_algorithm",
    "test_trap_algorithm",
]
