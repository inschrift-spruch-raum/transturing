"""
LLM-as-Computer: compiled transformer executor.

Three-layer architecture:
  - core: Zero-dependency ISA, types, programs, assemblers
  - backends.numpy: NumPy-based demo executor
  - backends.torch: PyTorch-based production executor
"""

from .core.isa import (
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
from .core.registry import get_executor, list_backends
from .core.wasm_binary import (
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
    "Instruction",
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
    "test_algorithm",
    "test_trap_algorithm",
]
