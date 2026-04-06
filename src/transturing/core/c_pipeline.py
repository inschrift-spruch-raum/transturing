# ruff: noqa: PLC0415
"""C → WASM → ISA compilation pipeline via the supported binary path."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from .isa import OP_CALL, OP_JNZ, OP_JZ, Instruction
from .isa import program as _make_prog
from .wasm_binary import compile_wasm


def _check_toolchain() -> dict[str, str | None]:
    """Check availability of external tools used by the C pipeline."""
    return {"clang": shutil.which("clang")}


def compile_c_to_wasm(
    source: str,
    *,
    opt_level: str = "-O1",
    extra_clang_args: list[str] | None = None,
) -> bytes:
    """
    Compile C source to raw WASM bytes.

    Args:
        source: C source code as a string.
        opt_level: Optimization level for clang (default '-O1').
        extra_clang_args: Additional arguments to pass to clang.

    Returns:
        Raw ``.wasm`` module bytes.

    Raises:
        OSError: If clang is not installed.
        RuntimeError: If clang invocation fails.

    """
    tools = _check_toolchain()
    clang_path = tools["clang"]
    if not clang_path:
        msg = (
            "clang not found. Install with: apt install clang lld\n"
            "clang must support --target=wasm32-unknown-unknown"
        )
        raise OSError(msg)

    with tempfile.TemporaryDirectory() as tmpdir:
        c_path = Path(tmpdir) / "input.c"
        wasm_path = Path(tmpdir) / "output.wasm"

        c_path.write_text(source)

        clang_cmd: list[str | Path] = [
            clang_path,
            "--target=wasm32-unknown-unknown",
            "-nostdlib",
            "-Wl,--no-entry",
            "-Wl,--export-all",
            opt_level,
            "-o",
            wasm_path,
            c_path,
        ]
        if extra_clang_args:
            insert_at = clang_cmd.index("-o")
            for arg in reversed(extra_clang_args):
                clang_cmd.insert(insert_at, arg)

        try:
            subprocess.run(  # noqa: S603
                clang_cmd,
                capture_output=True,
                text=True,
                timeout=30,
                shell=False,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            msg = f"clang compilation failed:\n{exc.stderr}"
            raise RuntimeError(msg) from exc

        return wasm_path.read_bytes()


def compile_c(
    source: str,
    *,
    func_name: str | None = None,
    opt_level: str = "-O1",
    extra_clang_args: list[str] | None = None,
) -> list[Instruction]:
    """
    Compile C source to ``list[Instruction]`` for execution.

    This is the supported C pipeline entrypoint. It compiles C to a binary
    ``.wasm`` module with clang, then hands off to ``compile_wasm()``.

    Args:
        source: C source code as a string.
        func_name: Export name of the function to compile. If ``None``, the
            first non-boilerplate exported function is selected automatically.
        opt_level: Optimization level for clang (default '-O1').
        extra_clang_args: Additional arguments to pass to clang.

    Returns:
        List[Instruction] ready for NumPyExecutor / TorchExecutor.

    Raises:
        OSError: If clang is not available.
        RuntimeError: If clang compilation fails.
        ValueError: For unsupported WASM features or missing functions surfaced
            by the binary compilation layer.

    """
    wasm_bytes = compile_c_to_wasm(
        source,
        opt_level=opt_level,
        extra_clang_args=extra_clang_args,
    )
    return compile_wasm(wasm_bytes, func_name=func_name)


def _offset_control_flow(
    prefix: list[Instruction],
    body: list[Instruction],
) -> list[Instruction]:
    """Prepend *prefix* to *body*, shifting absolute control-flow targets."""
    offset = len(prefix)
    adjusted: list[Instruction] = []
    for instr in body:
        if instr.op in (OP_JZ, OP_JNZ, OP_CALL):
            adjusted.append(Instruction(instr.op, instr.arg + offset))
        else:
            adjusted.append(instr)
    return prefix + adjusted


def compile_and_run(
    source: str,
    args: list[int],
    *,
    func_name: str | None = None,
    opt_level: str = "-O1",
    max_steps: int = 50000,
) -> int:
    """
    Compile C source and execute with given arguments.

    Convenience function that compiles, prepends PUSH instructions for
    arguments, executes, and returns the top-of-stack result.
    """
    # Imported lazily to keep the main module free of backend dependencies.
    from transturing.backends.numpy_backend import (  # pyright: ignore[reportMissingTypeStubs]
        NumPyExecutor,  # pyright: ignore[reportMissingTypeStubs]
    )

    prog = compile_c(source, func_name=func_name, opt_level=opt_level)

    arg_instrs = _make_prog(*[("PUSH", a) for a in args])
    full_prog = _offset_control_flow(arg_instrs, prog)

    trace = NumPyExecutor().execute(full_prog, max_steps=max_steps)
    return trace.steps[-1].top

