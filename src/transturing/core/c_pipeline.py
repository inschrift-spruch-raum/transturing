"""C → WASM → tokens end-to-end pipeline.

Compiles C source code to WASM via clang, converts to WAT via wasm2wat,
then parses through our WAT parser to produce List[Instruction] for execution.

Pipeline:
    C source → clang --target=wasm32 → .wasm binary → wasm2wat → WAT text
    → parse_wat() → List[Instruction] → execute

Required external tools:
    - clang (with wasm32 target support): apt install clang lld
    - wasm2wat (from wabt): apt install wabt
      or download from https://github.com/WebAssembly/wabt/releases

References:
    Issue #51: C → WASM → tokens end-to-end pipeline
    Issue #49: WAT parser (dependency)

"""

import os
import re
import shutil
import subprocess
import tempfile

from .isa import Instruction
from .wat_parser import parse_wat

# ─── Unsupported WASM features ────────────────────────────────────

_UNSUPPORTED_PATTERNS = [
    # 64-bit integer ops
    (r"\bi64\.\w+", "i64 (64-bit integer) operations"),
    # Floating point
    (r"\bf32\.\w+", "f32 (32-bit float) operations"),
    (r"\bf64\.\w+", "f64 (64-bit float) operations"),
    # SIMD
    (r"\bv128\.\w+", "v128 (SIMD) operations"),
    # Reference types
    (r"\b(ref\.null|ref\.is_null|ref\.func)\b", "reference type operations"),
    # Atomic operations
    (r"\b(memory\.atomic|i32\.atomic|i64\.atomic)\.\w+", "atomic operations"),
    # Bulk memory
    (
        r"\b(memory\.copy|memory\.fill|memory\.init|data\.drop)\b",
        "bulk memory operations",
    ),
    # Table operations
    (r"\b(table\.get|table\.set|table\.grow|table\.size)\b", "table operations"),
]


def _check_toolchain() -> dict:
    """Check availability of required external tools.

    Returns:
        dict with 'clang' and 'wasm2wat' keys, values are paths or None.

    """
    return {
        "clang": shutil.which("clang"),
        "wasm2wat": shutil.which("wasm2wat"),
    }


def _has_toolchain() -> bool:
    """Return True if both clang and wasm2wat are available."""
    tools = _check_toolchain()
    return tools["clang"] is not None and tools["wasm2wat"] is not None


def _check_unsupported_features(wat_text: str) -> list[str]:
    """Scan WAT text for unsupported WASM features.

    Returns list of human-readable error messages for each unsupported
    feature found.
    """
    errors = []
    for pattern, desc in _UNSUPPORTED_PATTERNS:
        matches = re.findall(pattern, wat_text)
        if matches:
            # Deduplicate
            unique = sorted(set(matches))
            examples = ", ".join(unique[:3])
            if len(unique) > 3:
                examples += f" (and {len(unique) - 3} more)"
            errors.append(f"Unsupported: {desc} — found: {examples}")
    return errors


def _extract_function_wat(wat_text: str, func_name: str) -> str:
    """Extract a single function's body from full module WAT text.

    Strips the module/export/memory/global boilerplate and returns
    just the (func ...) block for the named function, suitable for
    parse_wat().

    Args:
        wat_text: Full WAT module text from wasm2wat.
        func_name: Function name to extract (with or without $ prefix).

    Returns:
        WAT text containing just the target function.

    Raises:
        ValueError: If the function is not found.

    """
    if not func_name.startswith("$"):
        func_name = "$" + func_name

    # Find the function start
    # Pattern: (func $name (type ...) ...
    pattern = re.compile(
        r"\(func\s+" + re.escape(func_name) + r"\b",
        re.MULTILINE,
    )
    match = pattern.search(wat_text)
    if not match:
        # List available functions for better error message
        available = re.findall(r"\(func\s+(\$\w+)", wat_text)
        raise ValueError(
            f"Function {func_name} not found in WAT module. "
            f"Available functions: {available}",
        )

    # Find the matching closing paren
    start = match.start()
    depth = 0
    i = start
    while i < len(wat_text):
        if wat_text[i] == "(":
            depth += 1
        elif wat_text[i] == ")":
            depth -= 1
            if depth == 0:
                return wat_text[start : i + 1]
        i += 1

    raise ValueError(f"Unmatched parentheses for function {func_name}")


def _count_params(func_wat: str) -> int:
    """Count the number of i32 parameters in a WAT function declaration.

    Handles both:
        (param i32 i32)     — multiple params in one group
        (param i32) (param i32) — one param per group
    """
    count = 0
    for m in re.finditer(r"\(param\s+([^)]+)\)", func_wat):
        types = m.group(1).split()
        count += sum(1 for t in types if t == "i32")
    return count


def _count_locals(func_wat: str) -> int:
    """Count the number of additional local variables in a WAT function.

    These are (local ...) declarations beyond the params.
    """
    count = 0
    for m in re.finditer(r"\(local\s+([^)]+)\)", func_wat):
        types = m.group(1).split()
        count += sum(1 for t in types if t == "i32")
    return count


def compile_c_to_wat(
    source: str,
    *,
    opt_level: str = "-O1",
    extra_clang_args: list[str] | None = None,
) -> str:
    """Compile C source to WAT text.

    Args:
        source: C source code as a string.
        opt_level: Optimization level for clang (default '-O1').
            -O0 produces verbose code with memory-based stack frames.
            -O1 produces clean local-based code but may use i64 for
            loop optimizations. -O2/-O3 aggressively optimize away loops.
        extra_clang_args: Additional arguments to pass to clang.

    Returns:
        Full WAT module text.

    Raises:
        RuntimeError: If clang or wasm2wat invocation fails.
        EnvironmentError: If required tools are not installed.

    """
    tools = _check_toolchain()
    if not tools["clang"]:
        raise OSError(
            "clang not found. Install with: apt install clang lld\n"
            "clang must support --target=wasm32-unknown-unknown",
        )
    if not tools["wasm2wat"]:
        raise OSError(
            "wasm2wat not found. Install with: apt install wabt\n"
            "Or download from https://github.com/WebAssembly/wabt/releases",
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        c_path = os.path.join(tmpdir, "input.c")
        wasm_path = os.path.join(tmpdir, "output.wasm")

        with open(c_path, "w") as f:
            f.write(source)

        # C → WASM via clang
        clang_cmd = [
            tools["clang"],
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
            # Insert before -o
            idx = clang_cmd.index("-o")
            for arg in reversed(extra_clang_args):
                clang_cmd.insert(idx, arg)

        result = subprocess.run(
            clang_cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"clang compilation failed:\n{result.stderr}")

        # WASM → WAT via wasm2wat
        result = subprocess.run(
            [tools["wasm2wat"], wasm_path],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(f"wasm2wat conversion failed:\n{result.stderr}")

        return result.stdout


def compile_c(
    source: str,
    *,
    func_name: str | None = None,
    opt_level: str = "-O1",
    extra_clang_args: list[str] | None = None,
    strict: bool = True,
) -> list[Instruction]:
    """Compile C source to List[Instruction] for execution.

    This is the main entry point for the C → WASM → tokens pipeline.

    Args:
        source: C source code as a string.
        func_name: Name of the function to extract and compile.
            If None, uses the first exported non-boilerplate function.
        opt_level: Optimization level for clang (default '-O1').
        extra_clang_args: Additional arguments to pass to clang.
        strict: If True (default), raise on any unsupported WASM features.
            If False, attempt to parse anyway (may fail in parse_wat).

    Returns:
        List[Instruction] ready for NumPyExecutor / TorchExecutor.

    Raises:
        EnvironmentError: If clang or wasm2wat not available.
        RuntimeError: If compilation or conversion fails.
        ValueError: If unsupported WASM features are detected (strict mode),
            or if the target function is not found.

    Example::

        prog = compile_c('''
            int add(int a, int b) { return a + b; }
        ''', func_name='add')
        # Execute with arguments: push args, then run
        from transturing.core.isa import program
        from transturing.backends.numpy_backend import NumPyExecutor
        full = program(('PUSH', 3), ('PUSH', 5)) + prog
        result = NumPyExecutor().execute(full)

    """
    wat_text = compile_c_to_wat(
        source,
        opt_level=opt_level,
        extra_clang_args=extra_clang_args,
    )

    # Auto-detect function name if not specified
    if func_name is None:
        func_name = _auto_detect_function(wat_text)

    # Extract the target function's WAT
    func_wat = _extract_function_wat(wat_text, func_name)

    # Check for unsupported features
    if strict:
        errors = _check_unsupported_features(func_wat)
        if errors:
            raise ValueError(
                "C code compiles to WASM features not supported by our ISA:\n"
                + "\n".join(f"  - {e}" for e in errors)
                + "\n\nTips:\n"
                "  - Use -O0 or -O1 to reduce loop strength-reduction to i64\n"
                "  - Avoid float/double types\n"
                "  - Keep arithmetic to 32-bit integers",
            )

    n_params = _count_params(func_wat)
    n_locals = _count_locals(func_wat)

    prog = parse_wat(func_wat)

    # WASM calling convention: function parameters are passed on the
    # operand stack and become locals 0..n_params-1. Our executor's
    # LOCAL.SET/GET operates on a locals array, so we prepend instructions
    # to pop arguments from the stack into locals in reverse order
    # (stack is LIFO: last arg is on top).
    if n_params > 0:
        from .isa import program as make_prog

        # Initialize any extra locals to 0 (WASM semantics: locals default to 0)
        init_instrs = []
        for i in range(n_locals):
            init_instrs.extend([("PUSH", 0), ("LOCAL.SET", n_params + i)])
        # Pop args from stack into locals (reverse order: top of stack = last param)
        for i in range(n_params - 1, -1, -1):
            init_instrs.append(("LOCAL.SET", i))
        setup = make_prog(*init_instrs)
        # Fix: offset all jump targets in prog by len(setup)
        prog = _offset_jumps(setup, prog)

    return prog


def _offset_jumps(
    prefix: list[Instruction],
    body: list[Instruction],
) -> list[Instruction]:
    """Prepend prefix instructions to body, offsetting all jump targets in body.

    Jump instructions (JZ, JNZ) in body have absolute addresses that
    need to be shifted by len(prefix).
    """
    from .isa import OP_JNZ, OP_JZ

    offset = len(prefix)
    adjusted = []
    for instr in body:
        if instr.op in (OP_JZ, OP_JNZ):
            adjusted.append(Instruction(instr.op, instr.arg + offset))
        else:
            adjusted.append(instr)
    return prefix + adjusted


def _auto_detect_function(wat_text: str) -> str:
    """Find the first user-defined (non-boilerplate) exported function.

    Skips __wasm_call_ctors and other compiler-generated functions.
    """
    # Find all exported functions
    exports = re.findall(
        r'\(export\s+"(\w+)"\s+\(func\s+(\$\w+)\)',
        wat_text,
    )

    # Filter out compiler boilerplate
    _BOILERPLATE = {
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

    for export_name, func_name in exports:
        if export_name not in _BOILERPLATE:
            return func_name

    # Fallback: find any func that's not __wasm_call_ctors
    all_funcs = re.findall(r"\(func\s+(\$\w+)", wat_text)
    for f in all_funcs:
        if f != "$__wasm_call_ctors":
            return f

    raise ValueError("No user-defined functions found in compiled WASM")


def compile_and_run(
    source: str,
    args: list[int],
    *,
    func_name: str | None = None,
    opt_level: str = "-O1",
    max_steps: int = 50000,
) -> int:
    """Compile C source and execute with given arguments.

    Convenience function that compiles, prepends PUSH instructions for
    arguments, executes, and returns the top-of-stack result.

    Args:
        source: C source code.
        args: Integer arguments to pass to the function.
        func_name: Function to call (auto-detected if None).
        opt_level: Clang optimization level.
        max_steps: Maximum execution steps.

    Returns:
        Top-of-stack value after execution (the function's return value).

    """
    from transturing.backends.numpy_backend import NumPyExecutor
    from .isa import program as make_prog

    prog = compile_c(source, func_name=func_name, opt_level=opt_level)

    # Prepend argument pushes (WASM calling convention: args on stack)
    arg_instrs = make_prog(*[("PUSH", a) for a in args])
    full_prog = _offset_jumps(arg_instrs, prog)

    trace = NumPyExecutor().execute(full_prog, max_steps=max_steps)
    return trace.steps[-1].top


# ─── Self-test / main ─────────────────────────────────────────────


def main():
    """Run the C pipeline self-tests."""
    import sys

    if not _has_toolchain():
        tools = _check_toolchain()
        missing = [k for k, v in tools.items() if v is None]
        print(f"SKIP: Required tools not installed: {missing}")
        print("Install with: apt install clang lld wabt")
        print("Or download wabt from https://github.com/WebAssembly/wabt/releases")
        sys.exit(0)

    from transturing.backends.numpy_backend import NumPyExecutor
    from .isa import program as make_prog

    np_exec = NumPyExecutor()
    passed = 0
    failed = 0

    def check(name, got, expected):
        nonlocal passed, failed
        if got == expected:
            print(f"  PASS: {name} (got {got})")
            passed += 1
        else:
            print(f"  FAIL: {name}")
            print(f"    expected: {expected}")
            print(f"    got:      {got}")
            failed += 1

    # ── Test 1: Simple addition ──────────────────────────────────
    print("\n=== Test 1: int add(int a, int b) { return a + b; } ===")
    add_src = "int add(int a, int b) { return a + b; }"

    prog = compile_c(add_src, func_name="add")
    print(f"  Compiled to {len(prog)} instructions")

    # Execute: add(3, 5) = 8
    full = _offset_jumps(make_prog(("PUSH", 3), ("PUSH", 5)), prog)
    trace = np_exec.execute(full)
    check("add(3, 5)", trace.steps[-1].top, 8)

    # Execute: add(100, 200) = 300
    full = _offset_jumps(make_prog(("PUSH", 100), ("PUSH", 200)), prog)
    trace = np_exec.execute(full)
    check("add(100, 200)", trace.steps[-1].top, 300)

    # ── Test 2: Loop (Collatz steps) ─────────────────────────────
    # Note: simple sum loops get strength-reduced to i64 Gauss formula
    # by clang -O1. Collatz is non-linear so it compiles to a real loop.
    print("\n=== Test 2: loop (Collatz steps) ===")
    collatz_src = """
    int collatz_steps(int n) {
        int steps = 0;
        while (n != 1) {
            if (n % 2 == 0) {
                n = n / 2;
            } else {
                n = 3 * n + 1;
            }
            steps = steps + 1;
        }
        return steps;
    }
    """
    prog = compile_c(collatz_src, func_name="collatz_steps", opt_level="-O1")
    print(f"  Compiled to {len(prog)} instructions")

    # Collatz(6): 6→3→10→5→16→8→4→2→1 = 8 steps
    result = compile_and_run(collatz_src, [6], func_name="collatz_steps")
    check("collatz_steps(6)", result, 8)

    # Collatz(1) = 0 steps
    result = compile_and_run(collatz_src, [1], func_name="collatz_steps")
    check("collatz_steps(1)", result, 0)

    # Collatz(27) = 111 steps (famous long sequence)
    result = compile_and_run(collatz_src, [27], func_name="collatz_steps")
    check("collatz_steps(27)", result, 111)

    # ── Test 2b: Sum loop via manual WAT (demonstrates WAT path) ──
    print("\n=== Test 2b: sum loop (manual WAT) ===")
    loop_wat = """
    (func $sum_loop (param i32) (result i32)
      (local i32 i32)
      i32.const 0
      local.set 1
      i32.const 0
      local.set 2
      block $exit
        loop $loop
          local.get 2
          local.get 0
          i32.ge_s
          br_if $exit
          local.get 1
          local.get 2
          i32.add
          local.set 1
          local.get 2
          i32.const 1
          i32.add
          local.set 2
          br $loop
        end
      end
      local.get 1
    )
    """
    raw_prog = parse_wat(loop_wat)
    # Add parameter setup: init locals 1,2 to 0, then pop arg to local 0
    setup = make_prog(
        ("PUSH", 0),
        ("LOCAL.SET", 1),
        ("PUSH", 0),
        ("LOCAL.SET", 2),
        ("LOCAL.SET", 0),
    )
    prog = _offset_jumps(setup, raw_prog)
    print(f"  Compiled from WAT to {len(prog)} instructions (with setup)")

    full = _offset_jumps(make_prog(("PUSH", 10)), prog)
    trace = np_exec.execute(full)
    check("sum_loop(10)", trace.steps[-1].top, 45)

    full = _offset_jumps(make_prog(("PUSH", 5)), prog)
    trace = np_exec.execute(full)
    check("sum_loop(5)", trace.steps[-1].top, 10)

    # ── Test 3: Recursive factorial ──────────────────────────────
    print("\n=== Test 3: recursive factorial ===")
    fact_src = """
    int factorial(int n) {
        if (n <= 1) return 1;
        return n * factorial(n - 1);
    }
    """
    # clang -O1 turns tail-recursive factorial into a loop with locals,
    # which is perfect for our ISA
    prog = compile_c(fact_src, func_name="factorial", opt_level="-O1")
    print(f"  Compiled to {len(prog)} instructions")

    # Execute: factorial(5) = 120
    full = _offset_jumps(make_prog(("PUSH", 5)), prog)
    trace = np_exec.execute(full)
    check("factorial(5)", trace.steps[-1].top, 120)

    # Execute: factorial(1) = 1
    full = _offset_jumps(make_prog(("PUSH", 1)), prog)
    trace = np_exec.execute(full)
    check("factorial(1)", trace.steps[-1].top, 1)

    # Execute: factorial(10) = 3628800
    full = _offset_jumps(make_prog(("PUSH", 10)), prog)
    trace = np_exec.execute(full)
    check("factorial(10)", trace.steps[-1].top, 3628800)

    # ── Test 4: Unsupported features detection ───────────────────
    print("\n=== Test 4: unsupported feature detection ===")

    # Float code should fail in strict mode
    float_src = "float addf(float a, float b) { return a + b; }"
    try:
        compile_c(float_src, func_name="addf", strict=True)
        print("  FAIL: should have raised ValueError for floats")
        failed += 1
    except ValueError as e:
        if "f32" in str(e).lower() or "float" in str(e).lower():
            print(f"  PASS: float detection ({str(e)[:60]}...)")
            passed += 1
        else:
            print(f"  FAIL: wrong error: {e}")
            failed += 1

    # ── Test 5: Auto-detect function name ────────────────────────
    print("\n=== Test 5: auto-detect function name ===")
    prog = compile_c(add_src)  # no func_name specified
    full = _offset_jumps(make_prog(("PUSH", 7), ("PUSH", 8)), prog)
    trace = np_exec.execute(full)
    check("auto-detect add(7, 8)", trace.steps[-1].top, 15)

    # ── Test 6: compile_and_run convenience ──────────────────────
    print("\n=== Test 6: compile_and_run ===")
    result = compile_and_run(add_src, [10, 20], func_name="add")
    check("compile_and_run add(10, 20)", result, 30)

    result = compile_and_run(fact_src, [7], func_name="factorial")
    check("compile_and_run factorial(7)", result, 5040)

    # ── Test 7: compile_c_to_wat (intermediate) ──────────────────
    print("\n=== Test 7: compile_c_to_wat ===")
    wat = compile_c_to_wat(add_src)
    check("WAT contains func $add", "$add" in wat, True)
    check("WAT is module", wat.strip().startswith("(module"), True)

    # ── Summary ──────────────────────────────────────────────────
    print(f"\n{'=' * 50}")
    print(f"C Pipeline: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)
    print("All tests passed!")


if __name__ == "__main__":
    main()
