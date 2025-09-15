# tests/utils_loader.py
import ast
from pathlib import Path

def load_functions(file_path: Path, names: list[str]):
    """
    Dynamically loads and returns specified functions from a Python source file.
    Returns a dictionary mapping function names to function objects (or None if not found or on error).
    """
    src = Path(file_path).read_text(encoding="utf-8")
    try:
        tree = ast.parse(src)
    except SyntaxError:
        # If the file cannot be parsed, return None for all requested names
        return {n: None for n in names}

    spans = {}
    for node in tree.body:
        # Find the line numbers for each requested function definition
        if isinstance(node, ast.FunctionDef) and node.name in names:
            spans[node.name] = (node.lineno, node.end_lineno)

    # Prepare an environment with common imports and constants
    env = {}
    try:
        import pyspark.sql.functions as F
        env["F"] = F
    except Exception:
        pass
    try:
        import pyspark.sql.types as T
        env["T"] = T
    except Exception:
        pass
    try:
        import requests
        env["requests"] = requests
    except Exception:
        pass

    # Provide a dummy OMDB_API_KEY for functions that may require it
    env.setdefault("OMDB_API_KEY", "dummy")

    out = {}
    for name in names:
        span = spans.get(name)
        if not span:
            out[name] = None
            continue
        lineno, end = span
        # Extract the source code for the function
        func_src = "\n".join(src.splitlines()[lineno-1:end])
        code = compile(func_src, str(file_path), "exec")
        exec(code, env, env)
        out[name] = env.get(name)

    return out
