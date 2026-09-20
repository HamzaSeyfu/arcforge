from __future__ import annotations

import ast
import signal
import time
from collections import Counter, deque
from contextlib import contextmanager
from typing import Callable

import numpy as np

from .types import Grid, VerificationReport


_ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "range": range,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
    "Exception": Exception,
    "ValueError": ValueError,
}

_FORBIDDEN_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "open",
    "setattr",
    "vars",
}


class ProgramRejected(ValueError):
    pass


class ProgramTimeout(TimeoutError):
    pass


@contextmanager
def _time_limit(seconds: float):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def handler(signum, frame):
        raise ProgramTimeout(f"program exceeded {seconds:.2f}s")

    previous = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _validate_ast(source: str, source_char_limit: int) -> ast.Module:
    if len(source) > source_char_limit:
        raise ProgramRejected(
            f"source too large: {len(source)} chars > {source_char_limit}"
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ProgramRejected(f"syntax error: {exc}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ProgramRejected("imports are forbidden; numpy is available as np")
        if isinstance(node, ast.While):
            raise ProgramRejected("while loops are forbidden")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _FORBIDDEN_CALLS:
                raise ProgramRejected(f"forbidden call: {node.func.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ProgramRejected("dunder attribute access is forbidden")

    functions = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not any(f.name == "transform" for f in functions):
        raise ProgramRejected("source must define transform(grid)")

    return tree


def compile_transform(
    source: str,
    *,
    source_char_limit: int = 12000,
) -> Callable[[np.ndarray], np.ndarray]:
    tree = _validate_ast(source, source_char_limit)
    env = {
        "__builtins__": _ALLOWED_BUILTINS,
        "np": np,
        "Counter": Counter,
        "deque": deque,
    }
    exec(compile(tree, "<arcforge-program>", "exec"), env, env)
    fn = env.get("transform")
    if not callable(fn):
        raise ProgramRejected("transform is not callable")
    return fn


def _normalize_grid(value) -> Grid:
    arr = np.asarray(value)
    if arr.ndim != 2:
        raise ValueError(f"output must be rank 2, got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("output grid must be non-empty")
    if arr.shape[0] > 30 or arr.shape[1] > 30:
        raise ValueError(f"output shape {arr.shape} exceeds ARC 30x30 limit")

    if not np.issubdtype(arr.dtype, np.integer):
        if np.all(np.equal(arr, np.floor(arr))):
            arr = arr.astype(int)
        else:
            raise ValueError("output contains non-integer values")

    arr = arr.astype(int)
    if np.any(arr < 0) or np.any(arr > 9):
        raise ValueError("ARC colors must be integers 0..9")
    return arr.tolist()


def _soft_score(pred: Grid | None, truth: Grid) -> float:
    if pred is None:
        return 0.0
    pa = np.asarray(pred)
    ta = np.asarray(truth)
    if pa.shape != ta.shape:
        return 0.0
    if ta.size == 0:
        return 1.0
    return float(np.mean(pa == ta))


def verify_program(
    source: str,
    task: dict,
    *,
    timeout_seconds: float = 1.5,
    source_char_limit: int = 12000,
) -> VerificationReport:
    started = time.perf_counter()

    try:
        fn = compile_transform(source, source_char_limit=source_char_limit)
    except Exception as exc:
        n = len(task.get("train", []))
        return VerificationReport(
            exact_train=False,
            exact_examples=0,
            total_examples=n,
            mean_soft_score=0.0,
            per_example_soft=tuple(0.0 for _ in range(n)),
            train_outputs=tuple(None for _ in range(n)),
            test_outputs=tuple(None for _ in task.get("test", [])),
            errors=tuple(str(exc) for _ in range(n)),
            runtime_seconds=time.perf_counter() - started,
        )

    train_outputs = []
    soft_scores = []
    errors = []

    for pair in task.get("train", []):
        try:
            with _time_limit(timeout_seconds):
                raw = fn(np.asarray(pair["input"], dtype=int).copy())
            pred = _normalize_grid(raw)
            err = None
        except Exception as exc:
            pred = None
            err = f"{type(exc).__name__}: {exc}"
        train_outputs.append(pred)
        soft_scores.append(_soft_score(pred, pair["output"]))
        errors.append(err)

    exact_examples = sum(
        pred == pair["output"]
        for pred, pair in zip(train_outputs, task.get("train", []), strict=True)
    )
    total_examples = len(task.get("train", []))

    test_outputs = []
    for test in task.get("test", []):
        try:
            with _time_limit(timeout_seconds):
                raw = fn(np.asarray(test["input"], dtype=int).copy())
            test_outputs.append(_normalize_grid(raw))
        except Exception:
            test_outputs.append(None)

    return VerificationReport(
        exact_train=(total_examples > 0 and exact_examples == total_examples),
        exact_examples=exact_examples,
        total_examples=total_examples,
        mean_soft_score=sum(soft_scores) / len(soft_scores) if soft_scores else 0.0,
        per_example_soft=tuple(soft_scores),
        train_outputs=tuple(train_outputs),
        test_outputs=tuple(test_outputs),
        errors=tuple(errors),
        runtime_seconds=time.perf_counter() - started,
    )
