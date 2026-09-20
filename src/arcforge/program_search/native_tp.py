"""Native Transformers tensor-parallel helpers for Kaggle L4x4.

Use this path when vLLM is unavailable. Qwen3-MoE exposes a native TP plan in
Transformers; launch the worker with torchrun --nproc_per_node=4 and load with
tp_plan="auto" (never device_map="auto").
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def torchrun_command(worker: str | Path, *, nproc: int = 4) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "--max_restarts=0",
        str(worker),
    ]


def launch_native_tp(
    worker: str | Path,
    *,
    nproc: int = 4,
    env_overrides: dict[str, str] | None = None,
) -> int:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if env_overrides:
        env.update(env_overrides)

    return subprocess.call(torchrun_command(worker, nproc=nproc), env=env)
