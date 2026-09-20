"""Runtime notes/helpers for Qwen3-Coder on Kaggle 4xL4.

Transformers 5 exposes a DistributedConfig API for MoE expert parallelism.
For Qwen3-MoE, enable_expert_parallel=True shards experts across ranks and
automatically enables tensor parallelism for attention.

The Kaggle Qwen3-Coder 30B-A3B checkpoint has:
- 128 experts
- 8 experts activated per token
- 4 KV heads

Therefore a 4-rank topology divides both experts and KV heads evenly.
"""

from __future__ import annotations

import os


def build_distributed_config(world_size: int | None = None):
    """Create the Transformers 5 distributed config lazily."""
    try:
        from transformers.distributed.configuration_utils import DistributedConfig
    except Exception as exc:
        raise RuntimeError(
            "Transformers DistributedConfig is required for expert-parallel runtime"
        ) from exc

    size = world_size or int(os.environ.get("WORLD_SIZE", "1"))
    if size < 2:
        raise ValueError("expert-parallel runtime requires torchrun world_size >= 2")

    return DistributedConfig(
        tp_size=size,
        enable_expert_parallel=True,
    )
