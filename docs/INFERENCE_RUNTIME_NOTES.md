# ARCForge inference runtime notes

## 2026-09-20 Kaggle findings

1. Qwen3-Coder-30B-A3B-Instruct loads successfully on Kaggle L4 x4 using
   Transformers + device_map="auto", but generation through module placement is
   too slow for the agentic search loop.
2. Batching two long ARC prompts with that path can OOM GPU 0.
3. Reserving VRAM and generating sequentially fixes the OOM, but not the
   throughput bottleneck.
4. vLLM is not installed in the current Kaggle image, so ARCForge must not
   assume it exists.
5. Next runtime: native Transformers tensor parallelism under torchrun:
   - 4 processes
   - tp_plan="auto"
   - no device_map
   - identical prompts/seeds on all TP ranks
   - rank 0 owns reporting/checkpoints

Qwen3-MoE exposes a base_model_tp_plan and Qwen3-Coder has 4 KV heads, so
4-way tensor parallelism is structurally compatible with Kaggle L4 x4.
