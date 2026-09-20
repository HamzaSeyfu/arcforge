from __future__ import annotations

import time


class VLLMTextRuntime:
    """
    High-throughput local text generation for Kaggle L4x4.

    This runtime uses vLLM tensor parallelism instead of Transformers
    device_map='auto'. The latter is convenient for fitting large models but is
    poorly suited to repeated ARC refinement loops because every token traverses
    a pipeline-sharded model.

    The class keeps vLLM imports lazy so CPU-only tests remain lightweight.
    """

    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int = 4,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.90,
        max_num_seqs: int = 4,
        enforce_eager: bool = True,
        enable_prefix_caching: bool = True,
        dtype: str = "bfloat16",
    ):
        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError(
                "VLLMTextRuntime requires vLLM in the runtime"
            ) from exc

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            trust_remote_code=True,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            enable_prefix_caching=enable_prefix_caching,
            disable_log_stats=True,
        )

    def generate(
        self,
        prompts: list[str],
        *,
        seed: int,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 30,
    ) -> list[str]:
        if not prompts:
            return []

        params = self.SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            seed=seed,
        )
        outputs = self.llm.generate(prompts, params, use_tqdm=False)
        return [item.outputs[0].text for item in outputs]
