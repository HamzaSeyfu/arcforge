from __future__ import annotations

import os
import random


class LocalQwenTextRuntime:
    """
    Lazy local Hugging Face runtime for Kaggle-attached Qwen/Qwen-Coder weights.

    Heavy imports happen only when instantiated, so ARCForge's CPU tooling and
    tests stay lightweight.
    """

    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int = 1800,
        temperature: float = 0.85,
        top_p: float = 0.95,
        device_map: str = "auto",
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "LocalQwenTextRuntime requires torch + transformers in the runtime"
            ) from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _format_prompt(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        return prompt

    def __call__(self, prompts: list[str], seed: int) -> list[str]:
        if not prompts:
            return []

        rendered = [self._format_prompt(prompt) for prompt in prompts]
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        random.seed(seed)
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        return [
            self.tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            for row in generated
        ]
