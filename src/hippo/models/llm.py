"""Local Qwen 2.5 32B Instruct wrapper via mlx-lm.

Loads once per heavy-dream run, unloaded after. ~18GB resident at 4-bit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

LLM_MODEL_ID = "mlx-community/Qwen2.5-32B-Instruct-4bit"


@dataclass
class LocalLLM:
    model: Any
    tokenizer: Any

    @staticmethod
    def load() -> LocalLLM:
        from mlx_lm import load
        result = load(LLM_MODEL_ID)
        model, tokenizer = result[0], result[1]
        return LocalLLM(model=model, tokenizer=tokenizer)

    def generate(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 1024) -> str:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        return generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=temperature),
            verbose=False,
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)
