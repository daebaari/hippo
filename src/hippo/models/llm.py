"""LLM backends.

Two implementations behind the shared `LLMProto` contract:

- ``LocalLLM``: Qwen via mlx-lm. Loads the weights once per heavy-dream run.
- ``GeminiLLM``: Cloud Gemini via google-genai (optional extra).

``select_llm()`` dispatches on ``hippo.config.Config.backend``.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Any, Protocol


def _sleep(seconds: float) -> None:
    """Indirection so tests can patch sleep without touching `time`."""
    time.sleep(seconds)


LLM_MODEL_ID = "mlx-community/Qwen2.5-32B-Instruct-4bit"


class LLMProto(Protocol):
    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        thinking_level: str | None = None,
    ) -> str: ...


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
        thinking_level: str | None = None,  # ignored; LocalLLM has no thinking
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)


_RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})


@dataclass
class GeminiLLM:
    """Gemini backend.

    Constructed via ``GeminiLLM.load(api_key=..., model_id=..., default_thinking_level=...)``.
    No process-environment mutation: the API key is passed explicitly to the SDK client.
    """

    client: Any
    model_id: str
    default_thinking_level: str
    max_attempts: int = 5
    request_timeout_s: float = 60.0

    @staticmethod
    def load(
        *,
        api_key: str,
        model_id: str,
        default_thinking_level: str,
    ) -> GeminiLLM:
        try:
            from google import genai
        except ImportError as exc:
            from hippo.config import ConfigError
            raise ConfigError(
                "Gemini backend requires 'google-genai'. "
                "Install with: uv sync --extra gemini"
            ) from exc
        return GeminiLLM(
            client=genai.Client(api_key=api_key),
            model_id=model_id,
            default_thinking_level=default_thinking_level,
        )

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        thinking_level: str | None = None,
    ) -> str:
        from google.genai import types
        if len(messages) == 1 and messages[0].get("role", "user") == "user":
            contents: Any = messages[0]["content"]
        else:
            contents = "\n\n".join(
                f"{m.get('role', 'user').upper()}: {m['content']}" for m in messages
            )
        level = thinking_level or self.default_thinking_level
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            thinking_config=types.ThinkingConfig(thinking_level=level),  # type: ignore[arg-type]
        )
        return self._call_with_retry(contents=contents, config=config)

    def _call_with_retry(self, *, contents: Any, config: Any) -> str:
        import httpx
        from google.genai import errors
        _network_excs: tuple[type[BaseException], ...] = (
            OSError, TimeoutError, httpx.RequestError,
        )

        for attempt in range(self.max_attempts):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id, contents=contents, config=config
                )
                return resp.text or ""
            except errors.APIError as e:
                code = getattr(e, "code", None)
                if code in _RETRYABLE_HTTP and attempt < self.max_attempts - 1:
                    _sleep(2 ** attempt)
                    continue
                raise
            except _network_excs:
                if attempt < self.max_attempts - 1:
                    _sleep(2 ** attempt)
                    continue
                raise
        raise RuntimeError("unreachable")  # pragma: no cover


def select_llm(*, strict: bool = False) -> LocalLLM | GeminiLLM:
    """Return the configured LLM backend.

    See ``hippo.config.Config`` for the toggle, ``hippo.config.load_api_key``
    for the key resolution. If ``backend == "gemini"`` and no key is found,
    ``strict=True`` raises ``ConfigError``; ``strict=False`` warns and falls
    back to ``LocalLLM``.
    """
    from hippo.config import ConfigError, load_api_key, load_config
    cfg = load_config()
    if cfg.backend == "qwen":
        return LocalLLM.load()
    if cfg.backend == "gemini":
        key = load_api_key()
        if not key:
            msg = (
                "Gemini selected but no API key found. "
                "Set GOOGLE_API_KEY (or GEMINI_API_KEY) or write to "
                "~/.claude/hippo-secrets."
            )
            if strict:
                raise ConfigError(msg)
            print(f"WARNING: {msg} Falling back to qwen.", file=sys.stderr)
            return LocalLLM.load()
        return GeminiLLM.load(
            api_key=key,
            model_id=cfg.gemini_model_id,
            default_thinking_level=cfg.gemini_default_thinking_level,
        )
    raise ConfigError(f"Unknown backend {cfg.backend!r}")  # pragma: no cover
