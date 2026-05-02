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


LLM_MODEL_ID = "lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit"


class LLMProto(Protocol):
    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        thinking_level: str | None = None,
    ) -> str: ...

    def generate_chat_batch(
        self,
        message_lists: list[list[dict[str, str]]],
        *,
        temperature: float,
        max_tokens: int,
        thinking_level: str | None = None,
        batch_size: int = 8,
    ) -> list[str]:
        """Run multiple chat-style prompts and return one completion per input,
        in input order. Implementations may parallelize (e.g. mlx_lm.batch_generate)
        or fall back to a sequential loop. Output length must equal input length."""
        ...


@dataclass
class LocalLLM:
    model: Any
    tokenizer: Any
    # Lazily-built shared-prefix KV cache, keyed by the tokenized prefix.
    # Populated on the first generate_chat_batch call whose inputs share a long
    # prefix; reused on subsequent calls with the same prefix. None on a fresh
    # instance and after teardown.
    _prefix_tokens: list[int] | None = None
    _prefix_cache: Any = None

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

    def _format_chat_prompt(self, messages: list[dict[str, str]]) -> str:
        # enable_thinking=False auto-closes the model's thinking block in the prompt.
        # Required for Gemma 4 — its default thinking trace eats max_tokens before any
        # JSON/structured output is emitted. Tokenizers that don't accept the kwarg
        # fall back to plain template (older models without thinking mode).
        try:
            return str(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            ))
        except TypeError:
            return str(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            ))

    def generate_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        thinking_level: str | None = None,  # ignored; LocalLLM thinking is forced off
    ) -> str:
        prompt = self._format_chat_prompt(messages)
        return self.generate(prompt, temperature=temperature, max_tokens=max_tokens)

    def generate_chat_batch(
        self,
        message_lists: list[list[dict[str, str]]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        thinking_level: str | None = None,  # ignored; LocalLLM thinking is forced off
        batch_size: int = 8,
    ) -> list[str]:
        """Batched chat generation via mlx_lm.batch_generate, with a lazily-built
        shared-prefix KV cache when all prompts in the batch share a token prefix.

        Strategy per batch chunk:
          1. Tokenize each chunk member's full prompt.
          2. Compute the longest token prefix common to all chunk members.
          3. If the common prefix is the cached one, deepcopy the cache per
             sequence and pass the suffixes; otherwise rebuild or skip the cache.
          4. Run mlx_lm.batch_generate; collect responses in input order.
        """
        if not message_lists:
            return []
        from mlx_lm import batch_generate
        from mlx_lm.models.cache import make_prompt_cache
        import copy as _copy
        import mlx.core as mx

        # Pre-tokenize every prompt up front. Done once per call.
        token_lists: list[list[int]] = [
            list(self.tokenizer.encode(self._format_chat_prompt(m)))
            for m in message_lists
        ]

        results: list[str] = [""] * len(token_lists)
        for start in range(0, len(token_lists), batch_size):
            chunk = token_lists[start:start + batch_size]
            chunk_caches: list[list[Any]] | None = None

            if len(chunk) >= 2:
                # Longest token prefix common to every prompt in this chunk.
                common = chunk[0]
                for ids in chunk[1:]:
                    n = 0
                    upper = min(len(common), len(ids))
                    while n < upper and common[n] == ids[n]:
                        n += 1
                    common = common[:n]
                    if not common:
                        break

                if len(common) >= 32:  # only worth caching for non-trivial prefixes
                    if (
                        self._prefix_cache is None
                        or self._prefix_tokens != common
                    ):
                        self._prefix_cache = make_prompt_cache(self.model)
                        self.model(mx.array(common)[None], cache=self._prefix_cache)
                        mx.eval([c.state for c in self._prefix_cache])
                        self._prefix_tokens = list(common)
                    suffixes = [ids[len(common):] for ids in chunk]
                    chunk_caches = [
                        _copy.deepcopy(self._prefix_cache) for _ in suffixes
                    ]
                    chunk = suffixes

            resp = batch_generate(
                self.model, self.tokenizer,
                prompts=chunk,
                prompt_caches=chunk_caches,
                max_tokens=max_tokens,
                verbose=False,
            )
            for i, text in enumerate(resp.texts):
                results[start + i] = text
        return results


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

    def generate_chat_batch(
        self,
        message_lists: list[list[dict[str, str]]],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        thinking_level: str | None = None,
        batch_size: int = 8,  # ignored; Gemini calls go one-at-a-time
    ) -> list[str]:
        """Sequential per-message fallback. Gemini has no equivalent local-batch
        primitive; we keep the API surface so callers don't branch on backend."""
        return [
            self.generate_chat(
                m,
                temperature=temperature,
                max_tokens=max_tokens,
                thinking_level=thinking_level,
            )
            for m in message_lists
        ]

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
    if cfg.backend == "local":
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
            print(f"WARNING: {msg} Falling back to local.", file=sys.stderr)
            return LocalLLM.load()
        return GeminiLLM.load(
            api_key=key,
            model_id=cfg.gemini_model_id,
            default_thinking_level=cfg.gemini_default_thinking_level,
        )
    raise ConfigError(f"Unknown backend {cfg.backend!r}")  # pragma: no cover
