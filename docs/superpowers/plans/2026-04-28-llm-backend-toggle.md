# LLM Backend Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an easily toggleable Gemini backend (default qwen) to Hippo's heavy-dream pipeline, with a slash command for switching, file-based config + secrets that survive launchd's stripped env, and resolution of all known issues from the prior stash review.

**Architecture:** A `Config` dataclass loaded from `~/.claude/hippo-config.toml` and an API-key resolver that reads env then `~/.claude/hippo-secrets`. `select_llm(strict)` dispatches on the config; manual invocations are strict (hard fail on misconfig), launchd is non-strict (silent fallback to qwen). `LLMProto` consolidates into `hippo.models.llm`. A new slash command `/hippo-backend` reads/writes the config.

**Tech Stack:** Python 3.12, stdlib `tomllib` (read), hand-written TOML for write (3 fields, no extra dep), `google-genai>=1.0` as optional extra, mlx-lm for local Qwen, `httpx` (already a transitive dep) for typed network errors, pytest + monkeypatch for isolation.

**Spec:** `docs/superpowers/specs/2026-04-28-llm-backend-toggle-design.md`

---

## File Map

**New:**
- `src/hippo/cli/backend_toggle.py` — slash-command CLI module
- `tests/test_config_toggle.py` — covers the new `Config`/`load_config`/`write_config`/`load_api_key`
- `tests/test_llm_select.py` — covers `select_llm()` dispatch
- `tests/test_backend_toggle.py` — covers the slash-command CLI
- `~/.claude/commands/hippo-backend.md` — slash command markdown (lives outside repo)

**Modified:**
- `src/hippo/config.py` — extends with `Config` dataclass, `ConfigError`, TOML/secrets helpers
- `src/hippo/models/llm.py` — adds `LLMProto`, `GeminiLLM`, `select_llm`; adds `thinking_level` no-op to `LocalLLM`; replaces hardcoded retry list and adds timeout
- `src/hippo/cli/dream_heavy.py` — `--strict` flag, `select_llm`, `ConfigError` handling
- `src/hippo/cli/dream_bootstrap.py` — same
- `bin/dream-heavy`, `bin/dream-bootstrap` — append `--strict` to their delegated args
- `src/hippo/dream/{atomize,bootstrap,contradiction,edge_proposal,heavy,multi_head}.py` — delete local `LLMProto`, import from `hippo.models.llm`
- `src/hippo/dream/contradiction.py` and `edge_proposal.py` — pass `thinking_level="minimal"` at `llm.generate_chat`
- `tests/test_{atomize,bootstrap,contradiction,dream_heavy_orchestrator,edge_proposal,multi_head}.py` — fake `LLMProto`s gain `thinking_level=None` kwarg
- `tests/test_contradiction.py`, `tests/test_edge_proposal.py` — add assertion that `thinking_level="minimal"` was passed
- `pyproject.toml` — `google-genai` moves to `[project.optional-dependencies]`
- `KNOWN_ISSUES.md` — delete the stash-preserved entry
- `docs/operations.md`, `CLAUDE.md`, `README.md` — small modifications inside existing sections

**Deleted (after merge):**
- `stash@{0}: gemini-backend-wip-from-plan7-overreach` via `git stash drop`

---

### Task 1: Extend `hippo.config` with toggle paths and `ConfigError`

**Files:**
- Modify: `src/hippo/config.py`
- Test: `tests/test_config_toggle.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_toggle.py`:

```python
"""Tests for the toggle-related additions in hippo.config."""
from __future__ import annotations

import pytest

from hippo import config as cfg


class TestConfigPaths:
    def test_config_path_in_claude_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.config_path() == tmp_path / "hippo-config.toml"

    def test_secrets_path_in_claude_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.secrets_path() == tmp_path / "hippo-secrets"

    def test_default_config_dir_is_claude_home(self, monkeypatch):
        monkeypatch.delenv("HIPPO_CONFIG_DIR", raising=False)
        assert cfg.config_path().parent == cfg.CLAUDE_HOME


class TestConfigError:
    def test_config_error_is_runtime_error(self):
        assert issubclass(cfg.ConfigError, RuntimeError)

    def test_config_error_carries_message(self):
        err = cfg.ConfigError("boom")
        assert str(err) == "boom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_toggle.py -v`
Expected: FAIL — `AttributeError: module 'hippo.config' has no attribute 'config_path'` (etc.)

- [ ] **Step 3: Implement the additions**

Append to `src/hippo/config.py`:

```python

# === LLM backend toggle ===
import os as _os

HIPPO_CONFIG_FILENAME = "hippo-config.toml"
HIPPO_SECRETS_FILENAME = "hippo-secrets"


class ConfigError(RuntimeError):
    """Raised for malformed configuration or unrecoverable misconfiguration."""


def _config_dir() -> Path:
    override = _os.environ.get("HIPPO_CONFIG_DIR")
    return Path(override) if override else CLAUDE_HOME


def config_path() -> Path:
    return _config_dir() / HIPPO_CONFIG_FILENAME


def secrets_path() -> Path:
    return _config_dir() / HIPPO_SECRETS_FILENAME
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_toggle.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/hippo/config.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/hippo/config.py tests/test_config_toggle.py
git commit -m "config: add toggle paths and ConfigError"
```

---

### Task 2: `Config` dataclass + `load_config()` + `write_config()`

**Files:**
- Modify: `src/hippo/config.py`
- Test: `tests/test_config_toggle.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config_toggle.py`:

```python
class TestConfig:
    def test_load_missing_file_returns_defaults(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        c = cfg.load_config()
        assert c.backend == "qwen"

    def test_round_trip(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        cfg.write_config(cfg.Config(backend="gemini", gemini_model_id="x", gemini_default_thinking_level="low"))
        c = cfg.load_config()
        assert c.backend == "gemini"
        assert c.gemini_model_id == "x"
        assert c.gemini_default_thinking_level == "low"

    def test_load_malformed_toml_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-config.toml").write_text("not valid toml [[[")
        with pytest.raises(cfg.ConfigError):
            cfg.load_config()

    def test_load_unknown_backend_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-config.toml").write_text('backend = "pigeon"\n')
        with pytest.raises(cfg.ConfigError, match="backend must be"):
            cfg.load_config()

    def test_write_is_atomic(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        cfg.write_config(cfg.Config(backend="gemini"))
        # tmp file should not linger after successful write
        leftovers = list(tmp_path.glob("*.tmp*"))
        assert leftovers == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_toggle.py::TestConfig -v`
Expected: FAIL — `AttributeError: module 'hippo.config' has no attribute 'Config'`

- [ ] **Step 3: Implement `Config` dataclass + load/write**

Append to `src/hippo/config.py`:

```python

import tomllib as _tomllib
from dataclasses import dataclass as _dataclass


_VALID_BACKENDS: frozenset[str] = frozenset({"qwen", "gemini"})

DEFAULT_GEMINI_MODEL_ID = "gemini-3-flash-preview"
DEFAULT_GEMINI_THINKING_LEVEL = "high"


@_dataclass(frozen=True)
class Config:
    backend: str = "qwen"
    gemini_model_id: str = DEFAULT_GEMINI_MODEL_ID
    gemini_default_thinking_level: str = DEFAULT_GEMINI_THINKING_LEVEL


def load_config() -> Config:
    p = config_path()
    if not p.exists():
        return Config()
    try:
        data = _tomllib.loads(p.read_text())
    except _tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    backend = str(data.get("backend", "qwen"))
    if backend not in _VALID_BACKENDS:
        raise ConfigError(
            f"backend must be 'qwen' or 'gemini', got {backend!r} in {p}"
        )
    gemini = data.get("gemini") if isinstance(data.get("gemini"), dict) else {}
    return Config(
        backend=backend,
        gemini_model_id=str(gemini.get("model_id", DEFAULT_GEMINI_MODEL_ID)),
        gemini_default_thinking_level=str(
            gemini.get("default_thinking_level", DEFAULT_GEMINI_THINKING_LEVEL)
        ),
    )


def write_config(c: Config) -> None:
    p = config_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    body = (
        f'backend = "{c.backend}"\n'
        f"\n"
        f"[gemini]\n"
        f'model_id = "{c.gemini_model_id}"\n'
        f'default_thinking_level = "{c.gemini_default_thinking_level}"\n'
    )
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(body)
    _os.replace(tmp, p)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_toggle.py -v`
Expected: PASS (10 passed)

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/hippo/config.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/hippo/config.py tests/test_config_toggle.py
git commit -m "config: add Config dataclass + atomic TOML load/write"
```

---

### Task 3: `load_api_key()` with env-then-file precedence

**Files:**
- Modify: `src/hippo/config.py`
- Test: `tests/test_config_toggle.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_config_toggle.py`:

```python
class TestApiKey:
    def test_no_key_anywhere_returns_none(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.load_api_key() is None

    def test_google_env_wins(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GOOGLE_API_KEY", "from-google")
        monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-secrets").write_text("GOOGLE_API_KEY=from-file\n")
        assert cfg.load_api_key() == "from-google"

    def test_gemini_env_when_no_google(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv("GEMINI_API_KEY", "from-gemini")
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        assert cfg.load_api_key() == "from-gemini"

    def test_secrets_file_when_no_env(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-secrets").write_text(
            "# comment\n\nGOOGLE_API_KEY=from-file\n"
        )
        assert cfg.load_api_key() == "from-file"

    def test_secrets_file_gemini_key_alias(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-secrets").write_text("GEMINI_API_KEY=alias-file\n")
        assert cfg.load_api_key() == "alias-file"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config_toggle.py::TestApiKey -v`
Expected: FAIL — `AttributeError: module 'hippo.config' has no attribute 'load_api_key'`

- [ ] **Step 3: Implement `load_api_key`**

Append to `src/hippo/config.py`:

```python

def load_api_key() -> str | None:
    for env_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = _os.environ.get(env_name)
        if v:
            return v
    p = secrets_path()
    if not p.exists():
        return None
    for raw_line in p.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, val = line.partition("=")
        if key.strip() in ("GOOGLE_API_KEY", "GEMINI_API_KEY") and val.strip():
            return val.strip()
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config_toggle.py -v`
Expected: PASS (15 passed)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/config.py tests/test_config_toggle.py
git commit -m "config: load_api_key with env-then-secrets-file precedence"
```

---

### Task 4: Add `LLMProto` to `hippo.models.llm` and `thinking_level` no-op to `LocalLLM`

**Files:**
- Modify: `src/hippo/models/llm.py`

This task introduces no behavior change but lays the typing foundation for the next steps.

- [ ] **Step 1: Edit `src/hippo/models/llm.py`**

Replace the entire contents of `src/hippo/models/llm.py` with:

```python
"""LLM backends.

Two implementations behind the shared `LLMProto` contract:

- ``LocalLLM``: Qwen via mlx-lm. Loads the weights once per heavy-dream run.
- ``GeminiLLM``: Cloud Gemini via google-genai (optional extra).

``select_llm()`` dispatches on ``hippo.config.Config.backend``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `uv run pytest tests/ -x --ignore=tests/test_llm.py`
Expected: PASS (all currently-passing tests still pass; the dream-pipeline tests construct their own `FakeLLM`s that aren't typed against `LLMProto`, so they're unaffected)

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/hippo/models/llm.py`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add src/hippo/models/llm.py
git commit -m "llm: add LLMProto + thinking_level no-op kwarg on LocalLLM"
```

---

### Task 5: Implement `GeminiLLM` (explicit args, no env mutation, basic generate_chat)

**Files:**
- Modify: `src/hippo/models/llm.py`

- [ ] **Step 1: Append `GeminiLLM` to `src/hippo/models/llm.py`**

Add at the end of the file:

```python


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
        # Retry implementation lands in Task 6 (broadens exception list and adds timeout).
        # Minimal version for now: single attempt, propagate errors.
        resp = self.client.models.generate_content(
            model=self.model_id, contents=contents, config=config
        )
        return resp.text or ""
```

- [ ] **Step 2: Verify existing tests still pass**

Run: `uv run pytest tests/ -x --ignore=tests/test_llm.py`
Expected: PASS

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/hippo/models/llm.py`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add src/hippo/models/llm.py
git commit -m "llm: add GeminiLLM with explicit api_key (no env mutation)"
```

---

### Task 6: Harden `GeminiLLM` retry loop (broaden exceptions, add timeout)

**Files:**
- Modify: `src/hippo/models/llm.py`
- Test: `tests/test_llm_select.py` (we'll seed it with retry tests since llm_select.py doesn't exist yet — this avoids creating a one-off test file)

- [ ] **Step 1: Create the test file with retry-loop tests**

Create `tests/test_llm_select.py`:

```python
"""Tests for select_llm() dispatch and GeminiLLM retry hardening."""
from __future__ import annotations

import pytest

from hippo.models.llm import GeminiLLM


class _FakeAPIError(Exception):
    def __init__(self, code: int) -> None:
        self.code = code
        super().__init__(f"http {code}")


class _FakeModels:
    def __init__(self, errors_to_raise: list[Exception]) -> None:
        self._errors = list(errors_to_raise)
        self.calls = 0

    def generate_content(self, *, model, contents, config):
        self.calls += 1
        if self._errors:
            raise self._errors.pop(0)
        class _Resp:
            text = "ok"
        return _Resp()


class _FakeClient:
    def __init__(self, errors_to_raise: list[Exception]) -> None:
        self.models = _FakeModels(errors_to_raise)


class TestGeminiRetry:
    def _llm(self, errors):
        return GeminiLLM(
            client=_FakeClient(errors),
            model_id="x",
            default_thinking_level="high",
            max_attempts=4,
        )

    def test_retries_on_oserror(self, monkeypatch):
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        llm = self._llm([OSError("conn reset")])
        out = llm._call_with_retry(contents="hi", config=None)
        assert out == "ok"
        assert llm.client.models.calls == 2

    def test_retries_on_timeout(self, monkeypatch):
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        llm = self._llm([TimeoutError("slow")])
        out = llm._call_with_retry(contents="hi", config=None)
        assert out == "ok"

    def test_propagates_after_max_attempts(self, monkeypatch):
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        llm = self._llm([OSError("a"), OSError("b"), OSError("c"), OSError("d")])
        with pytest.raises(OSError):
            llm._call_with_retry(contents="hi", config=None)
        assert llm.client.models.calls == 4

    def test_does_not_retry_non_retryable_apierror(self, monkeypatch):
        # Hard 400-class errors propagate immediately
        from google.genai import errors  # type: ignore[import-not-found]
        monkeypatch.setattr(
            "hippo.models.llm._sleep", lambda *_a, **_k: None, raising=False
        )
        e = errors.APIError.__new__(errors.APIError)
        e.code = 401  # type: ignore[attr-defined]
        llm = self._llm([e])
        with pytest.raises(errors.APIError):
            llm._call_with_retry(contents="hi", config=None)
        assert llm.client.models.calls == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_select.py -v`
Expected: FAIL — current `_call_with_retry` does only one attempt and only catches `errors.APIError` (it doesn't even do that yet)

- [ ] **Step 3: Implement the hardened retry loop**

In `src/hippo/models/llm.py`, replace the placeholder `_call_with_retry` from Task 5 with:

```python
import time as _time


def _sleep(seconds: float) -> None:
    """Indirection so tests can patch sleep without touching `time`."""
    _time.sleep(seconds)


@dataclass
class GeminiLLM:
    # ... (existing fields unchanged) ...

    def _call_with_retry(self, *, contents: Any, config: Any) -> str:
        from google.genai import errors
        try:
            import httpx
            _network_excs: tuple[type[BaseException], ...] = (
                OSError, TimeoutError, httpx.RequestError,
            )
        except ImportError:
            _network_excs = (OSError, TimeoutError)

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
```

(Apply this as a surgical edit to the existing class — only `_call_with_retry` and the new `_sleep`/`_time` module-level additions change.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_select.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/hippo/models/llm.py`
Expected: clean (the `pragma: no cover` raise satisfies "function returns") — if mypy complains about the unreachable, change it to `assert False, "unreachable"`.

- [ ] **Step 6: Commit**

```bash
git add src/hippo/models/llm.py tests/test_llm_select.py
git commit -m "llm: harden GeminiLLM retry (broaden exceptions, exp backoff)"
```

---

### Task 7: Implement `select_llm(strict)` using config

**Files:**
- Modify: `src/hippo/models/llm.py`
- Test: `tests/test_llm_select.py`

- [ ] **Step 1: Append failing tests**

Append to `tests/test_llm_select.py`:

```python
class TestSelectLLM:
    def _patch_loaders(self, monkeypatch):
        from hippo.models import llm as llm_mod
        local_marker = object()
        gemini_calls: list[dict] = []

        def _fake_local_load() -> object:
            return local_marker

        def _fake_gemini_load(*, api_key, model_id, default_thinking_level):
            gemini_calls.append(
                dict(api_key=api_key, model_id=model_id,
                     default_thinking_level=default_thinking_level)
            )
            return ("gemini", api_key, model_id)

        monkeypatch.setattr(llm_mod.LocalLLM, "load", staticmethod(_fake_local_load))
        monkeypatch.setattr(llm_mod.GeminiLLM, "load", staticmethod(_fake_gemini_load))
        return local_marker, gemini_calls

    def test_default_returns_local(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        local, _ = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        assert select_llm() is local

    def test_qwen_explicit(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        (tmp_path / "hippo-config.toml").write_text('backend = "qwen"\n')
        local, _ = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        assert select_llm() is local

    def test_gemini_with_env_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("GOOGLE_API_KEY", "from-env")
        (tmp_path / "hippo-config.toml").write_text(
            'backend = "gemini"\n[gemini]\nmodel_id = "m1"\n'
        )
        _, calls = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        out = select_llm()
        assert out[0] == "gemini"
        assert calls[0]["api_key"] == "from-env"
        assert calls[0]["model_id"] == "m1"

    def test_gemini_with_secrets_file_key(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
        (tmp_path / "hippo-secrets").write_text("GOOGLE_API_KEY=from-file\n")
        _, calls = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        select_llm()
        assert calls[0]["api_key"] == "from-file"

    def test_gemini_no_key_strict_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
        self._patch_loaders(monkeypatch)
        from hippo.config import ConfigError
        from hippo.models.llm import select_llm
        with pytest.raises(ConfigError, match="API key"):
            select_llm(strict=True)

    def test_gemini_no_key_nonstrict_falls_back(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
        local, _ = self._patch_loaders(monkeypatch)
        from hippo.models.llm import select_llm
        out = select_llm(strict=False)
        assert out is local
        err = capsys.readouterr().err
        assert "WARNING" in err and "qwen" in err.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_llm_select.py::TestSelectLLM -v`
Expected: FAIL — `ImportError: cannot import name 'select_llm'`

- [ ] **Step 3: Implement `select_llm`**

Append to `src/hippo/models/llm.py`:

```python
import sys as _sys


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
            print(f"WARNING: {msg} Falling back to qwen.", file=_sys.stderr)
            return LocalLLM.load()
        return GeminiLLM.load(
            api_key=key,
            model_id=cfg.gemini_model_id,
            default_thinking_level=cfg.gemini_default_thinking_level,
        )
    raise ConfigError(f"Unknown backend {cfg.backend!r}")  # pragma: no cover
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_llm_select.py -v`
Expected: PASS (10 passed total)

- [ ] **Step 5: Commit**

```bash
git add src/hippo/models/llm.py tests/test_llm_select.py
git commit -m "llm: select_llm(strict) dispatches on hippo.config"
```

---

### Task 8: Consolidate `LLMProto` — delete 6 duplicates, update imports + test fakes

**Files:**
- Modify: `src/hippo/dream/atomize.py`, `bootstrap.py`, `contradiction.py`, `edge_proposal.py`, `heavy.py`, `multi_head.py`
- Modify: `tests/test_atomize.py`, `test_bootstrap.py`, `test_contradiction.py`, `test_dream_heavy_orchestrator.py`, `test_edge_proposal.py`, `test_multi_head.py`

- [ ] **Step 1: For each of the 6 dream files, replace the local `LLMProto` with an import**

In each of `src/hippo/dream/{atomize,bootstrap,contradiction,edge_proposal,heavy,multi_head}.py`:

Delete the block:
```python
class LLMProto(Protocol):
    def generate_chat(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str: ...
```

If the file imports `Protocol` only for `LLMProto`, remove that import too. Add at the top of the file (alongside other `hippo.*` imports):
```python
from hippo.models.llm import LLMProto
```

Keep any other Protocols in the file (e.g. `DaemonProto`) intact.

- [ ] **Step 2: Update each test fake to accept `thinking_level=None`**

In each of `tests/test_{atomize,bootstrap,contradiction,dream_heavy_orchestrator,edge_proposal,multi_head}.py`:

Change the fake's signature from
```python
def generate_chat(self, messages, *, temperature, max_tokens):
```
to
```python
def generate_chat(self, messages, *, temperature, max_tokens, thinking_level=None):
```

(No body change. Each fake gains the kwarg.)

- [ ] **Step 3: Run the suite**

Run: `uv run pytest tests/ -x --ignore=tests/test_llm.py`
Expected: PASS

- [ ] **Step 4: Type-check (strict)**

Run: `uv run mypy src`
Expected: clean

- [ ] **Step 5: Lint**

Run: `uv run ruff check src tests`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/hippo/dream tests/test_atomize.py tests/test_bootstrap.py \
    tests/test_contradiction.py tests/test_dream_heavy_orchestrator.py \
    tests/test_edge_proposal.py tests/test_multi_head.py
git commit -m "dream: consolidate LLMProto into hippo.models.llm"
```

---

### Task 9: Pass `thinking_level="minimal"` in short-classification call sites

**Files:**
- Modify: `src/hippo/dream/contradiction.py:61-63`
- Modify: `src/hippo/dream/edge_proposal.py:48-52`

- [ ] **Step 1: Edit `contradiction.py`**

Replace
```python
        raw = llm.generate_chat(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=400
        )
```
with
```python
        raw = llm.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
            thinking_level="minimal",
        )
```

- [ ] **Step 2: Edit `edge_proposal.py`**

Replace
```python
                raw = llm.generate_chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                )
```
with
```python
                raw = llm.generate_chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.1,
                    max_tokens=200,
                    thinking_level="minimal",
                )
```

- [ ] **Step 3: Run the existing tests (they should still pass — fakes accept the kwarg)**

Run: `uv run pytest tests/test_contradiction.py tests/test_edge_proposal.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add src/hippo/dream/contradiction.py src/hippo/dream/edge_proposal.py
git commit -m "dream: pass thinking_level=minimal to short classification calls"
```

---

### Task 10: Add regression assertions for `thinking_level`

**Files:**
- Modify: `tests/test_contradiction.py`
- Modify: `tests/test_edge_proposal.py`

- [ ] **Step 1: Capture `thinking_level` in `test_contradiction.py`'s fake**

In `tests/test_contradiction.py`, change the `FakeLLM.generate_chat` body to also record:
```python
def generate_chat(self, messages, *, temperature, max_tokens, thinking_level=None):
    self.calls.append(messages[-1]["content"])
    self.thinking_levels.append(thinking_level)
    return self.response
```
And initialize `self.thinking_levels: list[str | None] = []` in `__init__`.

In any existing test that runs the contradiction phase end-to-end, append:
```python
assert all(level == "minimal" for level in fake.thinking_levels)
```

- [ ] **Step 2: Same change in `test_edge_proposal.py`**

Apply the same pattern to `StubLLM` / whichever fake the tests use, asserting `thinking_level == "minimal"` on every call.

- [ ] **Step 3: Run the tests**

Run: `uv run pytest tests/test_contradiction.py tests/test_edge_proposal.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_contradiction.py tests/test_edge_proposal.py
git commit -m "tests: assert thinking_level=minimal at short call sites"
```

---

### Task 11: Update `dream_heavy.py` CLI (`--strict`, `select_llm`, ConfigError handling)

**Files:**
- Modify: `src/hippo/cli/dream_heavy.py`

- [ ] **Step 1: Edit `src/hippo/cli/dream_heavy.py`**

Replace the entire contents with:

```python
"""Heavy dream entry point. Run by launchd at 3am, or manually via /hippo-dream."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from hippo.config import ConfigError
from hippo.daemon.client import DaemonClient
from hippo.dream.heavy import run_heavy_dream_all_scopes
from hippo.models.llm import select_llm
from hippo.storage.multi_store import Scope


def _is_on_ac() -> bool:
    """macOS-only: returns True if on AC power."""
    try:
        out = subprocess.check_output(["pmset", "-g", "ps"], text=True)
        return "AC Power" in out
    except Exception:
        return True


def main() -> int:
    p = argparse.ArgumentParser(prog="dream-heavy")
    p.add_argument("--force", action="store_true", help="bypass AC check")
    p.add_argument("--project", action="append", default=[])
    p.add_argument("--global-only", action="store_true")
    p.add_argument(
        "--strict",
        action="store_true",
        help="hard-fail on backend misconfiguration (default off; on for manual invocation via bin/dream-heavy)",
    )
    args = p.parse_args()

    if not args.force and not _is_on_ac():
        sys.stderr.write("Not on AC power; skipping heavy dream. Use --force to override.\n")
        return 0

    scopes: list[Scope] = []
    if not args.project or args.global_only:
        scopes.append(Scope.global_())
    for proj in args.project:
        scopes.append(Scope.project(proj))
    if args.global_only:
        scopes = [Scope.global_()]

    daemon = DaemonClient(socket_path=Path.home() / ".claude" / "memory-daemon.sock")
    try:
        llm = select_llm(strict=args.strict)
    except ConfigError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 1
    stats = run_heavy_dream_all_scopes(scopes=scopes, llm=llm, daemon=daemon)
    print("heavy dream complete:")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the dream-heavy orchestrator test**

Run: `uv run pytest tests/test_dream_heavy_orchestrator.py -v`
Expected: PASS

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/hippo/cli/dream_heavy.py`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add src/hippo/cli/dream_heavy.py
git commit -m "cli: dream_heavy uses select_llm + --strict + ConfigError handling"
```

---

### Task 12: Update `dream_bootstrap.py` CLI (`--strict`, `select_llm`, ConfigError handling)

**Files:**
- Modify: `src/hippo/cli/dream_bootstrap.py`

- [ ] **Step 1: Edit `src/hippo/cli/dream_bootstrap.py`**

Add the import (replacing `from hippo.models.llm import LocalLLM` at line 21):

```python
from hippo.config import HEAVY_LOCK_FILENAME, ConfigError
from hippo.daemon.client import DaemonClient
from hippo.dream.bootstrap import atomize_legacy_files
from hippo.dream.contradiction import resolve_contradictions
from hippo.dream.edge_proposal import propose_edges
from hippo.dream.multi_head import expand_heads_for_eligible_bodies
from hippo.lock import LockHeldError, acquire_lock, release_lock
from hippo.models.llm import select_llm
from hippo.storage.multi_store import Scope, open_store
```

In the argparse setup, add:

```python
    p.add_argument("--strict", action="store_true", help="hard-fail on backend misconfiguration")
```

In the `try:` block at line 72, replace `llm = LocalLLM.load()` with:

```python
        try:
            llm = select_llm(strict=args.strict)
        except ConfigError as exc:
            sys.stderr.write(f"ERROR: {exc}\n")
            release_lock(g_handle)
            release_lock(p_handle)
            g_store.conn.close()
            p_store.conn.close()
            return 1
```

- [ ] **Step 2: Run the bootstrap-related tests**

Run: `uv run pytest tests/test_bootstrap.py -v`
Expected: PASS

- [ ] **Step 3: Type-check**

Run: `uv run mypy src/hippo/cli/dream_bootstrap.py`
Expected: clean

- [ ] **Step 4: Commit**

```bash
git add src/hippo/cli/dream_bootstrap.py
git commit -m "cli: dream_bootstrap uses select_llm + --strict + ConfigError handling"
```

---

### Task 13: Update `bin/dream-heavy` and `bin/dream-bootstrap` to pass `--strict`

**Files:**
- Modify: `bin/dream-heavy`
- Modify: `bin/dream-bootstrap`

- [ ] **Step 1: Edit `bin/dream-heavy`**

Replace the last line:
```bash
exec uv run --project "$SCRIPT_DIR/.." --quiet python -m hippo.cli.dream_heavy "$@"
```
with:
```bash
exec uv run --project "$SCRIPT_DIR/.." --quiet python -m hippo.cli.dream_heavy --strict "$@"
```

- [ ] **Step 2: Edit `bin/dream-bootstrap`**

Same change for `bin/dream-bootstrap` (substitute `dream_bootstrap` for `dream_heavy`).

- [ ] **Step 3: Smoke-test the shim**

Run: `bin/dream-heavy --help`
Expected: argparse usage output that includes `--strict` (the shim adds the flag, then --help short-circuits before doing real work).

- [ ] **Step 4: Commit**

```bash
git add bin/dream-heavy bin/dream-bootstrap
git commit -m "bin: shims pass --strict so manual invocation hard-fails on misconfig"
```

---

### Task 14: Make `google-genai` an optional extra in `pyproject.toml`

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (regenerated)

- [ ] **Step 1: Edit `pyproject.toml`**

Remove `"google-genai>=1.0",` from `[project] dependencies`. Add (or extend) the following block:

```toml
[project.optional-dependencies]
gemini = [
    "google-genai>=1.0",
]
```

- [ ] **Step 2: Regenerate the lockfile**

Run: `uv sync --extra gemini`
Expected: completes; `uv.lock` updates with the optional extra recorded.

- [ ] **Step 3: Verify `uv sync` (no extra) doesn't pull `google-genai`**

Run: `uv sync && uv run python -c "import google.genai; print('present')"`
Expected: the second command may either succeed (if the extra is still in the venv from step 2) or fail (if `uv sync` removed it). Either is acceptable for the lock; the important assertion is that fresh installers using `uv sync` without `--extra gemini` will not get the package.

- [ ] **Step 4: Verify `select_llm()` raises a clean error when gemini selected but package missing**

This is covered by `test_llm_select.py::TestGeminiLoad`-equivalent. Add the case if not already present:

```python
def test_gemini_missing_dep_raises_configerror(monkeypatch, tmp_path):
    monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
    monkeypatch.setenv("GOOGLE_API_KEY", "x")
    (tmp_path / "hippo-config.toml").write_text('backend = "gemini"\n')
    # Force ImportError in GeminiLLM.load
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "google", None)
    from hippo.config import ConfigError
    from hippo.models.llm import select_llm
    with pytest.raises(ConfigError, match="uv sync --extra gemini"):
        select_llm(strict=True)
```

Run: `uv run pytest tests/test_llm_select.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock tests/test_llm_select.py
git commit -m "deps: google-genai becomes [optional-dependencies] gemini"
```

---

### Task 15: Implement `backend_toggle.py` (status + qwen + gemini subcommands)

**Files:**
- Create: `src/hippo/cli/backend_toggle.py`
- Test: `tests/test_backend_toggle.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_backend_toggle.py`:

```python
"""Tests for the /hippo-backend slash-command CLI module."""
from __future__ import annotations

import pytest

from hippo.cli import backend_toggle


class TestStatus:
    def test_status_default(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        rc = backend_toggle.main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "backend: qwen" in out
        assert "api_key: not detected" in out

    def test_status_with_env_key(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("GOOGLE_API_KEY", "secret")
        rc = backend_toggle.main([])
        assert rc == 0
        out = capsys.readouterr().out
        assert "api_key: detected (env)" in out


class TestSwitch:
    def test_switch_to_qwen(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        rc = backend_toggle.main(["qwen"])
        assert rc == 0
        from hippo.config import load_config
        assert load_config().backend == "qwen"
        out = capsys.readouterr().out
        assert "switched to qwen" in out.lower()

    def test_switch_to_gemini_with_key(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        rc = backend_toggle.main(["gemini"])
        assert rc == 0
        from hippo.config import load_config
        assert load_config().backend == "gemini"

    def test_switch_to_gemini_without_key_warns(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        rc = backend_toggle.main(["gemini"])
        assert rc == 0  # write succeeds
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
        assert "no API key" in captured.err

    def test_invalid_backend(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setenv("HIPPO_CONFIG_DIR", str(tmp_path))
        with pytest.raises(SystemExit) as exc_info:
            backend_toggle.main(["pigeon"])
        assert exc_info.value.code == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backend_toggle.py -v`
Expected: FAIL — `ModuleNotFoundError: hippo.cli.backend_toggle`

- [ ] **Step 3: Create `src/hippo/cli/backend_toggle.py`**

```python
"""Slash-command CLI: /hippo-backend [qwen|gemini]"""
from __future__ import annotations

import argparse
import os
import sys

from hippo.config import (
    Config,
    config_path,
    load_api_key,
    load_config,
    secrets_path,
    write_config,
)


def _print_status() -> int:
    cfg = load_config()
    key = load_api_key()
    if key:
        if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
            key_status = "detected (env)"
        else:
            key_status = "detected (secrets file)"
    else:
        key_status = "not detected"
    print(f"backend: {cfg.backend}")
    print(f"gemini.model_id: {cfg.gemini_model_id}")
    print(f"gemini.default_thinking_level: {cfg.gemini_default_thinking_level}")
    print(f"api_key: {key_status}")
    print(f"config_path: {config_path()}")
    print(f"secrets_path: {secrets_path()}")
    print("logs (silent fallback warning): ~/.claude/debug/dream-heavy.err")
    return 0


def _switch(backend: str) -> int:
    current = load_config()
    new_cfg = Config(
        backend=backend,
        gemini_model_id=current.gemini_model_id,
        gemini_default_thinking_level=current.gemini_default_thinking_level,
    )
    write_config(new_cfg)
    print(f"switched to {backend} (config: {config_path()})")
    if backend == "gemini" and not load_api_key():
        print(
            "WARNING: no API key detected. Set GOOGLE_API_KEY in env or write to "
            f"{secrets_path()} (mode 600).",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hippo-backend")
    p.add_argument("backend", nargs="?", choices=["qwen", "gemini"])
    args = p.parse_args(argv)
    if args.backend is None:
        return _print_status()
    return _switch(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backend_toggle.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Type-check**

Run: `uv run mypy src/hippo/cli/backend_toggle.py`
Expected: clean

- [ ] **Step 6: Commit**

```bash
git add src/hippo/cli/backend_toggle.py tests/test_backend_toggle.py
git commit -m "cli: backend_toggle (status + qwen/gemini switch)"
```

---

### Task 16: Create slash command file `~/.claude/commands/hippo-backend.md`

**Files:**
- Create: `~/.claude/commands/hippo-backend.md`

This file lives outside the repo (under `~/.claude/`). It's a one-time setup, not committed.

- [ ] **Step 1: Resolve the absolute repo path**

Run: `echo "$(cd /Users/keon/code/hippo && pwd)"`
Expected: `/Users/keon/code/hippo`

- [ ] **Step 2: Write the slash command file**

Create `~/.claude/commands/hippo-backend.md`:

```markdown
---
description: Show or switch the Hippo LLM backend (qwen | gemini)
allowed-tools: Bash
---

Run the backend toggle:

!`uv run --project /Users/keon/code/hippo --quiet python -m hippo.cli.backend_toggle $ARGUMENTS`
```

- [ ] **Step 3: Test the slash command flows**

Run: `uv run --project /Users/keon/code/hippo --quiet python -m hippo.cli.backend_toggle`
Expected: status output (backend, model_id, api_key, paths, log location)

Run: `uv run --project /Users/keon/code/hippo --quiet python -m hippo.cli.backend_toggle qwen`
Expected: "switched to qwen" + config file written

- [ ] **Step 4: Verify the config file**

Run: `cat ~/.claude/hippo-config.toml`
Expected: valid TOML with `backend = "qwen"` (or `"gemini"` if you switched)

- [ ] **Step 5: No commit**

This file is in `~/.claude/`, not the repo. Skip.

---

### Task 17: Drop the stash, update docs

**Files:**
- Delete entry: `KNOWN_ISSUES.md` (the gemini-stash paragraph)
- Modify: `docs/operations.md` (one paragraph inside an existing section)
- Modify: `CLAUDE.md` (one line in existing section)
- Modify: `README.md` (one bullet in existing setup section)
- Drop: `stash@{0}`

- [ ] **Step 1: Edit `KNOWN_ISSUES.md`**

Open `/Users/keon/code/hippo/KNOWN_ISSUES.md` and delete the entire paragraph that begins around line 117 with the heading about the preserved Gemini stash. Do not add anything in its place.

Run: `grep -n "stash\|gemini" KNOWN_ISSUES.md` afterward to verify nothing about the stash remains.

- [ ] **Step 2: Edit `docs/operations.md`**

Find the existing section on heavy dream operations. Append (inside that section, no new heading) one paragraph:

```markdown
**Switching LLM backend.** Use `/hippo-backend` to view current backend
and key status, `/hippo-backend qwen` or `/hippo-backend gemini` to
switch. Choice persists in `~/.claude/hippo-config.toml`. Gemini also
needs a key in `GOOGLE_API_KEY`/`GEMINI_API_KEY` or `~/.claude/hippo-secrets`
(mode 600). On the launchd nightly run, missing key falls back to qwen
silently and writes a warning to `~/.claude/debug/dream-heavy.err`. See
`src/hippo/config.py` for current defaults.
```

- [ ] **Step 3: Edit `CLAUDE.md`**

Find the existing "Running services on this machine" section. Append at the end of that section (no new subsection):

```markdown
- LLM backend: switch with `/hippo-backend`; default qwen; see
  `~/.claude/hippo-config.toml` and `src/hippo/config.py`.
```

- [ ] **Step 4: Edit `README.md`**

Find the existing setup-instructions section. Append one bullet:

```markdown
- Optional: `uv sync --extra gemini` for cloud Gemini backend.
```

- [ ] **Step 5: Run all gates one last time**

Run in parallel:
```bash
uv run pytest
uv run ruff check src tests
uv run mypy src
```
Expected: all clean.

- [ ] **Step 6: Commit doc changes**

```bash
git add KNOWN_ISSUES.md docs/operations.md CLAUDE.md README.md
git commit -m "docs: switching backend, drop gemini-stash known issue"
```

- [ ] **Step 7: Drop the stash**

Run: `git stash drop 'stash@{0}'`
Expected: "Dropped stash@{0} (sha)"

- [ ] **Step 8: Final smoke**

Run:
```bash
git stash list
git log --oneline -8
```
Expected: stash list is empty (or no `gemini-backend-wip` entries); git log shows the new commits in order.

---

## Self-Review Notes

The plan covers all spec sections:

- §Architecture: Tasks 1-3 (config + secrets), Task 6-7 (select_llm), Tasks 11-13 (CLI + bin), Tasks 15-16 (slash command)
- §Components/`hippo.config`: Tasks 1-3
- §Components/`hippo.models.llm`: Tasks 4-7, with retry hardening at Task 6
- §Components/CLI entry points: Tasks 11-12 + 13 for shims
- §Components/`backend_toggle.py`: Task 15
- §Components/slash command: Task 16
- §Components/`pyproject.toml`: Task 14
- §Components/LLMProto consolidation: Task 8
- §Failure modes: covered by tests in Tasks 2, 3, 6, 7, 14, 15
- §Testing: new files at Tasks 1-3, 6-7, 14, 15; regressions at Task 10
- §Documentation: Task 17
- §Install impact: nothing to do (intentionally)
- §Migration: Task 17 step 7 drops the stash

No placeholders detected. Type names consistent (`Config`, `ConfigError`, `LLMProto`, `LocalLLM`, `GeminiLLM`, `select_llm`).
