"""Slash-command CLI: /hippo-backend [local|gemini]

"qwen" is accepted as a legacy alias for "local" so existing muscle memory
and config files keep working.
"""
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

_BACKEND_ALIASES: dict[str, str] = {"qwen": "local"}


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
    backend = _BACKEND_ALIASES.get(backend, backend)
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
    p.add_argument("backend", nargs="?", choices=["local", "qwen", "gemini"])
    args = p.parse_args(argv)
    if args.backend is None:
        return _print_status()
    return _switch(args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
