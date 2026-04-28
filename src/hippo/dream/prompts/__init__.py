"""Load prompt templates by name from this directory."""
from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def load_template(name: str) -> str:
    return (_DIR / f"{name}.md").read_text()


def render(template_name: str, **vars: object) -> str:
    """Simple {{var}} substitution. Doesn't try to be Jinja."""
    s = load_template(template_name)
    for k, v in vars.items():
        s = s.replace("{{" + k + "}}", str(v))
    return s
