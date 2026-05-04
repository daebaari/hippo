"""Benchmark atomize-task quality and latency across LLM backends.

Atomize is the most reasoning-heavy Hippo phase: it reads a multi-turn
transcript and emits a JSON array of "atom" objects (durable knowledge
chunks vs noise).

Unlike edge classification, atomize has no clean accuracy metric — output
is a variable-length list. We measure:
- Validity: did the model produce a parseable JSON array conforming to schema?
- Atom count and noise breakdown
- Latency
- Raw output snippet for human review

Run:
    uv run python scripts/bench_atomize.py
    uv run python scripts/bench_atomize.py --backends qwen3.5,gemini
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


# === Atomize prompt (mirrors src/hippo/dream/prompts/atomize.md) ===
ATOMIZE_PROMPT = '''You are extracting durable memory atoms from a Claude Code session transcript.

A "body" is one coherent piece of content that should be remembered. A "head" is a short keyword sentence that someone might use to recall this body. Bodies can have multiple heads (different angles into the same content).

Read the transcript below. Output a JSON array of atom objects. Each atom has the shape:
{{
  "title": "short title (under 60 chars)",
  "body": "full content",
  "scope": "global" | "project:{project}",
  "heads": ["1-2 sentence keyword summary 1", "1-2 sentence keyword summary 2"],
  "noise": true | false
}}

Rules for the `noise` field:
- noise=false: durable knowledge worth recalling (decisions, preferences, project facts, patterns, bug-fix learnings, spec statements).
- noise=true: in-the-moment chatter that does NOT generalize (terse procedural turns, acknowledgments, debug-loop chatter, trivial confirmations).
- Tiebreaker: if uncertain, output noise=true.

Other rules:
- Each atom must have at least 1 head and at most 5.
- Heads must be diverse — different angles of the body, not paraphrases.
- "scope" = "global" if the atom applies regardless of project. Otherwise "project:{project}".
- If nothing in the transcript is worth remembering, return [].

Return ONLY the JSON array. No prose, no markdown fences.

---

PROJECT: {project}
SESSION: {session_id}
TRANSCRIPT:
{transcript}'''


# === Test transcripts ===
@dataclass
class AtomizeFixture:
    name: str
    transcript: str
    expected_durable_atoms: int       # rough count of non-noise atoms expected
    expected_max_atoms: int            # upper bound for total atoms
    notes: str                         # what a good extractor should produce

FIXTURES: list[AtomizeFixture] = [
    AtomizeFixture(
        name='knowledge_rich',
        transcript='''USER: what is the cluster cosine threshold and why does it matter?
ASSISTANT: It's a configuration constant in src/hippo/config.py that controls how aggressively heads are grouped together by embedding similarity during edge proposal. Lower values produce larger clusters with more candidate pairs to evaluate per cluster. Higher values produce smaller, tighter clusters but may miss some legitimate edges between related-but-not-near-duplicate heads.
USER: so if i lower it, what happens to runtime?
ASSISTANT: Runtime grows roughly with the square of the cluster size, because edge proposal evaluates every within-cluster pair via an LLM call. So halving the threshold can quadruple the LLM call count if it doubles average cluster size. There's also a memory cost: cluster construction holds head embeddings in process memory.
USER: ok, makes sense. thanks.''',
        expected_durable_atoms=2,
        expected_max_atoms=4,
        notes='Two clear durable facts: (1) what cluster threshold controls, (2) runtime scaling. Final "thanks" should be noise.',
    ),
    AtomizeFixture(
        name='noise_only',
        transcript='''USER: status
ASSISTANT: Running.
USER: again
ASSISTANT: Still running.
USER: ok
ASSISTANT: Done.
USER: thanks
ASSISTANT: You're welcome.''',
        expected_durable_atoms=0,
        expected_max_atoms=2,
        notes='Pure procedural chatter. Should output [] or all noise=true.',
    ),
    AtomizeFixture(
        name='bugfix_mixed',
        transcript='''USER: hippo captures are showing up under the wrong project name
ASSISTANT: Let me check. Reading _resolve_project in capture/userprompt_hook.py — it walks up from cwd looking for .git or CLAUDE.md and uses the basename.
USER: yeah but i'm in a worktree
ASSISTANT: Right, that's the bug. In a git worktree, .git is a file not a directory — it contains a "gitdir:" pointer to the actual repo's worktree subdir. _resolve_project sees the worktree's .git file first and returns the worktree directory name as the project, not the main repo.
USER: ok fix it
ASSISTANT: Done. Updated _resolve_project to detect worktree pointers (.git as file with "gitdir:" prefix) and follow them to find the actual repo root, then return that basename as the project name.
USER: ok thanks''',
        expected_durable_atoms=1,
        expected_max_atoms=3,
        notes='One durable bug-fix learning (worktree .git is a file with gitdir: pointer; _resolve_project must follow it). Initial debug exchanges and "ok thanks" should be noise.',
    ),
]


def parse_atomize_output(raw: str) -> tuple[Any, str]:
    """Returns (parsed_obj_or_None, status_message). Tolerates fences and channel markers."""
    text = raw
    if '<channel|>' in text:
        text = text.split('<channel|>', 1)[1]
    text = re.sub(r'```(?:json)?\s*', '', text)
    text = text.replace('```', '')
    text = text.strip()
    # Empty array case
    if text == '[]':
        return [], 'empty_array'
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj, 'ok'
        return None, f'not_a_list: {type(obj).__name__}'
    except json.JSONDecodeError as e:
        # Try extracting [...]
        m = re.search(r'\[.*\]', text, re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    return obj, 'ok_after_extract'
            except json.JSONDecodeError:
                pass
        return None, f'json_error: {str(e)[:60]}'


# === Backends (mirror bench_relation_classifiers.py shape) ===
class Backend(Protocol):
    name: str
    def setup(self) -> None: ...
    def generate(self, prompt: str, max_tokens: int) -> tuple[str, float]: ...
    def teardown(self) -> None: ...


@dataclass
class QwenBackend:
    name: str = 'qwen-2.5-32b-mlx-4bit'
    llm: Any = None
    def setup(self) -> None:
        from hippo.models.llm import LocalLLM
        self.llm = LocalLLM.load()
    def generate(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        t0 = time.time()
        raw = self.llm.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=max_tokens,
        )
        return raw, time.time() - t0
    def teardown(self) -> None:
        self.llm = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


@dataclass
class Qwen35Backend:
    MODEL_ID: str = 'mlx-community/Qwen3.5-9B-4bit'
    name: str = 'qwen-3.5-9b-mlx-4bit'
    model: Any = None
    tokenizer: Any = None
    def setup(self) -> None:
        from mlx_lm import load
        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]
    def generate(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        try:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt + " /no_think"}],
                tokenize=False, add_generation_prompt=True,
            )
        t0 = time.time()
        raw = generate(self.model, self.tokenizer, prompt=chat,
                       max_tokens=max_tokens,
                       sampler=make_sampler(temp=0.2), verbose=False)
        return raw, time.time() - t0
    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


@dataclass
class GemmaBackend:
    MODEL_ID: str = 'lmstudio-community/gemma-4-26B-A4B-it-MLX-4bit'
    name: str = 'gemma-4-26b-moe-mlx-4bit'
    model: Any = None
    tokenizer: Any = None
    def setup(self) -> None:
        from mlx_lm import load
        result = load(self.MODEL_ID)
        self.model, self.tokenizer = result[0], result[1]
    def generate(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        t0 = time.time()
        raw = generate(self.model, self.tokenizer, prompt=chat,
                       max_tokens=max_tokens,
                       sampler=make_sampler(temp=0.2), verbose=False)
        return raw, time.time() - t0
    def teardown(self) -> None:
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import mlx.core as mx
            mx.metal.clear_cache()
        except Exception:
            pass


@dataclass
class GeminiBackend:
    name: str = 'gemini-3-flash'
    llm: Any = None
    def setup(self) -> None:
        from hippo.config import load_api_key, load_config
        from hippo.models.llm import GeminiLLM
        cfg = load_config()
        key = load_api_key()
        if not key:
            raise RuntimeError("No Gemini API key.")
        self.llm = GeminiLLM.load(
            api_key=key, model_id=cfg.gemini_model_id,
            default_thinking_level='minimal',
        )
    def generate(self, prompt: str, max_tokens: int) -> tuple[str, float]:
        t0 = time.time()
        raw = self.llm.generate_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.2, max_tokens=max_tokens, thinking_level='minimal',
        )
        return raw, time.time() - t0
    def teardown(self) -> None:
        self.llm = None
        gc.collect()


BACKENDS: dict[str, Callable[[], Backend]] = {
    'qwen':    QwenBackend,
    'qwen3.5': Qwen35Backend,
    'gemma':   GemmaBackend,
    'gemini':  GeminiBackend,
}


# === Runner ===
@dataclass
class AtomizeResult:
    backend: str
    fixture: str
    status: str
    n_atoms: int
    n_noise: int
    n_durable: int
    latency_s: float
    raw_snippet: str
    raw_full: str = ''


def run_one(backend: Backend, fixture: AtomizeFixture) -> AtomizeResult:
    prompt = ATOMIZE_PROMPT.format(
        project='hippo',
        session_id='bench-' + fixture.name,
        transcript=fixture.transcript,
    )
    raw, lat = backend.generate(prompt, max_tokens=1500)
    parsed, status = parse_atomize_output(raw)
    if parsed is None:
        return AtomizeResult(
            backend=backend.name, fixture=fixture.name, status=status,
            n_atoms=0, n_noise=0, n_durable=0, latency_s=lat,
            raw_snippet=raw[:120].replace('\n', ' '), raw_full=raw,
        )
    n_noise = sum(1 for a in parsed if isinstance(a, dict) and a.get('noise') is True)
    n_durable = sum(1 for a in parsed if isinstance(a, dict) and a.get('noise') is False)
    return AtomizeResult(
        backend=backend.name, fixture=fixture.name, status=status,
        n_atoms=len(parsed), n_noise=n_noise, n_durable=n_durable,
        latency_s=lat,
        raw_snippet=raw[:120].replace('\n', ' '), raw_full=raw,
    )


def report(results: list[AtomizeResult]) -> None:
    print("\n" + "=" * 100)
    print("ATOMIZE RESULTS")
    print("=" * 100)
    by_backend: dict[str, list[AtomizeResult]] = {}
    for r in results:
        by_backend.setdefault(r.backend, []).append(r)

    print(f"\n{'backend':<26} | {'fixture':<16} | {'status':<22} | {'atoms':>5} | {'noise':>5} | {'durable':>7} | {'lat':>7} | expected")
    print('-' * 130)
    for backend_name, rows in by_backend.items():
        for r in rows:
            fx = next(f for f in FIXTURES if f.name == r.fixture)
            expected = f"{fx.expected_durable_atoms} durable, ≤{fx.expected_max_atoms} total"
            print(f"{backend_name:<26} | {r.fixture:<16} | {r.status:<22} | {r.n_atoms:>5} | {r.n_noise:>5} | {r.n_durable:>7} | {r.latency_s:6.2f}s | {expected}")

    print("\nMean latency per backend:")
    for backend_name, rows in by_backend.items():
        lats = [r.latency_s for r in rows]
        print(f"  {backend_name:<26} mean={statistics.mean(lats):.2f}s  total={sum(lats):.1f}s")

    print("\n--- Raw outputs for human review ---")
    for r in results:
        print(f"\n=== {r.backend} on {r.fixture} ({r.status}, {r.latency_s:.1f}s) ===")
        print(r.raw_full[:2000])
        if len(r.raw_full) > 2000:
            print(f"... [truncated {len(r.raw_full) - 2000} more chars]")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--backends', default='gemini,qwen3.5,qwen,gemma')
    p.add_argument('--out', default='/tmp/bench_atomize.json')
    args = p.parse_args()

    selected = [b.strip() for b in args.backends.split(',') if b.strip()]
    for b in selected:
        if b not in BACKENDS:
            raise SystemExit(f"unknown backend: {b}. options: {list(BACKENDS)}")

    all_results: list[AtomizeResult] = []
    for b in selected:
        backend = BACKENDS[b]()
        print(f"\n=== Setting up {backend.name} ===", flush=True)
        try:
            backend.setup()
        except Exception as e:
            print(f"  setup failed: {e}", flush=True)
            continue
        for fixture in FIXTURES:
            print(f"  running {fixture.name}...", flush=True)
            try:
                result = run_one(backend, fixture)
            except Exception as e:
                result = AtomizeResult(
                    backend=backend.name, fixture=fixture.name,
                    status=f'exception: {str(e)[:50]}',
                    n_atoms=0, n_noise=0, n_durable=0, latency_s=0.0,
                    raw_snippet='', raw_full='',
                )
            all_results.append(result)
            print(f"    {result.status:<22} atoms={result.n_atoms} (noise={result.n_noise}, durable={result.n_durable}) lat={result.latency_s:.1f}s", flush=True)
        backend.teardown()

    report(all_results)
    with open(args.out, 'w') as f:
        json.dump({'fixtures': [{'name': f.name, 'expected_durable_atoms': f.expected_durable_atoms, 'expected_max_atoms': f.expected_max_atoms, 'notes': f.notes} for f in FIXTURES],
                   'results': [vars(r) for r in all_results]}, f, indent=2)
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()
