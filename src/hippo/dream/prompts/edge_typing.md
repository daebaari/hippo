You are deciding whether two memory heads are related, and if so, by what relation.

Head A: "{{head_a}}"
Head B: "{{head_b}}"

(Both heads point to bodies in the same memory store. They surfaced as candidates because their embeddings are similar.)

Possible relations:
- "causes" — A causes B (or B is a consequence of A). Asymmetric.
- "supersedes" — A replaces B as the current rule. Asymmetric.
- "contradicts" — A and B make incompatible claims. Symmetric.
- "applies_when" — B is a condition under which A applies. Asymmetric (A applies_when B).
- "related" — meaningfully connected but no specific relation. Symmetric.
- "none" — not actually related; the embedding similarity was misleading.

Output a single JSON object: {"relation": "<one of the above>", "weight": <float 0-1>, "reason": "<one sentence>"}. Return ONLY the JSON.
