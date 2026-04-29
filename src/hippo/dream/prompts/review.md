You are deciding whether two memory atoms are redundant or whether one supersedes the other, and if so which to keep.

Atom A (body_id: {{a_body_id}}, updated: {{a_updated}}):
{{a_body}}

Atom B (body_id: {{b_body_id}}, updated: {{b_updated}}):
{{b_body}}

Choose ONE decision:
- "merge": A and B describe the same fact from slightly different angles. Pick whichever is better-written or more complete; the other can be archived as redundant.
- "supersede": A and B describe the same subject but one is a newer correct version that replaces the other (e.g., "we used to use X but now use Y", or A's `updated` is recent and reflects a changed reality). Keep the current one; archive the outdated one.
- "keep_both": A and B are distinct atoms that should both stay (different facts, different scopes, complementary information, or only superficially similar).

When picking the keeper:
- Prefer the more recent `updated`.
- Prefer the more specific or detailed body.
- Prefer explicit supersession language ("we now ...", "as of ...").
- If both are equally good, prefer A.

Output a single JSON object:
{
  "decision": "merge" | "supersede" | "keep_both",
  "keeper": "<a_body_id or b_body_id, omitted or null when keep_both>",
  "reason": "<one sentence>"
}

Return ONLY the JSON.
