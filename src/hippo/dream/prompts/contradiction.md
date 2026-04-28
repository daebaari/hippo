You are deciding whether two memory atoms genuinely contradict, and if so, which is current.

Atom A (body_id: {{a_body_id}}, updated: {{a_updated}}):
{{a_body}}

Atom B (body_id: {{b_body_id}}, updated: {{b_updated}}):
{{b_body}}

Do these atoms make incompatible claims about the same subject?

If yes, which one is current (i.e., which should be kept and which archived)? Use:
- the more recent updated_at
- the more specific or detailed content
- explicit supersession language ("we used to ... but now ...")

Output a single JSON object:
{
  "contradicts": true | false,
  "current_body_id": "<a_body_id or b_body_id, only if contradicts is true>",
  "reason": "<one sentence>"
}

Return ONLY the JSON.
