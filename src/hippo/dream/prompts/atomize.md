You are extracting durable memory atoms from a Claude Code session transcript.

A "body" is one coherent piece of content that should be remembered. A "head" is a short keyword sentence that someone might use to recall this body. Bodies can have multiple heads (different angles into the same content).

Read the transcript below. Output a JSON array of atom objects. Each atom has the shape:
{
  "title": "short title (under 60 chars)",
  "body": "full content — can be a single fact, a paragraph of reasoning, or a long write-up. Whatever serves the concept.",
  "scope": "global" | "project:{{project}}",
  "heads": ["1-2 sentence keyword summary 1", "1-2 sentence keyword summary 2", ...]
}

Rules:
- ONLY extract genuinely durable facts: rules, decisions, learnings, preferences, important references. Skip in-the-moment chatter, debugging steps that don't generalize, or one-off task acknowledgments.
- Each atom must have at least 1 head and at most 5.
- Heads must be diverse — they capture different angles of the body, not paraphrases.
- "scope" = "global" if the atom applies regardless of project (user preferences, role, cross-project insights). Otherwise "project:{{project}}".
- If nothing in the transcript is worth remembering, return [].

Return ONLY the JSON array. No prose, no markdown fences.

---

PROJECT: {{project}}
SESSION: {{session_id}}
TRANSCRIPT:
{{transcript}}
