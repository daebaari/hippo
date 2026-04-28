You are deciding whether two memory bodies cover the same concept and should be merged.

Body A (title: "{{a_title}}"):
{{a_body}}

Body B (title: "{{b_title}}"):
{{b_body}}

Do these cover the same underlying concept (different phrasings, complementary detail, or near-duplicates)?

If yes, output a single merged body that incorporates both. The merged body should preserve nuance from both, deduplicate redundant claims, and stay under {{max_chars}} characters.

Output a single JSON object:
{
  "merge": true | false,
  "merged_title": "...",
  "merged_body": "...",
  "reason": "<one sentence>"
}

Return ONLY the JSON.
