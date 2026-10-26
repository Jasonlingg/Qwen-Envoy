# Research MCP paired decision pilot — saved evidence

This bundle supports the [predeclared protocol and results](../../docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md).
The protocol was committed as `092a647` before either arm ran on October 26,
2026. `summary.json` is the compact per-question score table. These are saved
assistant runs and model-assisted reviews, not independent human judgments.

| File | Purpose |
| --- | --- |
| `web_primary_answers.json`, `web_notes/` | Web-plus-Markdown arm, using the compliant replacement W1 and original W2–W5 |
| `web_original_answers.json`, `web_w1_replacement.json` | Original over-budget W1 and its fresh replacement, both kept for audit |
| `mcp_answers.json`, `mcp_records/` | MCP arm's answers and exported experiment/source records |
| `mcp_call_events.json` | MCP call arguments, timing and caller failures without the large raw responses |
| `blind_cases.json`, `blind_claim_review.json`, `unblinding_mapping.json` | Anonymized answer pack, independent assistant source review and hidden-arm mapping |
| `web_recovered.json`, `mcp_recovered.json`, `*_recovery_metadata.json` | Fresh-session decision recovery and tool-use notes |
| `blind_recovery_cases.json`, `blind_recovery_review.json` | Anonymized recovery pack and separate assistant review |
| `web_timing_note.json` | Limits of web-arm timing |
| `summary.json` | Per-case and aggregate results, gates and deviations |
| `MANIFEST.sha256` | Content hashes for these saved artifacts |

The `note_path` and `log_path` values inside original answer JSON are historical
temporary paths from the live run. Use the bundled `web_notes/` and
`mcp_call_events.json` instead. The MCP record export contains the saved
records used in this run; it is not a complete mirror of the arXiv source cache.
The independent assistant review checked cited sources during the pilot, and
the review file records its source URLs and judgments. The source-byte receipts
inside MCP records prove span provenance, not semantic support.

The web arm's original W1 used four searches, so the primary analysis uses a
fresh three-search W1. The MCP arm's first two searches for each question
executed but their responses were lost when its caller accessed camelCase SDK
attributes. All ten still count toward the three-search-per-question budget.
No paid model, GPU, or training run was used in this pilot.
