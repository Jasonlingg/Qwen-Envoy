# Atomic passage gap — scripted MCP/Docker example

**Synthetic responses and synthetic paper fixture, no live model output.** This
is the saved result of a real MCP client/server test using network-disabled
Docker for document Python. It does not evaluate Qwen's ability to follow the
new prompt or judge whether a real paper supports an answer.

The test reproduces the earlier ordering trap and recovers within four turns:

1. Inspect a passage through Docker.
2. Inspect a distinct subrange through Docker.
3. Try a separate Unanswerable submission; receive feedback to use `submit_gap`.
4. Call `submit_gap(missing=..., follow_up_query=...)`; save the gap and finish.

The ordinary path takes three turns: two reads and the terminal gap. The control
action is parsed as bounded literal data, not executed as Python. It contributes
no source inspection, launches no search and produces no accepted factual answer.
The model's gap report is unreviewed and does not prove absence from the paper
or literature. Inspected source observations remain available in the report.

- [Actual scripted MCP response and controller trajectory](scripted-report.json)
- [Persisted note, including missing evidence and unexecuted next query](saved-note.json)
- [New effective prompt](effective-prompt.txt)
- [Unchanged legacy prompt hashes](legacy-prompt-hashes.json)
- [Validation](validation.json)

The same note/request was retried through MCP and after service restart without
another policy call. The optional experiment MCP selects `passages-v2`; the
earlier baseline, focused and `passages-v1` prompts and base/V5 comparison records
remain unchanged. The four-request budget is unchanged. No training, reward
change, new GPU activation or paid inference is included.

[Implementation scope](../../docs/implementation/2026-10-22-atomic-passage-gap.md)
and [usage](../../docs/PASSAGE_READER.md).
