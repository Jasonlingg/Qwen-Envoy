# Qwen experiment as a learning-memory thread

This demo uses the [public September 25 evaluation report](../reports/qasper-scale-sft-2026-09-25.json),
not invented checkpoint results. A deterministic builder turns its fields into three linked
Obsidian notes: a **retrospective reconstruction** of the continuation strategy and reported
promotion rule, the measured result, and the rejection decision reported by the source. Each note
records the report path and SHA-256. None is marked as a user-approved memory revision.

The report compares three **SFT checkpoints on the same reserved 40 QASPER questions**: the
starting adapter passed 15, epoch 1 passed 10, and epoch 2 passed 4 under identity-blind,
**model-assisted** semantic review. Both continuation checkpoints were rejected. The source does
not prove when the promotion rule was set, and this is neither a base-Qwen comparison nor a
personal-vault result. The earlier 9/40-to-19/40 improvement used a different question set and
must not be folded into this three-arm result.

## Try the browser thread

Install the viewer dependencies, then launch a disposable, local-only copy:

```bash
python -m pip install -e '.[viewer]'
python scripts/launch_qwen_experiment_demo.py
```

Open the printed URL (default `http://127.0.0.1:8766`). Under **Recent result notes**, open
**Qwen QASPER continuation measured result**. The local impact review should show the
reconstructed earlier strategy and the report-derived result. Inspect both full notes and the
reported decision before drawing a conclusion. A proposed correction is only a scaffold; replace
it with your own interpretation and approve it explicitly if you want a new linked note. That
revision supersedes the *next-step strategy*, not the validity of the promotion rule or the
historical run. The launcher deletes the disposable copy on exit.

Ask the chat, for example, “Why were the Qwen QASPER continuation checkpoints rejected?” In
this launcher the chat returns **exact source passages only** through the lexical baseline. It
does not run Qwen or Nemotron or spend GPU/API credits. A larger assistant can synthesize an answer
from those passages, but the local browser does not pretend that retrieval alone is a generated
answer.

Two live Nemotron chat turns were also run against **only these public notes**. Their saved
[first](../reports/qwen_experiment_nemotron_smoke_20260927.json) and
[second](../reports/qwen_experiment_nemotron_smoke_20260928.json) reports include served model,
source passages, token usage, and a source-by-source Codex review. The first turn misstated the
provenance of the promotion rule. After clarifying the imported note, the second got the rejection
and tradeoff right but grouped citations at the end and omitted a factual sentence from its
structured claims. Both are marked **partial**, not a semantic-quality pass. A new citation guard
rejects the second saved answer on local replay, returning to evidence-only mode rather than
presenting that format as fully cited. The guard checks citation placement, not claim meaning.

To keep an editable vault after the server exits, build it at a new, empty path and launch the
ordinary web app:

```bash
python scripts/build_qwen_experiment_vault.py --output out/qwen-experiment-vault
python scripts/personal_memory_web.py \
  --vault out/qwen-experiment-vault --state-dir out/qwen-experiment-state --port 8766
```

The builder refuses to overwrite existing notes. To see native backlinks and the graph, open
`out/qwen-experiment-vault` in Obsidian. A human-approved revision uses the ordinary app path and
will be a fourth, append-only note; the public report-derived notes remain unchanged.

## Query the same evidence through MCP

Freeze the vault **after** any approved revision, then start the read-only stdio server:

```bash
python scripts/personal_memory.py snapshot \
  --vault out/qwen-experiment-vault --output out/qwen-experiment-snapshot
python scripts/personal_memory_mcp.py --snapshot out/qwen-experiment-snapshot
```

From an MCP host, call `search_memory` with a specific question such as “Qwen QASPER continuation
checkpoint rejection,” then `get_memory_source` using a returned `review_id` and `doc_id`. This
MCP server exposes source passages, not a model answer; its snapshot stays fixed until rebuilt and
the server restarted. The [local memory guide](PERSONAL_MEMORY_LOCAL_DEMO.md) shows the stdio host
configuration. The product integration test exercises both tools against this report-derived
thread.

The remaining model milestone is a **separate** paired base-versus-trained Qwen run on named,
reviewed personal-vault code-execution questions. This demo does not satisfy that gate.
