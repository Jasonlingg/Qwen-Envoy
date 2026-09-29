# Repository map for the research-library build

The active product is a weekly public-paper research library. Qwen is the bounded Python-tool investigator; the host verifies its evidence before a separate drafter can prepare an Obsidian note. Only a person promotes an `_inbox/` draft into `library/`. The [harness design](RESEARCH_LIBRARY_HARNESS_DESIGN.md) is the target contract.

| Area | Role now |
| --- | --- |
| `src/harness/` | New fixed workflow boundary: exact-span bundle, structural draft gate, external run record, and atomic `_inbox/` staging. `pipeline.py` currently runs an **offline fixture replay**, not live Qwen/Nemotron. |
| `src/product/qwen_investigator.py`, `src/env/`, `src/policies/` | Existing multi-step code-execution worker and unchanged Qwen-facing document tools. Preserve this interface for a paired base/checkpoint comparison. |
| `src/research/papers.py`, `src/research/agent.py`, `src/research/tools_runtime.py` | Public-paper snapshot and read-only source handling reused by the new harness. |
| `scripts/research_library_mcp.py` | Existing read-only paper search/source MCP facade; the shared source broker and live harness query route remain to build. |
| `src/product/weekly_digest.py`, `src/product/memory.py`, `src/research/vault.py` | Existing product/demo flows. Their direct vault writers still need migration to the single staging writer; do not treat them as the new harness write path. The generic vault importer now excludes `_inbox/`. |
| `scripts/`, `tests/` | Entry points and focused regression tests. Training and historical QASPER scripts remain because they support reproducibility and existing code paths. |
| `data/`, `reports/`, `release/`, ignored `out/` | Paper questions, review material, experiment evidence, and local outputs. `out/` includes unique model and evaluation artifacts; it is not disposable build cache. |
| `docs/archive/` | Superseded plans for different projects, retained for history and kept out of the active design path. |

The first CPU-only vertical slice is tested with:

```bash
pytest -q tests/test_harness_staging.py tests/test_harness_evidence.py \
  tests/test_harness_draft.py tests/test_harness_run_record.py \
  tests/test_harness_pipeline.py tests/test_harness_source.py \
  tests/test_harness_vault_index_boundary.py
```

It proves that a fixture-controlled tool result can become a checked evidence bundle and an unreviewed note under `_inbox/`, with a run record outside the vault. It does **not** prove live Qwen tool calls are independently logged, that Nemotron is integrated, that a human found the note useful, or that trained Qwen beats base Qwen. The next implementation boundary is a host tool broker/ledger that preserves the current Qwen tool outputs, followed by a live public-paper drafter and migration of remaining direct writers.
