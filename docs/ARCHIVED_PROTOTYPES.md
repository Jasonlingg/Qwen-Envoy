# Archived personal-memory and map prototypes

October 25, 2026. The user authorized removing historical product prototypes to
reduce the maintained code surface. The complete earlier implementation remains
in Git at **`5240090`**; this cleanup does not duplicate that code in another active
package or keep deprecated launchers as wrappers.

Removed feature groups:

- Personal-memory web UI, its MCP/CLI, synthetic-vault evaluation and worker smoke.
- Learning Lab thread/revision routes and the older Nemotron chat coordinator.
- Transformer/research maps and the disposable synthetic-note/report demos.
- Their HTML templates and tests that exercise only those retired features.

The retired launchers include `personal_memory_web.py`, `personal_memory_mcp.py`,
`personal_memory.py`, `personal_memory_eval.py`, `run_personal_memory_qwen_eval.py`,
`smoke_qwen_memory_worker.py`, `launch_sample_demo.py`,
`launch_research_map_demo.py`, `launch_transformer_map_demo.py`,
`launch_qwen_experiment_demo.py`, and `build_qwen_experiment_vault.py`, formerly in
`scripts/`. Their dedicated Learning Lab, impact, map and chat-coordinator modules
were removed from `src/product/`.

## Recover an earlier prototype

From this repository, create a separate historical checkout:

```bash
git worktree add --detach ../qwen-envoy-prototypes 5240090
```

Read and run its instructions there, using that revision's dependencies and
explicit model configuration. The command only checks out source; it does not
launch a server, model, GPU or training job. The historical software retains its
original limitations and is not the maintained project interface.

The dated personal-memory, Learning Lab, question-desk, systems-map and
report-derived memory documents remain historical records. Their launch commands
are for the earlier checkout, not the current tree.

## Retained boundaries and evidence

Current research/experiment MCP entry points, Studio, Qwen investigation, SFT,
native Prime-RL, reward audits and paired evaluation remain. All saved datasets,
reports, release bundles, source annotations and evaluation judgments are kept
byte-for-byte. Removing a prototype does not alter its historical results or
convert a failed gate into a success.

Qwen still reads dated front matter and imports frozen vault fixtures, and the
research-library development runner still verifies exact source packets. The
small `memory.py` and `chat.py` helpers remain at their existing import paths;
their retained function bodies are unchanged. Vault graph/import, weekly digest
and staging utilities with other callers are outside this first cleanup pass.

The lower test count after cleanup reflects removal of tests for deleted
features. Frozen execution hashes, surviving source-integrity checks and the
remaining full test suite must continue to pass; this is a maintenance change,
not a new model evaluation.

## Verified cleanup result

| Measure | Before | After |
| --- | ---: | ---: |
| Implementation lines | 69,707 | 61,828 |
| Test lines | 34,424 | 31,108 |

Counts include comments and blank lines in source/CLI/benchmark/Studio code;
documentation, datasets and saved release artifacts are excluded. Net reduction:
**7,879 implementation lines (11.3%) and 3,316 test lines**. Thirty-seven
prototype files were deleted, and two shared modules were reduced to their
remaining helpers. Source can be recovered from the baseline commit above.

Verification: **68 focused surviving-workflow tests**, **seven shared provenance
checks**, **160 native training/evaluation tests**, and the **full 2,141-test suite**
passed. There are 17 optional skips and the same existing warnings. Ninety test
cases for retired features were removed; seven direct provenance checks retain
coverage for the surviving source-review boundary. This explains the full-suite
count reduction from 2,224 without implying broader test coverage or model gains.

No surviving Python imports reference the removed modules. The three retained
function bodies match their original ASTs; all 29 frozen runtime hashes and the
saved controller plan still validate. Every tracked `data/`, `examples/` and
`release/` file remains byte-identical, and the offline case-study checker reports
`verified`. Ruff, maintained documentation links and Git whitespace checks pass.
No model calls, GPU provisioning or paid experiments were used for cleanup.
