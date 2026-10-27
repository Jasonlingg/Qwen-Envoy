# Archived product prototypes

## October 27 follow-up: isolated experiment scripts

Fourteen one-off diagnostic, comparison, and review scripts were removed from
`scripts/` (1,848 Python lines). None had a surviving script, test, or documented
command that invokes its CLI. Their source is recoverable from commit `e5b0e6d`:

- QASPER GRPO analysis and review: `analyze_qasper_grpo_eval.py`,
  `judge_qasper_grpo_semantic_review.py`,
  `prepare_qasper_grpo_semantic_review.py`,
  `prepare_qasper_multi_system_review.py`, and
  `score_qasper_grpo_semantic_review.py`.
- Earlier comparisons and diagnostics: `compare_claude_qwen_qasper.py`,
  `prepare_qwen_learning_diagnostic.py`,
  `run_learning_loop_inference_diagnostic.py`,
  `run_learning_loop_model_sweep.py`, `run_qwen_learning_diagnostic.py`,
  `run_research_json_ablation.py`, and `score_qasper_smoke.py`.
- Historical teacher and escalation helpers: `make_teacher_review_template.py`
  and `resolve_envoy_escalations.py`.

The shared scoring, training, replay, MCP, and Studio modules remain. The
Nebius continuation and merge scripts remain because their provenance tests
exercise the published reference run. No saved artifact or experiment result
was changed by this cleanup.

## October 26 follow-up: old live viewer

The earlier FastAPI/WebSocket trajectory viewer (`scripts/viewer.py`), its
HTML template, and `Dockerfile.web` are no longer maintained. QASPER Agent
Studio is the current run/trace interface. No current MCP/Studio code or
GitHub Pages workflow uses the old viewer. Recover its source at `92df497`;
the dated [walkthrough](DEMO_AND_EVIDENCE.md) is retained as a historical
record. The recorded replay JSON examples and their offline builder remain
available for inspecting the earlier SFT continuation case.

The `src/research/knowledge_tool.py` wrapper stays: several saved experiment
plans pin its source hash, even though current runtime code does not call it.
This removal cuts 1,053 HTML/Python/Dockerfile lines. Forty-one focused
Studio, MCP, wrapper, and release tests pass. Release hashes and the case-study
recount pass. The full clean-checkout suite remains at 1,995 passes, 19 skips,
34 failures and 26 errors, with the exact same 60 failing/error test identities
as before this viewer cleanup; see the preceding section for the baseline causes.

## October 26 follow-up: weekly publication path

The weekly digest writer, weekly paper-radar collector, and frozen-vault graph
were also retired from the maintained checkout. These were isolated historical
product paths, not dependencies of QASPER Agent Studio, the paper Research MCP,
the experiment-record MCP, or the Qwen investigator. Their implementations and
dedicated tests remain recoverable at commit `a67d97a`:

- `src/product/weekly_digest.py`, `scripts/weekly_digest.py`,
  `tests/test_weekly_digest.py`;
- `src/research/radar.py`, `scripts/weekly_radar_candidates.py`,
  `tests/test_research_radar.py`;
- `src/product/graph.py`, `tests/test_product_graph.py`.

The same cleanup retired `src/harness/`, the former CPU-only
evidence-to-Obsidian draft/staging prototype, and its six dedicated tests.
It is distinct from the maintained Docker code-execution and V1/V2 replay
harnesses. The current replay-checkpoint and vault-import boundary tests remain.

Their historical design notes and saved data remain in the repository. Commands
in those notes require the earlier checkout. This retirement does not mean the
weekly digest usability gate was achieved. The current arXiv paper discovery is
implemented separately in `src/product/paper_discovery.py`.

This follow-up removes 2,441 implementation lines and 1,899 dedicated test
lines across 21 files. It leaves all tracked `data/`, `examples/`, `release/`,
`benchmarks/`, and `reports/` files unchanged. Fifty focused tests for the
surviving replay, vault boundary, Qwen investigator, MCPs, and case study pass.
The full clean-checkout suite has 1,995 passes, 19 optional skips, 34 failures,
and 26 errors. The pre-cleanup checkout has the **same 60 failing/error test
identities** (2,099 passes before removing 104 retired-feature cases). Those
existing failures include missing ignored QASPER datasets and run artifacts,
historical frozen-runtime hash mismatches, and a stale corpus-count assertion.
They are not a passing full-suite result; the removed files do not account for
the failures. Release hashes and the saved case-study recount still pass.

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
