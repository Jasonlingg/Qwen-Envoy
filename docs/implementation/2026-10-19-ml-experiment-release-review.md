# ML experiment workflow — Task 6 review and verification

Task 6 was explicitly approved after the Tasks 1–5 implementation and real-data
walkthrough. Scope: independent code review, fixes, repository tests, setup/demo
documentation, and preservation of existing work. It did not authorize the
proposed Qwen intervention, a Codex comparison, paid inference, or GPU provisioning.

## Independent assistant code review

A fresh reviewer inspected the experiment models, storage, source boundaries,
comparison/provenance logic, MCP surface, demonstration, tests and documentation.
It edited no files and made no network/model/GPU calls. This is independent
assistant code review, not independent human review of scientific claims or answers.

Two P2 findings were reproduced and fixed in commit `e3d2f5b`:

| Finding | Reproduction and fix |
| --- | --- |
| Store ancestor symlinks could redirect writes after construction | Replacing the selected store's parent with a symlink redirected a later save. Root opening now traverses every ancestor with directory descriptors and `O_DIRECTORY | O_NOFOLLOW`; initialization creates missing components relative to verified descriptors. The same helper protects source roots against an ancestor replacement between preliminary checks and opening. |
| Canonical counts accepted negative/fractional values | A candidate count of −999.5 could produce a passing relative-improvement result. `unit="count"` now requires a nonnegative integral value. Generic signed scores remain supported under other units; event counts are not arbitrarily capped by sample size. |

Outcome summaries additionally expose `data_kind`, so synthetic entries remain
identifiable without selecting their full record. This changes the presentation,
not frozen-plan or stored result bytes.

The reviewer rechecked the fixes and ran **91 focused tests**. Its original
ancestor-swap reproduction was rejected with zero files outside the selected
store; negative and fractional counts were rejected. No blocking findings remained.
Regression tests cover both attacks and synthetic-summary labeling. Original
valid frozen records and result hashes remain readable; malformed historical
counts would now be rejected rather than treated as successful measurements.

## Full repository verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest -q --tb=short
python -m ruff check src/product/ml_experiment*.py \
  scripts/ml_experiment_mcp.py scripts/demo_ml_experiment_workflow.py \
  scripts/export_ml_experiment_case_study.py tests/test_ml_experiment*.py \
  tests/ml_experiment*fixtures.py
git diff --check
```

Result: **1,478 passed in 50.75 seconds**, with three SWIG/FAISS dependency
deprecation warnings; Ruff and whitespace checks passed. Provider API keys were
removed from the test subprocess environment and model downloads disabled.
The synthetic corpus fixture directory was backed up and restored around the
full suite. No training or paid inference job was started.

The repository run includes the 188 focused experiment/arXiv MCP cases, existing
V1/V2 tool and harness compatibility, Docker replay-boundary unit checks, source
network/redirect/size/throttle checks, isolated reader failures, and saved-study
reproduction tests. Passing these is a software verification result, not a new
model run or a repeated replay experiment. The 40-question Nemotron replay was
not rerun in Task 6; its fixed-action interpretation limitations remain unchanged.

The local full log is `out/ml-experiment-task6-full-suite.log`. A machine-readable
record of tests, runtime source hashes and read-only MCP checks is retained at
`out/ml-experiment-task6-validation.json` and copied into
`examples/ml-experiment-qwen-v5/task6-validation.json`.

## Configured-server checks and preserved records

Both project-local MCP entries were initialized through their configured stdio
commands: five source tools and five experiment tools were advertised. Read-only
calls retrieved the actual Task 5 proposal and historical result after the fixes.

- Proposal: still a draft, no plan digest, no intervention results.
- Historical record: same plan/result digests and same retrospective/inconclusive
  assessment as the published walkthrough; original proxy checks met, independent
  support/complete timing/all-in cost checks unknown.
- No new records were created by the Task 6 configured-server check. Test writes
  used temporary stores. No source fetch, Qwen request, paid call or GPU launch
  was needed for this read-back.

The planning tools are still absent from this native host session's tool
inventory. Their actual stdio implementation was verified; native tool discovery
needs a client reload and is not represented as completed. The source connector
was used natively during Task 5.

Pre-existing modifications to the arXiv guide, arXiv/pilot scripts and model
configuration, the three untracked October 16 plan/spec files, and the protected
training/research files were left unchanged. Their raw hashes were checked
against the task baseline; no unrelated work was staged in Task 6 commits.

## What is available, and what remains unproved

An MCP user can ask an ML engineering question, have the host inspect selected
exports and paper evidence, save/revise a brief, freeze an agreed plan, and later
record measurements against unchanged criteria. The host supplies research and
judgment; the server supplies bounded persistence, source receipts and mechanical
comparisons. Qwen is an optional research reader, not a requirement for this flow.

The reproducible example, schema inputs, exact review prompts and actual
walkthrough are in `examples/ml-experiment-qwen-v5`. Opening its three public
papers can require network access; replaying with existing caches makes no model
calls. Corpus/cache-dependent tests skip explicitly when those local inputs are
absent; they never fetch them automatically.

User review of the proposal's usefulness remains pending. No newly generated
Qwen answers, intervention outcome, independent human semantic validation,
fine-tuning benefit, ordinary-Codex advantage, or cost/time saving is established.
The separate comparison benchmark and any model execution require their own
approved tasks and budgets. Broader Obsidian usability and held-out model-quality
gates in `AGENTS.md` have not been achieved by this implementation.
