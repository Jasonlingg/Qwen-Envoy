# Repository map

The submission-facing product is a personal **ML experiment second brain** for
research evidence, measured runs and later decisions. The research MCP is its
working tool interface; the public research page inspects a saved run and makes
no live model call. **QASPER Agent Studio** is the separate, inspectable Qwen
base/v5 case study. Opening either saved-run page never starts a model run.

The [script entry-point guide](../scripts/README.md) gives the shortest path
to the maintained local commands. Other `scripts/` files include dated
experiment drivers and audits, not additional product launchers.

| Area | Responsibility |
| --- | --- |
| `scripts/research_justification_mcp.py` | Main research MCP: paper discovery, source inspection, experiment records and decision recall. Nebius explanation is opt-in. |
| `scripts/experiment_mode_extended_mcp.py` | Current combined experiment tools beneath the research MCP. `experiment_mode_mcp.py` retains the byte-for-byte historical replay implementation. |
| `scripts/launch_experiment_workbench.py` | Localhost view of the atlas, experiment results and append-only decision threads. |
| `scripts/demo_hackathon_research_mcp.py`, `scripts/demo_experiment_mode.py` | No-key paper and restart demos; the first script has a separate opt-in live Nebius path. |
| `release/hackathon-research-mcp/` | Public inspector of a recorded paper/Nemotron run; it does not call the MCP or model live. |
| `scripts/launch_studio.py` | Default local launcher. Loads the packaged Qwen runs and separately labeled Nebius reference run. `--live-nebius` enables an operator-configured live reader. |
| `benchmarks/envoybench/demo.py` | FastAPI routes for the Studio, full saved traces, paper reader, and human review. |
| `benchmarks/envoybench/saved_payload.py` | Validates saved run artifacts and builds the read-only Studio payload. |
| `benchmarks/envoybench/paper_api.py` | Optional live paper upload, web fetch, bounded run, and saved-history API. |
| `benchmarks/envoybench/frozen_dataset.py` | Loads and hash-checks the frozen QASPER-derived question split and corpus. |
| `benchmarks/envoybench/runtime_support.py` | Validates model configuration, checks the Docker sandbox, redacts secrets, and writes run files atomically. |
| `benchmarks/envoybench/run.py` | Paired inference runner. Historical imports from this module remain compatible. |
| `benchmarks/envoybench/score.py`, `qasper_official.py` | Provisional mechanical/review scoring and the pinned official QASPER Answer F1 path. These measure different things. |
| `src/env/`, `src/policies/`, `src/product/qwen_investigator.py` | Multi-step Python-action environment, model policies, and bounded paper investigator. The code-execution tool boundary stays `search()` / `read()` / `extract()`. |
| `docs/EXPERIMENT_ATLAS.md`, `src/product/experiment_atlas.py` | Curated index of 11 previous runs and bounded, read-only access to their saved source artifacts. This does not import every ignored local log. |
| `scripts/export_research_vault.py` | One-way Obsidian export of the atlas and saved research/experiment records; it does not watch or synchronize a personal vault. |
| `src/training/`, `environments/envoy_reader/` | Source-bound reader tasks, Docker rollout collection, Verifiers plugin, native Prime-RL orchestration and paired evaluation. |
| `src/product/memory.py`, `src/product/chat.py` | Small retained snapshot/metadata and provenance-review helpers. These modules no longer implement a memory-writing product or chat server. |
| `release/` | Recorded training, run, review, and Studio artifacts. The case-study checker verifies their hashes; cleanup must not rewrite them. |

The personal-memory app and Learning Lab/map demos were removed from the
maintained tree; their source can be recovered from Git at `5240090`. See
[archived prototypes](ARCHIVED_PROTOTYPES.md). Historical reports, data and saved
evidence remain in place. Commands in those dated product documents refer to the
earlier checkout.

Shared vault import and research utilities remain where retained callers need
them. The weekly digest, graph, draft-staging harness, and earlier live viewer
are archived in Git. The current Studio and MCPs do not use those prototypes;
the Obsidian research-history export is a separate maintained script.
This cleanup does not change model weights, active rewards, frozen execution
code or published results.

Run the no-key MCP demos from the [project guide](../PROJECT_GUIDE.md) or the
offline case-study check with `python -I -S scripts/reproduce_case_study.py`.
The submission status and limits are in the [hackathon draft](HACKATHON_SUBMISSION_2026.md).
