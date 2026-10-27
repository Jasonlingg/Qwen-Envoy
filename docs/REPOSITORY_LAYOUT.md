# Repository map

The current release is **QASPER Agent Studio**, a saved, inspectable case study
of the Qwen base/v5 paper-agent evaluation. The local Studio opens recorded runs
without an endpoint or GPU. Live paper investigation is a separate, opt-in path;
opening Studio never starts a model run.

| Area | Responsibility |
| --- | --- |
| `scripts/launch_studio.py` | Default local launcher. Loads the packaged Qwen runs and separately labeled Nebius reference run. `--live-nebius` enables an operator-configured live reader. |
| `benchmarks/envoybench/demo.py` | FastAPI routes for the Studio, full saved traces, paper reader, and human review. |
| `benchmarks/envoybench/saved_payload.py` | Validates saved run artifacts and builds the read-only Studio payload. |
| `benchmarks/envoybench/paper_api.py` | Optional live paper upload, web fetch, bounded run, and saved-history API. |
| `benchmarks/envoybench/frozen_dataset.py` | Loads and hash-checks the frozen QASPER-derived question split and corpus. |
| `benchmarks/envoybench/runtime_support.py` | Validates model configuration, checks the Docker sandbox, redacts secrets, and writes run files atomically. |
| `benchmarks/envoybench/run.py` | Paired inference runner. Historical imports from this module remain compatible. |
| `benchmarks/envoybench/score.py`, `qasper_official.py` | Provisional mechanical/review scoring and the pinned official QASPER Answer F1 path. These measure different things. |
| `src/env/`, `src/policies/`, `src/product/qwen_investigator.py` | Multi-step Python-action environment, model policies, and bounded paper investigator. The code-execution tool boundary stays `search()` / `read()` / `extract()`. |
| `scripts/research_justification_mcp.py` | Optional combined research MCP entry point: curated prior experiments, question trails, experiment records and paper tools. The source and experiment MCP files remain reusable components. |
| `docs/EXPERIMENT_ATLAS.md`, `src/product/experiment_atlas.py` | Curated index of 11 previous runs and bounded, read-only access to their saved source artifacts. This does not import every ignored local log. |
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
are archived in Git. The current Studio and MCPs do not use those prototypes.
This cleanup does not change model weights, active rewards, frozen execution
code or published results.

Run the offline artifact check with `python scripts/reproduce_case_study.py`.
The main launcher and release status are described in the [README](../README.md).
