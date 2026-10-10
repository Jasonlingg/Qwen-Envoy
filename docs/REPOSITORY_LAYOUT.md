# Repository map

The current release is **QASPER Agent Studio**, a saved, inspectable case study
of the Qwen base/v5 paper-agent evaluation. The local Studio opens recorded runs
without an endpoint or GPU. Live paper investigation is a separate, opt-in path;
opening the viewer never starts a model run.

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
| `release/` | Recorded training, run, review, and Studio artifacts. The case-study checker verifies their hashes; cleanup must not rewrite them. |

The earlier Obsidian digest and research-library prototype lives in
`src/harness/`, `src/research/`, `src/product/weekly_digest.py`, and related
scripts. It remains available as historical work, but it is not a completed
product gate for this release. Training and reward code is also retained for
reproducibility; the current cleanup does not change model weights, rewards, or
the published evaluation.

Run the offline artifact check with `python scripts/reproduce_case_study.py`.
The main launcher and release status are described in the [README](../README.md).
