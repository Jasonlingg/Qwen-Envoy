# Qwen Envoy

[![Case study checks](https://github.com/Jasonlingg/Qwen-Envoy/actions/workflows/release-check.yml/badge.svg)](https://github.com/Jasonlingg/Qwen-Envoy/actions/workflows/release-check.yml)

**QASPER Agent Studio** is a case study of a Qwen3-8B paper-reading agent. The agent writes Python against `search()`, `read()`, and `extract()` inside Docker. This repository contains the environment, rank-4 QLoRA training recipe, saved paired evaluation, and a read-only trace inspector. An optional research MCP connects paper passages to saved ML experiment results and decisions.

[Open Studio](https://jasonlingg.github.io/Qwen-Envoy/) · [Findings](docs/QASPER_AGENT_STUDY.md) · [Technical case study](docs/TECHNICAL_CASE_STUDY.md) · [Project guide](PROJECT_GUIDE.md)

## Start with three files

1. [`src/env/repl.py`](src/env/repl.py) is the Docker boundary for model-authored Python. Its cumulative-script design re-runs earlier actions, a known efficiency and side-effect limitation. The optional MCP reader uses stronger settings in [`src/env/hardened_repl.py`](src/env/hardened_repl.py).
2. [`scripts/train_sft.py`](scripts/train_sft.py) turns reviewed trajectories into per-action examples and checks the inference chat prefix.
3. [`scripts/reproduce_case_study.py`](scripts/reproduce_case_study.py) verifies packaged hashes and recounts the paired result without a model, GPU, API key, or network.

```bash
python -I -S scripts/reproduce_case_study.py
python -m pip install -e '.[dev,viewer,mcp,vault,qasper]'
python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data
python -m pytest -q
```

The first command checks saved results offline. The remaining commands set up
the full test suite; QASPER paper text is generated from a pinned source and
kept out of Git.

For a training-code review, start at [`scripts/train_sft.py`](scripts/train_sft.py):
it validates the split before loading a model, expands each trajectory into
next-action targets, verifies token-for-token inference-prefix alignment, masks
the prior context from the loss, then constructs the TRL/PEFT trainer and writes
a run manifest. Its GPU-free data preflight is:

```bash
python scripts/train_sft.py check-data \
  --data data/sft/qasper-v5/train.jsonl \
  --val-data data/sft/qasper-v5/val.jsonl \
  --corpus-manifest data/sft/qasper-v5/manifest.json
```

The [published training record](release/qwen-v5-training/README.md) pins the
observed v5 run. The current script has changed since that run; this preflight
does not retrain the model or prove its historical weights are reproducible.

## What the saved study shows

Base Qwen3-8B and the v5 SFT adapter attempted the same 40 selected QASPER questions, each with an assigned paper. QASPER is by Dasigi et al. (2021), CC BY 4.0. This is a development study, not an original dataset or an official full-test leaderboard result.

| Recorded result | Base | v5 SFT |
| --- | ---: | ---: |
| Episodes with a tool, runtime, or endpoint error | 22/40 | 1/40 |
| Provisional supported-answer passes on 20 answerable questions | 3/20 | 3/20 |
| Official Answer F1, all 40 selected questions | 19.92% | 30.69% |
| Official Answer F1, 20 answerable questions | 29.83% | 21.37% |

The adapter improved execution reliability but did **not** establish better source-supported answers on answerable questions. Support judgments were model-assisted, not independently human reviewed. Overall F1 rose largely through refusals. See the [findings](docs/QASPER_AGENT_STUDY.md).

The dataset comprises 44 teacher-seeded, AI-repaired, replay-verified demonstrations (35 train, nine validation). Of these, 37 contain Codex repairs; seven retain original teacher actions. Source and stopping review was performed by the editing assistant with labels visible. See the [dataset card](data/sft/qasper-v5/README.md). A later Prime-RL/Verifiers pilot did not establish answer-quality improvement; see the [run report](docs/READER_RL_PILOT_20261026.md).

## Repository map

| Path | Purpose |
| --- | --- |
| [`src/env/`](src/env/) and [`src/policies/`](src/policies/) | Paper tools, Docker boundary, Qwen policy. |
| [`scripts/train_sft.py`](scripts/train_sft.py) and [`src/training/`](src/training/) | SFT and experimental Prime-RL/Verifiers integration. |
| [`data/sft/qasper-v5/`](data/sft/qasper-v5/) | Training examples and provenance. |
| [`release/qasper-agent-study/`](release/qasper-agent-study/) | Packaged traces and Studio assets. |
| [`benchmarks/envoybench/`](benchmarks/envoybench/) | Frozen split, runner, scoring, and Studio API. |
| [`scripts/research_justification_mcp.py`](scripts/research_justification_mcp.py) and [`docs/EXPERIMENT_ATLAS.md`](docs/EXPERIMENT_ATLAS.md) | Optional research MCP and curated experiment index. |

The main branch keeps the runnable case study, optional MCP, their evidence, and tests. Historical experiments and one-off runners remain in the [`archive-full` snapshot](https://github.com/Jasonlingg/Qwen-Envoy/tree/archive-full). The [repository map](docs/REPOSITORY_LAYOUT.md) gives entry points.

## Optional research MCP

The MCP searches public arXiv papers and a curated [experiment atlas](docs/EXPERIMENT_ATLAS.md), opens bounded source excerpts, and records question → evidence → experiment → result → decision trails across sessions. It runs without Qwen or a GPU; Nemotron explanation is opt-in. Exact quotation proves provenance, not semantic support. A [five-question paired pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md) found no measured advantage over ordinary web research plus Markdown notes. Its demonstrated capability is a recoverable, source-linked experiment record.

[Saved MCP inspector](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/) · [MCP client instructions](docs/RESEARCH_JUSTIFICATIONS.md#connect-an-mcp-client)
