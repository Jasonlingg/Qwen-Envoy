# Qwen Envoy

A reproducible case study of a **Qwen3-8B research agent** that writes Python to
inspect papers in a Docker sandbox. It includes QLoRA training, paired saved
evaluation, full action traces, and a read-only Studio. A separate personal
research MCP connects papers to later experiment decisions.

[Open Agent Studio](https://jasonlingg.github.io/Qwen-Envoy/) ·
[Read the findings](docs/QASPER_AGENT_STUDY.md) ·
[Browse the experiment atlas](docs/EXPERIMENT_ATLAS.md) ·
[Inspect the implementation](docs/TECHNICAL_CASE_STUDY.md)

## Qwen agent research case study

Base Qwen3-8B and a rank-4 QLoRA adapter attempted the **same 40 selected
QASPER paper questions**. Every question named its target paper; this tests
multi-step evidence inspection within a known document. QASPER's questions and
annotations are by Dasigi et al. (2021), CC BY 4.0. This is a development
study, not an original dataset or an official full-test leaderboard result.

| Recorded result | Base | Trained v5 |
| --- | ---: | ---: |
| Episodes with a tool, runtime, or endpoint error | 22/40 | 1/40 |
| Provisional supported-answer passes on answerable questions | 3/20 | 3/20 |
| Official Answer F1 on all selected questions | 19.92% | 30.69% |
| Official Answer F1 on the 20 answerable questions | 29.83% | 21.37% |

The trained run was substantially more reliable at executing actions, but it
did **not** establish better source-supported answers on answerable questions.
The provisional support judgments are model-assisted, not independently human
reviewed. The overall F1 increase is driven by refusals and masks the lower
answerable-only F1. The [findings report](docs/QASPER_AGENT_STUDY.md) explains
the scoring, early-stopping pattern, and limits. A later reader GRPO pilot
completed three verified updates but stopped before held-out evaluation; it
provides no model-quality improvement claim.

In the [Studio](https://jasonlingg.github.io/Qwen-Envoy/), choose a question to
inspect the prompt, Python actions, tool observations, submitted answer, and
saved assessment. The [training record](release/qwen-v5-training/README.md)
pins the model and adapter identities; the
[dataset card](data/sft/qasper-v5/README.md) describes the 35 training
conversations. The [technical case study](docs/TECHNICAL_CASE_STUDY.md) walks
through representative successes and failures.

## What is here

| Path | Purpose |
| --- | --- |
| [`src/env/`](src/env/) | Paper tools and the Docker execution boundary for model-authored Python. |
| [`src/policies/`](src/policies/) | Qwen policy and bounded model transport. |
| [`data/sft/qasper-v5/`](data/sft/qasper-v5/) | Training/validation trajectories and provenance. |
| [`release/qasper-agent-study/`](release/qasper-agent-study/) | Read-only Studio and packaged comparison evidence. |
| [`release/envoybench-v0.1/`](release/envoybench-v0.1/) | Frozen paired runs and review artifacts. |
| [`docs/EXPERIMENT_ATLAS.md`](docs/EXPERIMENT_ATLAS.md) | Eleven prior ML experiments, each linked to its saved result and open question. |
| [`scripts/research_justification_mcp.py`](scripts/research_justification_mcp.py) | Optional personal research-memory MCP. |

The atlas is a direct way to review previous SFT, retrieval, reader, RL, serving,
and research-tool attempts. It is curated rather than an automatic import of
every raw local log. The [repository map](docs/REPOSITORY_LAYOUT.md) separates
maintained components from historical prototypes.

## Reproduce the results

From the repository root, verify packaged hashes and recount the saved paired
study with Python's standard library; this makes no model, GPU, or network call:

```bash
python -I -S scripts/reproduce_case_study.py
```

To recompute the selected-question official QASPER Answer F1 with project
dependencies installed:

```bash
python -m benchmarks.envoybench.qasper_official score \
  --run-dir release/envoybench-v0.1/run
```

The recount validates artifacts and arithmetic; it does not independently
judge semantic support. For local Studio setup and full reproduction details,
see the [project guide](PROJECT_GUIDE.md).

## Optional research MCP

The personal MCP searches the [experiment atlas](docs/EXPERIMENT_ATLAS.md),
opens bounded excerpts from its listed source artifacts, and saves new
question → paper → experiment → result trails. The paper/source and memory
workflow works without Qwen or a GPU. Exact source matches establish
provenance, not whether a quote supports a conclusion.

[See the saved MCP demo](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/) ·
[Connect a client](docs/RESEARCH_JUSTIFICATIONS.md#connect-an-mcp-client) ·
[Read the workflow](docs/EXPERIMENT_MODE.md)

In a [five-question paired pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md),
this MCP did **not** improve decisions or recall over ordinary web research plus
Markdown notes. Its demonstrated capability is a recoverable, source-linked
experiment record; a broader productivity advantage remains unproven.

Historical weekly-digest, Obsidian, and retired demo code is documented in
[archived prototypes](docs/ARCHIVED_PROTOTYPES.md). Those paths are not part of
the maintained Studio or research MCP runtime.
