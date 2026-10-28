# Qwen Envoy

A personal **ML experiment second brain** for deciding what to try next with
small language model agents. Its research MCP connects paper evidence, saved
experiment results, and decisions across sessions. The Qwen3-8B paper agent is
the first case study: it writes Python in Docker, with QLoRA training, paired
evaluation, full action traces, and a read-only Studio. Engineers still choose
and run experiments; Envoy does not autonomously retrain or promote a model.

[Open Agent Studio](https://jasonlingg.github.io/Qwen-Envoy/) ·
[Read the findings](docs/QASPER_AGENT_STUDY.md) ·
[Browse the experiment atlas](docs/EXPERIMENT_ATLAS.md) ·
[Inspect the implementation](docs/TECHNICAL_CASE_STUDY.md)

## One experiment decision, end to end

Suppose you want to know whether two Qwen readers can share one engine and
reduce latency. The MCP can connect a verified PagedAttention paper passage to
the saved two-reader serving run, recall that run after a restart, and record
the decision to **defer a latency-win claim**: concurrent admission worked, but
the measured pair times did not establish a reliable speedup. The paper is
motivation, not proof of the run's outcome. [Try the local test build](PROJECT_GUIDE.md) ·
[See the saved MCP inspector](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/) ·
[Read the source-linked record](docs/EXPERIMENT_ATLAS.md#paper--attempt--result-example).

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
the scoring, early-stopping pattern, and limits. Later Prime-RL/Verifiers
experiments have not established an answer-quality improvement.

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
| [`scripts/launch_experiment_workbench.py`](scripts/launch_experiment_workbench.py) | Local browser view for prior attempts, saved results and durable decisions. |
| [`scripts/export_research_vault.py`](scripts/export_research_vault.py) | One-way export of the atlas and saved records as linked Obsidian notes. |

The atlas is a direct way to review previous SFT, retrieval, reader, RL, serving,
and research-tool attempts. It is curated rather than an automatic import of
every raw local log. The [repository map](docs/REPOSITORY_LAYOUT.md) separates
maintained components from historical prototypes.

To try the complete local record in a browser, with no API key or GPU:

```sh
python -m pip install -e '.[viewer,mcp,vault]'
python scripts/demo_experiment_mode.py --output out/my-experiment-demo
python scripts/launch_experiment_workbench.py --store-root out/my-experiment-demo/records
```

Open `http://127.0.0.1:8976`, search **Qwen reader latency**, open the saved
experiment and its measured run, then open the decision thread. It records
**inconclusive**, not a speedup. Use a new output directory if you repeat the
demo. The [workbench guide](docs/EXPERIMENT_MODE.md#open-the-local-workbench)
explains the source and authorship boundaries.

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

To browse the same history in Obsidian, run
`python scripts/export_research_vault.py` and open `out/envoy-research-vault` as
a folder vault. Refresh it after new records are saved; this is a reading copy,
not live sync or an Obsidian plugin. See the [export guide](docs/EXPERIMENT_MODE.md#browse-the-research-history-in-obsidian).

The earlier weekly-digest and personal-vault demos are documented in
[archived prototypes](docs/ARCHIVED_PROTOTYPES.md). They are separate from the
current research-history export.
