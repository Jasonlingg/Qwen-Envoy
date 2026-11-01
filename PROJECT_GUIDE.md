# Qwen Envoy project guide

Start with the [README](README.md) for scope, results, and three code entry points.

## Reproduce the saved QASPER study

From the repository root, verify packaged files and recompute the paired counts with the Python standard library:

```bash
python -I -S scripts/reproduce_case_study.py
```

To recompute official QASPER Answer F1 after installing project dependencies:

```bash
python -m benchmarks.envoybench.qasper_official score --run-dir release/envoybench-v0.1/run
```

To open Studio locally:

```bash
python -m pip install -e '.[viewer]'
python scripts/launch_studio.py
```

Studio loads saved runs by default; opening it does not launch inference. The [release README](release/qasper-agent-study/README.md) lists artifacts and checks. The [technical case study](docs/TECHNICAL_CASE_STUDY.md) explains the action loop, dataset, prefix bug, and failures. The [training record](release/qwen-v5-training/README.md) identifies the adapter. A new live run requires a compatible model endpoint, Docker, and the recorded configuration. Saved replay cannot measure how the model would respond to new feedback.

The full test suite and new live QASPER runs also need the pinned paper corpus:

```bash
python -m pip install -e '.[dev,viewer,mcp,vault,qasper]'
python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data
python -m pytest -q
```

The generated `dev/` and `test_candidate/` directories are ignored by Git.
The saved Studio and offline case-study recount do not need this step.

## Research MCP

The optional personal MCP exposes paper discovery, bounded source inspection, experiment records, and decision recall to a larger assistant:

```bash
python -m pip install -e '.[mcp]'
python scripts/research_justification_mcp.py
```

It is a stdio server; an MCP client starts that command and communicates over stdin/stdout. See [client configuration and tool workflow](docs/RESEARCH_JUSTIFICATIONS.md). New arXiv searches use the public network; local records do not need a model endpoint. Nemotron explanation uses a separately configured Nebius API key. The [saved browser inspector](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/) is a recorded demo, not a live MCP connection.

## Experimental RL continuation

The [October 26 Prime-RL/Verifiers pilot](docs/READER_RL_PILOT_20261026.md) completed nine of ten planned updates from v5. A generation timeout interrupted update ten. Exploratory paired evaluation found one complete supported answer in each arm on ten clear-reference questions; no quality gain was established. The [protocol](docs/READER_PAIRED_EVALUATION.md) and [training code](src/training/) are retained for inspection.

## Evidence boundaries

The QASPER agent reads an assigned paper. Hash checks establish artifact integrity; exact quote matching establishes provenance; neither establishes semantic support. Support judgments were model-assisted and the 40 questions were a selected development set. The MCP pilot did not beat ordinary web research plus Markdown notes. See the [findings](docs/QASPER_AGENT_STUDY.md) and [MCP pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md).
