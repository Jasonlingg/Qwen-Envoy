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

## Open the research history in Obsidian

Run `python scripts/export_research_vault.py` to create a linked reading vault at
`out/envoy-research-vault`, or add `--refresh` to update an existing export.
In Obsidian, choose **Open folder as vault** and select that folder, then open
`Home.md`. Keep your own annotations in `My Notes`, outside the generated folders.
The home note shows the refresh time and source checkout. Refresh preserves your
Obsidian settings and personal folders; edited generated notes block replacement.
The current research MCP refreshes the vault automatically after successful
saves; restart the server once after updating its code. Read calls and unchanged
retries do not rebuild it. A refresh failure returns a warning while preserving
the saved record. Repository edits still need the manual command. This is not
two-way sync or MCP indexing of personal notes. See
[the refresh and backup details](docs/EXPERIMENT_MODE.md#browse-the-research-history-in-obsidian).

## Experimental RL continuation

The [October 26 Prime-RL/Verifiers pilot](docs/READER_RL_PILOT_20261026.md) completed nine of ten planned updates from v5. A generation timeout interrupted update ten. Exploratory paired evaluation found one complete supported answer in each arm on ten clear-reference questions; no quality gain was established. The [protocol](docs/READER_PAIRED_EVALUATION.md) and [training code](src/training/) are retained for inspection.

## Evidence boundaries

The QASPER agent reads an assigned paper. Hash checks establish artifact integrity; exact quote matching establishes provenance; neither establishes semantic support. Support judgments were model-assisted and the 40 questions were a selected development set. The MCP pilot did not beat ordinary web research plus Markdown notes. See the [findings](docs/QASPER_AGENT_STUDY.md) and [MCP pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md).
