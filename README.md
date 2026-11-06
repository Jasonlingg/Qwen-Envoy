# Envoy

[![Case study checks](https://github.com/Jasonlingg/Qwen-Envoy/actions/workflows/release-check.yml/badge.svg)](https://github.com/Jasonlingg/Qwen-Envoy/actions/workflows/release-check.yml)

**Research memory for ML experiments.**

Envoy is a personal second brain for keeping track of papers, experiments, and
the decisions between them. It connects to your assistant through MCP and
exports linked notes to Obsidian.

Before starting another training run, you can ask:

> What have we already tried to reduce Qwen's false refusals? Show me the results
> and the evidence behind our last decision.

The assistant retrieves previous attempts, inspects their source artifacts,
looks up papers, and saves useful findings or decisions for a later session.
Envoy supplies the records and tools; the assistant interprets the evidence.

## What works today

- **Find previous work:** search the curated experiment history and saved records.
- **Inspect paper evidence:** discover arXiv papers and read bounded source passages.
- **Keep a research trail:** connect questions, findings, results, and decisions
  while preserving revisions and source references.
- **Browse in Obsidian:** successful MCP saves refresh linked notes, preserving
  personal folders and Obsidian settings.

```text
Your assistant → Envoy MCP → Paper tools + saved experiment records
                                      ↓
                              Linked Obsidian notes
```

Recall currently uses word overlap. Personal Obsidian notes do not sync back
into MCP recall. A PyTorch memory ranker is under investigation and is not yet
integrated.

## Try it

From the repository root, with Python 3.10 or newer:

```bash
python -m pip install -e '.[mcp,vault]'
```

Configure your MCP client to launch:

```bash
python /absolute/path/to/rlm-explorer/scripts/research_justification_mcp.py
```

See [client setup](docs/RESEARCH_JUSTIFICATIONS.md#connect-an-mcp-client) for
configuration examples. Core tools need no GPU or model API key. New paper
searches use the public network; Nemotron tools are opt-in.

To create the Obsidian view before saving any records:

```bash
python scripts/export_research_vault.py
```

Open `out/envoy-research-vault` as a vault in Obsidian and start at `Home.md`.
Keep annotations in `My Notes`, outside the generated folders. For an existing
export, add `--refresh`. See [refresh and backup details](docs/EXPERIMENT_MODE.md#browse-the-research-history-in-obsidian).

## The Qwen case study

Envoy grew out of a Qwen3-8B paper agent that writes Python inside Docker.
Its training recipe, data, saved evaluation, and **QASPER Agent Studio** remain
available to inspect.

[Open Studio](https://jasonlingg.github.io/Qwen-Envoy/) ·
[Findings](docs/QASPER_AGENT_STUDY.md) ·
[Technical walkthrough](docs/TECHNICAL_CASE_STUDY.md)

On 40 selected QASPER development questions, episodes without recorded execution
errors increased from **18/40 to 39/40** after SFT. Provisional, model-assisted
supported-answer passes on the 20 answerable questions remained **3/20 in both
runs**. The 44 demonstrations include AI repairs. QASPER is from Dasigi et al.
(2021); this is not an official full-test result or independently human-reviewed
answer-quality improvement. The later Prime-RL pilot established no quality gain.

Recount the saved comparison and verify packaged hashes without dependencies,
network access, or a model:

```bash
python -I -S scripts/reproduce_case_study.py
```

For a code review, start with the [training script](scripts/train_sft.py),
[Docker boundary](src/env/repl.py), and
[result checker](scripts/reproduce_case_study.py).
The [project guide](PROJECT_GUIDE.md) covers training preflight, local Studio,
and tests. See the [repository map](docs/REPOSITORY_LAYOUT.md) for other entry
points and [`archive-full`](https://github.com/Jasonlingg/Qwen-Envoy/tree/archive-full)
for historical prototypes.

## What is established

Envoy can recover source-linked experiment records. A small
[paired pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md) did not establish a
workflow advantage over web research plus Markdown notes. Source hashes and
exact quotations establish provenance, not whether a conclusion is correct.
