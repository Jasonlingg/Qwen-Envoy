# Research justifications for experiment decisions

The experiment-mode MCP can now keep a source-bound comparison of papers in an
append-only research note. It helps an ML engineer challenge a proposed method
before spending compute. It does not decide whether the papers are correct or
whether the method will transfer to a different workload.

Run the companion entry point to add `get_research_justification` to the
existing experiment-mode tools while preserving the historical server file
used by pinned replay artifacts:

```bash
python scripts/research_justification_mcp.py \
  --store-root ~/.envoy/ml-experiments \
  --library-root ~/.envoy/arxiv-library
```

Point your MCP client at this script instead of `experiment_mode_mcp.py` when
you want the packet tool. It uses the same local record and paper-cache roots.
This companion entry point does not configure optional Qwen or Nemotron tools.

## Connect an MCP client

From the repository root, after `pip install -e '.[mcp,vault]'`, Claude Code can
register the complete local workflow with the Python environment you installed:

```bash
claude mcp add envoy-research -- "$(command -v python)" \
  "$(pwd)/scripts/research_justification_mcp.py"
```

For Codex, register the same server from the repository root:

```bash
codex mcp add envoy-research -- "$(command -v python)" \
  "$(pwd)/scripts/research_justification_mcp.py"
```

The repository's [experiment-decision skill](../.codex/skills/envoy-research-decision/SKILL.md)
guides Codex on when to use these tools. It does not connect the server by
itself, start a model, or save a record until the assistant calls an MCP tool.

For another stdio MCP host, use its equivalent of this configuration. Replace
both absolute paths with your installed Python and checkout paths:

```json
{
  "mcpServers": {
    "envoy-research": {
      "command": "/absolute/path/to/python",
      "args": ["/absolute/path/to/rlm-explorer/scripts/research_justification_mcp.py"]
    }
  }
}
```

The default local stores are `~/.envoy/ml-experiments` and
`~/.envoy/arxiv-library`; pass explicit `--store-root` and `--library-root`
arguments if you want other locations. Register this one server rather than a
second copy of `experiment_mode_mcp.py`, since both expose the same record and
paper tools. Starting the server makes no model or network request. Source
search and open calls can fetch public arXiv pages; local record and recall
calls work from saved data.

For a substantive proposal, first define the local decision and the cheapest
plausible baseline. Search for the target paper, then deliberately search for
stronger baselines, conflicting results and relevant scope differences. Open the
papers and inspect exact passages. Save a `record_experiment_research` note with
the optional `justification` object. The note must compare at least two distinct,
versioned papers and include a qualifying, challenging or different-setting
finding. Each comparison names an exact `paper_span` reference and states the
intervention, comparator, setting, reported result, transfer limit and potential
evidence dependence. It also records the simpler baseline, transfer gaps,
cheapest matched test and decision rule. `get_research_justification` retrieves
those claims with the saved quotes, source URLs and hashes after an MCP restart,
even if the source cache is later removed.

The host chooses papers and writes these interpretations. The server verifies
that the quoted bytes came from the cached, versioned documents and refuses an
unchecked URL or one paper presented as two. It does **not** verify that a quote
supports the attributed result, that a comparison is fair, that papers are
independent, or that the search found all important counterevidence. Its output
always says `semantic_review: not_reviewed`. A reviewer should read each full
method/results section and check the claim, comparator, population, effect and
possible shared dataset or evaluation before relying on the packet. A private
author comment can be stored only as explicitly user-reported context, separate
from the verified paper findings. The author label is caller-declared.

## Example: WebOrganizer versus quality curation

The [WebOrganizer paper](https://arxiv.org/abs/2502.10341) reports that adding
domain mixing to FineWeb-Edu raises its selected average from 54.2% to 56.2%.
[DataComp-LM](https://arxiv.org/abs/2406.11794) reports that model-based
filtering was important to its data-curation recipe, and
[FineWeb](https://arxiv.org/abs/2406.17557) reports stronger knowledge and
reasoning results for its educational-quality subset. These latter papers
qualify a recommendation to invest in mixing first: quality selection is a
serious baseline. They are **not** a matched cross-paper experiment showing
that quality filtering always beats mixing. WebOrganizer's own comparisons
matter more for that narrower question, and shared corpora or benchmarks mean
three papers are not three independent votes.

For Qwen post-training, the key transfer gap is that these are web-text
pretraining results, with different models, objectives and evaluations. A
candidate local test is quality-only data selection against quality selection
plus mixing, with the same training pool, tokens, steps and held-out questions.
Set a minimum useful supported-answer gain and error/cost ceilings before
training; keep quality-only if the added mixing misses them. This is a proposed
test, not a measured Qwen benefit.

Reproduce the no-model public-source walkthrough:

```bash
python scripts/demo_research_justification.py \
  --library-root ~/.envoy/arxiv-library \
  --output out/my-justification-demo
```

Use a fresh output directory. It downloads the three public papers on a cold
cache, makes nine MCP calls, and writes `report.json`. No Qwen, Nemotron, GPU or
paid model request is made. The saved packet is an example of provenance and
structured challenge, not a validated research recommendation. The first local
run on October 26 opened HTML text for all three versioned papers and completed
all nine MCP calls. An implementing-assistant spot check found that each of the
three short attributed findings is stated in its selected quote. The labels
`qualifies` on DCLM and FineWeb are an interpretation: those quotes establish
that quality filtering matters in their own settings, not a direct comparison
with WebOrganizer's domain mixing. The broader transfer and paper-dependence
claims were not independently validated. Independent human claim-level review
was not performed.

The [data-mixing workflow evaluation](RESEARCH_DECISION_EVAL_20261026.md) adds
a real search probe, fresh-process recovery and claim-by-claim assistant review.
It also records the missing numerical decision threshold before any training.
