# Qwen Envoy — saved experiments and research tools

**Start with the experiments we actually ran.** The
[experiment atlas](docs/EXPERIMENT_ATLAS.md) connects Qwen training, evaluation,
retrieval, RL, serving and research-tool pilots to their saved results and open
questions. Read it directly in this repository; no MCP connection is needed.
The [QASPER Agent Studio](https://jasonlingg.github.io/Qwen-Envoy/) lets you
inspect the saved model traces.

## Optional research MCP

An assistant can also remember a new question, find and inspect papers, record
why one seemed relevant, and link it to a later experiment result through the
research MCP. It can search the atlas's 11 prior experiments and read their
listed source artifacts through bounded, read-only tools. Those prior runs are
indexed from the repository, not copied into mutable experiment records. A
person still has to judge whether a paper supports a claim or led to an outcome.

[Inspect the Research MCP demo](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/) ·
[Connect an MCP client](docs/RESEARCH_JUSTIFICATIONS.md#connect-an-mcp-client) ·
[Use the Codex decision skill](.codex/skills/envoy-research-decision/SKILL.md) ·
[Read the workflow and limits](docs/EXPERIMENT_MODE.md) ·
[See the hackathon run](docs/HACKATHON_SUBMISSION_2026.md)

The core source-and-memory workflow runs without a model endpoint. An optional
Nemotron explanation or question draft uses Nebius Token Factory only when
explicitly invoked; sharing local result summaries needs a per-call flag.
The saved demo includes that Nemotron call; **it does not run Qwen**. The
[five-question paired pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md) did not
show better decisions or recall than ordinary web research plus Markdown notes.
The demonstrated value is an inspectable, source-bound experiment record, not
a proven research-quality advantage.

To run the source-and-memory demo locally without a model or API key:

```bash
pip install -e '.[mcp,vault]'
python scripts/demo_hackathon_research_mcp.py \
  --library-root ~/.envoy/arxiv-library \
  --output out/my-research-demo
```

For a concrete paper → attempted method → measured result → next question trail,
run `python scripts/demo_research_question.py --output out/my-question-demo`.
It uses the saved Qwen serving diagnostic and retrieves one versioned public
paper; the recorded latency result remains **inconclusive**. The
[question-first guide](docs/EXPERIMENT_MODE.md#question-first-research-trail)
explains the MCP tools and the optional Nemotron boundary.

The first run may fetch public arXiv text. To connect an assistant to the full
workflow, use the [MCP client configuration](docs/RESEARCH_JUSTIFICATIONS.md#connect-an-mcp-client).
If a macOS Python installation reports `CERTIFICATE_VERIFY_FAILED` on a cold
paper fetch, set `SSL_CERT_FILE` to a trusted CA bundle before rerunning. For
example, the installed `certifi` package provides one:

```bash
export SSL_CERT_FILE="$(python -c 'import certifi; print(certifi.where())')"
```

## Qwen agent research case study

**A reproducible ML-agent engineering study.** I built a Qwen3-8B agent that
writes Python to search, read, and cite papers inside Docker, fine-tuned it with
QLoRA, and evaluated base and trained checkpoints on the same 40 QASPER paper
questions. The saved actions, observations, answers, and scoring artifacts are
open to inspect.

[Open the interactive Agent Studio](https://jasonlingg.github.io/Qwen-Envoy/) ·
[Inspect the saved RL lab](https://jasonlingg.github.io/Qwen-Envoy/rl.html) ·
[Read the findings](docs/QASPER_AGENT_STUDY.md) ·
[Inspect the training evidence](release/qwen-v5-training/README.md) ·
[Reproduce the saved analysis](#reproduce-the-results)

| In the paired 40-question development run | Base Qwen3-8B | QLoRA v5 |
| --- | ---: | ---: |
| Episodes with a recorded tool, runtime, or endpoint error | 22 | 1 |
| Provisional supported answers on the 20 answerable questions | 3 | 3 |

**Finding:** the trained checkpoint's run had far fewer execution and endpoint
errors, but did not establish better source-supported answers on answerable
questions.
The support judgments are model-assisted and await independent human review.
Official QASPER Answer F1, refusal behavior, and the limitations of this
comparison are reported separately in the [findings](docs/QASPER_AGENT_STUDY.md).
The three-update GRPO pilot did not reach its held-out quality evaluation;
it provides no additional model-improvement claim.
This study is separate from the Research MCP demo. A Qwen passage reader exists
as an experimental, explicitly configured tool, but it is not required for the
MCP workflow or established as a reliable automatic evidence-card writer.

## What is here

- **Agent and environment:** Qwen writes one Python action at a time against
  `search()`, `read()`, `extract()`, and passage tools. The controller executes
  model-authored code in Docker, returns observations, and saves the trace.
- **Post-training:** QLoRA SFT on 35 replay-verified training trajectories;
  the [dataset card](data/sft/qasper-v5/README.md) and
  [run record](release/qwen-v5-training/README.md) document how they were built.
- **Evaluation:** matched saved base/v5 runs, mechanical execution diagnostics,
  official QASPER answer scoring, and separately attributed provisional support
  review. The [technical case study](docs/TECHNICAL_CASE_STUDY.md) explains
  where these measurements disagree.
- **Studio:** a read-only public view of runs, questions, answers, and full
  action traces. Opening it makes no model calls and needs no GPU.

QASPER's questions and annotations are by Dasigi et al. (2021), CC BY 4.0.
This project adapts them to a **known-paper** code-tool task: each question
names its paper. These 40 inspected questions are development evidence, not an
original benchmark or an official full-test QASPER leaderboard result.
[Data provenance](benchmarks/envoybench/data/README.md) and the
[release bundle](release/qasper-agent-study/README.md) give the exact scope.

## Reproduce the results

From the repository root, Python 3.10+ and the standard library are enough to
recompute counts from saved outputs; no GPU, API key, or download is required:

```bash
python scripts/reproduce_case_study.py
python -m benchmarks.envoybench.qasper_official score \
  --run-dir release/envoybench-v0.1/run
```

The first command validates release hashes and reaggregates recorded review
judgments; it does not independently judge answer support. The second runs the
pinned official QASPER answer scorer on the selected 40 questions. To inspect
the snapshot offline, download
[studio.html](release/qasper-agent-study/studio.html) and open it in a browser.
For local source review and optional live investigations, use the
[full Studio setup](PROJECT_GUIDE.md#open-the-demo).

For setup instructions, source maps, historical experiments, and detailed
limitations, see the [full project guide](PROJECT_GUIDE.md). For the shortest
technical narrative, read the [case study](docs/TECHNICAL_CASE_STUDY.md).
Retired weekly-digest, Obsidian drafting, and live-viewer code is documented in
[archived prototypes](docs/ARCHIVED_PROTOTYPES.md); it is not part of the
current Studio or Research MCP runtime.
