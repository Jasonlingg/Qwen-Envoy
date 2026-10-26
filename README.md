# Qwen Envoy — training and evaluating a research agent

**A reproducible ML-agent engineering study.** I built a Qwen3-8B agent that
writes Python to search, read, and cite papers inside Docker, fine-tuned it with
QLoRA, and evaluated base and trained checkpoints on the same 40 QASPER paper
questions. The saved actions, observations, answers, and scoring artifacts are
open to inspect.

[Open the interactive Agent Studio](https://jasonlingg.github.io/Qwen-Envoy/) ·
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

## Companion research MCP

The repository also contains a personal MCP for finding public papers, saving
source-linked ML experiment decisions, recording outcomes, and recalling them
later. [Try the saved workflow](docs/EXPERIMENT_MODE.md) or
[inspect its demo](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/).
A [five-question paired pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md)
did **not** show better decisions or recovery than ordinary web research plus
Markdown notes. The MCP is a working experiment-record interface, not a proven
research-quality improvement.

For setup instructions, source maps, historical experiments, and detailed
limitations, see the [full project guide](PROJECT_GUIDE.md). For the shortest
technical narrative, read the [case study](docs/TECHNICAL_CASE_STUDY.md).
