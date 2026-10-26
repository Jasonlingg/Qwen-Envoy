# Qwen Envoy — full project guide

This is the detailed guide previously at the repository root. For the concise
project and evaluation overview, start with the [main README](README.md).

## Research MCP — from paper evidence to experiment results

For the Nebius × NVIDIA hackathon, this repository includes a personal MCP
workflow for ML engineers: find a public paper, inspect an exact passage, save
what it does and does not support, plan a test, attach the measured result, and
link related attempts and stated decisions across sessions. The optional NVIDIA
Nemotron 3.5 Lightning explanation runs through Nebius Token Factory. Source-byte and
artifact checks establish provenance; a person still has to assess whether a
claim is supported and whether a method helps their workload.

**Start with the research MCP:** [public saved-run inspector](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/) ·
[standalone inspector](release/hackathon-research-mcp/index.html) ·
[local demo and submission evidence](docs/HACKATHON_SUBMISSION_2026.md) ·
[experiment-mode guide](docs/EXPERIMENT_MODE.md) ·
[cross-paper justifications](docs/RESEARCH_JUSTIFICATIONS.md). The inspector shows a saved
run; the MCP service is the working tool interface.

To try the public-paper and local-memory path without an API key or model call:

```bash
pip install -e '.[mcp,vault]'
python scripts/demo_hackathon_research_mcp.py \
  --library-root ~/.envoy/arxiv-library \
  --output out/my-research-demo
```

To use the complete research workflow from an MCP host, register one stdio
server from this checkout (with the same Python environment used above):

```bash
claude mcp add envoy-research -- "$(command -v python)" \
  "$(pwd)/scripts/research_justification_mcp.py"
```

The [generic MCP client configuration](docs/RESEARCH_JUSTIFICATIONS.md#connect-an-mcp-client)
works for hosts other than Claude. This server includes paper discovery, source
inspection, experiment records, decision recall and cross-paper justification.
It starts without a model endpoint or API key; searching or opening a new paper
uses the public arXiv network.

The first run may download public arXiv text. Use a fresh output directory for
each run; `report.json` records the MCP calls and source passage. The separate
`python scripts/demo_experiment_mode.py --output out/my-completed-experiment-demo`
example replays a complete research-to-result record without a model call.
The Nebius explanation requires an explicit `--live-nebius` opt-in and an API
key; see the [demo instructions](docs/HACKATHON_SUBMISSION_2026.md).
For a no-model comparison that challenges one paper against quality-curation
baselines, run [the WebOrganizer walkthrough](docs/RESEARCH_JUSTIFICATIONS.md).
The [single-case decision evaluation](docs/RESEARCH_DECISION_EVAL_20261026.md)
records discovery, claim review, restart recovery and the remaining training gate;
it is not a measured win over ordinary browsing.
The [five-question paired pilot](docs/RESEARCH_MCP_PAIRED_PILOT_20261026.md)
compares the MCP with ordinary web research plus Markdown notes. Both arms
recovered all five decisions; the MCP did not meet the predeclared claim-support
or recovery advantage gates.

## QASPER Agent Studio — a separate Qwen Envoy case study

Qwen Envoy is a bounded document-research worker: **Qwen3-8B writes Python to
search, inspect, and cite research papers inside Docker**. The surrounding
controller supplies observations, enforces limits, and records each action.
QASPER Agent Studio makes saved answers, source spans, errors, and complete
**Runs → Questions → Trace** inspectable. Companion MCP servers expose paper
research and experiment planning to a larger assistant.

This repository connects the worker, execution harness, post-training, evaluation,
and inspection interfaces. Its demonstrated benchmark task is answering questions
about an **assigned paper**, rather than finding papers across the open web.

The questions and paper annotations come from **QASPER, by Dasigi et al.
(2021), licensed CC BY 4.0**. This project adapts QASPER to a bounded code-tool
workflow; it does not introduce an original question benchmark and is not
affiliated with or endorsed by QASPER or AllenAI. See the
[data provenance](benchmarks/envoybench/data/README.md). Existing `envoybench`
package names and repository/demo URLs remain unchanged.

**October 25 training update:** a native **Prime-RL + Verifiers GRPO** pilot
completed **three of ten planned updates**, starting from the original v5
adapter. All three passed checkpoint reload, sampled-policy alignment, and
optimizer-state checks. Collection for update four failed with a transport timeout
coinciding with laptop lid sleep. The GPU and temporary volume were deleted;
adapters and receipts were saved. **The held-out comparison did not run, and no
RL answer-quality improvement is established.** See the
[run report](docs/READER_RL_PILOT_20261025.md). Moving the controller into the cloud
is [scoped but not deployed](docs/READER_CLOUD_CONTROLLER.md).

**October 13 SFT follow-up:** a fresh-base v6 QLoRA pilot used replayed trajectories
and a new, paper-disjoint 24-question QASPER test subset. Blinded,
model-assisted review found **5/17** supported answers for v6, versus **4/17**
for v5 and **2/17** for base. V6 passed only **2/7** insufficient-evidence
questions versus v5's **7/7** and missed its preregistered supported-answer
gain by one question. The decision is sensitive to one ambiguous source
interpretation; it is not a human-validated win. See the
[pilot report](docs/QASPER_V6_PILOT_RESULTS.md) and
[audit bundle](benchmarks/envoybench/pilots/qasper-v6-20261013/README.md).

**Earlier development finding:** on the same 40 saved questions, official QASPER Answer F1 is
**19.92% for base Qwen3-8B and 30.69% for v5 SFT**. On the fixed 20 answerable
questions, however, F1 falls from **29.83% to 21.37%**; the overall increase
comes from the unanswerable group. Error-affected episodes fell from 22/40 to
1/40, while provisional supported-answer passes stayed at **3/20 for both**.
Better protocol execution and more refusals did not establish better substantive
paper answers.

**Start here:** [short findings report](docs/QASPER_AGENT_STUDY.md) ·
[public demo](https://jasonlingg.github.io/Qwen-Envoy/) ·
[research MCP demo](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/) ·
[research MCP portfolio walkthrough](docs/PORTFOLIO_SHOWCASE.md) ·
[Nebius hackathon research MCP demo](docs/HACKATHON_SUBMISSION_2026.md) ·
[technical case study](docs/TECHNICAL_CASE_STUDY.md) ·
[research-to-decision log](docs/RESEARCH_DECISION_LOG.md) ·
[training evidence](release/qwen-v5-training/README.md) ·
[latest GRPO attempt](docs/READER_RL_PILOT_20261025.md) ·
[detailed protocol](benchmarks/envoybench/COMPARISON_2026_09_30.md).

**Scope:** a text-only QASPER agent study with reproducible saved results.
The original 40 questions have been inspected and are development evidence;
the later 24-question pilot is paper-disjoint from training and those earlier
sets, but still has unreviewed QASPER references and provisional answer
judgments. Earlier Obsidian and systems-map prototypes remain historical work.
The [study release](release/qasper-agent-study/README.md) separates original
QASPER answer scoring from provisional support judgments.

## How the parts fit together

| Part | Responsibility | Implementation / evidence |
| --- | --- | --- |
| Data | QASPER papers, questions, answers and evidence annotations; separate training and evaluation splits | [SFT data](data/sft/qasper-v5/README.md), [RL task pool](data/rl/reader-facts-v1/README.md) |
| Qwen worker and harness | Model chooses Python actions; controller executes them in restricted Docker and returns observations | [Investigator](src/product/qwen_investigator.py), [tools](src/env/tools.py), [sandbox](src/env/repl.py) |
| Post-training | TRL/rank-4 QLoRA SFT; native Prime-RL/Verifiers GRPO continuation | [SFT recipe](release/qwen-v5-training/README.md), [native launcher](src/training/prime_run.py), [Docker rollout adapter](src/training/reader_rollout.py) |
| Evaluation | Matched saved runs, execution diagnostics, official QASPER Answer F1 and separately attributed support judgments | [Comparison](benchmarks/envoybench/COMPARISON_2026_09_30.md), [paired RL protocol](docs/READER_PAIRED_EVALUATION.md) |
| Interfaces | MCP tools for another assistant; Studio for inspecting saved evaluation runs | [Research MCP](docs/ARXIV_RESEARCH_MCP.md), [experiment planning](docs/ML_EXPERIMENT_MCP.md), [Studio launcher](scripts/launch_studio.py) |

```text
QASPER training split → SFT / GRPO → Qwen checkpoint
                                          |
Assigned-paper question → controller ↔ model inference endpoint
                              |
                       Docker document tools
                              |
                    answer + evidence + trace

Saved evaluation runs → Agent Studio
Larger assistant → research MCP → source tools / optional Qwen investigator
```

Training changes model weights; the harness controls execution; evaluation checks
the resulting behavior; Studio displays saved results. They are connected parts
of one project. The newest GRPO artifacts are saved separately and are **not
automatically loaded into Studio**. The MCP's source tools can also operate
without a Qwen endpoint. Exact citation spans establish provenance, not semantic
support or answer correctness.

In the latest training deployment, Docker, collection and the resource watchdog
ran on the Mac, while native inference and training ran on a Runpod GPU. That
layout requires the Mac to stay **plugged in, awake, with its lid open and Docker
running** throughout the job. Cloud placement of the controller is pending;
neither SSH detachment on the GPU nor a local watchdog removes this dependency.
Opening the saved Studio requires no GPU and makes no model calls.

## Post-training: what has actually run

The pinned v5 checkpoint was trained using **TRL SFT and rank-4 QLoRA** on
**35 curated, replay-verified training conversations**, with nine validation
conversations. They expand into 106 training and 26 validation next-action targets.
Of all 44 conversations, seven retain original teacher trajectories and 37 contain
assistant-authored repairs that were executed again against the tools. Calling
all 35 training conversations unmodified Claude demonstrations would be inaccurate.
Per-action supervision masks earlier context and checks token-for-token equality
between training and inference prefixes. See the
[data card](data/sft/qasper-v5/README.md) and
[observed training evidence](release/qwen-v5-training/README.md).

The newer RL path uses the existing Docker investigator through a Verifiers
plugin, with Prime-RL owning the native GRPO trainer and optimizer. The latest
attempt verified all **504 optimizer states at steps 0 → 1 → 2 → 3**, preserved
policy/checkpoint continuity, and retained raw traces separately from admitted
training batches. It attempted 23 training questions and recorded 92 episodes;
91 were infrastructure-healthy and one failed in transport. Only three complete
mixed-reward groups were admitted. The training-only reward checks finite accepted
short answers and reviewed inspected evidence; it is not a general semantic judge.

The ten-update pilot and its 12-question paired comparison remain incomplete.
Original v5 remains the reference checkpoint; there is no automatic promotion of
the partial RL adapters. Native training checks and increased reward alone cannot
establish better research answers. [Actual results and limitations](docs/READER_RL_PILOT_20261025.md)
are separate from the [prospective protocol](examples/reader-rl-pilot-v2/protocol.json).

## Open the demo

For the separate **personal ML research and experiment workflow**, see
[experiment mode](docs/EXPERIMENT_MODE.md). It connects paper research, saved
experiment context and measured outcomes through a local MCP. Run the complete
recorded-data example with `python scripts/demo_experiment_mode.py --output
out/my-experiment-demo`; it needs the repository dependencies with `mcp` and
`vault` extras, and makes no model calls or GPU launches. This personal tooling
does not change the QASPER study's results or establish a research-quality win.
An optional [Qwen passage reader](docs/PASSAGE_READER.md) now connects selected
source windows to saved experiment notes. A [repaired-harness live rerun](examples/passage-reader-rerun-1/README.md)
returned two supported answers under assistant source review and one incomplete
gap case; the overall test still did not pass. This is not a model-training gain.
A [matched base-model smoke](examples/passage-reader-base/README.md) completed
one answerable case versus V5's two, mainly differing in protocol compliance;
three exposed development cases do not establish a general fine-tuning gain.
Gap completion now uses a [single terminal action](examples/atomic-passage-gap/README.md),
verified locally through MCP/Docker; live Qwen adoption of that change is pending.

**No setup:** [open the public Studio](https://jasonlingg.github.io/Qwen-Envoy/).
It includes the saved runs and makes no model calls; no GPU, Docker, login,
or API key is needed. You can also download [the standalone HTML](release/qasper-agent-study/studio.html)
and open it offline. GitHub's file preview does not run the HTML.

**Recompute the recorded numbers:** after cloning this repository, Python 3.10+
alone is enough. No package installation, network access, GPU, or model call:

```bash
python scripts/reproduce_case_study.py
```

This checks the release hashes, binds the saved blind reviews to the runs,
and recomputes the table below from the recorded artifacts. Add `--json` for
machine-readable output. It reaggregates existing provisional judgments; it
does not independently grade answers or reconstruct the paper corpus.

To recompute Answer F1 against all original QASPER annotations for these
questions, also without inference or downloads:

```bash
python -m benchmarks.envoybench.qasper_official score \
  --run-dir release/envoybench-v0.1/run
```

The [saved F1 report](release/qasper-agent-study/qwen-official-score.json) uses
the unchanged official evaluator on this selected subset, not the full QASPER
test split or an official leaderboard submission.

**Local Studio:** with Python 3.10+, run from a checkout:

```bash
git clone https://github.com/Jasonlingg/Qwen-Envoy.git
cd Qwen-Envoy
python -m venv .venv
source .venv/bin/activate
pip install -e '.[viewer,qasper]'
python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data
python scripts/launch_studio.py
```

Open <http://127.0.0.1:8765>. The data builder downloads the pinned QASPER
revision if needed; viewing the recorded runs does not perform inference.
The [release guide](release/envoybench-v0.1/README.md) documents artifact
verification and the distinction between saved and live runs.

For a dependency-free local demo server, run
`python -m http.server 8765 --bind 127.0.0.1 --directory release/qasper-agent-study`
and open <http://127.0.0.1:8765/studio.html>. This serves the saved snapshot only;
the full local Studio above adds source review and optional paper investigations.

## September 30 base/v5 comparison

The September 30 run compared base Qwen3-8B with the v5 SFT adapter on the same
40 QASPER-derived candidate questions. Each question names its paper, so this
tests known-paper answering and evidence selection, not discovery of papers.

| Measure | Base Qwen3-8B | v5 SFT |
| --- | ---: | ---: |
| Official QASPER Answer F1, selected 40 questions | 19.92% | 30.69% |
| Answer F1, fixed 20 answerable questions | 29.83% | 21.37% |
| Answer F1, fixed 20 unanswerable questions | 10.00% | 40.00% |
| Provisional passes, all questions | 5/40 | 15/40 |
| Supported-answer passes on answerable questions | 3/20 | 3/20 |
| Correct abstentions on unanswerable questions | 2/20 | 12/20 |
| Episodes with a recorded tool, runtime, or endpoint error | 22/40 | 1/40 |
| Final answers submitted | 29/40 | 40/40 |

**Answer-quality judgments are model-assisted and provisional.** References and
answers have not been independently human-reviewed. V5 improved correct
abstention and execution reliability in this run; supported answering on the
answerable subset did not improve. An always-refuse policy would be expected
to pass 20/40 under these labels, illustrating why the aggregate score alone
is misleading. That is an analytical baseline, not another recorded run.

A separate [fixed BM25 run](release/qasper-agent-study/README.md#fixed-retrieval-baseline)
retrieved passages from each named paper before one Qwen answer request. It
scored **41.52%** for base and **58.86%** for v5 on the same selected questions.
V5 falsely refused 9/20 answerable questions; the higher overall F1 does not
establish better grounded answers. The Studio shows both saved runs and their
retrieved passages.

Answer F1 uses the literal submitted answer and the best match among all
original annotations: 86 references across these 40 questions. Missing answers
score zero, including the eleven failed base episodes. It measures normalized
token overlap, not semantic support; it does not replace the separate judgments.
The two subgroup F1 means each retain all 20 questions, including missing
predictions. They differ from the upstream answer-type breakdown, whose
denominators exclude missing answers.

All 11 missing base submissions hit the shared 8,192-token serving cap.
Historical timing and usage are incomplete, so no full cost or latency win is
claimed. See the [case-study report](benchmarks/envoybench/COMPARISON_2026_09_30.md)
for protocol details, source bindings, and limitations.

V5 submitted at step three on **38/40** questions and at step four on two.
That is consistent with short training demonstrations: **27/35** training
episodes end by step three, and **30/35** were assistant-repaired. For those
30, mean length fell from **5.033 original actions to 3.067 repaired actions**.
This suggests a stopping bias worth investigating; it does not prove that
repairs caused the behavior or that a short investigation found enough evidence.
The [short report](docs/QASPER_AGENT_STUDY.md) records the provenance and limits.

The October 9 **Nemotron Ultra/Nebius reference run completed under an amended
protocol**. A $2 local cost guard stopped the original attempt during question
39; the user then authorized a **$25 total** cap. Question 39 restarted at step
1 because its live state could not be restored, and question 40 received its
first attempt. The [derived 40-question result](release/qasper-agent-study/nebius-amended-run/results.json)
has **28 submissions and 12 missing predictions**. Its
[official QASPER Answer F1](release/qasper-agent-study/nemotron-official-score.json)
is **19.03%**, counting missing predictions as zero. Cumulative catalog-rate
estimated usage is **$2.035422 for 383 requests**, not an invoice. The
[original partial traces](release/qasper-agent-study/nebius-run/results.partial.json)
remain preserved; see the [reference-run record](release/qasper-agent-study/README.md#bounded-hosted-reference)
for lineage and comparison limits.

Studio shows the amended Nemotron run as a fifth row. Its saved traces and
Answer F1 can be inspected alongside the four earlier Qwen runs. The hosted
answers have no independent support review, and differences in provider,
model, context cap, and runner prevent a controlled Qwen–Nemotron comparison.

The separate October 3 two-question token diagnostic is **unscored**. Its
generated-token logprobs are likelihoods of emitted tokens, not probabilities
that answers are correct. Neither it nor the interrupted larger diagnostic
changes the September 30 scores. See the [logprob analysis](benchmarks/envoybench/LOGPROB_ANALYSIS.md).

## Read your own paper

The [paper reader](benchmarks/envoybench/PAPER_READER.md) fetches PDFs from
arXiv/direct HTTPS links and accepts PDF/Markdown/text uploads. It saves model
answers, exact source spans, code-tool traces, and observed usage separately
from the benchmark. These answers are unreviewed and receive no benchmark score.

**PDF support is text extraction only.** Images and diagrams are not passed
to a vision model; table rows/columns are not reliably preserved. Scanned PDFs
need OCR, which is not implemented. Figures, tables, and equations may be
missing from the model's input.

Live investigations require Docker and a configured model endpoint. To use
the optional Nebius setup, put `NEBIUS_API_KEY` in your local `.env` or
environment, build the sandbox, and launch explicitly:

```bash
docker build -f benchmarks/envoybench/Dockerfile -t rlm-sandbox .
python scripts/launch_studio.py --live-nebius
```

Only an explicit **Run** on `/papers` starts inference and consumes API credits.
The [live validation record](release/envoybench-v0.1/LIVE_VALIDATION.md) covers
four NVIDIA Nemotron/Nebius attempts: three Lightning failures and one Ultra
completion with a correct answer but incomplete citations. This verifies a
local integration, not reliable semantic support or a Qwen improvement.

The [v5 model card](benchmarks/envoybench/MODEL_CARD.md) records the checkpoint,
QLoRA recipe, and data. The public technical release is separate from a
hackathon submission; the historical [submission draft](release/envoybench-v0.1/DEVPOST_DRAFT.md)
does not constitute a completed entry.

## arXiv research MCP

Discover papers, follow citations, and inspect exact version-pinned passages
through five model-free source tools.
```bash
claude mcp add arxiv-research -- python "$(pwd)/scripts/arxiv_research_mcp.py" --tool-profile sources
```
The full profile adds an optional investigation tool. Configured Qwen/Nemotron
readers run Python in Docker; quotes establish provenance, not semantic support.
See [setup, limits and evaluation](docs/ARXIV_RESEARCH_MCP.md).

## ML experiment planning through MCP

Ask an engineering question such as “Would fine-tuning address these failures?”
The host assistant inspects selected run exports, researches relevant methods,
and saves a proposed experiment behind the conversation. The companion MCP
keeps versioned briefs, immutable agreed plans, evidence receipts, and supplied
results compared against the original criteria. Missing measurements stay unknown.

```bash
python scripts/ml_experiment_mcp.py \
  --store-root ~/.envoy/ml-experiments \
  --library-root ~/.envoy/arxiv-library \
  --artifact-root /absolute/path/to/selected-run-exports
```

Use the same paper cache as the research server and a dedicated directory of
sanitized JSON exports. The planning server makes no model/network calls and
runs no experiments. Qwen readers are optional and require a separately
configured endpoint; this workflow needs no GPU to plan or review saved results.

See [setup and conversation flow](docs/ML_EXPERIMENT_MCP.md) and the
[real Qwen development walkthrough](examples/ml-experiment-qwen-v5/README.md).
That walkthrough leaves its proposed intervention as an unrun draft. Neither
Qwen improvement nor an advantage over ordinary assistant research has been
established; the Codex comparison is a separate proposed benchmark.

## What the agent does

Each episode gives the model a question and a bounded Python tool session:

```text
Question
   |
   v
Qwen writes Python  ──>  search / read / extract / verify
   ^                              |
   |                              v
   +──────────── stdout and errors
   |
   v
SUBMIT: <answer> CITATIONS: [<document IDs>] EVIDENCE: [<source spans>]
```

The model can filter search results, inspect documents, run regular expressions,
and revise queries using earlier output. Execution depends on the backend:

- The local backend keeps one Python worker alive per episode and executes each action once.
- The released benchmark and live paper reader require Docker. Its current backend reconstructs
  state by replaying prior successful actions; it is not equivalent to the local worker, and
  measured episode time includes replay overhead.

An illustrative episode looks like this (the final placeholders must be replaced
with the actual answer, source ID, and observed character offsets):

```text
# action 1
hits = search("retrieval token masking ablation", top_k=5)
print([(h["doc_id"], h["title"]) for h in hits])

# action 2, after seeing stdout
text = read(hits[0]["doc_id"])
print(extract(hits[0]["doc_id"], r"(?i).{0,180}mask.{0,240}"))

# final action
SUBMIT: <answer> CITATIONS: ["<paper_id>"] EVIDENCE: [{"doc_id":"<paper_id>","start":<start>,"end":<end>}]
```

## Project status

| Component | Status |
| --- | --- |
| Recorded Studio | Runs, questions, full transcripts, provisional scores; standalone HTML available |
| September 30 base/v5 comparison | Inference complete; model-assisted review only; no validated promotion |
| October 9 Nemotron/Nebius reference | Amended 40-question result: 19.03% official Answer F1; 28 submissions; original budget-stopped trace preserved |
| Source and anonymous answer review | Local review interfaces available; independent review remains incomplete |
| Paper reader | Frozen text snapshots, optional endpoint inference, saved evidence and traces |
| Token logprobs | Recorded in a separate small diagnostic; no calibrated correctness score |
| Qwen v5 training recipe | Pinned adapter and observed QLoRA settings documented in the model card |
| Native Prime-RL / Verifiers continuation | Three verified updates; ten-update pilot and held-out comparison incomplete; no RL quality gain claimed |
| Cloud controller | Proposed; current training placement still depends on an awake Mac running Docker |
| Retrieval, thinking, and prefix-cache comparisons | Prepared protocols; no completed comparative result claimed |
| Historical training and Obsidian/map demos | Training evidence retained; personal-memory and map apps archived in Git |
| Hackathon submission | Separate historical draft; not a completed submission |

## Historical training experiments

These experiments use different questions, checkpoints, and protocols from the
September 30 release. Their pass counts are not directly comparable. Historical
QASPER semantic judgments were model-assisted; development results do not
establish an independently validated held-out improvement.

<details>
<summary>Expand earlier MuSiQue, QASPER, SFT, and GRPO results</summary>

### Historical MuSiQue experiment

The original model experiment used 50 held-out MuSiQue development questions. All three policies
used the same corpus, prompt, step budget, decoding settings, and reward implementation.

| Policy | Outcome reward | Answer F1 | Citation precision | Citation recall |
| --- | ---: | ---: | ---: | ---: |
| Qwen2.5-7B base | 0.158 | 0.127 | 0.350 | 0.215 |
| Qwen2.5-7B + SFT | **0.176** | 0.140 | **0.400** | **0.238** |
| Qwen2.5-7B + SFT + GRPO | 0.172 | **0.146** | 0.350 | 0.207 |

SFT improved the measured task. The surviving GRPO artifact did not beat SFT, and the two
published GRPO model IDs turned out to contain the same adapter. See [RESULTS.md](RESULTS.md) for
the full protocol and per-run manifests.

### Qwen3-8B research baseline

Base Qwen3-8B was evaluated before domain training. These datasets differ from MuSiQue and from
each other, so their reward values should not be compared across rows.

| Evaluation | Questions | Outcome reward | Citation precision | Citation recall |
| --- | ---: | ---: | ---: | ---: |
| Frozen AI-paper pilot | 10 | 0.445 | 0.850 | 0.950 |
| QASPER test conversion | 20 | 0.355 | 0.800 | 0.950 |

The model usually found the right paper but was less reliable at using it. It guessed on
unanswerable questions, sometimes answered the general topic instead of the requested fact, and
occasionally mischaracterized a correctly cited passage. That diagnosis led to the earlier
QASPER-grounded SFT experiment. See
[docs/QWEN3_BASELINE_PILOT.md](docs/QWEN3_BASELINE_PILOT.md).

### Qwen3-8B SFT and harness diagnosis

An earlier base model and targeted QASPER SFT adapter were compared on a locked
40-question set subsequently used during development. SFT raised outcome reward
from 0.152 to 0.203 and provisional model-assisted semantic passes from 9/40 to
19/40. The adapter did not pass the full promotion rule. These are not the
September 30 v5 results.
It also eliminated malformed tool actions in this run and increased required-paper recall from
0.65 to 1.00. The adapter is better at operating the research environment, although 7 of the 20
answerable questions were still refused incorrectly.

A follow-up ablation separated model failures from harness failures. In several cases, Qwen wrote
a reasonable query but `search_within()` returned only the three highest-scoring windows. Those
windows often overlapped, while the passage containing the answer ranked fourth through eighth.
The model cannot reason from evidence the harness never places in its context.

On a 25-question development diagnostic selected from the earlier failures, changing only the
number of returned windows from three to eight increased outcome reward from 0.266 to 0.314 and
semantic passes from 10/25 to 12/25 without breaking any of the ten control cases. A longer prompt
performed worse: it encouraged more tool calls and increased episodes containing an exact repeated
action from 1 to 7 at top three, while semantic passes fell from 10 to 9.

These diagnostic numbers are not held-out performance because the questions were selected after
examining failures. Their value is causal: retrieval depth explains part of the remaining error,
while prompt wording alone does not.

An oracle-evidence follow-up removed retrieval entirely on five valid cases by handing both models
the same answer-bearing passages. SFT produced 4 passes, 1 partial, and 0 failures; base produced 2
passes, 2 partials, and 1 failure. This small diagnostic says the adapter can use good evidence and
the next change belongs in retrieval/passage presentation, not another training run. It also
exposed a metric problem: token-overlap reward ranked base higher even though semantic review
favored SFT, because complete yes/no sentences often share no tokens with references such as
`Yes`. See
[docs/QASPER_FAILURE_ATTRIBUTION.md](docs/QASPER_FAILURE_ATTRIBUTION.md) for the transcripts,
controls, and next experiment.

The passage-presentation experiments also produced useful negative results. Aggressively merging
overlapping windows raised automatic reward to 0.330 and rescued two target failures, but broke one
previously correct control; a conservative merger preserved the controls but rescued nothing. A
`ranked_diverse` mode then kept the original top three and added three non-overlapping windows. It
looked strong offline and scored 0.328 end to end, but semantic review found only 7/10 controls
still correct. None of these modes became the default. In that diagnostic, raw
top eight was the only tested retrieval change that rescued two target failures
while retaining all ten controls. The recorded query-planning follow-up is a
historical proposal, not the current release plan.

### Qwen3-8B GRPO pilot

Starting from the targeted SFT adapter, a bounded QASPER GRPO run found a useful early checkpoint
and then regressed with more updates. Standard GRPO peaked at 0.5233 automatic reward after five
updates, versus 0.4563 for unchanged SFT, before falling to 0.4535 at update 20. A one-variable
follow-up removed per-group standard-deviation scaling. Its second checkpoint reached **0.5485**,
with 15 correct abstentions, 3 false refusals, 39/40 valid submissions, and the same three
execution-error episodes as SFT.

The centered checkpoint beat SFT by 0.0922 mean reward on the paired 40-question development set;
a question-level bootstrap gave a 95% interval of [0.0146, 0.1914]. Its smaller 0.0252 lead over
the earlier standard-GRPO checkpoint was inconclusive. Reviewing the seven outputs that changed
from SFT found four clear improvements and one clear regression, plus two cases where the overlap
reward did not reflect semantic quality cleanly. This is evidence of improvement over SFT on the
development task, not a final held-out or product-readiness result. See the complete setup,
failure analysis, artifacts, and limits in
[docs/qasper-grpo/README.md](docs/qasper-grpo/README.md).

### Historical training approach

The targeted SFT run distilled QASPER-grounded research trajectories into Qwen3-8B with rank-4 QLoRA:

1. QASPER supplies paper questions, evidence, and expert answerability labels.
2. A stronger teacher produces multi-step Python trajectories against the frozen corpus.
3. The pipeline replays trajectories and checks that actions execute and cited documents exist;
   QASPER's labels provide the answerability target.
4. Each assistant action becomes a next-action example containing its complete preceding history.
5. Loss is applied only to the action tokens, using the same non-thinking Qwen3 chat prefix used at
   inference.
6. Base and trained Qwen are compared with identical tools, prompts, questions, decoding, and
   step limits.

The per-action representation fixed a real failure in the first Qwen3 run: intermediate code
actions had been formatted differently during training and inference, while the inference prefix
appeared only before `SUBMIT` in the training data. The model learned that accidental correlation
and submitted too early. The diagnosis and token-level checks are documented in
[docs/QWEN3_SFT_PREFIX_DIAGNOSIS.md](docs/QWEN3_SFT_PREFIX_DIAGNOSIS.md).

The project also found and fixed an independent GRPO bug. PPO ratios were computed from warped
generation scores instead of fresh policy logits, which suppressed gradients for some samples.
Regression coverage lives in `tests/test_grpo_loss.py`.

### Scale-SFT continuation: negative result

A later run continued the targeted Qwen3-8B adapter on 995 QASPER-derived code-execution
conversations. The starting adapter, epoch 1, and epoch 2 were evaluated on the **same new**
40-question QASPER validation set, with identical tools, corpus, prompt, decoding, and seed.
The identity-blind, model-assisted semantic review found 15, 10, and 4 passes respectively.
Mean automatic reward fell from 0.363 to 0.179 and 0.127. Neither continuation checkpoint met
the preregistered promotion rule. This is a separate evaluation from the earlier 9/40 versus
19/40 base-to-targeted-SFT comparison, so the two sets of pass counts are not directly comparable.

The failure was concentrated in answerability: epoch 1 improved on answerable questions (5 to
8 passes out of 20) but fell on insufficient-evidence questions (10 to 2 passes out of 20).
The continuation training set had only 31 abstention examples among 796 training conversations.
That imbalance is a plausible mechanism, not a causal conclusion. No additional
training is part of the current Studio release. See the
[public experiment report](reports/qasper-scale-sft-2026-09-25.json) and
[demo walkthrough](docs/DEMO_AND_EVIDENCE.md).

</details>

## Evaluation

EnvoyBench stores question and corpus identities, declared model revisions,
policy settings, full trajectories, and review provenance. Protocol hashes
bind paired comparisons. Recorded gaps, including missing historical timings
and unverified endpoint weight identity, remain explicit.

Verify the packaged case study after building the corpus with the demo setup:

```bash
python -m benchmarks.envoybench.verify \
  --run-dir release/envoybench-v0.1/run \
  --review-dir release/envoybench-v0.1/review-prepared \
  --judged-review release/envoybench-v0.1/review-model-assisted/review.json \
  --provisional
```

This checks artifact integrity, paired completeness, exact spans, and review
bindings. It does not independently judge answer correctness or validate the
references. The candidate split remains distinct from the older development
sets; upstream model pretraining exposure to these papers is unknown.

Historical overlap rewards are separate from supported-answer judgments.
`src/env/reward.py` defines the active MuSiQue reward; document-ID overlap and
exact quote matching do not establish that a claim follows from its evidence.
The earlier [abstention preregistration](docs/SFT_ABSTENTION_PREREGISTRATION.md)
is preserved as experiment history, not the current release's validation status.

## Repository map

| Path | Purpose |
| --- | --- |
| `benchmarks/envoybench/` | Candidate data protocol, paired runner, review tools, verifier, and Studio |
| `benchmarks/envoybench/saved_payload.py`, `demo.py` | Saved-run validation and Studio web routes, respectively |
| `benchmarks/envoybench/frozen_dataset.py`, `runtime_support.py` | Shared frozen-data and model/sandbox utilities used by the runner and Studio |
| `release/envoybench-v0.1/` | Saved comparison, review artifacts, standalone Studio, and release notes |
| `scripts/launch_studio.py` | Main local demo entry point; live Nebius inference is opt-in |
| `src/env/` | Gym-style document environment, persistent REPL, corpus, tools, and reward |
| `src/policies/` | Qwen base/SFT/GRPO policies, Claude reference policy, and RAG baselines |
| `src/eval/` | Evaluation harness, manifests, scoring, and abstention metrics |
| `src/research/` | Paper/vault ingestion and the earlier structured research-agent path |
| `src/product/qwen_investigator.py` | Bounded Docker-only paper investigation used by the live reader |
| `scripts/train_sft.py` | Qwen3 QLoRA training with per-action target masking |
| `scripts/train_grpo_custom.py` | Custom GRPO loop for the historical MuSiQue experiment |
| `scripts/run_eval.py` | Shared policy evaluation entry point |
| `scripts/research_vault.py` | Immutable Markdown/PDF/Obsidian snapshot import and export |
| `scripts/viewer.py` | Earlier general-purpose trajectory viewer; separate from Studio |

See the [current code map](docs/REPOSITORY_LAYOUT.md) for the saved-viewer,
live-reader, and historical-prototype boundaries.

The local worker executes each action once. The optional Docker backend still rebuilds state by
replaying successful actions and is not considered behaviorally equivalent yet.

Core stack: Python, PyTorch, Hugging Face Transformers, TRL, PEFT/QLoRA, bitsandbytes, FAISS,
sentence-transformers, Docker, and pytest.

The saved Studio does not require FAISS or sentence-transformers. They are in
the optional `vector` extra for indexed RAG; the `dev` extra includes them for
the full test suite.

## Development and historical commands

Use **Open the demo** above for the current release. The commands here are for
environment development and earlier experiments. From an existing checkout:

```bash
source .venv/bin/activate
pip install -e ".[dev]"

# Build the small synthetic corpus used by local tests and examples.
python scripts/setup_corpus.py

# Run the test suite.
pytest -q
```

Import a folder from an Obsidian vault into an immutable corpus snapshot:

```bash
pip install -e ".[vault]"
python scripts/research_vault.py import \
  --vault /path/to/MyVault \
  --collection "AI Research" \
  --output out/research/my-vault-snapshot
```

Run base Qwen3-8B on the historical AI-paper pilot (GPU and the named local
corpus snapshot required; it is not created by the Studio setup):

```bash
pip install -e ".[training]"
BASE_MODEL_PATH=Qwen/Qwen3-8B python scripts/run_eval.py \
  --policy qwen_base_policy \
  --questions data/research/code_exec_pilot_v1.json \
  --corpus out/research/starter-2026-09-12/corpus \
  --max-steps 10 \
  --seed 42 \
  --require-evidence \
  --no-vector-index \
  --run-label qwen3_base \
  --output out/research/qwen3-baseline/base.json
```

Re-run an earlier Qwen3 adapter experiment with its original local artifacts
(these questions have since been used during development):

```bash
CHECKPOINT_PATH=/path/to/adapter/final \
  ./scripts/run_qwen3_qasper_abstention_eval.sh
```

See [docs/GPU_TRAINING_READINESS.md](docs/GPU_TRAINING_READINESS.md) before starting a GPU run and
[docs/EVAL_RUNBOOK.md](docs/EVAL_RUNBOOK.md) before comparing checkpoints.

### Earlier trajectory UI

This is the older general-purpose viewer, not the Studio release entry point.

The viewer can run an agent live, stream every Python action and tool result, and browse saved
evaluation transcripts. It discovers the synthetic, MuSiQue, QASPER, AI-paper, and imported-vault
corpora available under `out/research/`. Select a saved question or type a new one. **Watch
Replay** runs bundled, recorded trajectories without loading a model or using an API key. The
selector includes two paired three-checkpoint Qwen cases and a Claude reference trace; each is
labeled by source and review status in the UI.

```bash
pip install -e ".[viewer]"
python scripts/viewer.py
# Open http://127.0.0.1:8000
```

The UI reads its policy list from the shared registry rather than hardcoding Qwen or Claude. To use
an OpenAI-compatible vLLM, Ollama, LM Studio, or hosted endpoint:

```bash
export ENVOY_MODEL_ENDPOINT=http://localhost:8000/v1
export ENVOY_MODEL_ID=Qwen/Qwen3-8B

# Useful for a Qwen3 model served by vLLM:
export ENVOY_MODEL_EXTRA_JSON='{"chat_template_kwargs":{"enable_thinking":false}}'

python scripts/viewer.py --port 8001
# Viewer: http://127.0.0.1:8001 (separate from the model endpoint on port 8000)
```

The same adapter works in the command-line evaluator:

```bash
python scripts/run_eval.py \
  --policy openai_compatible \
  --questions data/research/code_exec_pilot_v1.json \
  --corpus out/research/starter-2026-09-12/corpus \
  --question-only-observation \
  --no-vector-index
```

At the harness boundary, a policy only needs `act(observation) -> action` and `reset()`. The
environment, REPL, tools, transcript format, UI, and scoring code do not depend on the provider or
model family.

## Agent tools

| Tool | Description |
| --- | --- |
| `search(query, top_k=5)` | Passage BM25, returning documents ranked by their best passage |
| `search(query, method="chunk")` | Legacy TF-IDF scoring over overlapping text windows |
| `read(doc_id)` | Read a complete document |
| `passage(doc_id, start, length)` | Return an exact passage with stable character offsets |
| `extract(doc_id, pattern)` | Run a regular expression over one document |
| `scan(doc_id, pattern)` | Return bounded context around regex matches across a full document |
| `aggregate(doc_ids, field)` | Collect a metadata field across documents |
| `search_within(doc_id, query)` | Rank passages inside one document |
| `verify(doc_id, claim)` | Check claim keywords and return an excerpt; not semantic entailment |
| `list_docs()` | List document IDs, titles, and lengths |

## Current limitations

- Scores and reference answerability labels lack independent human validation.
  No held-out supported-answer improvement or model promotion is established.
- Exact source spans verify provenance, not whether a passage supports a claim.
- The reader is text-only; figures, structured tables, and scanned pages are not reliably parsed.
- Known-paper questions do not measure open-ended research or paper discovery.
- Token logprobs are diagnostic telemetry, not calibrated answer confidence.
- Historical timing/usage gaps prevent a complete cost or latency comparison.
- Local Qwen training/evaluation used CUDA GPUs. Hosted inference needs a configured endpoint;
  saved Studio runs can be viewed without a GPU or API key.
- Docker execution does not yet match the persistent local worker exactly.

## Earlier product prototypes

The personal-memory web/MCP apps, Learning Lab, systems maps, and their dedicated
launchers and evaluation scripts are **archived in Git**, outside the maintained
source tree. Their reports, fixtures and saved evidence remain unchanged.
[Archived prototypes](docs/ARCHIVED_PROTOTYPES.md) lists the removed features and
how to recover the complete earlier implementation at commit `5240090`.
Commands in their dated documentation require that earlier checkout.

The current research and experiment MCPs, Qwen investigator, training/evaluation
pipeline and Studio remain available. Shared snapshot metadata/link helpers and
the earlier staging harness remain where current callers use them; this cleanup
does not establish the historical weekly-digest product gates or a cost advantage
for the small research worker. See the [repository layout](docs/REPOSITORY_LAYOUT.md).

The original code-execution environment was inspired by
[arXiv:2512.24601](https://arxiv.org/abs/2512.24601).

## Documentation

- [Release bundle](release/envoybench-v0.1/README.md): saved demo and verification instructions
- [Benchmark protocol](benchmarks/envoybench/README.md): splits, runners, scoring, and review
- [Latest comparison](benchmarks/envoybench/COMPARISON_2026_09_30.md): provisional results and failures
- [Paper reader](benchmarks/envoybench/PAPER_READER.md): optional live endpoint setup
- [V5 model card](benchmarks/envoybench/MODEL_CARD.md): observed training recipe and checkpoint identity
- [Code-execution second brain](docs/CODE_EXECUTION_SECOND_BRAIN.md) and
  [weekly research radar](docs/WEEKLY_RESEARCH_RADAR.md): historical product plans
- [Qwen3 baseline pilot](docs/QWEN3_BASELINE_PILOT.md): measured research-agent failures
- [Qwen3 SFT prefix diagnosis](docs/QWEN3_SFT_PREFIX_DIAGNOSIS.md): failed run analysis and repair
- [QASPER GRPO experiment](docs/qasper-grpo/README.md): historical checkpoint curve and optimization diagnosis
- [Evaluation runbook](docs/EVAL_RUNBOOK.md): reproducible checkpoint comparison
- [Obsidian workflow](docs/OBSIDIAN_WORKFLOW.md): snapshot import and cited-note export
- [Demo and evidence](docs/DEMO_AND_EVIDENCE.md): recorded Qwen traces, reproducibility, and
  recruiter-facing scope
- [Results](RESULTS.md): dated experiment history

## License

MIT
