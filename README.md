# Qwen Envoy — training and inspecting a code-executing research agent

Qwen Envoy is a small-model training and evaluation case study. Qwen writes
Python to search, read, and cite a frozen paper collection; a bounded execution
environment returns tool observations and errors. EnvoyBench records the
comparison, and Studio lets you inspect **Runs → Questions → Trace**.

**Main finding:** in the recorded base-versus-v5 comparison, error-affected
episodes fell from 22/40 to 1/40, while provisional supported-answer passes on
answerable questions stayed at 3/20 for both models. The higher aggregate score
comes from correct abstention. This is a useful failure analysis, not evidence
that fine-tuning produced a better general research assistant.

**Start here:** [technical case study](docs/TECHNICAL_CASE_STUDY.md) ·
[saved-run demo](release/envoybench-v0.1/studio.html) ·
[training evidence](release/qwen-v5-training/README.md) ·
[detailed protocol](benchmarks/envoybench/COMPARISON_2026_09_30.md).

**Current scope, October 9, 2026:** publish the text-only technical case study
and inspectable v0.1 evaluation candidate. Recommendations, new memory products,
graph-aware model training and partial-paper reasoning tests are outside this
release. Earlier Obsidian and systems-map prototypes remain as historical work.

EnvoyBench builds on **QASPER's existing paper questions**, with a bounded
code-tool protocol and inspectable failure diagnostics. It is a candidate
evaluation protocol, not an independently validated leaderboard or a new
collection of original questions. See the [benchmark guide](benchmarks/envoybench/README.md)
and [data provenance](benchmarks/envoybench/data/README.md).

## Open the demo

**No setup:** download [the standalone Studio](release/envoybench-v0.1/studio.html)
and open it in a browser. It includes the saved runs and makes no model calls;
no GPU, Docker, or API key is needed. GitHub's file preview does not run the HTML.

**Recompute the recorded numbers:** after cloning this repository, Python 3.10+
alone is enough. No package installation, network access, GPU, or model call:

```bash
python scripts/reproduce_case_study.py
```

This checks the release hashes, binds the saved blind reviews to the runs,
and recomputes the table below from the recorded artifacts. Add `--json` for
machine-readable output. It reaggregates existing provisional judgments; it
does not independently grade answers or reconstruct the paper corpus.

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
`python -m http.server 8765 --bind 127.0.0.1 --directory release/envoybench-v0.1`
and open <http://127.0.0.1:8765/studio.html>. This serves the saved snapshot only;
the full local Studio above adds source review and optional paper investigations.

## Latest recorded comparison

The September 30 run compared base Qwen3-8B with the v5 SFT adapter on the same
40 QASPER-derived candidate questions. Each question names its paper, so this
tests known-paper answering and evidence selection, not discovery of papers.

| Measure | Base Qwen3-8B | v5 SFT |
| --- | ---: | ---: |
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

All 11 missing base submissions hit the shared 8,192-token serving cap.
Historical timing and usage are incomplete, so no full cost or latency win is
claimed. See the [case-study report](benchmarks/envoybench/COMPARISON_2026_09_30.md)
for protocol details, source bindings, and limitations.

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
QLoRA recipe, and data. The release is not a completed hackathon submission;
public hosting, video, sponsor feedback, and submission remain pending in the
[release checklist and draft](release/envoybench-v0.1/DEVPOST_DRAFT.md).

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
| Source and anonymous answer review | Local review interfaces available; independent review remains incomplete |
| Paper reader | Frozen text snapshots, optional endpoint inference, saved evidence and traces |
| Token logprobs | Recorded in a separate small diagnostic; no calibrated correctness score |
| Qwen v5 training recipe | Pinned adapter and observed QLoRA settings documented in the model card |
| Retrieval, thinking, and prefix-cache comparisons | Prepared protocols; no completed comparative result claimed |
| Historical training and Obsidian/map demos | Retained for reproducibility; outside the current release scope |
| Hackathon submission | Local artifacts and draft ready; public hosting, video, feedback, and submission pending |

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

The local worker executes each action once. The optional Docker backend still rebuilds state by
replaying successful actions and is not considered behaviorally equivalent yet.

Core stack: Python, PyTorch, Hugging Face Transformers, TRL, PEFT/QLoRA, bitsandbytes, FAISS,
sentence-transformers, Docker, and pytest.

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

The repository retains earlier Obsidian, memory, and map experiments. These are
outside the current Studio release scope; their dated plans are not the active
shipping roadmap. The proposed cost benefit of a small research worker has not
been established by a complete cost comparison.

- [Learning-memory API](docs/PERSONAL_MEMORY_LOCAL_DEMO.md) and
  [Learning Lab](docs/LEARNING_LAB_VERTICAL_SLICE.md): local notes, exact source
  receipts, and approval-gated revisions. `python scripts/launch_sample_demo.py`
  opens a disposable synthetic-note demo; `/chat` is its older chat page.
- [Question desk](docs/QUESTION_DESK_PRODUCT_DIRECTION.md) and
  [systems-map prototype](docs/SYSTEM_MAP_PROTOTYPE.md): earlier interaction
  experiments. `python scripts/launch_research_map_demo.py` uses curated maps
  and updates, not model-generated diagrams.
- [Report-derived memory demo](docs/QWEN_EXPERIMENT_MEMORY_DEMO.md): recorded
  Qwen experiment results presented as a dated Obsidian thread.
- [Research-library layout](docs/REPOSITORY_LAYOUT.md) and
  [harness design](docs/RESEARCH_LIBRARY_HARNESS_DESIGN.md): the earlier staging
  harness has an offline fixture replay; it is separate from the live Studio
  paper reader. The read-only MCP bridge serves snapshot passages; unattended
  weekly scheduling and end-to-end product validation remain incomplete.

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
