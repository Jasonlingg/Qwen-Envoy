# EnvoyBench (candidate v0.1)

**Start with the [v0.1 release bundle](../../release/envoybench-v0.1/README.md)**
for a self-contained, read-only Studio snapshot and the recorded base/v5 run.
The bundle includes the exact artifacts needed to regenerate the corpus and
verify the provisional comparison from a fresh clone. Commands below that use
`out/envoybench/` describe the original local experiment paths; use the release
bundle paths on a fresh checkout. No independently validated leaderboard is
claimed.

**Studio now includes a [paper reader](PAPER_READER.md) and
[training recipe](MODEL_CARD.md).** Upload a PDF/Markdown/text source at `/papers`,
ask the configured Qwen checkpoint, inspect its source spans and code trace,
and export the saved run. `/model` records the observed v5 recipe and limitations.
Without an endpoint, uploads and saved views work but live inference is disabled.
These product runs are unreviewed and separate from the frozen benchmark below.

Studio also has a separate [web-to-evidence development pilot](WEB_EVIDENCE.md)
at `/web-evidence`. It replays frozen HTML/PDF responses to measure literal
fact retention and top-three passage retrieval. Its assistant-checked labels
and component scores are **not** Qwen answer-quality results.

EnvoyBench tests whether a language model can **investigate a scientific paper
through several executable tool steps and give a source-supported answer**. A
model sees a question, writes Python that calls the same `search()`, `read()`,
`search_within()`, `passage()`, and `extract()` tools as its competitors, inspects
the returned text, and submits an answer, paper IDs, and exact source spans.
The benchmark records the whole trajectory, including unsuccessful actions.

EnvoyBench is an *agent evaluation protocol built on* [QASPER](https://huggingface.co/datasets/allenai/qasper),
not a new collection of original paper questions. QASPER's dataset card lists
its license as CC BY 4.0; retain attribution to its authors and the dataset in
any redistributed derivative. The contribution here is the bounded code-tool
environment, paper-disjoint split audit, evidence and failure diagnostics, and
paired model comparison. It is intended for people deciding whether a
fine-tuned research agent is actually better than its base model. Related
scientific-agent benchmarks already exist; [the prior-art review](PRIOR_ART.md)
states what this protocol does and does not add.

## Current status

The [September 30 case study](COMPARISON_2026_09_30.md) records the complete
40-question base/v5 inference run, observed tool failures, and current grading
status. Its raw traces are available in the local Studio. It is separate from
the historical 9/40 versus 19/40 development result described below.
Use the Studio's question table to compare base/v5 verdicts and execution errors,
then inspect both complete Python/action transcripts for a selected question.
The live viewer loads complete saved responses on demand; standalone HTML exports
label long excerpts. Thinking was disabled in this run, so no reasoning text was
recorded.

To verify that saved run without model calls or a GPU:

```bash
python -m benchmarks.envoybench.verify \
  --run-dir out/envoybench/provisional-full-20260930 \
  --review-dir out/envoybench/provisional-review-20260930 \
  --judged-review out/envoybench/codex-reviewed-20260930/review.json \
  --provisional
```

This checks the frozen split and corpus, complete paired results, exact source
spans, and review/report bindings. It reports answerable passes separately from
passes on unanswerable questions. A valid span does not prove semantic support;
the saved model-assisted grades remain provisional.

The current workbench adds an analytical refusal baseline, separate behavior
and evidence diagnostics, three inspectable case studies, a prepared retrieval
comparator, and a small blank calibration packet. These changes analyze saved
results; they do not constitute another inference or training run. The existing
base/v5 quality grades remain provisional and unchanged.

New EnvoyBench runs use the **fail-closed-v2** verifier protocol. It gives up to one
recovery message, then records a structurally invalid submission or duplicate
action as an `escalated` episode with no accepted prediction or reward. The
archived September/October runs used **legacy-v1**: after feedback was spent,
an invalid submission could be accepted. Their saved answers and scores remain
unchanged. Use `--verifier-protocol legacy-v1` only when deliberately reproducing
that historical behavior; the protocol choice is bound to each new run's
manifest and comparison ID. This structural verifier does not judge whether a
quoted passage semantically supports the answer.

**Candidate release, no independently validated leaderboard yet.** The
historical 40-question QASPER runs in this repository were used during
development. Their review was model-assisted, not independent human review.
They illustrate the system, but they are not an untouched test result. The
new `test_candidate` split is frozen by ID and excluded-paper rules before
benchmark model runs. It can support a **provisional automated base-versus-v5
comparison** without asking the user to judge every paper. Any model-graded
answer quality remains provisional: the converted references and final answers
have not been independently human-reviewed. Do not describe a model-graded
score as a validated held-out result or a passed promotion gate.

The existing positive development result is instructive: a targeted Qwen3-8B
adapter increased provisional semantic passes from 9/40 to 19/40 on one
QASPER-derived set. It still failed its full promotion rule, and later training
and transfer tests exposed regressions. Those observations motivate the
benchmark's emphasis on supported answers, correct abstention, executable
tool reliability, and per-question failures rather than one aggregate reward.
See [the demo and evidence guide](../../docs/DEMO_AND_EVIDENCE.md) for the
recorded-trace walkthrough. The recorded replays do not call a model live.

## Offline diagnostics and next comparisons

The [prefix-cache experiment](INFERENCE_ABLATION.md) prepares a small same-v5
serving comparison: caching OFF versus ON on three fixed development questions,
with per-request timing and workload checks. It is **prepared, not run**; no
speedup or model-quality gain is claimed. It does not change the base/v5 grades.

The [Qwen thinking smoke](THINKING_ABLATION.md) prepares a matched v5
thinking OFF/ON comparison. New runs can record the model's generated reasoning
beside each code turn for inspection. No live thinking run or improvement claim
exists yet; the saved September 30 run had thinking disabled.

An **always-refuse policy would be expected to pass 20/40** under this split's
frozen labels and rubric, while producing **zero useful substantive answers**.
That is an analytical expectation, not an executed or model-graded baseline.
It illustrates why a higher aggregate pass rate does not by itself establish
better answering. The Studio separates supported answers on answerable
questions from refusals on unanswerable questions.

The [curated case notes](case_study_20260930.json) explain three question pairs:
a useful refusal, a correct dataset name with the wrong citation, and an answer
that misses the requested architectural change. They distinguish factual answer
correctness from support in the **submitted** evidence. These six diagnoses are
unblinded, model-assisted, post-hoc examples, not a representative accuracy
sample. Their result hashes and exact paper quotations bind them to the saved
run. Unreviewed component judgments stay unknown; a failed end-to-end verdict
is never automatically converted into an incorrect-answer label.

A separate response-only, model-assisted review classifies all 80 saved outputs
as substantive answers, abstentions, mixed responses, or missing submissions. This identifies
natural-language refusals as well as the literal word `Unanswerable`; it does
not re-grade truthfulness, citation support, or the validity of QASPER labels.
The classification is distinct from the original quality verdicts. A refusal
on a question labeled answerable is an **apparent unnecessary refusal under the
frozen labels**, not independent proof that the paper contains an answer.

Mechanical diagnostics count returned tool-validation errors as well as Python
exceptions, and separate context-limit failures from ordinary exploration.
Timing and token-usage coverage remain visible. Missing historical timings and
usage are not replaced with invented values; incomplete coverage cannot support
an efficiency win. The original results are preserved.

### Deterministic retrieval comparator: prepared, not run

The [known-paper retrieval comparator](RETRIEVAL_BASELINE.md) prepares a simple
alternative: fixed BM25 passage selection inside the already named paper,
followed by one answer request to the same base or v5 checkpoint. It tests the
value of the **agent architecture**, rather than treating a prompt/tool change
as a training improvement.

Forty gold-free packets are prepared in
`out/envoybench/retrieval-baseline-prepared-20260930/`. They preserve the frozen
questions and corpus, use the pinned local tokenizer, and reserve output tokens
inside the explicit context limit. **Zero inference requests have been made;
80 are planned for the paired comparator.** No retrieval-baseline accuracy,
latency, or cost result exists yet.

```bash
python -m benchmarks.envoybench.retrieval_baseline \
  --prepare \
  --source-run out/envoybench/provisional-full-20260930 \
  --output out/envoybench/retrieval-baseline-prepared-copy \
  --context-window 8192
```

This command uses no model endpoint or credentials. The separate `--execute`
mode requires explicit model configuration and may incur inference cost; see
the comparator guide for its protocol and command.

### Small independent calibration packet

`out/envoybench/calibration-20260930/review-packet.json` contains **eight questions
and sixteen anonymous answers**, with full paper text and blank decisions. No
independent review has been completed. The packet hides model identities and
prior verdicts and permits `unsure`; an independent reviewer need not guess.

To prepare another copy without any model calls:

```bash
python -m benchmarks.envoybench.calibration \
  --review out/envoybench/codex-reviewed-20260930/review.json \
  --questions 8 --seed 42 \
  --output out/envoybench/calibration-review-copy.json
```

The sample covers answerability and prior-verdict strata, so it is deliberately
not population-representative. Its purpose is to expose grading disagreements
and confusing references, not to produce a replacement score or bypass the full
human promotion gate. Use a reviewer who has not seen the named model outputs;
disclose any AI assistance instead of marking it as independent human review.

### Share the recorded Studio without a GPU

Export a standalone, read-only HTML snapshot containing the saved traces and
their current grading status:

```bash
python -m benchmarks.envoybench.export \
  --run-dir out/envoybench/provisional-full-20260930 \
  --review-dir out/envoybench/provisional-review-20260930 \
  --judged-review out/envoybench/codex-reviewed-20260930/review.json \
  --provisional \
  --output out/envoybench/studio-shareable.html
```

Open the exported file directly in a browser. It makes no model or review API
calls and cannot save review decisions. It preserves the provisional labels and
source provenance; exporting does not validate the scores. These offline tools
require new output paths and refuse to overwrite existing artifacts.

## Local EnvoyBench Studio

From the repository root:

```bash
pip install -e '.[viewer]'
python -m benchmarks.envoybench.demo
```

Open <http://127.0.0.1:8765>. The recorded-results view shows the frozen candidate's
status, then a **separate historical September 2026 development comparison**
with two recorded, same-question base/v5 traces. It shows a success and a
regression, their actual code and tool observations, answers, and the original
model-assisted review labels. The historical 9/40 versus 19/40 result predates
EnvoyBench v0.1 and is **not** a human-reviewed held-out benchmark score. Some
long observations are visibly marked as excerpts. No GPU, model API, or key is
needed to view the demo.

The Studio also has an **optional human source-review workspace**. It shows each frozen
candidate question, QASPER annotations and evidence, and the full paper text.
A person records `reference_valid` or `unscorable` with a reason, then explicitly
finalizes the sheet after all 40 decisions. The initial GET only reads; the
first saved decision creates `out/envoybench/test-reference-review.json` if it
does not exist. Saves check the frozen benchmark/corpus hashes and use a revision
token to avoid overwriting another edit. A finalized sheet is locked. Do not
mark model-assisted screening as an independent human decision. See the
[review rubric](REVIEW.md) before using this workspace. This step is for a
stronger, independently validated result; it does **not** block the first
provisional automated comparison.

After you have a complete **development** run, the same viewer can inspect all
of its saved question trajectories. Without a completed review bundle it shows
automatic diagnostics and leaves semantic scores blank:

```bash
python -m benchmarks.envoybench.demo --run-dir out/envoybench/dev-run
```

For an independently validated `test_candidate` comparison, the Studio requires
both a finalized source-review sheet and a completed blind **human** answer
review before showing named model outputs. This prevents the review interface
from revealing model identities before verdicts are locked:

```bash
python -m benchmarks.envoybench.demo \
  --run-dir out/envoybench/test-run \
  --reference-review out/envoybench/test-reference-review.json \
  --review-dir out/envoybench/test-review
```

The provisional route instead uses an explicit `--provisional` viewer option
with a paired candidate run. It can show named outputs and, if a model-graded
review has been completed, clearly labeled provisional answer verdicts. Do not
use this route for a later blind human review of the same exposed answers: it
reveals model identities. In provisional mode the Studio hides and disables its
human-review workspaces, and the server rejects review edits. Independent
validation needs a separate blind Studio session and a reviewer who has not
seen the named outputs. The viewer checks the frozen split and paired run
before showing it, serves on loopback only, and never runs a model on page
load.

The first live base–v5 provisional run follows the
[predeclared comparison protocol](PROVISIONAL_COMPARISON_2026_09_30.md).

## What the scores mean

| Measure | How it is assessed | What it does **not** prove |
| --- | --- | --- |
| Provisional answer verdict (`pass` / `partial` / `fail`) | Model grading against the question, QASPER reference annotations, and source evidence | A model grader may miss unsupported claims; this is not independent validation. |
| Human-reviewed supported-answer verdict | Blind human review against the question and actual cited text | One reviewer does not establish general benchmark validity. |
| Citation and span validity | Deterministic lookup in the frozen corpus | A real quotation may still be irrelevant or misinterpreted. |
| Tool reliability | Recorded execution errors, repeated actions, completion and abstention | A syntactically valid program need not answer the question. |
| Efficiency | Steps and wall time on a named hardware/serving setup | A faster run is not necessarily more accurate or cheaper. |

The MuSiQue reward in `src/env/reward.py` is **not** the primary research-answer
metric. Deterministic checks are diagnostics. A model-graded answer-quality
comparison is explicitly provisional; a human-reviewed comparison requires
the blind review to be complete. Compare models on identical questions, corpus
revision, tool version, step budget, prompt, and decoding.
The run artifact must keep the exact model and adapter revisions, seed,
hardware, question IDs, and source hashes.

For the first **base Qwen3-8B versus v5** candidate experiment, we will call
v5 a substantive improvement only if all of these predeclared conditions hold
on the same human-reviewed, scorable questions: at least a 15 percentage-point
increase in `pass` rate; a two-sided exact McNemar (p < 0.05) on paired
`pass` versus non-`pass` verdicts; no more than two additional execution-error
episodes or no-submission episodes across the 40 questions; and mean episode
duration and estimated per-question compute cost no more than 1.5 times base.
The scorer reports the paired test. For cost, record the serving GPU, runtime,
and hourly price for each arm in the model configuration before running. The
price-times-episode-duration figure is a comparable compute estimate on a
matched server, not an invoice. If cost is unreported or fewer than 35
questions pass reference validation, the promotion decision is inconclusive.
We will report a failed gate and use development cases for diagnosis rather
than tune against the candidate test answers.

## Reproduce the dataset and run protocol

From the repository root, install the project's QASPER dependencies and
materialize the frozen split from the pinned QASPER source. A local copy of the
full paper text is generated under `benchmarks/envoybench/data/` and ignored by
Git; the tracked split specification contains IDs, exclusions, and source
hashes. The generator refuses changed source rows. It is intended to be run
once in a fresh checkout; use a new output directory rather than overwriting
an existing frozen snapshot.

```bash
pip install -e '.[qasper]'
python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data
docker build -f benchmarks/envoybench/Dockerfile -t rlm-sandbox .
```

The runner accepts the [example model configuration](models.example.json).
It points both Qwen3-8B and the pinned v5 LoRA adapter at the **same**
OpenAI-compatible vLLM server, with thinking disabled for both arms. Start
`scripts/serve_vllm.sh` on a GPU host and point the client at it; an SSH tunnel
to a local port is preferable to exposing the server publicly. The adapter's
served model name is `envoy` by default. No API key or GPU is needed for the
configuration check:

```bash
export ENVOY_QWEN_ENDPOINT=http://127.0.0.1:8000/v1
python -m benchmarks.envoybench.run \
  --dataset benchmarks/envoybench/data --split dev \
  --models benchmarks/envoybench/models.example.json \
  --output out/envoybench/check --validate-only
```

When the endpoint is actually running, omit `--validate-only` and use a new
output directory to run the full split. This makes billable model calls; the
offline validation command above does not. A `--question-id` run is explicitly
marked as a subset smoke test and cannot be reported as the full benchmark.

```bash
python -m benchmarks.envoybench.run \
  --dataset benchmarks/envoybench/data --split dev \
  --models benchmarks/envoybench/models.example.json \
  --output out/envoybench/dev-run
```

Each complete run writes `manifest.json` and `results.json`: exact source and
implementation hashes, selected question IDs, declared model/adapter identity,
decoding, environment and hardware metadata, duration, evidence, and full
step-by-step actions. Endpoints and API-key values are never put in the public
run artifact. A remote server's model revision is operator-declared, not
cryptographically attested by its OpenAI-compatible API.

To collect generated-token logprobs on a small diagnostic run, add
`"logprobs": true, "top_logprobs": 0` to `extra_body` for **both** models in a
copy of the model configuration. The runner saves a bounded prefix of sampled
token logprobs and a numeric summary on each recorded turn when the endpoint
returns them. Missing or invalid values remain unavailable. This telemetry
does not change action execution or benchmark scores, and it cannot be
recovered from older runs. It describes the probability of generated text,
not the probability that an answer is correct or that evidence is sufficient.
With a thinking parser, the provider's token stream may include reasoning and
is not aligned to the cleaned Python action. Record the vLLM server's
`--logprobs-mode` with the run, since raw and sampling-processed probabilities
mean different things.

## First provisional base-versus-v5 comparison

After the base and v5 endpoints are configured, run the frozen candidate with
the same model configuration and tool budget. This invokes the models and can
incur GPU or API cost:

```bash
python -m benchmarks.envoybench.run \
  --dataset benchmarks/envoybench/data --split test_candidate \
  --models benchmarks/envoybench/models.example.json \
  --output out/envoybench/test-run
```

Prepare an **automated, provisional** scoring bundle from the complete paired
run. This route uses the existing QASPER annotations without claiming that a
human has checked each converted reference:

```bash
python -m benchmarks.envoybench.score prepare-provisional \
  --benchmark benchmarks/envoybench/data/test_candidate/benchmark.json \
  --corpus benchmarks/envoybench/data/test_candidate/corpus \
  --manifest out/envoybench/test-run/manifest.json \
  --results out/envoybench/test-run/results.json \
  --output out/envoybench/test-provisional
```

The Studio can inspect the complete paired run immediately, without a human
review or model judge. Its answer-quality fields remain blank until a review
is supplied; the execution, quotation, and duration diagnostics are still
useful:

```bash
python -m benchmarks.envoybench.demo \
  --run-dir out/envoybench/test-run --provisional
```

Preview the model judge's request count and rough input size **without making
model calls**:

```bash
python -m benchmarks.envoybench.judge \
  --review out/envoybench/test-provisional/review.json \
  --output-dir out/envoybench/test-provisional-judged
```

Adding `--execute` makes paid Anthropic model calls and writes a new anonymous
`review.json` plus `judge-report.json` in the output directory. The judge sees
the question, existing QASPER reference answer and annotated evidence, and
both anonymous answers with their submitted evidence. It does **not** read
the full paper or the identity key, so its judgments remain provisional. Once
the full judged file is complete, score it against the original blind key:

```bash
python -m benchmarks.envoybench.judge \
  --review out/envoybench/test-provisional/review.json \
  --output-dir out/envoybench/test-provisional-judged --execute
python -m benchmarks.envoybench.score score \
  --review out/envoybench/test-provisional-judged/review.json \
  --key out/envoybench/test-provisional/blind-key.json \
  --results out/envoybench/test-run/results.json \
  --output out/envoybench/test-provisional-score.json
```

Open the named base-versus-v5 traces and clearly marked model-graded verdicts
without modifying the original provisional bundle:

```bash
python -m benchmarks.envoybench.demo \
  --run-dir out/envoybench/test-run \
  --review-dir out/envoybench/test-provisional \
  --judged-review out/envoybench/test-provisional-judged/review.json \
  --provisional
```

Model-graded `pass`/`partial`/`fail` verdicts are never independent human
judgments or a model promotion decision. Keep the grader identity, prompt hash,
token usage, and per-question verdict notes with the run, and inspect the individual
model answers and source spans before interpreting a score. Do not feed the
candidate questions or graded answers back into training and then call a
later rerun held out.

## Optional human validation of a candidate run

Validate the **source references first**, before looking at model outputs. The
template includes all 40 candidate questions and their QASPER annotations.
Complete every decision as `reference_valid` or `unscorable`, with a reason for
each exclusion; then set `status`, `reviewer_id`, and `reviewed_at`. See the
[review rubric](REVIEW.md). This is a human review step, not an automatic label.

```bash
mkdir -p out/envoybench
python -m benchmarks.envoybench.score reference-template \
  --benchmark benchmarks/envoybench/data/test_candidate/benchmark.json \
  --output out/envoybench/test-reference-review.json
```

After a **full** paired `test_candidate` run, generate a blinded review bundle.
Keep `blind-key.json` and `automatic.json` closed until answer verdicts are
locked. The scorer refuses an incomplete reference sheet, mismatched source
hashes, missing model/question pairs, or a `--question-id` smoke run.

```bash
python -m benchmarks.envoybench.score prepare \
  --benchmark benchmarks/envoybench/data/test_candidate/benchmark.json \
  --corpus benchmarks/envoybench/data/test_candidate/corpus \
  --manifest out/envoybench/test-run/manifest.json \
  --results out/envoybench/test-run/results.json \
  --reference-review out/envoybench/test-reference-review.json \
  --output out/envoybench/test-review
```

The Studio can then present the prepared answers **without model identities**:

```bash
python -m benchmarks.envoybench.demo \
  --reference-review out/envoybench/test-reference-review.json \
  --blind-review-dir out/envoybench/test-review
```

The reviewer records `pass`, `partial`, or `fail` for every anonymous answer,
with a reason and relevant passage or absence note for each non-pass, then
explicitly finalizes the sheet. The editor never opens `blind-key.json` or
`automatic.json`; it writes only `review.json`. Set `reviewer_kind` to `human`
only for an actual source-checked human review; model-assisted judgments are
provisional. Once the review is complete:

```bash
python -m benchmarks.envoybench.score score \
  --review out/envoybench/test-review/review.json \
  --key out/envoybench/test-review/blind-key.json \
  --results out/envoybench/test-run/results.json \
  --output out/envoybench/test-reviewed-score.json
```

`automatic.json` reports execution and provenance diagnostics separately. A
valid span means the quoted bytes occur in the frozen paper, not that the
answer is supported. The candidate set deliberately balances answerable and
unanswerable questions, so its pass rate is an abstention stress-test result,
not an estimate of naturally sampled QASPER accuracy.

The environment requests exact evidence and gives one verifier-feedback turn;
it can still record a final submission with unresolved evidence problems.
Those failures remain visible to the reviewer and in the automatic diagnostics.
The current Docker REPL replays earlier Python actions on each turn. Therefore
`trajectory_steps` counts **model turns**, not actual tool invocations, and
episode duration includes replay overhead. Compare latency only under the same
backend and disclose this limitation when reporting efficiency. Some failed
requests can have no positive observed episode time in the artifact. The Studio
reports how many episodes were timed; a mean from incomplete timing coverage
cannot establish a latency or compute-cost advantage.

## Execution boundary

The benchmark's [Dockerfile](Dockerfile) contains Python tool dependencies but
**no papers, questions, or answer keys**. The runner requires that image and
mounts only the selected frozen corpus read-only for model-authored code. Do not
serve an arbitrary public code-execution endpoint on the strength of this local
sandbox alone. A public recruiter demo can show recorded trajectories without
running submitted code.

## What is still needed for a public benchmark release

1. Regenerate and audit the frozen candidate split from the pinned QASPER
   revision; publish exact IDs and hashes, not copyrighted paper copies without
   checking their source terms.
2. Independently review converted references and blinded model answers. Record
   the review rubric and disagreements. Keep development cases separate.
3. Run at least base Qwen, the trained checkpoint, and a simple retrieval
   baseline under a matched protocol. A Nemotron run through Nebius Token
   Factory can provide a useful stronger-model comparison and a real hackathon
   runtime role.
4. Publish an inspectable report and reproduction commands. Do not claim a
   model win from citation syntax, MuSiQue reward, or a few selected replays.

The benchmark package does not itself fulfill the Nebius hackathon's application
requirement. A submission needs a working user-facing evaluator with a live
Nebius Token Factory inference call or Nebius AI Cloud deployment and an NVIDIA
open-source model, in addition to the public repo and video. See the
[official rules](https://nebiusglobalaihackathon.devpost.com/rules).
