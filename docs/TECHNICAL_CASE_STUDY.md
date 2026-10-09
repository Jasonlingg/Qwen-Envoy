# QASPER Agent Studio — a Qwen Envoy case study

This study adapts QASPER's existing questions and paper annotations (Dasigi et
al., 2021; CC BY 4.0) to executable Python investigation. It is an independent
project, not an original question benchmark or an official QASPER/AllenAI
product. [Data provenance](../benchmarks/envoybench/data/README.md) records the
source revision and adaptations; the [short findings report](QASPER_AGENT_STUDY.md)
summarizes the final comparison and stopping diagnosis.

## What changed after supervised fine-tuning?

**The saved comparison shows more reliable tool use and more refusals, while
supported answers on answerable questions remain unchanged.** On September 30,
2026, base Qwen3-8B and the v5 adapter attempted the same 40 paper questions.
Provisional model-assisted passes increased from **5/40 to 15/40**, but both
models passed **3/20 answerable questions**. Correct refusals on the remaining
20 questions account for the entire aggregate gain. Independent review of the
references and answers is pending; this is a technical case study, not a
validated model-improvement result.

A separate re-score of these unchanged outputs uses the official QASPER
evaluator and all original annotations for the same questions: **Answer F1 is
19.92% for base and 30.69% for v5**. On the fixed 20 answerable questions,
including missing answers as zero, F1 instead falls from **29.83% to 21.37%**.
The unanswerable group's mean rises from 10% to 40%. This selected-subset token-overlap metric
does not validate evidence support or replace the provisional pass judgments.
The [short report](QASPER_AGENT_STUDY.md#original-qasper-answer-scoring) explains
the distinction and links the exact scoring artifacts.

The public deliverable is an inspectable experiment: training data and recipe,
pinned checkpoint identities, a frozen evaluation protocol, complete saved
trajectories, and a runnable Studio. Start with the
[public saved-run demo](https://jasonlingg.github.io/Qwen-Envoy/), or use the
commands below. The [comparison report](../benchmarks/envoybench/COMPARISON_2026_09_30.md)
contains the detailed analysis behind this account.

## The task and the system

Envoy tests whether a small language model can investigate documents by writing
Python. The intended role is a bounded research worker that returns evidence
to a larger assistant. Each action can search for a passage, inspect the result,
change the query, or submit an answer with source IDs and character offsets.
The model receives the actual observation from each action before choosing its
next one. This keeps retrieval decisions and failures visible in a multi-step
trajectory.

The [tool implementation](../src/env/tools.py) exposes `search()`, `read()`,
`extract()`, `search_within()`, and `passage()`. The
[environment](../src/env/document_env.py) handles steps and submissions; the
[execution backend](../src/env/repl.py) runs Python. The evaluated Docker backend
disables container networking and uses cumulative-script replay to carry state
between turns. The local backend instead uses a long-lived worker. That
implementation difference matters: the comparison used Docker, and backend
parity is not assumed.

September 30 evaluated **known-paper question answering**. Every question named
the target paper and document ID, with 40 target papers inside a frozen
363-paper corpus. The model still had to find and select evidence within that
paper, but relevant-paper discovery was given away. The result therefore does
not measure open-ended literature discovery, multi-paper synthesis, or transfer
to a personal Obsidian vault. Those broader goals motivated the project; the
release demonstrates the narrower experiment.

## What was trained

The [v5 dataset card](../data/sft/qasper-v5/README.md) records 44 QASPER training
conversations, split by paper with seed 42: 35 for training and nine for
validation. They expand into 106 training actions and 26 validation actions.
The training targets contain only **4,812 supervised tokens**. Seven
conversations retain original teacher trajectories; 37 contain assistant-made
repairs re-executed against real source observations. The editing assistant saw
the references, so this is reviewed demonstration construction, not independent
evaluation. The expansion also missed its declared acceptance target: 23 of
32 candidates were accepted instead of the requested 24, with only eight
accepted insufficient-evidence cases instead of ten.

The student predicts one next Python action or `SUBMIT:` answer at a time.
Earlier instructions, actions, and tool observations remain in context but are
masked out of the loss. In [train_sft.py](../scripts/train_sft.py),
`_expand_per_action()` constructs those examples and `_tokenize_per_action()`
checks that each training prefix equals Qwen3's inference prefix with thinking
disabled. It refuses a prefix mismatch or a sequence that would truncate the
target. This is a concrete training/inference alignment constraint, separate
from whether the learned answer is good.

The [model card](../benchmarks/envoybench/MODEL_CARD.md), its
[machine-readable record](../benchmarks/envoybench/model_card.json), and the
[published training evidence](../release/qwen-v5-training/README.md) pin the
observed September 19 run:

| Item | Recorded value |
| --- | --- |
| Base | `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` |
| Adapter | `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@27b912a863ff914ad45baa03624d8911dc1e17fb` |
| Selected weights | `artifacts/full/checkpoint-50` |
| Adapter SHA-256 | `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3` |
| Method | QLoRA: 4-bit base, LoRA rank 4, alpha 8, dropout 0.05 |
| Adapted projections | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Optimization | Learning rate 0.0002; microbatch 1; accumulation 4; seed 42 |
| Sequence limit | 8,192 tokens |
| Full training run | Four epochs, 108 optimizer steps, NVIDIA A40 |
| Recorded training duration | 579.49 seconds, excluding setup |

Checkpoint 50 was selected near the epoch-two validation-loss minimum and
checked on development questions. It is not the final step-108 checkpoint.
Nine validation conversations provide little evidence for broad selection
claims. The original manifest records a script hash but has a blank git commit,
and the current training script has a different hash. The recipe and identities
are inspectable; bit-for-bit reproduction of the original training run is not
established. Upstream Qwen exposure to these papers is unknown.

The public bundle preserves the original
[run manifest](../release/qwen-v5-training/run-manifest.json),
[launch script](../release/qwen-v5-training/run-sft-v5.sh),
[package versions](../release/qwen-v5-training/environment.txt), and
[final trainer state](../release/qwen-v5-training/trainer-state-step-108.json).
The launch script exposes the recorded training arguments, including historical
`/workspace` paths; it is not a portable fresh-clone launcher. The bundle's
[training summary](../release/qwen-v5-training/training-summary.json) records
the final metrics and original log hash. Training and validation loss are
optimization diagnostics, not evidence of supported-answer improvement.

## How the comparison was evaluated

The [predeclared protocol](../benchmarks/envoybench/PROVISIONAL_COMPARISON_2026_09_30.md)
asked whether the existing v5 adapter improved supported answers under matched
tools and decoding. A gain of at least 15 percentage points, with no more than
two additional execution-error or no-submission episodes, would justify seeking
independent review and a new untouched split. It did not authorize a validated
promotion from provisional grades alone.

Both arms used an A40 48 GB, vLLM 0.30.0, temperature 0, top-p 1, thinking
disabled, a 15-step budget, and at most 1,024 generated tokens per action.
The runner seed was 42; the remote endpoint did not receive an inference seed,
so remote determinism was not guaranteed. The
[run manifest](../release/envoybench-v0.1/run/manifest.json) records all question
IDs, model declarations, implementation hashes, and corpus hash
`bc71958267aa4ef6216085ffc42d364423760f9f042fc1da661822e87bd10fe5`.
The [data manifest](../benchmarks/envoybench/data/manifest.json) pins QASPER
revision `13b496d2a5359329b110e3419628de3cf791843b` and the split recipe.

Three different measurements must stay separate:

1. **Training objective:** next-action supervised token loss for v5. This
   experiment did not train v5 with a research-quality reward.
2. **Mechanical diagnostics:** submission status, exceptions, returned tool
   errors, exact source IDs/spans, and trajectory length. These can be checked
   against saved artifacts.
3. **Answer quality:** the [review rubric](../benchmarks/envoybench/REVIEW.md)
   requires a useful answer supported by its own submitted evidence, or an
   appropriate refusal when the paper cannot answer. This needs semantic
   judgment; a real quote can still be irrelevant.

The legacy [MuSiQue reward](../src/env/reward.py), `outcome-v1`, combines
0.8 × answer token-overlap F1, 0.1 × citation precision, and 0.1 × citation
recall. It remains visible in saved execution observations for compatibility.
The manifest explicitly marks it as **not the benchmark score**. Neither that
number nor quote-integrity validation is evidence of supported research answers.

The planned identity-blind Sonnet grader stopped after eight questions when
credits ran out. A separate Codex-assisted review covered all 40 paired
questions, once each, across two fresh-context reviewers. They saw anonymized
answers and the rubric, without the model key or previous grades. Their exact
provider model identities, decoding, and token usage were not exported. The
[full provisional review](../release/envoybench-v0.1/review-model-assisted/review.json)
is kept separately from the incomplete Sonnet judgments. This fallback and the
unvalidated QASPER references limit reproducibility and confidence in the grades.

| September 30 result | Base | v5 |
| --- | ---: | ---: |
| Final answers submitted | 29/40 | 40/40 |
| Episodes with any recorded tool, runtime, or endpoint error | 22/40 | 1/40 |
| Endpoint context-limit failures | 11/40 | 0/40 |
| Mean trajectory steps | 11.00 | 3.05 |
| Provisional passes, all questions | 5/40 | 15/40 |
| Passes on answerable questions | 3/20 | 3/20 |
| Correct refusals on unanswerable questions | 2/20 | 12/20 |

These are model-plus-harness observations from one run. The shared serving
context cap was **8,192 tokens**, including the 1,024-token completion reserve.
Base exhausted it on eleven questions; six v5 passes occurred on questions
where base hit that cap. The cap was recovered from serving configuration and
actual errors, having been omitted from the original prose protocol and run
manifest. A larger context or different context management could change the
comparison.

Refusal also changes the aggregate score. V5 abstained on eight questions
labeled answerable. An always-refuse policy would be expected to pass **20/40**
under the frozen labels while supplying **zero useful answers**. That is an
analytical expectation, not an executed baseline. It demonstrates why the
15/40 aggregate cannot stand in for useful answering ability.

Historical base timing is missing for all eleven failed episodes: the harness
wrote zero placeholders despite retaining their actions. Later diagnostics
exclude those placeholders and preserve incomplete coverage. The evidence does
not support a complete latency or cost advantage. Likewise, error accounting
was corrected to count structured tool-error returns as well as exceptions;
v5 had one recovered tool error and should not be described as error-free.

## Why does v5 usually stop at step three?

The saved v5 run submits at **step three on 38/40 questions**, and step four on
the other two. This is an observed action count, not proof that three actions
were sufficient. In the training split, **27/35 conversations (77.1%)** also
end by step three. Thirty of the 35 training demonstrations were rewritten by
the editing assistant; comparing those same questions before and after repair
gives mean lengths of **5.033 versus 3.067 actions**, including submission.
These original-length counts come from the archived source trajectories;
the public [v4 review](../data/research/qasper_sft_v4_review.json) and
[v5 review](../data/research/qasper_sft_v5_review.json) preserve parent hashes,
repair decisions, and authored replacement actions.

The preparation process deliberately selected
[2–6-action demonstrations](QWEN3_SFT_DATA_ITERATION.md#changes-in-the-preparation-pipeline)
and removed redundant searches. It did not preserve a neutral sample of the
teacher's natural behavior. Across all 44 train/validation examples, 37 were
repaired and seven were unchanged. The editing assistant could see references
and inspect sources; the initial Sonnet pilot also supplied training-only gold
answers/passages for answerable questions, then disabled that guidance after
leakage failures. A separate legacy source batch did not record the Claude
model identity. Describing all final examples as Sonnet independently finding
answers in three steps would therefore be inaccurate.

The exact three-action sequence varies: the training split has eight
`search_within → search_within → SUBMIT` examples, six with `read` as the middle
action, three with `extract`, and one with `passage`. Sixteen of 35 training
questions have insufficient-evidence targets, but those targets explain a
specific evidence gap rather than simply saying `Unanswerable`. None of the
44 final training/validation targets includes an `EVIDENCE` span field.

The data distribution makes learned short stopping plausible. The saved
comparison does not isolate its cause from prompt, protocol, checkpoint
selection, or other data changes. No controlled repair-versus-original training
ablation was run, and the current questions cannot serve as an untouched test
for a later intervention. See the [short report](QASPER_AGENT_STUDY.md).

There is also a runtime caveat. A pre-run source archive contains an environment
file whose hash matches the September 30 manifest, alongside a check for
refusals before two successful document actions. That check issued bounded
feedback, not an absolute minimum: the run allowed one feedback event and no
mandatory escalation. None of the 80 saved trajectories contains a warning
about a premature refusal. The evidence-state module itself was not hash-pinned in that
manifest, so the archive does not fully attest its run-time version. This is
not evidence that the runtime forced v5 to stop at step three.

## Three traces worth inspecting

All actions and observations below are in the packaged
[80-result inference record](../release/envoybench-v0.1/run/results.json).
The [case annotations](../benchmarks/envoybench/case_study_20260930.json) bind
each diagnosis to exact result hashes and source passages. These cases were
selected after seeing outcomes and analyzed with model identities visible.
They illustrate failure mechanisms, not a representative sample or independent
review.

**A useful refusal: human judgment agreement.**
`qasper_test_fd556a038c36abc88a800d9d4f2cfa0aef6f5aba` asks for the percentage
of human judgment agreement in *Modeling German Verb Argument Structures:
LSTMs vs. Humans* (`qasper_1912_00239`). Base repeatedly searches similar
phrases, receives a repetition warning, and ends after ten recorded steps with
a context-limit error. V5 searches within the paper, reads a 1,600-character
passage starting at 5,000, and submits `Unanswerable` on step three. The refusal
matches the frozen label; it does not independently prove that the requested
number is absent from the paper.

**Right dataset, wrong submitted passage.**
`qasper_test_7ae95716977d39d96e871e552c35ca0753115229` asks which dataset
*Represent, Aggregate, and Constrain: A Novel Architecture for Machine Reading
from Noisy Sources* uses (`qasper_1610_09722`). Both name the Stanford Plane
Crash Dataset. Both submit offsets **15,600–16,000**, which describe belief
propagation; the sentence naming the dataset is at **16,027–16,174**. V5 also
calls `passage(doc_id, 15600, 16000)`, confusing the length argument with an
end offset, then recovers after the tool returns a length error. It later reads
the relevant text but still submits the wrong span. This one trace separates
API recovery, answer correctness, and citation support.

**A shorter trajectory that misses the question.**
`qasper_test_0b5a7ccf09810ff5a86162d502697d16b3536249` asks what architectural
simplification preserved performance in *SEPT: Improving Scientific Named
Entity Recognition with Span Representation* (`qasper_1911_03353`). Base
identifies pruner removal, under-sampling, and simple pooling, but cites only
the phrase “simplify the origin network architecture” at **3,593–3,633**. V5
finishes in three steps with “combines the span extractor with BERT,” missing
the requested simplifications and supplying inadequate evidence. Fewer steps
and successful submission do not guarantee a better answer.

## Negative results and unresolved questions

The October 9 hosted reference run also remains incomplete. Under its
[predeclared protocol](../release/qasper-agent-study/EXPERIMENT.md),
`nvidia/Nemotron-3-Ultra-550b-a55b` through Nebius attempted the same frozen
questions with a 15-action budget and requested thinking off. The $2 local
cost guard stopped it after 38 terminal episodes, during question 39 at
11 saved actions; question 40 was not attempted. The
[partial results](../release/qasper-agent-study/nebius-run/results.partial.json)
contain 27 submissions, 11 episodes without submission, and the interrupted
episode. The [usage record](../release/qasper-agent-study/nebius-run/usage-budget.json)
reports 366 requests and a **$1.947862 catalog-rate estimate**, not an invoice.
No comparable 40-question Nemotron Answer F1 is reported. Its larger serving
context, provider-managed model, and newer runner would also prevent a causal
training comparison even if all questions had completed. The
[reference release](../release/qasper-agent-study/README.md#bounded-hosted-reference)
preserves the failure and configuration separately from the Qwen comparison.

An earlier, separate continuation experiment assembled 995 QASPER-derived
conversations: 796 for training and 199 for paper-disjoint validation. Its
[September 25 report](../reports/qasper-scale-sft-2026-09-25.json)
records 15/40 semantic passes for the starting adapter, 10/40 after epoch one,
and 4/40 after epoch two on a different reserved validation set. Both
continuation checkpoints failed the declared decision rule and were rejected.
This was identity-blind model-assisted review, not a base-versus-v5 comparison.
The record shows that more continuation training did not reliably improve the
selected evaluation; it does not isolate a causal explanation for the decline.

For September 30, answerable quality did not improve, the grader changed, and
independent reference/answer review remains unfinished. No model promotion is
claimed. A [deterministic within-paper retrieval comparator](../benchmarks/envoybench/RETRIEVAL_BASELINE.md)
has prepared packets but no inference result, so the agent loop has not shown
that it earns its complexity over one fixed retrieval step. The eight-question
independent calibration packet remains blank. The separate
[October token diagnostic](../benchmarks/envoybench/LOGPROB_ANALYSIS.md) did not
establish generated-token likelihood as a useful error-warning signal and does
not change these grades.

The release stays text-only. Extracted paper text, including flattened table
content, is the available evidence; figures, visual layouts, and equation
rendering are not reliably interpreted. A useful weekly Obsidian digest and
queryable evidence library remain broader product gates, not consequences of
this known-paper result. No new Qwen training is required to reproduce the saved
analysis; new reference-model inference is recorded separately from these
September 30 outputs and grades.

## Reproduce the saved analysis and inspect the implementation

From the repository root, use the standard-library-only checker:

```bash
python scripts/reproduce_case_study.py
python scripts/reproduce_case_study.py --json
python -m benchmarks.envoybench.qasper_official score \
  --run-dir release/envoybench-v0.1/run
```

The first two commands verify packaged checksums and recompute September 30 counts from
saved results and review assignments. They do not download the corpus, execute
Qwen, regrade answers, or validate spans against full paper text. The third
computes official Answer F1 from the saved answers and original annotations. Open
[studio.html](../release/envoybench-v0.1/studio.html) directly for the saved
**Runs → Questions → Trace** walkthrough, including all 562 September 30 turns.

For the deeper frozen-corpus and review-binding check, the existing
[release workflow](../release/envoybench-v0.1/README.md) is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[viewer,qasper]'
python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data
python -m benchmarks.envoybench.verify \
  --run-dir release/envoybench-v0.1/run \
  --review-dir release/envoybench-v0.1/review-prepared \
  --judged-review release/envoybench-v0.1/review-model-assisted/review.json \
  --provisional
python scripts/launch_studio.py
```

The builder may download the pinned QASPER source if uncached. The verifier
checks frozen corpus identity, same-question coverage, source spans, and review
bindings; it still cannot establish semantic support. Viewing saved runs needs
no model endpoint or Docker. The launcher serves Studio at
<http://127.0.0.1:8765>.

For a code review or interview, these are concrete entry points:

| Inspect | Engineering question |
| --- | --- |
| [Training preprocessing](../scripts/train_sft.py) and [prefix/masking tests](../tests/test_train_sft.py) | Does the loss supervise the intended next action under the same prefix used at inference? |
| [Tool API](../src/env/tools.py) and [REPL](../src/env/repl.py) | How are reads bounded, offsets returned, observations truncated, and episode state executed? |
| [Evaluation runner](../benchmarks/envoybench/run.py) | Which model, corpus, decoding, sandbox, and implementation identities are recorded? |
| [Scoring and artifact bindings](../benchmarks/envoybench/score.py) | Can a changed result, missing response, or mismatched review silently enter a score? |
| [Diagnostics](../benchmarks/envoybench/diagnostics.py) and [artifact verifier](../benchmarks/envoybench/verify.py) | How are refusal behavior, citation integrity, returned errors, and missing measurements kept distinct? |
| [Saved answers and trajectories](../release/envoybench-v0.1/run/results.json) | Does the submitted evidence actually support the answer, including failures? |
