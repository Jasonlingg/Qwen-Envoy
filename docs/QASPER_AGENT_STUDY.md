# QASPER Agent Studio — findings and limitations

QASPER Agent Studio is a Qwen Envoy case study of paper-question answering
through executable Python tools. The questions, answers, and paper annotations
come from **QASPER (Dasigi et al., 2021; CC BY 4.0)**. This independent project
adapts them to known-paper investigation; it is not a new question benchmark or
an official QASPER/AllenAI product. The [data provenance](../benchmarks/envoybench/data/README.md)
records the pinned source and split construction.

## What the Qwen comparison established

Base Qwen3-8B and its v5 QLoRA adapter attempted the same 40 questions on
September 30, 2026. Each question disclosed the target paper. The task measured
finding evidence and answering within that paper, not discovering relevant
papers. The [saved results](../release/envoybench-v0.1/run/results.json) retain
both successes and failures.

## Original QASPER answer scoring

Scoring those unchanged outputs with the official QASPER evaluator gives:

| Model | Answer F1, all 40 | Answerable 20 | Unanswerable 20 | Missing predictions |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3-8B | 19.92% | 29.83% | 10.00% | 11/40 |
| v5 SFT | 30.69% | 21.37% | 40.00% | 0/40 |

**The overall gain masks an 8.46-point F1 decline on the answerable group.**
These are fixed groups of 20, retaining zero scores for missing submissions:
base misses two answerable questions and nine unanswerable questions. The
subgroups use the source answerability flags; they are descriptive aggregates,
not new human review. They differ from upstream answer-type means, which omit
missing predictions and can select different reference types per model.

The [score artifact](../release/qasper-agent-study/qwen-official-score.json)
uses **all 86 original annotations** for the selected 40 question IDs; 37
questions have multiple references. The unchanged
[official evaluator](../benchmarks/envoybench/vendor/qasper/README.md) takes
the best normalized token-overlap F1 among each question's references, then
averages over all 40. Failed or missing submissions score zero. Submitted
strings receive only upstream normalization; refusal paraphrases are not
rewritten into `Unanswerable`.

This is the official scoring method on a selected, known-paper agent workload,
not a full-QASPER score or leaderboard comparison. F1 does not check whether
cited evidence supports the answer. No Evidence F1 is claimed: the agent's
character spans were not converted to QASPER paragraph strings.

## Separate evidence-support judgments

| Measure | Base Qwen3-8B | v5 SFT |
| --- | ---: | ---: |
| Submitted answers | 29/40 | 40/40 |
| Episodes with a recorded tool, runtime, or endpoint error | 22/40 | 1/40 |
| Provisional supported-answer passes on answerable questions | 3/20 | 3/20 |
| Provisional correct refusals on unanswerable questions | 2/20 | 12/20 |
| Provisional passes, all questions | 5/40 | 15/40 |

The aggregate pass increase comes from refusals. It does not establish better
substantive answers. V5 also refused eight questions labeled answerable. An
always-refuse policy would be expected to pass 20/40 under these labels while
providing zero useful answers; that is an analytical expectation, not another
recorded run.

These support judgments are model-assisted and have not received independent
human validation. Their answerability subgroups use the frozen source labels,
not a new adjudication of every paper. The planned Sonnet review stopped after
eight questions;
the complete Codex-assisted review was collected separately. Base also hit the
shared 8,192-token serving cap on eleven questions, and its failed-episode
timing is missing. The experiment therefore does not establish a general
reasoning, latency, or cost advantage. The [full comparison](../benchmarks/envoybench/COMPARISON_2026_09_30.md)
records these qualifications.

## The stopping pattern

V5 submits on **step three for 38/40 questions**, and step four for two. The
training data also concentrate on short trajectories: **27/35 (77.1%)** finish
by step three. Of the 35 training examples, **30 were assistant-repaired**.
Those same 30 questions have 151 actions in their original archived trajectories
and 92 after repair: means of **5.033 and 3.067**, including submission.

This was partly a construction choice. The preparation plan selected
[2–6-action episodes](QWEN3_SFT_DATA_ITERATION.md#changes-in-the-preparation-pipeline)
and explicitly removed redundant work. The [v4 review](../data/research/qasper_sft_v4_review.json)
and [v5 review](../data/research/qasper_sft_v5_review.json) publish parent hashes,
decisions, and edited actions. The original-length aggregate was recomputed
from the archived Sonnet pilots and legacy Claude batch; their full raw
trajectories remain local. The final [training](../data/sft/qasper-v5/train.jsonl)
and [validation](../data/sft/qasper-v5/val.jsonl) conversations are public.

It would be inaccurate to attribute the short paths simply to Sonnet solving
everything quickly. Across the combined 44 conversations, only seven preserve
unchanged teacher episodes; 37 were repaired with source/reference information
available to the editor. Twenty-one trace back to recorded Sonnet generation;
23 trace back to a legacy Claude batch that did not record its model identity.
The first Sonnet pilot gave answerable questions gold answers and annotated
passages, then disabled that guidance after reference-leakage failures. These
were demonstrations constructed with more information than the student has at
inference. See the [historical generation record](QWEN3_SFT_DATA_ITERATION.md#september-18-follow-up-source-guided-teacher-generation).

Short targets do not all share one exact template. Among the 18 three-action
training examples, eight search twice, six search then read, three search then
extract, and only one searches then calls `passage()`. Sixteen training
questions have insufficient-evidence answers, all explaining a specific gap;
none simply says `Unanswerable`. No final train/validation target includes
character-offset `EVIDENCE` fields, although evaluation requires them.

The observed distribution supports a **hypothesis of learned early stopping**.
It does not prove that repairs caused v5's behavior: data content, prompt,
protocol, and checkpoint selection were not independently varied. A real
citation also does not prove support—the saved dataset-name example answers
correctly while citing the adjacent, irrelevant passage. The
[technical case study](TECHNICAL_CASE_STUDY.md#three-traces-worth-inspecting)
walks through that failure and two contrasting traces.

A pre-run runtime archive also has a two-successful-action refusal check, but
this was bounded corrective feedback, not a hard minimum or a stopping rule.
The environment-file hash matches the run manifest; the evidence-state module
was not separately hash-pinned. None of the 80 saved trajectories records a
premature-refusal rejection, so it did not visibly force the three-step pattern.

## Bounded reference run: amended completion

A separate October 9 run used `nvidia/Nemotron-3-Ultra-550b-a55b` through
Nebius on the same frozen question list. Its
[predeclared protocol](../release/qasper-agent-study/EXPERIMENT.md) allowed
15 actions and 1,024 output tokens per action, requested thinking off, and
set a $2 local estimated-cost ceiling. That guard stopped after **38 terminal
episodes**, during question 39 after 11 actions; question 40 was not attempted.
The [original partial traces](../release/qasper-agent-study/nebius-run/results.partial.json)
and incomplete manifest are preserved unchanged.

The user then authorized a **$25 total** local estimated-cost ceiling. The
[recorded amendment](../release/qasper-agent-study/CONTINUATION_2026_10_09.md)
restarted question 39 at step 1 because its live state could not be restored,
and ran question 40 for the first time. The [derived 40-row result](../release/qasper-agent-study/nebius-amended-run/results.json)
retains the original first 38 outcomes. Question 39 used 15 steps without
submitting; question 40 submitted `85%` after two steps, scoring zero F1.
Across the full selected set there are **28 submissions and 12 missing
predictions**. The [official score artifact](../release/qasper-agent-study/nemotron-official-score.json)
reports **19.03% Answer F1**. This is normalized answer-token overlap, not an
evidence-support grade; no independent support review was done for Nemotron.

The [cumulative usage record](../release/qasper-agent-study/nebius-amended-run/usage-budget.json)
contains **383 provider requests**, 1,903,542 prompt tokens, and 43,960
completion tokens. Its catalog-rate estimate is **$2.035422**, before cache
discounts; this is not an invoice. The initial $2 limit and no-retry term were
amended only for the interrupted question, and the
[derived manifest](../release/qasper-agent-study/nebius-amended-run/manifest.json)
records that lineage. The $25 figure is a local guard, not a provider cap.

The reference configuration also differs from September 30 in model,
provider, serving context, and runner version. It cannot isolate the effect of
Qwen's training or establish a general model ranking. Raw failures are
preserved, without changing the original Qwen answers or support grades.

## Verifier correction for future runs

The saved runs used a one-feedback verifier configuration that could accept a
structurally invalid submission once feedback was exhausted. A trace audit found
one such accepted submission in base Qwen, none in v5, and nine in the amended
Nemotron run. This concerns accepted-submission and evidence-integrity status;
the reported official Answer F1 still scores the literal saved answer strings.
It does not establish whether any cited passage supports an answer.

New EnvoyBench runs default to a versioned, fail-closed verifier: an invalid
submission after the recovery turn ends as `escalated`, with zero reward and no
accepted prediction. The old artifacts and scores remain untouched, and the
historical behavior is available only through an explicit `legacy-v1` run mode.

## Reproduce and interpret

The [study release](../release/qasper-agent-study/README.md) documents the saved
artifacts and commands. The original comparison can be checked without a
network, GPU, endpoint, or package installation:

```bash
python scripts/reproduce_case_study.py
python -m benchmarks.envoybench.qasper_official score \
  --run-dir release/envoybench-v0.1/run
```

The first command recomputes mechanical counts and aggregates existing support
judgments; the second runs original-reference Answer F1. Neither makes new
semantic judgments. The
[public Studio](https://jasonlingg.github.io/Qwen-Envoy/) exposes the actual
answers and actions, and the [training evidence](../release/qwen-v5-training/README.md)
pins the observed recipe while disclosing the missing historical script version.

The 40 questions are now inspected development evidence. They can diagnose
failures, but cannot serve as a fresh holdout for a future intervention.
No new Qwen training or model promotion is part of this closing study.
