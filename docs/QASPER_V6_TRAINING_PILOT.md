# QASPER v6: protocol-aligned SFT pilot

Registered October 13, 2026, before a new optimizer run or inspection of the
new holdout's answers. This is one bounded attempt to improve the executable
paper agent, not a claim that the next adapter will be better.

## Why change the data

The [v5 training set](../data/sft/qasper-v5/README.md) contains 35 training
conversations and 4,812 supervised action tokens. Twenty-seven of those 35
episodes end by action three; 30 were assistant-repaired. On the previously used
40-question development set, v5 submitted by action three on 38 questions and
passed the provisional supported-answer review on only 3 of 20 answerable
questions. It refused eight questions labeled answerable. These observations
make early stopping and over-refusal plausible failure modes, not proven causal
effects of the repairs.

V5 was also trained with the shorter `SYSTEM_PROMPT`, a question-only first user
message, ten-step tool observations, and final actions without `EVIDENCE` spans.
The current EnvoyBench runtime uses `QASPER_SYSTEM_PROMPT`, a tool/evidence
preamble in the first user message, a 15-step limit, and exact evidence spans.
V5 nevertheless produced span fields during the saved evaluation, so this
format mismatch is a controlled risk to remove, not an established explanation
of its score. A separate 995-conversation continuation worsened semantic passes
on its reserved set; more rows alone are not the intervention.

## Hypothesis and expected signal

**Hypothesis:** starting a fresh rank-4 QLoRA adapter from the pinned Qwen3-8B
base, then supervising replayed, source-reviewed trajectories in the *exact*
runtime prompt and tool protocol, will increase supported answers without
teaching premature refusal. This keeps the model, tool interface, and loss
masking fixed relative to the existing trainer; the learning variable is the
curated data and its protocol alignment. It does not isolate which individual
data change causes a gain.

**Expected signal:** the new adapter follows search-to-passage recovery when a
first search is weak, submits relevant exact spans when evidence is present,
and abstains only after a genuine investigation. On the frozen questions, it
should win paired supported-answer reviews against base Qwen and v5 while
keeping refusals, execution failures, and latency acceptable.

## Frozen data boundary

The [holdout lock](../data/research/qasper-v6-holdout-lock.json) identifies 24
questions on 23 official QASPER *test* papers: 17 labeled sufficient and seven
insufficient. The dataset revision, selection seed, source benchmark SHA-256,
corpus hash, prior-paper exclusions, exact question IDs, and paper IDs are
recorded there. These papers do not overlap the 137 reviewed training
candidates or the previously used test/validation snapshots. This holdout is
for one final candidate comparison, not data selection or prompt tuning.
An initial unrun selection overlapped six packaged EnvoyBench test papers; its
hash and rejection reason are recorded in the lock, and the replacement
excludes it as well. No answers or model outputs from either selection were
inspected while freezing the replacement.
QASPER may appear in the base model's pretraining; paper-disjointness prevents
fine-tuning leakage here but cannot prove pretraining novelty.

The starting pool is the existing 137 mechanically checked `qasper-span-v1`
trajectories under
`out/research/qasper-synthetic-reliability-v1/accepted/`: 107 answerable
(including 69 recovery traces) and 30 source-reviewed abstentions. Twenty of
the 107 retained answerable rows received individual source-visible semantic
review; the others passed exact QASPER answer/evidence-annotation checks and
fresh replay, which do not by themselves establish semantic support. All
semantic review was model-assisted, not independent human review. Its 29-row
paper-disjoint validation partition has already informed an earlier checkpoint
comparison, so it is only training/development validation, never a new promotion
test. We will not automatically accept additional QASPER-unanswerable labels:
10 of 40 raw synthetic refusals were rejected after inspecting their papers.

## Data gate before GPU use

Replay the accepted action sequences against the current runtime configuration:
`QASPER_SYSTEM_PROMPT`, `max_steps=15`, required evidence, initial preamble,
and fail-closed evidence verifier. The exported first user message and every
tool observation must come from that replay. Require:

1. No change in executed actions, source-visible tool payloads, answer, cited
   document, or exact evidence spans relative to the reviewed parent;
2. No verifier intervention, execution error, missing submission, fabricated
   observation, or raw-action/executed-action mismatch;
3. A strict final `SUBMIT … CITATIONS … EVIDENCE` parse and hash-bound review
   provenance, with all refusals source-reviewed;
4. No train/validation/holdout paper overlap and at least 80 answerable and 20
   reviewed-abstention candidates after filtering; and
5. Exact system-prompt and initial-observation equality with the evaluation
   protocol, plus per-action assistant-only loss-mask and token-limit checks.

If any gate fails, diagnose and repair the data before training. Do not append
gold answers to failed traces or select the shortest trace as a proxy for
quality. Record every rejection and retain the original artifacts.

## Training and decision rule

If the data gate passes, run **one** fresh-from-base one-epoch rank-4 QLoRA
candidate with the existing per-action SFT trainer, pinned Qwen3-8B revision,
seed 42, and no v5 adapter continuation. Record dataset/prompt hashes, token
counts, optimizer settings, software versions, GPU, wall time, and actual cost.
First run a tiny optimizer/reload smoke and stop on masking, format, or memory
failure. Read current GPU price and credit balance before setting a spend cap;
reserve capacity for all three held-out inference arms.

Compare base, v5, and the new adapter on the same 24 locked questions, corpus,
decoding, verifier, 15-step budget, and GPU class. Blind the model identities
for a source-visible pass/partial/fail review. Report the paired question table,
answerable and insufficient strata, false refusals, verifier events, Python
errors, submissions, tokens, latency, and cost. The automatic QASPER F1 and
span validity are diagnostics; a real quote is not necessarily support.

**Promotion rule for this small pilot:** at least four more fully supported
answers than base, more than v5, no increase in answerable-question false
refusals or Python-error episodes versus base, and mean latency/cost at most
twice base under matched hardware. These thresholds demand a visible effect;
24 questions cannot establish a precise generalization rate. If the candidate
misses the rule, keep the saved analysis and do not call it a better agent.
Any pass remains provisional until independent review or a second fresh set.
