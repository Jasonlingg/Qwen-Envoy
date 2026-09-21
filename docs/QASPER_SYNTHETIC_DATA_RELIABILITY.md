# QASPER synthetic trajectory reliability pilot

Date: 2026-09-21

## Decision

The programmatic **answerable-data lane** is reliable enough to scale with the
audit and filtering pipeline kept in the loop. The programmatic abstention lane
is not reliable enough to scale in its current form, and the raw generator output
is not reliable enough to train on without filtering.

The pilot began with 148 accepted trajectories. Semantic review rejected 11:

- 10 of 40 generated abstentions were unsafe because the frozen paper text
  contained a plausible answer.
- 1 of 21 reviewed answerable trajectories had a QASPER `Yes` label whose cited
  evidence did not support the wording of the question.

After those rows were removed, all 137 retained trajectories passed fresh REPL
replay, provenance, leakage, evidence, split-isolation, and semantic coverage
checks. The final `ready_to_scale` gate is `true`.

A subsequent autonomous rollout also found that the current Qwen3-8B SFT adapter
transfers the learned behavior to the 29 retained, paper-disjoint validation
questions. In an identity-hidden semantic review, SFT produced 15 passes, 6
partials, and 8 failures, compared with 7 passes, 4 partials, and 18 failures for
base Qwen3-8B. SFT won 13 paired questions, base won 2, and 14 tied. This is a
strong development signal that the task is learnable and that the adapter learned
the execution protocol; it is not yet a clean test-set claim or an attribution of
the gain solely to the new programmatic rows.

This supports generating the answerable portion of the larger dataset in audited
batches. It does not support automatically generating hundreds of abstentions or
generating 600--1,000 rows once and trusting them without further review.

## Pre-registered test

**Hypothesis:** QASPER annotations can be converted into executable research-agent
trajectories without leaking the answer into search, fabricating evidence, or
teaching false abstentions, provided an independent verifier and a semantic filter
run after generation.

**Expected signal:**

1. Every accepted action sequence replays exactly in a fresh environment.
2. Every answerable response matches a QASPER train annotation and cites an exact
   annotated span that the trajectory observed before submission.
3. Corrupted fixtures are rejected.
4. Training and validation papers are disjoint and reserved evaluation papers do
   not appear.
5. All abstentions and at least 20 answerable examples receive source-visible
   semantic review.

**Decision rule:** do not scale unless all five checks pass. Questionable semantic
examples are excluded; they are not counted as passes because a mechanical score
succeeds.

## Results

| Check | Result |
|---|---:|
| Raw pilot trajectories | 148 |
| Semantically rejected | 11 |
| Retained trajectories | 137 |
| Retained answerable / unanswerable | 107 / 30 |
| Fresh exact REPL replays | 137 / 137 |
| Mechanically valid retained rows | 137 / 137 |
| Answerable semantic review | 20 pass / 1 reject |
| Semantically reviewed retained abstentions | 30 / 30 |
| Raw abstention semantic review | 30 pass / 10 reject |
| Train / validation rows | 108 / 29 |
| Train / validation paper overlap | 0 |
| Reserved evaluation paper overlap | 0 |
| Duplicate IDs or trajectories | 0 |
| Deliberate corruptions rejected | 5 / 5 |
| Final scaling gate | **PASS** |

The five corruption tests cover a hidden answer term inserted into a query, a
shifted evidence offset, a changed observation, submission without investigation,
and a citation to a missing document.

The final accepted-candidate SHA-256 is
`87630b733b83d13689bd67b5d0920241a2e30412d0834ff00f4936eadce7af42`.
The frozen corpus hash is
`233445316f342d5652744bce395c2654372b84b91efcfb12e7cc03f67dd247ae`.

## Autonomous Qwen transfer check

After the data gate passed, base Qwen3-8B and the behaviorally selected targeted
SFT adapter (`checkpoint-150`) independently attempted every retained validation
question through the real persistent REPL. These 29 questions are disjoint from
the SFT training papers and contributed no gradients. They did contribute to
next-action validation loss during training, so this is development evidence,
not the untouched confirmation set.

The comparison used the same exact configuration for both arms:

- Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`
- deterministic decoding, seed 42, and at most 10 model turns
- raw `search_within()` with its top-three default
- the QASPER exact-span system prompt used by the training exporter
- corpus hash `233445316f342d5652744bce395c2654372b84b91efcfb12e7cc03f67dd247ae`
- one NVIDIA A40 with PyTorch 2.8.0+cu128

Both arms first passed a fixed five-question smoke gate. The complete runs then
submitted all 29 episodes without harness-level run failures.

| Measure | Base Qwen3-8B | Targeted SFT | Change |
|---|---:|---:|---:|
| Semantic pass | 7 / 29 | **15 / 29** | +8 |
| Semantic partial | 4 / 29 | **6 / 29** | +2 |
| Semantic fail | 18 / 29 | **8 / 29** | -10 |
| Paired wins | 2 | **13** | +11 net |
| Automatic outcome reward | 0.305 | **0.549** | +0.243 |
| Answer token F1 | 0.248 | **0.474** | +0.226 |
| Citation precision | 0.448 | **0.828** | +0.379 |
| Citation recall | 0.621 | **0.862** | +0.241 |
| Sufficient-question false refusals | 6 / 20 | **4 / 20** | -2 |
| Insufficient-question false answers | 5 / 9 | **2 / 9** | -3 |
| Episodes containing a Python execution error | 21 / 29 | **4 / 29** | -17 |
| Mean model turns | 5.83 | **5.21** | -0.62 |
| Mean wall time per question | **11.5 s** | 20.6 s | +9.1 s |

On the 20 answerable questions, semantic passes rose from 3 to 8; another 6 SFT
answers were useful but incomplete or paired with an inadequate exact span. On
the 9 expert-labeled unanswerable questions, correct abstentions rose from 4 to
7. The clearest behavioral change was tool reliability: base Qwen repeatedly
called `search()` with an unsupported `doc_id` argument, while SFT generally used
the trained `search_within()` → recovery search → `passage()` → `SUBMIT` sequence.

The semantic review was frozen before opening the system-identity key. It was
performed by Codex against QASPER's official answers and materialized evidence,
not by an independent human panel. Correct answers with irrelevant, invalid, or
truncated evidence spans received partial rather than full credit. The paired
score difference was positive on 13 questions and negative on 2, with 14 ties.

This result answers the immediate student test: Qwen can autonomously execute the
programmatically represented tasks, and the SFT adapter is materially better than
the base model at them. It does not isolate the marginal effect of the filtered
108-row training split because the evaluated adapter continued from the earlier
QASPER SFT checkpoint and was trained on the raw pre-filter training set. A clean
training-data attribution requires retraining from the same starting adapter on
the filtered rows and comparing both candidates on a new paper-disjoint test set.

Local transcripts, manifests, smoke logs, and the blind-review packet are under
`out/research/qasper-synthetic-reliability-v1/student-rollout-eval/`. The full-run
transcript hashes are:

- base: `870a7439782baf5d39a48db1d0fdd84a92ea6624662712a88afe54af12e3d795`
- SFT: `75b3d01ed08a77feed83b2b998ae9056d83f9122c7b70482bdc22c5c469935cb`

## What the semantic audit caught

These examples show why exact-answer and exact-citation checks alone were not
enough:

- A trajectory abstained on whether self-attention can reproduce convolution.
  The paper explicitly describes a constructive proof and a reparameterization.
- A trajectory abstained on how NMT keeps generated reviews on topic. The paper
  explicitly lists the conditioning context: rating, restaurant, city, state, and
  food tags.
- A trajectory abstained on the computational saving of a mixture-of-experts
  model. The paper reports using 6% of the comparison computation, supporting an
  approximate 94% saving.
- One answer said that tweets came from “any individual,” but its cited passage
  only established that public tweets were collected through the Twitter API.

QASPER's official unanswerable label is useful supervision, but it is not
sufficient evidence for teaching an agent to abstain. Some questions labeled
unanswerable still have a plausible answer in the frozen paper text, and a
question-answer annotation can occasionally be looser than the cited passage.

## What can scale

Of the 21 answerable examples reviewed, 20 passed and one was rejected. All 107
retained answerable examples also passed exact annotation, evidence, leakage, and
replay checks. This is enough to proceed cautiously with programmatic answerable
trajectories, auditing every row mechanically and at least 10% semantically in
each batch.

Only 30 of 40 automatic abstentions passed. A 25% rejection rate is too high to
scale. The larger build must therefore do one of the following for abstentions:

1. reuse the 30 reviewed pilot abstentions;
2. add teacher-authored abstentions that receive the same source-visible review;
3. improve the absence verifier and re-run a new pilot before accepting more
   programmatic abstentions.

The immediate 600--1,000-row generation should use option 1 plus programmatic
answerable trajectories. This gives Qwen many grounded search/read/cite examples
without teaching it to refuse questions that the paper actually answers.

## Audit implementation

- `scripts/prepare_qasper_synthetic_semantic_review.py` creates a deterministic,
  source-visible review packet containing every abstention and a balanced sample
  of answerable trajectories.
- `scripts/filter_qasper_synthetic_data.py` removes rejected or ambiguous rows
  while preserving the existing paper-disjoint split.
- `scripts/verify_qasper_synthetic_data.py` independently recomputes the checks;
  it does not trust the generator's acceptance flag. With `--replay`, it executes
  every saved action again and requires matching actions, observations, and
  terminal states.
- `tests/test_verify_qasper_synthetic_data.py` checks known-good trajectories and
  deliberate corruptions.

The final local artifacts are under
`out/research/qasper-synthetic-reliability-v1/`. Reproduce the final gate with:

```bash
python scripts/verify_qasper_synthetic_data.py \
  --candidates out/research/qasper-synthetic-reliability-v1/accepted/candidates.jsonl \
  --benchmark out/research/qasper-sonnet-expansion-20260919/benchmark.json \
  --corpus out/research/qasper-rl-v1/train-v2/corpus \
  --train out/research/qasper-synthetic-reliability-v1/accepted/train.jsonl \
  --validation out/research/qasper-synthetic-reliability-v1/accepted/val.jsonl \
  --semantic-review data/research/qasper_synthetic_reliability_v1_review.json \
  --output out/research/qasper-synthetic-reliability-v1/final-audit.json \
  --replay
```

## Claim boundary

The answerable rows inherit their semantic target from QASPER's expert
annotations. The additional source-visible audit here was performed by Codex, not
by an independent human panel. It is strong evidence that the conversion and
filtering machinery work and that obvious semantic failures are removed. It is
not a measurement of inter-annotator agreement or proof that every retained
natural-language label is indisputable.

For the 600--1,000-row build, generate answerable rows in batches, run the full
mechanical audit on every row, and sample at least 10% semantically per batch.
Stop the build if a batch's answerable semantic rejection rate exceeds 5% and fix
the generator before adding more data. Do not accept newly generated abstentions
from the current programmatic method.
