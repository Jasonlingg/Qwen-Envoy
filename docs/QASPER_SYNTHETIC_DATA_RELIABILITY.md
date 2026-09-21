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
