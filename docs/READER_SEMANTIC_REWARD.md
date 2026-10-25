# QASPER-grounded semantic reward candidate — October 23, 2026

The offline RLAIF candidate is implemented and an audit pack is prepared. **No
live judge has run, no independent human review has occurred, and the candidate
is not enabled for training.** The current reference reward and frozen reader
harness are unchanged. No GPU rental, paid request, or Git commit was made.

## Why QASPER's references are enough to start, but not a string whitelist

The frozen bundle already contains the original QASPER answer annotations and
supporting evidence. We reuse every available text annotation, including multiple
annotations for a question. We do not replace them with an invented reference
answer or send the old finite accepted-answer list to the judge. The paper and
scope remain authoritative: conflicting or insufficient references should yield
an uncertain judgment rather than guessed reward.

The current reward can reject a semantically equivalent answer or alternative
valid evidence. Our candidate separates:

1. Is the answer correct and complete for the question, preserving numbers,
   units, negation, conditions and dataset/method bindings?
2. Does the candidate's actual cited evidence support every material claim?
3. Are all its citations relevant?

A genuine quote proves provenance only. Reference evidence or another displayed
paper passage cannot substitute for missing candidate support. The judge must
attribute supported claims to candidate evidence IDs. Code checks source bounds,
controller-owned inspection receipts, actual quote clipping, strict response
schema, candidate/rubric hashes and evidence identities. These checks cannot
establish semantic truth or ensure an AI judge resists prompt injection.

The candidate reward is 1 only when all three semantic judgments are true and
all enumerated claims are supported with attributed citations. A complete known
failure gives 0. Uncertainty gives null; malformed/contradictory judgments are
errors, never automatic zero-reward training examples. A future trainer must
exclude the entire group when any member has unavailable reward.

## Hypothesis, control and decision rule

Hypothesis: a QASPER-reference-grounded AI judge can accept valid paraphrases
that the frozen finite checker rejects, without accepting unsupported or wrong
answers. The control is the existing unchanged reference reward on the same cases.

Use the eight existing reward-validation papers, not the twelve locked future
model-evaluation papers. Compare judge verdicts with separately attributed human
labels on correctness, support and citation relevance. The preliminary development
gate requires complete case coverage, both positive and negative human labels,
zero false acceptances, improved correct acceptance over the finite checker, no
lost correct acceptances, and agreement on the three axes. Missing labels or
judgments produce null error counts, not a claim of zero errors. Passing a small
control pack would not establish general judge reliability or authorize training;
`training_ready` remains false.

## Prepared audit

[The audit pack](../examples/reader-semantic-reward-v1/README.md) has **36 cases**:
32 authored controls across eight reward-validation papers and four saved real
Docker/Qwen episodes from the successful GPU smoke. The authored controls include
an original reference answer, a proposed paraphrase, an incorrect answer and a
correct answer with a genuine one-character citation. Their inspections are
explicitly assumed for fixture construction; they are not model rollouts.

The pack contains 52 original annotation entries across the 36 cases. All eight
proposed paraphrases receive zero from the old checker. This measures the finite
checker's restriction, not independently validated paraphrase correctness. Human
fields are blank. No authored expectations or baseline scores enter judge inputs.
The four already-exposed training-paper episodes are diagnostics only and are
excluded from the reward-validation gate. Their packets are reverified against
frozen sources and the original prompt before export.

Judge requests are prepared one case at a time. No network runner is included:
we have not selected/validated a hosted judge, established a new paid allowance,
or measured its behavior. The previous Nemotron Super batch diagnostic returned
no usable judgments after consuming its reasoning budget; that historical allowance
remains consumed and is not reused. A fresh live diagnostic must freeze model,
reasoning controls, request/token limits and a reliable spending bound first.
Temperature zero is a decoding setting, not a guarantee of semantic correctness
or deterministic serving. Human validation remains a separate requirement.

## Files and commands

- `src/training/reader_semantic_reward.py`: source-bound input construction,
  semantic verdict schema, candidate reward and human comparison.
- `scripts/prepare_reader_semantic_reward.py`: offline export and saved-verdict
  comparison. It cannot call an API or launch training. A bad case remains unknown
  while other valid case scores are retained.
- `tests/test_reader_semantic_reward.py`: protocol/gating tests with explicitly
  authored verdicts; these are not actual judge-accuracy measurements.

Prepare into a new directory (existing exports are not overwritten):

```sh
python scripts/prepare_reader_semantic_reward.py --output out/semantic-audit \
  --saved-rollouts out/prime-rl-smoke-20261023/rollouts
```

Omit `--saved-rollouts` to export only the 32 controls. `requests.json` contains
blinded candidate requests with the model explicitly unselected. After a separately
bounded judge run, put its parsed verdict objects into `judgments.json`. A human
reviewer separately fills the three boolean axes, notes, identity and timestamp
in `human-review.json`; model-assisted labels must not be represented as human.

```sh
python scripts/prepare_reader_semantic_reward.py --output out/semantic-audit --score
```

This writes `scored-comparison.json` exclusively. It does not verify the identity
of an external live judge run; `live_judge_run_verified` remains false. A complete
training integration additionally needs validated judge identity/receipts, group
admission, fixed reward version and a separately authorized training experiment.

## Actual verification

Focused semantic/reward/data checks: **36 passed**. The authored fixtures exercise
input integrity, original multi-annotation preservation, malformed verdicts,
unknown rewards, forbidden evaluation roles, missing human labels and partial
failure retention. Full repository results are recorded in the audit README.
Full suite: **2,117 passed, 16 skipped**, three existing SWIG warnings. After
the final request-template hash binding, **eight contract tests passed**. Ruff and
Git whitespace checks pass. All 29 frozen runtime files and all frozen
dataset/reference hashes are preserved.

The design follows QASPER's distinction between answers and supporting evidence
([original paper](https://arxiv.org/abs/2105.03011)). AI judges have documented
biases and must be checked against human judgments
([MT-Bench judge study](https://arxiv.org/abs/2306.05685)); that research does not
validate our candidate judge or establish that RL will improve Qwen.
