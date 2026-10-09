# QASPER Agent Studio — a Qwen Envoy case study

This release accompanies the [short findings report](../../docs/QASPER_AGENT_STUDY.md)
and [technical case study](../../docs/TECHNICAL_CASE_STUDY.md). It adapts existing
QASPER paper questions to a bounded Python-tool agent, preserving source
attribution and separating answer overlap from evidence support.

**QASPER is by Dasigi et al. (2021), licensed CC BY 4.0.** This is an independent
adaptation, not an original question benchmark, official QASPER leaderboard
submission, or affiliated AllenAI product. The [data provenance](../../benchmarks/envoybench/data/README.md)
pins the source revision and explains the deliberate 20/20 answerability
balance. All 40 questions disclose their target paper and have now been
inspected, so they are development evidence for future experiments.

## Original Qwen results

The September 30 base/v5 inference records and provisional support judgments
remain in the [original release](../envoybench-v0.1/README.md). They are not
rewritten by later scoring or reference-model runs.

- [Inference manifest](../envoybench-v0.1/run/manifest.json) and
  [all 80 model results](../envoybench-v0.1/run/results.json).
- [Provisional model-assisted review](../envoybench-v0.1/review-model-assisted/review.json)
  and [reported score](../envoybench-v0.1/review-model-assisted/score.json).
- [Standalone Studio](../envoybench-v0.1/studio.html), also available as the
  [public demo](https://jasonlingg.github.io/Qwen-Envoy/).
- [Observed v5 training evidence](../qwen-v5-training/README.md).

From the repository root, with Python 3.10+:

```bash
python scripts/reproduce_case_study.py
python scripts/reproduce_case_study.py --json
```

This validates original release hashes and recomputes counts from existing
results and bound reviews, using only the Python standard library. It neither
runs a model nor independently judges answer support. The
[original release guide](../envoybench-v0.1/README.md#rebuild-the-corpus-and-verify-the-artifacts)
documents the optional frozen-corpus/source-span verification and local Studio.

## Scoring boundary

The [Qwen Answer F1 report](qwen-official-score.json) scores the unchanged
September 30 outputs: **19.9154% base, 30.6873% v5**. The
[reference export](references.json) contains all 86 original annotations for
40 selected questions, including multiple annotations on 37 questions.
The [official evaluator and license](../../benchmarks/envoybench/vendor/qasper/README.md)
are pinned to upstream commit `e996b6c7b1b5f95d9308a74e3586416c6e780df1`.

Grouping the report's per-question F1 values by fixed source answerability,
with 20 questions in each mean and missing predictions retained as zero,
gives **29.8307% base / 21.3746% v5 on answerable questions**, and
**10% / 40% on unanswerable questions**. These derived fixed-group means are
reported in the [short findings report](../../docs/QASPER_AGENT_STUDY.md).
They are not the upstream by-answer-type means described below.

Recompute offline, from the repository root:

```bash
python -m benchmarks.envoybench.qasper_official score \
  --run-dir release/envoybench-v0.1/run
```

The command prints its JSON report. Add `--output <new-report.json>` to save a
copy. It uses only the standard library and the published reference export;
no dataset download or model endpoint is needed. To check operation in an
isolated Python environment without installed site packages:

```bash
python -I -S benchmarks/envoybench/qasper_official.py score \
  --run-dir release/envoybench-v0.1/run
```

Original QASPER Answer F1 is a token-overlap measure against annotated answers.
It must be reported separately from the provisional rubric that asks whether
the answer's own cited passages support its claims. Neither metric turns a
valid character span into a semantic-support judgment. Exact spans and source
hashes are integrity checks, not independent review.

Scoring preserves the literal submitted answer, applies upstream normalization,
and takes the best match across references. It does not rewrite explanatory
refusals. Missing/failed/escalated episodes and submitted empty strings score
zero in the overall 40-question mean. The official by-answer-type means omit
missing predictions and use the first maximizing reference type, so their
denominators differ by model and are included in the JSON. Evidence F1 is not
reported because no conversion from character spans to official paragraphs was
performed.

## Bounded hosted reference

**Status: incomplete, stopped by the local cost guard.** Thirty-eight episodes
reached a terminal outcome; question 39 was interrupted after 11 recorded
actions, and question 40 was not attempted. The
[partial results](nebius-run/results.partial.json) contain 27 submissions,
11 episodes without submission, and the budget-interrupted episode. The
[manifest](nebius-run/manifest.json) preserves the incomplete status and
configuration. No comparable 40-question Nemotron Answer F1 is reported.

The [usage budget record](nebius-run/usage-budget.json) reports **366 provider
requests, 1,819,534 prompt tokens, and 42,776 completion tokens**, with a
**$1.947862 catalog-rate estimate**. This is not an invoice and applies no
cache discounts. The guard refused the next request because its conservative
reservation would exceed the $2 limit; the final local rejection is not an
additional provider request.

The [predeclared experiment](EXPERIMENT.md) specifies a separate
`nvidia/Nemotron-3-Ultra-550b-a55b` run through Nebius on the same 40 questions.
It freezes the prompt/tool task and a 15-action, 1,024-output-token budget,
with requested thinking off, no retry selection, and a $2 local estimated-cost
ceiling. Its larger serving context, different model and provider, and newer
runner make it a descriptive reference configuration, not a controlled test
of Qwen's training. This release preserves the incomplete run and its failures
under that protocol.

Re-running that experiment invokes a paid API; the commands in the scoring
sections above only inspect saved data. The original Qwen outputs and support
grades remain separate and unchanged.

The release contains a known-paper diagnostic, not proof of useful open-ended
research, reliable personal recommendations, or a trained-model promotion.
The reference-model run cannot establish the effect of Qwen's training.
