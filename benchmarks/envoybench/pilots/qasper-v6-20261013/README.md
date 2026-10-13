# QASPER v6 pilot audit bundle

This is the complete saved 24-question, three-model EnvoyBench comparison from
October 13, 2026. The method, limitations, and decision are in
[`docs/QASPER_V6_PILOT_RESULTS.md`](../../../../docs/QASPER_V6_PILOT_RESULTS.md);
the pre-run hypothesis and threshold are in
[`docs/QASPER_V6_TRAINING_PILOT.md`](../../../../docs/QASPER_V6_TRAINING_PILOT.md).

`manifest.json` and `results.json` are the original complete run matrix.
`models.json` declares the pinned base/v5/v6 identities and decoding settings;
endpoint weight identity and hardware are operator-declared, not remotely
attested. `training-manifest.json` hashes the replayed SFT data and trainer.
The v6 adapter weights are not stored in Git; their SHA-256 is in
`models.json`, and a local backup is at
`out/research/qasper-v6-pilot-20261013/trained-adapter/`.

`review-original.json` was prepared with anonymous row IDs before any
verdicts. Three model-assisted, source-visible reviewers independently filled
`review-part-{1,2,3}.json` while blind to `blind-key.json` and automatic
scores. `review-completed.json` locks those 72 verdicts and 51 explicit
answerable-question false-refusal decisions. `review-provenance.json` hashes
the parts and completed review. `reviewed-score.json`, `answer-f1.json`, and
`promotion-diagnostics.json` are the derived results. This is **not a human
reviewed score**: QASPER references and source-support verdicts remain
provisional, including the one ambiguous case that can flip the numeric
promotion threshold.

The question lock is
[`data/research/qasper-v6-holdout-lock.json`](../../../../data/research/qasper-v6-holdout-lock.json).
The paper corpus is regenerated from the pinned QASPER source via
`python -m benchmarks.envoybench.package_v6_holdout`; it is not duplicated in
this folder. With the packaged holdout at
`out/research/qasper-v6-envoybench-holdout-20261013`, the mechanical and
review scores can be checked without a GPU:

```bash
python -m benchmarks.envoybench.score_v6_holdout \
  --dataset out/research/qasper-v6-envoybench-holdout-20261013 \
  --run-dir benchmarks/envoybench/pilots/qasper-v6-20261013 \
  --output /tmp/qasper-v6-answer-f1-check.json

python -m benchmarks.envoybench.score score \
  --review benchmarks/envoybench/pilots/qasper-v6-20261013/review-completed.json \
  --key benchmarks/envoybench/pilots/qasper-v6-20261013/blind-key.json \
  --results benchmarks/envoybench/pilots/qasper-v6-20261013/results.json \
  --output /tmp/qasper-v6-review-check.json

python scripts/analyze_qasper_v6_promotion.py \
  --automatic benchmarks/envoybench/pilots/qasper-v6-20261013/automatic.json \
  --review benchmarks/envoybench/pilots/qasper-v6-20261013/review-completed.json \
  --blind-key benchmarks/envoybench/pilots/qasper-v6-20261013/blind-key.json \
  --results benchmarks/envoybench/pilots/qasper-v6-20261013/results.json \
  --base-key qwen_base --v5-key qwen_v5 --candidate-key qwen_v6 \
  --output /tmp/qasper-v6-promotion-check.json
```

The run uses selected QASPER test questions and source text under QASPER's
recorded CC BY 4.0 license. It is a narrow paper-agent case study, not a
score on the full QASPER benchmark.
