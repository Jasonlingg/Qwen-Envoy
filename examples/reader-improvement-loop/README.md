# Saved local reader improvement queue

This example imports the completed [hybrid V5 smoke](../qasper-hybrid-v5/README.md)
without changing its harness or making any model call. All three cases are retained.
Original evaluation questions and papers remain regression-only, excluded from
training. The original Qwen outputs are unchanged.
Data: AllenAI QASPER, Dasigi et al. (2021), CC BY 4.0, revision
`13b496d2a5359329b110e3419628de3cf791843b`; adapted saved development examples,
not an original benchmark or an official QASPER result.

| Case | Implementing-assistant source review | Proposed final-action repair |
| --- | --- | --- |
| Humor cues | Fail: available features were denied | Extract the three stylistic features and four latent semantic aspects, with attribution to the related-work discussion. |
| Translation models | Pass: LSTMs, attention and dropout are supported | None; keep the original as a control. |
| Racial/ethnic bias | Fail: scoped uncertainty was submitted as an answer, without the required gap/query | Use the existing `submit_gap()` protocol and a specific suggested search query. |

Artifacts:

- `queue.json`: original source, multi-turn traces, actual final model inputs,
  mechanical signals and frozen reproducibility metadata. Semantic verdicts null.
- `review-template.json`: blank review, never treated as a completed judgment.
- `implementing-assistant-review.json`: explicitly model-assisted source review,
  including correction proposals. **No independent human review was performed.**
- `correction-drafts.json`: actual checked output, with two quarantined drafts.
  They are not SFT examples: no corrected observations or training conversations
  are manufactured. Both have `training_eligible: false`, `training_ready: false`
  and `docker_replay: required_not_performed`.

The correction actions have not been executed. The proposed bias query has not
been searched. The checked quotes establish provenance; the support/completeness
judgments are the attributed reviewer's, not the mechanical checker's. No new
Qwen answer, fine-tuning result, held-out result or model improvement is claimed.

Reproduce with fresh output directories:

```bash
python scripts/reader_improvement.py prepare \
  --run examples/qasper-hybrid-v5 --output out/reader-queue-reproduction
python scripts/reader_improvement.py check \
  --run examples/qasper-hybrid-v5 \
  --queue out/reader-queue-reproduction \
  --review examples/reader-improvement-loop/implementing-assistant-review.json \
  --output out/reader-drafts-reproduction
```

Queue and check output are deterministic for the same frozen bundle/review.
The CLI never overwrites an existing output directory. See the
[workflow guide](../../docs/READER_IMPROVEMENT_LOOP.md) and
[scope](../../docs/implementation/2026-10-22-reader-improvement-loop.md).

Validation: **55 focused tests passed; full suite 2042 passed in 272.17 seconds**,
no skips and three existing SWIG warnings. Ruff/diff passed. All 29 frozen
implementation files and 1,680 prior file/marker hashes remain unchanged.
`validation.json` binds the packaged artifacts; `full-tests.txt` is the actual
standalone suite output. This validates the local workflow, not answer correctness
or improved model performance.
