# Prepared QASPER semantic-judge audit

Status: **prepared, no live judgments, no human review, not training-ready**.

- 32 authored controls on the eight frozen reward-validation papers.
- Four actual saved Qwen/Docker episodes from the GPU smoke, diagnostic only.
- 52 original QASPER annotation entries; all text annotations are retained.
- Eight proposed paraphrases rejected by the finite checker; human confirmation pending.
- Zero model/API requests, zero paid spend and zero GPU launches for this audit.

`cases.json` preserves source/role/origin and baseline scores. `requests.json`
contains only judge-visible inputs and the strict response schema, with no expected
labels or worker-model identity. No provider/model is selected. The source-paper
passages and candidate quotations are exact frozen text. Multiple QASPER annotations
are alternative references, not a requirement to imitate every reference string.

`human-review.json` is an intentionally blank independent-human template for the
32 reward-validation cases. Read the corresponding question, annotations, candidate
and evidence in `cases.json`, then label correctness/completeness, claim support,
and citation relevance independently. Enter reviewer identity/time and notes.
Do not fill these fields with model judgments. The four previously exposed training
cases are excluded from this calibration gate.

`comparison.json` reports all validation cases as pending, with unmeasured error
counts set to null. `judgments.json` is empty. No judge improvement, RL gain, or
semantic reliability is claimed. Authored fixture labels and mocked verdicts used
by tests are not independent validation evidence.

See [design and commands](../../docs/READER_SEMANTIC_REWARD.md).

Verification: 36 focused tests passed; full suite 2,117 passed, 16 skipped,
three existing SWIG warnings. After adding request-template hash binding,
the final eight contract tests passed. All 29 frozen runtime files and frozen
dataset/reference hashes remain unchanged. No Git commits.
