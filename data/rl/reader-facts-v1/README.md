# QASPER short-fact RL pilot

This is an adaptation of [QASPER](https://github.com/allenai/qasper), introduced
by Dasigi et al., *A Dataset of Information-Seeking Questions and Answers Anchored
in Research Papers* (2021). The source is `allenai/qasper` at revision
`13b496d2a5359329b110e3419628de3cf791843b`. It is not an original question dataset
or an official full-test QASPER evaluation. Preserve the upstream dataset's
licensing and attribution when redistributing these extracts.

The pool has 82 official-train questions on distinct papers for a bounded RL
pilot, eight separate official-validation papers for reward-contract checks,
and 12 more official-validation papers locked for future model evaluation.
The latter references are automatically prepared, not manually source-reviewed,
and have not been used for model calls or tuning in this pass. No reserved
`research_reader_v1` material was opened.

Every question receives this extra instruction:

> For this short-fact task, answer with only the requested name, number, or fact
> phrase; do not add explanation. Use the normal SUBMIT answer and citation format.

Passage windows use QASPER evidence annotations. Thus this curriculum evaluates
answering from a constrained scope, not retrieval quality or general research.
The policy still writes Python, inspects source through the frozen V2 Docker
runtime and submits an answer with a citation. Gold references are not policy
inputs. A finite set of source-reviewed variants defines the experimental reward;
correct unlisted paraphrases or shorter valid citations may receive zero.

`manifest.json` binds cases, inputs, keys, paper snapshots, review, history and
role locks by SHA-256. `source-review.json` records implementing-assistant review,
with `human_review` null. It is not independent human review. The 84 reward
controls use eight separate papers but the same implementing assistant; positive
variants are explicit references, not a blind semantic-judge calibration.

Review rejected 28 candidate labels. Historical question-to-paper closure then
excluded 11 otherwise accepted training papers conservatively; prior roles
include curation and unknown-purpose artifacts, not only demonstrated evaluation
use. `history-exclusions.json` preserves those decisions. Previous training-pool
reuse is explicitly allowed and disclosed; neither a fresh corpus nor unknown
model exposure is claimed. Both raw QASPER question IDs and canonical IDs are
joined to whole-paper exclusions. `historical-evaluation-reservations/` preserves
311 conservative prior question-to-paper reservations for future inventories.

The size is an engineering-pilot choice, not evidence that 82 questions suffice
for generalization. No real Qwen3-8B inference or optimizer update was performed
while preparing this pool. See `docs/READER_RL_READY.md` for validation, the frozen
bounded first experiment and its remaining GPU proof.
