# Local reader improvement queue

The user approved a continuous improvement loop after choosing to freeze the
tested harness at `cce82c2`. This first increment is offline trace triage and
reviewable final-action correction drafts. No inference, GPU activation, training,
reward change, automatic model promotion or runtime harness change is included.

Hypothesis: source-bound review records make the two observed reader failures
(ignoring supplied facts and failing structured gap reporting) actionable without
moving the evaluation baseline. Expected signal: reproduce all three saved hybrid
smoke cases, keep the successful translation as a control, and prepare corrections
for source review without labelling them training-ready or semantically verified.
Decision rule: original trace/source/input hashes remain intact; changed evidence
or review bindings are rejected; review attribution remains explicit; incomplete
reviews fail closed; every candidate requires fresh Docker replay; baseline
questions/papers remain regression-only even after review. This is an engineering
workflow check, not an experiment showing model improvement.

Implement a small standalone CLI using saved QASPER passage smoke bundles. Prepare
one immutable queue and an editable review template. Preserve original multi-turn
actions, observations, exact final model input, source text/offsets, checkpoint,
prompt, decoding, corpus revision, seed and sandbox image. Include every case,
not only failures; automated signals describe mechanics, never semantic verdicts.
The reviewer supplies a diagnosis, notes and (optionally) a replacement final
action with source spans. Only final-action replacements are in scope; earlier
tool failures remain diagnostic and cannot be repaired by fabricating observations.

Checking a completed review produces quarantined correction drafts, not chat/SFT
files. Review checks distinguish span provenance from reviewer-attested semantic
support. No independent human judgment is generated. The adapter accepts frozen
bundles of the current format only; broader trace formats, continuous collection,
separate training data, Docker replay/export integration, candidate evaluation and
promotion are later increments. Current baseline cases must not enter training.

Verify tamper rejection, incomplete/misattributed reviews, uninspected or incorrect
spans, malformed/unsafe terminal actions and regression-only eligibility. Exercise
the complete local CLI with the actual three saved cases. Run focused tests and
the full relevant suite. Preserve all existing dirty files and historical results.

## Outcome

Implementation `01615f9` adds a standalone prepare/check CLI, reusing the existing
packet verifier and literal gap parser. The actual local run retains all three
saved hybrid cases. Explicitly implementing-assistant source review proposes two
final-action repairs and keeps translation as a passing control. Both drafts are
quarantined regression examples pending Docker replay, never training-ready.
No automatic semantic grade or independent human judgment is generated. Source
queue and checked output reproduce exactly from the original run and review.
All 29 frozen implementation files and 1,680 prior file/marker hashes are unchanged.
No model, GPU or cloud call, training, reward change or promotion occurred.

Focused verification: 55 passed (30 new queue/check tests plus existing SFT
quality and reader-comparison review checks). Full suite with offline model
settings and the pinned Docker image: **2042 passed in 272.17 seconds**, no skips,
three existing SWIG warnings. Ruff and diff checks passed. Actual CLI preparation,
source review/check and deterministic artifact reproduction all succeeded.
