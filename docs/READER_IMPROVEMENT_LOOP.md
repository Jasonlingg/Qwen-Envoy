# Reader improvement loop

This local workflow turns saved passage-reader traces into a source-bound review
queue and final-action correction drafts. It is the first part of an improvement
loop, not a self-training system. The tested harness baseline remains `cce82c2`.
It makes no model calls and does not start a GPU, execute Python from a model,
change rewards, train a checkpoint or promote one.

```
Saved run → review queue → attributed source review → quarantined correction draft
                                                        ↓
                                            fixed-action Docker regression replay
                                                        ↓ future work
                                            separate reviewed training data
                                            → training → candidate evaluation → promotion
```

## Use it

From the repository root, choose a new output directory:

```bash
python scripts/reader_improvement.py prepare \
  --run examples/qasper-hybrid-v5 \
  --output out/my-reader-queue
```

Open `queue.json` alongside `review-template.json`. The queue retains all cases,
including successful controls: exact source text, original actions/observations,
the actual final model request, mechanical signals and reproducibility metadata.
The frozen plan includes checkpoint/revision, source revision, system prompt and
implementation hashes, decoding, seed, sandbox image and limits. The runtime
commit and the lack of an enforced remote inference seed remain recorded.

Copy the template to a separate review file. Set `reviewer_kind` to `human` or
`model_assisted`, give the actual reviewer identity and review time, and complete
every row. Choose a verdict (`pass`, `partial`, `fail`), a failure category and
specific notes. Successful controls use category `none` and no correction.
Automated signals never assign semantic verdicts. Human attribution is
self-reported; the program never asserts independent human review.

For a failure, optionally supply a `correction` using the template's format:

- `replacement_final_action`: one `SUBMIT: ... CITATIONS: [...] EVIDENCE: [...]`
  line or a literal standalone `submit_gap(missing=..., follow_up_query=...)`.
- `rationale`: explain why the correction addresses the diagnosed failure.
- `source_support_reviewed`: explicit reviewer attestation, not an automatic grade.
- `supporting_spans`: exact `{doc_id, start, end, quote}` objects from source text
  actually inspected before the original final action. For a gap, these show the
  inspected scope; they do not prove absence in the full paper or literature.

Check the review against its original saved run:

```bash
python scripts/reader_improvement.py check \
  --run examples/qasper-hybrid-v5 \
  --queue out/my-reader-queue \
  --review out/my-completed-reader-review.json \
  --output out/my-reader-correction-drafts
```

The checker rebuilds the queue from the hash-bound original bundle and rejects
changed records, missing cases/notes, incomplete attribution, invented quotes,
uninspected spans, oversized spans and malformed terminal corrections. Drafts
retain bindings to the original multi-turn prefix and actual final request.
The original observations are never rewritten. Output directories cannot be
overwritten or placed inside the frozen source bundle.

## What the checks mean

`correction-drafts.json` is deliberately not an SFT/chat training file. Every
draft has `training_ready: false`, `training_eligible: false`, regression-only
usage and `docker_replay: required_not_performed`. Evaluation questions and papers
must stay out of training, even after someone reviews their corrected actions.
The current adapter enforces regression-only usage for every imported record.
A future training adapter needs a separate source set and paper/question exclusion
checks; changing a flag in this queue is rejected when it is rebuilt.

Exact source matching proves provenance. The reviewer must still assess factual
support, completeness, attribution, numeric conditions and useful scoped gaps.
Checking the review does not establish those judgments automatically. In
particular, the frozen runtime still accepts some cited prose non-answers as
`evidence_found` and clips citations to 1,200 characters; this workflow does not
silently fix either baseline behavior.

Only saved, complete QASPER passage-smoke bundles with a bound plan, ledger,
source snapshots and one raw request per action are currently supported. No
automatic runtime collection, arbitrary trace adapter, correction generator,
training export or model deployment is included.
Earlier tool errors can be diagnosed, but repairing them requires new trajectories
with actual observations. Never append invented successful observations.

## Next increments and decision rule

Reviewed final-action corrections can now be replayed in the existing Docker
boundary as **regression demonstrations**:

```bash
python scripts/replay_reader_corrections.py \
  --run examples/qasper-hybrid-v5 \
  --queue examples/reader-improvement-loop \
  --review examples/reader-improvement-loop/implementing-assistant-review.json \
  --drafts examples/reader-improvement-loop/correction-drafts.json \
  --output out/my-reader-correction-replay
```

The exact pinned Docker image must already exist locally. There is no LocalREPL
fallback, model/tokenizer download, credential lookup or inference request. First
all three original trajectories must reproduce, comparing the entire packet
except its fresh run UUID and every generated request against the historical
payload. Only then are the two reviewed final-action replacements allowed.
Original source-inspection actions and actual observations must remain identical.
The runner records per-action/input receipts, sandbox configuration and verified
removal of owned containers. Input bundles and output directories are immutable.

A fixed reply does not show how Qwen would choose actions or react to new
feedback. The replay cannot measure model quality, a post-training gain or
generalization. Passing replay does not change the earlier draft files or
their training eligibility. See the [saved replay](../examples/reader-correction-replay/README.md).

Next collect a separate, reviewed training set
with no overlap in questions or papers with evaluation. Train only under a
separate scoped plan with named questions, reviewed outputs and a budget.

Evaluate candidate and current V5 using the same frozen tools, prompts, source
snapshots, decoding and budgets. Set concrete thresholds before that experiment:
supported complete answers and useful gaps must improve without unacceptable
execution, latency or token-cost regressions. A trained-versus-base comparison
is also required for a fine-tuning improvement claim. Never promote based only
on valid code, quoted spans or the model's own grades.

The [saved local example](../examples/reader-improvement-loop/README.md) demonstrates
the queue/review/check path. It does not demonstrate an improved Qwen checkpoint.
