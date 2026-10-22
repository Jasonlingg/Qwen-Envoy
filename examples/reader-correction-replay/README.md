# Actual Docker replay of reviewed reader corrections

This is an **offline fixed-action regression replay**, not a new Qwen inference
run or a post-training result. The actual CLI run used the frozen `cce82c2`
runtime implementation, source snapshots, prompts, decoding payloads and Docker
image. Scope `2674209`, implementation `dc350a8`. No model/tokenizer was loaded,
credential read, cloud resource activated or external model request sent.

All three [original hybrid smoke](../qasper-hybrid-v5/README.md) episodes replayed
first. Each entire packet matched the saved original except its newly generated
run UUID. Every generated policy payload matched the recorded model input.
Only after that gate passed did the runner replace the final actions in the two
[source-reviewed drafts](../reader-improvement-loop/correction-drafts.json).

| Episode | Replay result | What it establishes |
| --- | --- | --- |
| Original humor | Exact original reproduced, including its incorrect refusal as `evidence_found` | The old failure remains reproducible; no semantic repair is hidden in the baseline. |
| Original translation | Exact original reproduced, including LSTM/attention/dropout answer | Passing control is retained. |
| Original bias | Exact original reproduced, including prose uncertainty as `evidence_found` | The old gap-classification limitation remains reproducible. |
| Corrected humor | `evidence_found`, reviewed answer with exact source span 2400..3050 | Reviewer-authored feature extraction fits the existing submission interface. |
| Corrected bias | `no_evidence`, terminal gap and specific follow-up query retained | Reviewer-authored structured gap fits the existing interface after the two reads. No search was executed. |

**Five episodes, 15 matched policy payloads, 10 source-inspection steps.** All
source-inspection actions and observations before either corrected final action
are identical to the originals. There were no replay failures. The actual output
is in `summary.json`; each episode has its packet, verified evidence/gap result,
per-action receipt and request-hash receipt under `originals/` or `corrections/`.

The runner uses the real policy formatter, investigator, document tools and
packet verifier directly. This does not test fresh online discovery, inference,
MCP network transport, multi-paper synthesis or training. Raw packets retain
historical checkpoint identity and the core's `model_requests_attempted` counter;
here those attempts are local scripted policy turns, not API inference. Historical
responses carry historical usage only; corrected replies carry no invented usage.

The five owned containers used image
`sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382`,
network mode `none` and exact read-only corpus mounts. Per-episode sandbox receipts
record the configuration. `cleanup.json` and a separate fresh inspection in
`fresh-container-check.json` confirm all five containers are absent. Only owned
containers were inspected/removed; other resources were untouched.

These are the same already exposed QASPER development cases: AllenAI QASPER,
Dasigi et al. (2021), CC BY 4.0, revision
`13b496d2a5359329b110e3419628de3cf791843b`. They are not an original question
dataset, held-out validation or an official QASPER result. Source review remains
explicitly implementing-assistant review, not independent human review.

Passing fixed-action replay **cannot measure how Qwen would select actions or
respond to new feedback**. It proves interface/provenance compatibility, not
semantic correctness or model improvement. Every correction remains excluded
from training, with `training_eligible: false` and `training_ready: false`.
The earlier pending draft files stay unchanged; this is a separate replay receipt.
Next work needs separate reviewed training data and a predeclared comparison.

Reproduce with a fresh output directory and the pinned local Docker image:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/replay_reader_corrections.py \
  --run examples/qasper-hybrid-v5 \
  --queue examples/reader-improvement-loop \
  --review examples/reader-improvement-loop/implementing-assistant-review.json \
  --drafts examples/reader-improvement-loop/correction-drafts.json \
  --output out/my-reader-replay
```

Packet UUIDs, container IDs and sandbox receipt filenames change on a fresh run.
Original comparisons explicitly exclude only the UUID. The semantic review is
not regenerated or relabeled. See the [workflow guide](../../docs/READER_IMPROVEMENT_LOOP.md).

Verification: **39 focused tests passed; full suite 2051 passed in 137.22 seconds**,
no skips and three existing SWIG warnings. Ruff/diff passed. Preservation checks
confirm 29 frozen implementation files and 1,692 prior files/markers unchanged.
`validation.json` binds this actual output; `full-tests.txt` is the standalone
suite output. No training-ready example or model improvement is established.
