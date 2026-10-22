# Reviewed passage-reader pilot data — October 22, 2026

**Data ready for a small SFT pilot; no new model trained or evaluated.**
This adapts official QASPER questions, annotations and paper text (Dasigi et al.,
2021, CC BY 4.0); it is not an original benchmark or an official leaderboard
result. See [dataset attribution and provenance](../../benchmarks/envoybench/data/README.md).

The hypothesis is that paired evidence states can teach the existing worker when
to extract a supported fact and when to issue a specific scoped gap. The frozen
`cce82c2` runtime, prompt, tools, retrieval and decoding were not changed.
Source review is explicitly **implementing-assistant/model-assisted**, with
independent human review null. Exact quotes and successful execution alone do
not establish semantic support.

| Local preparation result | Actual |
| --- | ---: |
| Historical files inventoried by identifier | 213 |
| Conservatively excluded prior papers / questions | 956 / 1,370 |
| Source-review rejected question/paper pairs | 13 |
| Accepted training papers / paired trajectories | 12 / 24 |
| Separate development papers / paired episodes | 6 / 12 |
| Accepted papers overlapping the exclusion registry | 0 |
| Successful fixed-action Docker episodes | 36 / 36 |
| Actual source inspections / local policy turns | 72 / 108 |
| Train next-action targets, exact prefix/mask checks | 72 / 72 |
| Training input / supervised tokens | 225,496 / 3,018 |
| Longest tokenized action sequence / limit | 3,729 / 8,192 |
| New model requests / GPU activations / optimizer updates | 0 / 0 / 0 |

Every pair uses one question and two ordered, disjoint 1,600-character source
windows (3,200 characters per episode), with a shared distractor. The missing
condition changes one window. Original QASPER answerability remains sufficient;
only the **selected evidence scope** changes. The controller metadata does not
count as inspection. All approved targets perform two actual reads before a
supported `SUBMIT` or terminal `submit_gap`.

The scopes are annotation-assisted controlled conditions, not results from live
retrieval. Source review rejected 13 pairs before any Qwen outputs existed:
several supposedly missing scopes answered by paraphrase, others had ambiguous
positive labels. [Reasons](source-review-rejections.json) are retained. The four
local draft rounds remain under `out/passage-pilot-data-20261022/`; accepted IDs
were never selected or replaced using model outcomes. Full selected text was
reviewed for support, attribution, complete requested facts, numeric conditions,
and gap/query specificity. Review notes acknowledge partial evidence and qualify
answers when the excerpt does not establish an exhaustive claim.

Row balance is not token balance. Supervised tokens are 784 present-read, 1,028
present-terminal, 749 missing-read and 457 missing-terminal: approximately 60%
present and 40% missing. No loss weighting or optimizer update was performed.

Focused tests: **28 passed**. Full suite: **2,075 passed** in 113.11 seconds,
no skips, three existing SWIG deprecation warnings. Ruff and diff checks passed.
[Commands and counts](checks.json), [full output](full-suite.log),
[preservation](preservation.json): all 1,747 prior files and all 29 frozen runtime
files unchanged.

## Saved evidence

- [Cases, roles and exact windows](prepared/cases.json),
  [source snapshots](prepared/snapshots),
  [frozen exclusions](prepared/exclusions.json),
  [selection](prepared/selection.json).
- [Attributed source review and final actions](prepared/implementing-assistant-review.json).
  This is not independent human grading or a reward.
- [Replay summary with per-case/action results](replay/summary.json),
  [actual packets and policy payloads](replay/cases),
  [pinned code/image and artifact hashes](replay/manifest.json).
- [Training conversations](../../data/sft/passage-reader-pilot-v1/train.jsonl),
  [scoring-only development references](../../data/sft/passage-reader-pilot-v1/development-scoring-only.json),
  [token allocation, initialization pins and export manifest](../../data/sft/passage-reader-pilot-v1/manifest.json).
- [Preparation hardware/software](runtime.json),
  [bundle hash inventory](validation.json),
  [fresh inspection of all owned containers](fresh-cleanup-inspection.json).

The first [attempt](failed-attempt/summary.json) completed two cases, then the
frozen five-second Docker image-availability probe returned unavailable before
case three could act. The two owned containers were removed. Its old cleanup
`all_absent` field conflated completion count with cleanup, and its adapter raised
a missing-counter error. Commit `44f87b5` fixes only the new adapter's reporting:
owned-container absence and case completeness are separate. The failed artifacts
are preserved. Three subsequent image checks took 0.156/0.018/0.016 seconds; the
second full attempt passed all 36 episodes. The transient local probe failure is
not attributed to Qwen; its underlying Docker Desktop cause was not isolated.

All successful containers used image
`sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382`,
network `none`, and exact read-only source mounts. All 38 containers owned across
both attempts were separately inspected as absent. No host Python fallback,
credential read, cloud call or paid teacher endpoint was used.

## Reproduce locally

Strict re-export requires the original inventoried workspace artifacts, including
historical local `out/` files. The committed registry preserves identifier lists
and file hashes, not every historical experiment. The saved replay and source
bundle can be inspected without those older files; do not mistake archive
integrity checks for re-proving the historical exclusion inventory.

With the original workspace, cached pinned QASPER source/tokenizer and pinned
Docker image available, use new output directories:

```sh
PATH=/Users/jasonling/.local/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/replay_passage_pilot.py replay \
  --prepared examples/passage-reader-pilot-data/prepared \
  --review examples/passage-reader-pilot-data/prepared/implementing-assistant-review.json \
  --output out/passage-pilot-repeat

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/replay_passage_pilot.py export \
  --prepared examples/passage-reader-pilot-data/prepared \
  --review examples/passage-reader-pilot-data/prepared/implementing-assistant-review.json \
  --replay-dir out/passage-pilot-repeat \
  --output out/passage-pilot-repeat-export
```

The frozen exclusion registry predates this new dataset. A new inventory now
also treats this dataset's papers as exposed; do not overwrite the old registry
to reproduce this selection. The reserved `benchmarks/research_reader_v1` pack
was not opened. The inventory covers declared workspace history, not unknown
remote jobs or Qwen pretraining.

Fixed actions cannot measure how Qwen chooses tools or responds to changed
feedback. This work proves neither a fine-tuning gain nor the research product's
success. The [pilot proposal](../../docs/PASSAGE_READER_SFT_PILOT.md) still requires
one separately authorized, budgeted training run and a matched candidate/V5/base
development comparison, followed by fresh evaluation before any promotion.
Current paid-request and GPU-training allowance remains zero.

Prepared cases retain their original pending-review/false training flags as frozen
drafts. Acceptance lives in the separate attributed review; only the gated
export manifest records the 24 eligible new training trajectories. Historical
evaluation corrections were not relabeled as training data.
