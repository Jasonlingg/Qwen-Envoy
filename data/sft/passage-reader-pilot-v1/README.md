# Passage-reader pilot training export

24 annotation-assisted, implementing-assistant source-reviewed trajectories from
12 QASPER train papers. Each has two real Docker passage reads followed by a
supported answer or a specific scoped gap. This is data preparation, not a new
trained checkpoint or evidence of improved model performance.

`train.jsonl` is the only optimizer input. `development-scoring-only.json` holds
12 separate episodes on six different official validation papers and must never
be passed as training data or trainer validation-loss data. It is reserved for
the single proposed matched model comparison; it is not an untouched final test.

[Source review, replay receipts, limitations and attribution](../../../examples/passage-reader-pilot-data/README.md)
record the gates. [Manifest](manifest.json) binds both files, IDs, token checks,
base/V5 initialization and proposed settings. Independent human review is null.
No model inference, paid teacher call, GPU activation or training occurred.
