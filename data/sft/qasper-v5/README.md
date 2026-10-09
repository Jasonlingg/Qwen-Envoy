# QASPER code-execution SFT data: v5

Prepared September 18, 2026. **44 assistant-reviewed conversations**, built from the
21-example v4 seed plus 23 accepted episodes from a fixed 32-episode review batch.
Every example is from QASPER's official training split and targets a different paper.
Qwen was not retrained in this pass.

| Split | Conversations | Answerable | Insufficient | Next-action targets |
|---|---:|---:|---:|---:|
| Train | 35 | 19 | 16 | 106 |
| Validation | 9 | 7 | 2 | 26 |

The student receives the question and actual tool observations, and predicts Python
actions or a `SUBMIT:` answer with document citations. Teacher-only labels and reference
answers are not student inputs. Use the existing per-action preprocessing in
`scripts/train_sft.py` to mask preceding context and supervise only the next action.

## Files

- `train.jsonl`, `val.jsonl`: complete conversations in the current student protocol.
- `candidates.jsonl`: all 44 accepted full trajectories, including repair provenance.
- `manifest.json`: source/review hashes, exact question IDs, paper-group split, corpus
  identity, seed, system prompt hash, and excluded evaluation benchmarks.
- `tokenization-check.json`: actual Qwen3-8B tokenization and inference-prefix checks.
- `review-template.json`: generated reference/template artifact; **not** the completed
  review. Completed decisions are in the review files linked below.

The split uses seed 42 and is disjoint by paper. The official-validation development
snapshot, consumed test snapshot, and earlier validation target papers are excluded.
This version repartitions the previously unused seed data; v4/v5 validation loss would
not be a same-split comparison. The v2 and v4 files were preserved.

## Review and limitations

Seven examples across the combined set are unchanged original teacher episodes; 37
contain explicit Codex repairs, with every revised action sequence executed anew.
Actual observations were never fabricated or manually rewritten. All accepted episodes
have exact replay records. Source/stopping review was performed by the editing assistant
with labels visible, so this is neither independent nor blind evaluation.

The expansion accepted 23/32 episodes (15 answerable, eight insufficient), missing its
declared target of 24 overall and ten per class. Nine questions were excluded for
malformed wording, ambiguous interpretation, reference conflict, or inadequate source
coverage. The accepted examples remain useful candidates for supervised training; the
failed gate is not a claim that the data-generation process is ready to scale unattended.

All 132 action examples passed the current preprocessing with the cached Qwen3-8B
tokenizer at revision `b968826d9c46dd6066d109eabc6255188de91218` and local Transformers
5.3.0. Longest sequence: 2,579 tokens. Longest supervised target: 177 tokens. Training
contains only 4,812 supervised tokens. These checks establish format compatibility,
not a trained model improvement or sufficient data volume; repeat preprocessing in
the pinned GPU environment before any future training run.

This set primarily teaches known-paper investigation, concise supported answers, and
bounded uncertainty. It does not establish cross-paper synthesis or personal-vault
usability. Short two-action episodes still execute a real evidence-retrieval action
before submitting. Nine validation conversations are too few for a broad quality claim.

No teacher API credits or GPU time were used for the v5 expansion. The v4 seed involved
previously recorded Claude API generation.

See the [experiment record](../../../docs/QWEN3_SFT_DATA_ITERATION.md),
[v4 review](../../research/qasper_sft_v4_review.json), and
[v5 review](../../research/qasper_sft_v5_review.json). Full source snapshots and replay
artifacts are local under `out/research/`; archive them with any future experiment.
