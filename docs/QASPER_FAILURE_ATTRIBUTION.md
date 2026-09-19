# QASPER failure attribution

**Run date:** September 19, 2026

**Status:** development-set diagnosis; not a held-out model result

## Question

The Qwen3-8B SFT checkpoint improved tool execution but still failed or partially answered 15 of
20 answerable questions in the 40-question QASPER development evaluation. The diagnostic question
was whether those errors came from the SFT weights, the runtime prompt, or the retrieval harness.

The end-to-end score cannot answer that question because all three components determine the
trajectory. This audit therefore intervenes on retrieval while holding the saved model trajectory
fixed.

## Hypothesis and decision rule

Hypothesis: the failures mix three causes: relevant evidence never reaches the model, the model
chooses weak queries or stops early, and the model misreads evidence that is already visible.

For each non-passing answerable response, the audit locates QASPER's gold evidence in the frozen
paper and measures whether it overlaps:

1. the windows actually shown to the model;
2. the top eight results for the model's original queries;
3. wider context around the original top-three results; and
4. the top-three results for diagnostic oracle queries made from the gold answer/evidence.

The oracle query uses gold information and is only a diagnostic. It must never be exposed to an
evaluated policy.

Decision rule:

- Evidence already shown: interpretation, completeness, or stopping error.
- Same query succeeds at greater depth: harness ranking-depth error.
- Only the oracle query succeeds: query-planning or early-stopping error.
- Oracle query fails: retrieval-backend or evidence-alignment error.

## Result

| Attribution | Count | Meaning |
|---|---:|---|
| Evidence already surfaced | 6 | Qwen overlooked, misread, or incompletely used visible evidence. |
| Original query succeeds at top eight | 4 | The three-window default hid relevant evidence below repetitive or stronger lexical hits. |
| Only an oracle query succeeds | 5 | The existing retriever can find the evidence, but Qwen's query or stopping decision prevents it. |
| Backend cannot retrieve gold evidence | 0 | No audited failure currently justifies replacing the retrieval engine. |

The strongest examples are behaviorally distinct:

- The tweet-count question returned a window containing both `19,300` and `2500`, but Qwen said
  the total was not specified.
- The English-data question returned the passage naming three English datasets, but Qwen still
  refused to answer Yes.
- The Europarl/MultiUN question used one generic query, received an abstract-level result, and
  stopped. A targeted query retrieves the gold evidence with the current backend.
- Several list questions surface their missing evidence when the existing query's depth increases
  from three to eight windows.

This established that the next intervention should combine a prompt ablation with a small
retrieval-depth ablation. It did not establish whether the SFT weights themselves were
over-calibrated toward refusal; an oracle-evidence model run is still required for that claim.

## Two-by-two ablation result

The four conditions ran on September 19, 2026 with Qwen3-8B, the QASPER v5 SFT
`checkpoint-50` adapter, deterministic decoding, seed 42, and one RTX 4090. Each condition used
the same 25-question diagnostic slice: the 15 non-passing answerable cases above, five passing
answerable controls, and five passing insufficient-evidence controls.

| Condition | Outcome reward | Answer F1 | Semantic pass / partial / fail | Previously failing cases rescued | Mean steps | Mean observation chars | Episodes with an exact repeated action |
|---|---:|---:|---:|---:|---:|---:|---:|
| Original prompt, top 3 | 0.266 | 0.108 | 10 / 4 / 11 | 0 / 15 | 2.68 | 2,642 | 1 |
| Recovery prompt, top 3 | 0.273 | 0.116 | 9 / 2 / 14 | 0 / 15 | 3.24 | 3,353 | 7 |
| Original prompt, top 8 | 0.314 | 0.167 | **12 / 3 / 10** | **2 / 15** | 2.68 | 6,732 | 2 |
| Recovery prompt, top 8 | **0.332** | **0.190** | 11 / 5 / 9 | 2 / 15 | 3.48 | 10,254 | 8 |

Semantic review used the repository's pass/partial/fail rubric. It was an assistant self-review
after system identities were known, not an independent blind review. The diagnostic slice was
selected after inspecting prior failures, so these numbers diagnose behavior and are not held-out
performance estimates.

The original prompt with top-eight retrieval is the clean winner. It turned two previously
non-passing answers into passes, preserved all ten passing controls, and did not increase the
number of tool calls. The cost is 2.5 times as much observation text per episode. The combined
condition has the highest token-overlap reward, but it has one fewer semantic pass and almost four
times the control condition's observation text.

The recovery prompt should not be adopted. Its instruction to try distinct queries increased tool
actions from 42 to 56 at top three and from 42 to 62 at top eight, while exact repeated-action
episodes rose from 1 to 7 and from 2 to 8 respectively. Qwen often repeated the same query despite
the explicit instruction. This is a concrete example of why reward alone cannot select the
production configuration: the combined run has the best automatic reward but not the best
semantic pass rate.

## Oracle-evidence result

The follow-up oracle test ran on September 19, 2026. It replaced each paper with the QASPER gold
evidence excerpts and forced the first action to read that document. Base Qwen3-8B and the v5 SFT
`checkpoint-50` adapter therefore received exactly the same answer-bearing text, prompt, decoding,
and five-step budget. The run used seed 42 and one RTX 4090. All 12 episodes submitted after the
forced read, in two steps.

One of the six fixtures was invalid: the reference names ten datasets, but the converted
`grader_notes` contained evidence for only the first four. The code-execution converter had
silently capped these notes at five paragraphs. That cap is now removed, and the affected case is
excluded from the semantic comparison rather than counted against either model.

| Policy | Pass | Partial | Fail | Excluded |
|---|---:|---:|---:|---:|
| Base Qwen3-8B | 2 | 2 | 1 | 1 |
| Qwen3-8B + SFT | **4** | **1** | **0** | 1 |

The two decisive SFT wins were concrete. For the tweet-count question, base returned only 19,300;
SFT included the additional 2,500 tweets and the correct 21,800 total. For the English-only
question, base answered `No` while SFT correctly said the paper reported only English data. Both
models remained incomplete on the classifier-list question, returning only the best SVM setup
instead of all four evaluated classifier families.

The automatic metric gives the opposite ranking: base outcome reward is 0.504 and SFT is 0.411
over all six cases. This is a scoring artifact. Exact token F1 awards base a perfect score for the
one-word answer `No`, but gives SFT zero for the semantically correct sentence “The paper reports
results only on English data” when the reference is `Yes`. These oracle results must therefore be
read from the answers, not selected by the MuSiQue-oriented token-overlap reward.

This is a small, selected development diagnostic, not a held-out performance estimate. Its causal
signal is still useful: SFT does not become worse when retrieval is removed, and on these cases it
uses supplied evidence better than base. The remaining end-to-end failures should be addressed in
retrieval and evidence presentation before generating another SFT batch.

## Passage-presentation follow-up

The next experiment changed `search_within()` rather than the model. The original implementation
ranked overlapping 500-character windows independently, so near-duplicate passages could consume
most of the three result slots. The hypothesis was that merging or diversifying those windows
would expose more answer-bearing text without paying the full raw top-eight context cost.

Two merged-passage variants were run end to end on the same checkpoint and 25-question diagnostic:

| Condition | Outcome reward | Semantic pass / partial / fail | Target cases rescued | Controls retained | Mean steps |
|---|---:|---:|---:|---:|---:|
| Original raw top 3 | 0.266 | 10 / 4 / 11 | 0 / 15 | 10 / 10 | 2.68 |
| Aggressive merged top 3 | **0.330** | 11 / 5 / 9 | 2 / 15 | 9 / 10 | **2.56** |
| Rank-bounded merged top 3 | 0.288 | 10 / 4 / 11 | 0 / 15 | 10 / 10 | 2.72 |

Neither merged variant passed the pre-registered acceptance rule. The aggressive version rescued
two failures but flipped a correct control answer about whether hashtag prediction was an
established task. The conservative version kept every control correct but rescued no target case.
The automatic reward would have selected the aggressive version, again demonstrating why semantic
review and controls are required for harness changes.

A third mode, `ranked_diverse`, is implemented but remains experimental. It preserves the original
top-three windows byte for byte, then fills three additional slots only with non-overlapping
windows. On the saved queries from the 15 failures, this raised mean gold-evidence coverage from
0.330 to 0.617. It returned 4,356 characters on average versus 5,833 for raw top eight, while
exposing almost the same amount of unique text (3,103 versus 3,156 characters). Its end-to-end GPU
run is still pending, so it is not the default.

These figures are assistant-reviewed development diagnostics, not held-out results. The aggressive
and conservative transcripts and disclosed reviews are stored under
`out/research/qasper-search-within-dedupe-v1-gpu/` and
`out/research/qasper-search-within-dedupe-v2-gpu/` respectively.

## Reproduction

```bash
python scripts/audit_qasper_failure_attribution.py \
  --benchmark out/research/qasper-code-dev-v2/benchmark.json \
  --run out/research/qwen3-qasper-v5-sft-20260919/eval40/sft.json \
  --review out/research/qwen3-qasper-v5-sft-20260919/eval40/blind-review/review.json \
  --blind-key out/research/qwen3-qasper-v5-sft-20260919/eval40/blind-review/blind-key.json \
  --corpus out/research/qasper-code-dev-v2/corpus \
  --output out/research/qwen3-qasper-v5-sft-20260919/eval40/failure-attribution.json
```

## Ablation command

Use the same checkpoint, questions, decoding, hardware, and seed for a two-by-two development
ablation:

| Condition | Prompt | `search_within` default |
|---|---|---:|
| Control | Original | 3 |
| Prompt only | Failure-recovery suffix | 3 |
| Harness only | Original | 8 |
| Combined | Failure-recovery suffix | 8 |

The prompt suffix is `data/prompts/qasper_failure_recovery_v1.txt`. `scripts/run_eval.py` now
accepts `--system-prompt-suffix` and `--search-within-top-k`, and records both settings in the run
manifest.

The frozen diagnostic slice is `data/research/qasper_failure_ablation_v1.json`: all 15 non-passing
answerable questions, five passing answerable controls, and five correctly refused insufficient
controls. Run the four conditions with:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-8B \
CHECKPOINT_PATH=/path/to/checkpoint-50 \
./scripts/run_qasper_failure_ablation.sh
```

The saved transcripts, manifests, GPU log, and disclosed semantic self-review are under
`out/research/qasper-failure-ablation-v1-gpu/`.

Run the offline passage-presentation comparison with:

```bash
python scripts/compare_search_within_modes.py
```

Run the pending `ranked_diverse` behavioral condition with:

```bash
BASE_MODEL_PATH=/path/to/Qwen3-8B \
CHECKPOINT_PATH=/path/to/checkpoint-50 \
./scripts/run_qasper_dedupe_ablation.sh
```

## Decision and next experiment

Do not adopt the recovery prompt or either merged-passage mode, and do not start another SFT run
yet. Raw top eight remains the only end-to-end condition that rescued two target cases while
preserving all ten controls, but it costs 2.5 times the original observation text. The pending
`ranked_diverse` top-six condition is the bounded attempt to retain that coverage more cheaply.

Accept `ranked_diverse` only if its end-to-end run rescues at least the same two target cases,
preserves all ten controls, and does not materially increase tool loops. If it passes, make that
mode the inference default and rerun the locked 40-question base/SFT comparison. If it fails, keep
raw retrieval and move the next experiment to query planning rather than more passage formatting
or more SFT data.
