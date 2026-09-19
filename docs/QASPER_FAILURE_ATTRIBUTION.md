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

## Decision and next experiment

Do not adopt the recovery prompt and do not make top eight the global default yet. Top eight is a
useful candidate, but it did not solve the central yes/no interpretation failures and its context
cost is substantial.

Before training another adapter, run the pre-registered oracle-evidence comparison on the six
`evidence_surfaced__interpretation_or_completeness` cases. Give base and SFT the same gold evidence
windows while keeping decoding fixed:

- If SFT refuses or misreads evidence that base uses correctly, add targeted answerable
  trajectories and retrain.
- If both models fail, improve the answer-reasoning prompt or use a stronger policy model.
- If both models succeed, keep the weights and improve retrieval presentation, ideally with
  deduplicated or section-aware windows rather than a permanent top-eight expansion.
