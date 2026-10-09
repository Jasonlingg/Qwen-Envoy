# Qwen Envoy v5 — recorded training recipe

This adapter teaches Qwen3-8B to investigate papers with Python document tools
and submit answers with source spans. It is an experimental worker, not a
validated autonomous research assistant. The Studio training page is `/model`;
its machine-readable source is `model_card.json`.

## Checkpoint and data

- Base: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`.
- Adapter: `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5`, revision
  `27b912a863ff914ad45baa03624d8911dc1e17fb`.
- Subfolder: `artifacts/full/checkpoint-50`.
- Adapter SHA-256: `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
- Training: 35 QASPER conversations, 106 next-action targets, 4,812 supervised tokens.
- Validation: 9 paper-disjoint conversations, 26 next-action targets.
- The data contain recorded teacher trajectories and assistant-reviewed repairs
  re-executed against real source observations. Review was not independent or blind.
  See the [dataset card](../../data/sft/qasper-v5/README.md).

## Observed recipe

The September 19 full-run manifest records a 4-bit base with LoRA adapters
(QLoRA), rank 4, alpha 8, dropout 0.05. Targets are `q_proj`, `k_proj`, `v_proj`,
`o_proj`, `gate_proj`, `up_proj`, and `down_proj`. The objective supervises the
next Python action or submission while masking preceding context.

Learning rate was 0.0002, microbatch 1, accumulation 4, sequence limit 8,192,
and seed 42. The full run completed 4 epochs / 108 optimizer steps in 579.49
seconds on an NVIDIA A40, excluding setup. The displayed adapter is step 50,
selected near the epoch-two validation-loss minimum and checked on development
questions; it is not the final checkpoint. Nine validation conversations provide
limited evidence for checkpoint selection.

The original manifest is local at
`out/research/qwen3-qasper-v5-sft-20260919/artifacts/full/run-manifest.json`.
Its training-script SHA-256 is recorded, but its git commit is blank and the
current script differs. This is an observed recipe, not a guarantee that the
current working tree reproduces the original weights exactly. The archived
`run-sft-v5.sh` records the original launch arguments. No new training was run
to prepare this card.

## Latest comparison

On the [September 30 comparison](COMPARISON_2026_09_30.md), error-affected
episodes decreased from 22/40 to 1/40. Provisional supported answers on
answerable questions stayed at 3/20 for both models. The aggregate pass change
from 5/40 to 15/40 is driven by correct refusals, not improved substantive
answering. No promotion is claimed. The shared 8,192-token serving cap contributed
to 11 base failures, and semantic grades/references lack independent validation.

Uploaded-paper runs use a separate public-paper product prompt, have no gold
answers, and cannot be added to these benchmark scores. Quote matching confirms
source integrity, not semantic support. Upstream training exposure is unknown.

## Inspect and run

Use [the reader setup](PAPER_READER.md) for the live workflow. The model card
includes the saved-comparison command, requiring no inference. The optional
[prefix-cache experiment](INFERENCE_ABLATION.md) remains prepared, not measured.
Historical base timing coverage is incomplete; no full cost or latency win is
inferred from the missing measurements.
