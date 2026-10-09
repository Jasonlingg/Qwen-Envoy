# Qwen v5 training evidence

These records document the September 19, 2026 QASPER code-execution SFT run.
They are published for inspection, not instructions to start a new GPU job.
The evaluated adapter is **checkpoint-50**, not the final step-108 checkpoint.

| File | Origin |
| --- | --- |
| `run-manifest.json` | Byte-identical original training manifest: data hashes, model revision, LoRA settings, hardware, seed and training-script hash |
| `run-sft-v5.sh` | Byte-identical original launch script, including historical `/workspace` paths; not a portable launcher |
| `environment.txt` | Byte-identical original package/GPU version record |
| `trainer-state-step-108.json` | Byte-identical saved final-checkpoint trainer state, including intermediate training and validation losses |
| `training-summary.json` | Final metrics dictionary extracted from the original local training log; the log's hash and extraction method are recorded |

The original files were copied from
`out/research/qwen3-qasper-v5-sft-20260919/`. No training was rerun or results
regraded during packaging. `SHA256SUMS` pins the published records.

The manifest binds the public [training conversations](../../data/sft/qasper-v5/train.jsonl),
[validation conversations](../../data/sft/qasper-v5/val.jsonl), and
[data manifest](../../data/sft/qasper-v5/manifest.json). The full run reports
579.4858 seconds of training on one A40, excluding setup. Its recorded epoch-two
validation loss is lower than the other recorded validation points; that does
not establish downstream answer quality or prove checkpoint-50 is optimal.

## Reproducibility boundary

The original training manifest has a blank Git commit. Its recorded training
script hash differs from the current `scripts/train_sft.py`, and the exact
historical script is not included in this bundle. Package versions are
observations, not a full dependency lock. The log summary's source hash records
provenance but cannot independently reconstruct the unpublished full log.

Consequently, this is an **auditable observed recipe**, not a guarantee that a
fresh training run reproduces the weights bit for bit. The pinned adapter,
base-model revision, selection caveats and latest evaluation are documented in
the [model card](../../benchmarks/envoybench/MODEL_CARD.md).

Read the [technical case study](../../docs/TECHNICAL_CASE_STUDY.md) for the
training objective, environment and interpretation of the measured results.
