---
record_id: qwen-qasper-measured-result-report-2026-09-25
kind: result
effective_date: '2026-09-25'
status: reported_measurement
review_status: source_derived_unreviewed
origin: public_eval_report
source_repo_path: reports/qasper-scale-sft-2026-09-25.json
source_sha256: 1d9a9511950d2f78c8ed1d844775b738698b48198fa0d993c20b18a453c98cc4
---

# Qwen QASPER continuation measured result

The public report's identity-blind, model-assisted semantic review found **15/40** passes for the starting SFT checkpoint, **10/40** after epoch 1, and **4/40** after epoch 2 on the same reported question IDs. This result is linked to the reconstructed continuation strategy [[01-2026-09-25-qwen-continuation-strategy]]; the link identifies what to inspect, not a proven causal relationship.

**Answerability tradeoff (reported semantic pass counts):** on sufficient-evidence items, starting SFT / epoch 1 / epoch 2 passed 5 / 8 / 4; on insufficient-evidence items they passed 10 / 2 / 0. The public report supplies pass counts by subgroup but not subgroup denominators. More answerable passes at epoch 1 coincided with fewer insufficient-evidence passes; this does not by itself establish the cause of the tradeoff.

**Review:** identity-blind model-assisted review; not independent human review; reviewer model `claude-sonnet-5`. The review covered 40 questions and 120 candidate responses.

**Scope:** This is not a personal-vault product evaluation or a base-Qwen comparison. The starting SFT checkpoint is not base Qwen, and this QASPER result is not a personal Obsidian-vault evaluation. Base model: `Qwen/Qwen3-8B` at revision `b968826d9c46dd6066d109eabc6255188de91218`; starting adapter SHA-256 `70143ec6933494efe60cab726e792fcc182965fd13a8561453202b41bba9bfb4`.

**Source:** `reports/qasper-scale-sft-2026-09-25.json`; SHA-256 `1d9a9511950d2f78c8ed1d844775b738698b48198fa0d993c20b18a453c98cc4`. Benchmark SHA-256 `d5d4cff9a4a4ff711d022792c7b520c5b948231a1afbc684f1ccbbb405513a83` and corpus-content SHA-256 `6abb65821917c11d9cda4fa82415c31623a8566e1ad3b9bd74a09dd6df1743b2` identify the reported evaluation.
