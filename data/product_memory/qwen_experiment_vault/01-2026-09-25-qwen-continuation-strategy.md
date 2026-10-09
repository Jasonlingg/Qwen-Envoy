---
record_id: qwen-qasper-continuation-strategy-report-2026-09-25
kind: decision
effective_date: '2026-09-25'
status: retrospective_reconstruction
review_status: source_derived_unreviewed
origin: public_eval_report
source_repo_path: reports/qasper-scale-sft-2026-09-25.json
source_sha256: 1d9a9511950d2f78c8ed1d844775b738698b48198fa0d993c20b18a453c98cc4
---

# Qwen QASPER continuation strategy (retrospective reconstruction)

This note reconstructs the next-step strategy after the run and separately records the promotion rule stated in the public evaluation report. It is **not** a contemporaneous preregistration, a user-approved memory revision, or proof that thresholds were fixed before results. The effective date here is the report date; the source does not supply an independent timestamp for the rule.

**Inferred next-step strategy and experiment question:** Test one or two further SFT epochs on QASPER paper QA through persistent executable Python tools and ask whether those epochs improve the starting SFT checkpoint. The report compares `epoch_1` and `epoch_2` to `starting_sft` on the same 40 `validation` questions. The report-recorded gates include: semantic mean exceeds starting_sft; at least two more paired wins than losses; at least 28 of 40 semantic passes. Only the strategy and question are retrospective inferences, not a quoted original plan. The listed gates are report fields, not inferred from the reported rejection decision.

**Promotion rule recorded in the report:**
- semantic mean exceeds starting_sft
- at least two more paired wins than losses
- at least 28 of 40 semantic passes
- automatic mean reward exceeds starting_sft
- at least 39 of 40 protocol-valid submissions
- no more than starting_sft plus one false refusal
- no more than starting_sft plus one execution-error episode

The report sets `no_posthoc_threshold_changes: true`. That is a statement in the report, not independent verification of when the rule was established. A later revision could supersede the inferred next-step strategy; it would not invalidate this rule or rewrite the historical run.

**Source:** `reports/qasper-scale-sft-2026-09-25.json`; SHA-256 `1d9a9511950d2f78c8ed1d844775b738698b48198fa0d993c20b18a453c98cc4`. The report is the authority for these reconstructed fields.
