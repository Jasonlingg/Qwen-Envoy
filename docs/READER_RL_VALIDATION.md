# Reader RL local validation — October 22, 2026

Scope: validate a narrow reward specification and the requirements for a future
V2 training adapter. No paid inference, GPU activation, optimizer update, runtime
change or training-data export is authorized by this step. The existing 24-example
export remains on hold. This follows the [RL investigation](READER_RL_INVESTIGATION.md).

## Hypothesis and decision rule, before implementation

The old overlap objective rewards some unsupported answers. For a bounded
answerable-fact task, a finite set of source-reviewed complete answer variants,
paired with required supporting excerpts, should reject those exposed exploits.
The mechanism is reference matching against attributed review, not a general
semantic judge. Exact quote matching alone never establishes support.

Use the existing 150 exposed diagnostic records, without changing their labels
or bytes. Review the 14 answerable positive controls claim by claim; record any
disagreement rather than forcing the new scorer to accept an old label. Missing
scope/gap examples are explicitly outside this proposed training objective:
their usefulness cannot be graded by a parser. They must receive no task reward,
and must not enter relative-advantage groups for this objective. False gaps on
answerable tasks receive zero. Reference-string controls are not automatically
approved answer variants. Reviewed paraphrases may be added explicitly.

Expected signal: every answerable known-negative record scores zero; every
currently supported positive scores one; controls needing source-support repairs
are reported separately and their narrower repairs tested. The score must reject
changed relations, units, numbers, negation, extra unsupported clauses, uninspected
or irrelevant citations, mismatched source identity, and support lost to the
existing 1,200-character clipping. Compare against the old scorers per record.
Passing this exposed development audit is necessary, never sufficient for training.

The local preflight must stay blocked while independent reward-validation cases,
a reviewed eligible training pool, exact V2 Docker/raw-token rollout parity,
GPU update/save-reload checks, and an enforceable paid budget are missing. Record
the actual old trainer's generation-config fix separately from its incompatible
host-execution/prompt path; do not call all historical optimizer updates broken.
Do not import or launch that trainer to inspect it. A report must not imply it
can prevent a user directly invoking the historical training script.

Verification: focused reward/source-binding/preflight tests, saved-record
recomputation, existing optimizer/config regressions, then the full relevant
offline suite. Preserve frozen runtime, active rewards, old trainer, existing
artifacts, protected files, original user changes and the reserved question pack.
This is an assistant-authored development review; independent human review stays
null. A later experiment must lock separate reward-validation cases and model
evaluation papers before model outcomes, declare IDs, objective, hardware and
budget, and measure source-supported quality rather than training reward alone.
