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

## Implemented local experiment

The candidate lives separately in `src/eval/reader_rl_validation.py`; neither
active reward nor historical training code was changed. It matches a complete
answer against explicitly reviewed variants and checks that every claim's
required reference excerpt appears in the inspected, clipped submitted evidence.
Extra unrelated citations fail. References bind the entire question, source and
scope. Only NFC/whitespace are normalized: case is preserved because `MB`/`Mb`
and `mW`/`MW` differ. A genuine shorter supporting excerpt or a correct unlisted
paraphrase can still score zero. This deliberately restrictive objective is not
a general semantic-support evaluator or an automatic label generator.

The [saved keys](../examples/reader-rl-validation/reviewed-keys.json) contain
14 source-reviewed cards, 20 claim/support bindings and 28 complete variants.
The implementing assistant authored this review with sources visible; independent
human review is null. Twelve cards are from the exposed pilot training-paper
diagnostics, two from already exposed evaluation regressions. All are marked
diagnostic-only, ineligible for training. Gap quality has no reward in this
prototype. `None` means exclude that task from its RL objective, never turn it
into a zero-reward failed rollout.

One old positive control needs narrower wording: the cited excerpt reports
perplexity on test sets, but does not establish the claimed **independence** of
those sets. Its original record remains unchanged and scores zero; two narrower
reviewed variants pass. This is a support insufficiency in the cited excerpt,
not a finding that independent sets are absent elsewhere in the paper.

## Results and interpretation

All 150 old records remain intact and are reported individually in the
[saved results](../examples/reader-rl-validation/results.json).

| Diagnostic | Actual result |
| --- | ---: |
| Answerable records scored | 100 |
| Known unacceptable answerable records getting positive reward | **0 / 74** |
| Original answerable positive controls accepted | **13 / 14** |
| Explicitly reviewed complete variants accepted | **28 / 28** |
| Missing-scope records excluded from the objective | **50** |
| New Qwen inference / GPU activations / Qwen updates | **0 / 0 / 0** |

On the same 74 answerable negatives, the old legacy, scoped-overlap and
scoped-exact scorers give positive reward to **50, 38 and 12**, respectively.
The 12 original reference-string controls also receive zero from this candidate:
their wording was not automatically promoted into the reviewed variant list.
These are not new negative semantic labels. This highlights the false-negative
risk of a narrow allowlist, not superiority on an independent distribution.

The exposed development gate passes. That outcome is partly by construction:
complete reviewed answer matching blocks all unlisted outputs. It does not
establish reward recall, semantic-judge calibration, useful exploration variance,
training-data sufficiency or model improvement. Separate source-reviewed reward
validation must include unseen acceptable paraphrases as well as bad answers;
passing only authored exploits is inadequate.

## Training readiness and next executable work

The [preflight](../examples/reader-rl-validation/preflight.json) records the pinned
V5 identity, source revision, effective prompt, Docker image, all 29 runtime
hashes, V1 preamble hash and original data hold. Its paid request/token/time/dollar
budget is zero. It distinguishes the already-present generation-defaults fix
from the incompatible host-execution/old-prompt path. The existing optimizer and
generation-config regression tests pass; this does not reproduce a real current
V5 GPU update or establish why the historical confirmation failed.

`scripts/validate_reader_rl.py` is a **report-only** local entry point. It exits
2 even when the exposed diagnostic gate passes, because training readiness is
blocked. It never launches a trainer, and cannot stop someone directly invoking
the historical trainer. Do not interpret its report as a hardened launch guard.

Remaining work before any budgeted rollout: agree a disclosed training-only
data/reuse policy and retain evaluation-paper exclusions; lock separately
reviewed reward-validation cases; then implement the exact V2 Docker-only
raw-token rollout adapter. Prove actual prefix/token/log-prob/loss-mask parity,
complete-group handling and infrastructure-error separation. Only then freeze
a bounded sampling probe and require valid trajectories plus useful within-group
reward variation before an update. A later GPU test must verify policy change,
unchanged reference, finite gradients and save/reload identity. Neither a new
optimizer choice nor this allowlist resolves those missing checks.

## Verification and preservation

- Focused reward/audit/generation/optimizer checks: **60 passed**. CPU toy-model
  optimizer checks are not current-checkpoint GPU training evidence.
- Final relevant offline suite: **2,116 passed**, zero skips, three existing SWIG
  warnings, 109.23 seconds. It includes local pinned-Docker fixtures. An earlier
  full run before the unit-case correction also passed 2,116 in 174.59 seconds.
- Saved 150-row scores, summaries and all 28 reviewed-variant checks reproduce
  exactly; input/output/source bindings verify. The CLI returns expected exit 2.
- The first saved focused attempt failed because `platform.platform()` invoked
  `uname -p` under a no-subprocess test. Hardware reporting now uses `os.uname()`;
  the failure and later green logs are retained. It was a local reporting bug,
  unrelated to historical Qwen training or paid infrastructure.
- Ruff and diff checks pass. All 29 frozen runtime files and V1 preamble bytes
  remain unchanged. Of 2,041 pre-existing recorded files, only this pass's
  intentional `CODEX_HANDOFF.md` update changed; 2,040 are unchanged, including
  original user modifications, old audit/replay/data artifacts and protected
  training/reward files. No reserved question pack was inspected.

The [bundle](../examples/reader-rl-validation/README.md) contains raw logs,
per-record results, source bindings and preservation/recomputation receipts.
