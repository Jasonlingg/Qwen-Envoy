# Base Qwen passage-reader comparison

User instruction: “can we try running base also”, after the V5 repaired-harness
rerun. One new bounded base-model attempt is authorized. No training or reward
change, new model, new cases or further V5 inference is included.

Hypothesis: the unchanged base Qwen3-8B can recover some or all of the same
source-supported facts under the repaired passage harness. Expected signal:
per-case supported facts, sufficient submitted citations, action/inspection
errors and gap completion compared descriptively with the saved V5 rerun.
Do not change the awkward gap contract before this comparison.

Decision rule: use the same two answerable cases and one missing-measurement
case, source windows, effective prompt bytes, temperature 0, top-p 1, seed 42,
thinking disabled, 1024 output tokens/action and four actions/case. Judge each
answer against its own evidence and retain incomplete reports. Both answerable
cases must cover the requested facts with adequate citations; the gap must
complete the required procedure without fabricating the target measurement and
return useful partial findings. Independent human review stays blank. This tiny
exposed-development sequential comparison cannot establish model improvement,
statistical superiority, discovery ability or training value in general.

Use the already configured `envoy-base-only` route on the pinned same base
revision. The existing worker also loads V5, but base requests must use and
receive the base alias, with `adapter_applied: false` in the research record.
No worker image/bootstrap/env change is needed. All model-authored Python stays
inside network-disabled Docker; source-inspection/citation gates stay identical.

Limits: one existing A5000, 900 seconds including startup, twelve aggregate
requests, 98,304 reserved tokens and four requests per case; no inference
resubmissions. Refresh price and parked/empty state before activation. The
established $0.50 local allowance uses time/request/token guards, not a provider
invoice cap. Consume only `out/passage-reader-three-case-BASE-1`, preserve every
prior artifact and marker, use native startup identity/readiness verification,
restore the full original configuration and independently verify shutdown.

Preparation must verify each first payload matches the saved V5 repaired-rerun
payload after changing only `model` to `envoy-base-only`. Save checkpoint,
corpus/prompt/source hashes, decoding, hardware, per-action results and claim
review. Compare factual quality separately from execution/protocol compliance.
