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

## Actual result

`6c84969` adds explicit model-arm selection and tests base identity through the
MCP/policy/queue/record path. `4ad24e3` freezes the matched inputs before activation.
No product prompt, source tools, verifier, gap contract, training or reward code
changed. First actual inputs matched V5 exactly apart from model alias; all twelve
requests and responses used `envoy-base-only`, and saved identities declare
`adapter_applied: false`. The worker still loaded both routes; no endpoint
image/bootstrap/env mutation was needed.

Base: QLoRA returned all requested facts with adequate citations at step four.
Its first two-read action was rejected, then it recovered with two single reads
and a final submission. AWQ remained incomplete: every response combined read/
search code and an Unanswerable submission. The unchanged policy cleaner selected
the submission line, so no AWQ source read executed. Four premature abstentions
were rejected; semantic AWQ reading was not assessed. The gap case read both
windows after a rejected two-read action, then abstained without a gap report
at its last step. It remained incomplete, invented no target measurement and
returned no useful partial findings. The overall rule was not met.

Under implementing-assistant source review, supported answerable cases were
base one / saved V5 two; gap completions were zero for both. Base had two action
and five submission rejections, versus V5's one submission rejection. Both had
zero Docker Python execution errors. This is a descriptive tiny development
comparison, not a validated general fine-tuning gain. V5 complied better with
the current action protocol; the base AWQ failure cannot establish weak domain
knowledge. Training can target repeatable protocol errors, but these observations
alone do not justify another job. Independent human review remains blank.

Base usage: 33,974 input + 672 output = 34,646 reported tokens across twelve calls,
98,304 reserved tokens. V5's saved usage was nine calls / 26,302 reported tokens.
All base jobs completed; no inference resubmissions or new failed/retried jobs.
Native startup verified identity/readiness with thirteen received snapshots.
Supervisor time before cleanup was 203.884 seconds; watchdog did not fire.
Fresh reads at 00:54:54 UTC on October 22 verified full configuration restored,
workers 0/0, no workers/pending jobs; all six owned Docker containers were absent.
Actual invoice cost is unknown. The base allowance is consumed, without another
paid attempt or new V5 calls. Eleven protected files and 507 prior artifacts
remained unchanged.

Focused checks: 46 passed in 50.30 seconds. Initial full run: 1928 passed, one
Docker startup failure before any scripted call, three warnings, 258.96 seconds.
A separate Docker rehearsal overlapped that run; exact cause is unproven.
Without code/timeout changes, four standalone Docker tests passed in 7.40 seconds,
then the standalone full suite passed 1929 tests in 152.10 seconds, no skips and
three existing SWIG warnings. Preserve both logs. [All results and source review](../../examples/passage-reader-base/README.md).
