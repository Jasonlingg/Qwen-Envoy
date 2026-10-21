# Two-reader serving diagnostic

The user asked to continue after the real experiment-mode walkthrough. Prepare
and verify the fixed-input diagnostic. During preparation, a GPU budget question
was pending; no paid activation was permitted until authorization arrived. Do not reuse any
consumed reader/research live-attempt marker or previous paid allowance.

Hypothesis: the existing single-engine queue worker can overlap two fixed Qwen
reader requests and reduce pair completion time compared with serial dispatch.
Expected signal: two measured rounds per mode, matched input hashes and output
counts/finish reasons, phase telemetry and observed engine overlap. The same
engine/runtime/cache settings apply to both modes. Two rounds are diagnostic,
not independent task trials or a tail-latency/productivity benchmark.

Decision rule: report both rounds and every failure. Missing telemetry, changed
identity/cache state, or differing outputs make the performance comparison
inconclusive. Fixed inputs cannot measure reactions to feedback, complete reader
tasks, research usefulness or fine-tuned Qwen improvement.

Limits: one A5000, an absolute 900-second activation window including loading,
ten model requests (two warmups plus eight measured), 7168 input and 1024 output
tokens reserved per dispatch (81920 total), no uncertain inference retry, and
read-back of shutdown. A local price/time guard is not a provider dollar cap.

Read-only preflight on October 21 found endpoint `q05pjrvguakieg` parked at min/max
0/0, zero workers and zero queued/in-progress jobs. It already has max engine
sequences 2, worker concurrency 2 and prefix caching false. There are no attached
volumes. Do not create new infrastructure or change these serving settings.

The pinned image corresponds to worker-vllm source
`b7212acc6d60d5ada997f453783c2e09df8e6e49`. Its generic route proxy decodes
non-streamed responses as JSON, so Prometheus `/metrics` cannot be obtained
through that proxy unchanged. Add an opt-in, fixed-operation telemetry hook;
ordinary worker startup must remain unchanged. The hook must enforce the two
input hashes, worker-local request allowance and one active telemetry window.
Preserve the upstream chat path, keep sampling bounded, and expose no arbitrary
code execution. The client must reserve requests durably and stop across worker
identity changes or uncertain transport outcomes.

## Prepared implementation

`18e3fd8` adds the optional worker hook, fixed-input client, preparation/guarded
runner and local regression checks. `9d94ef7` preserves returned telemetry/model
diagnostics before response validation can fail. The original worker bootstrap,
endpoint entrypoint/settings and prior live attempt allowances were not modified.

The [prepared-run guide](../../examples/two-reader-profile/README.md) records
the exact local plan, read-only endpoint state, execution limits, possible
comparison confounds and pending budget selection. The cached tokenizer
reproduced both historical input counts (1400 and 2043). Preparation/check made
zero model calls. After the final trace-preservation change, the complete suite
passed: 1867 tests in 109.08 seconds, with three existing SWIG deprecation warnings
and no skipped tests. Twelve new focused checks also passed.

Live telemetry compatibility remains unverified. The only cloud operations in
the preparation phase were read-only endpoint/worker/health and catalog queries;
the local tests and prepared JSON did not authorize activation.

## Authorized attempt outcome

After the user said “ok go,” one attempt used the smaller $0.50 allowance with
the original 900-second/ten-request limits. It stopped during startup identity
verification: one log read returned without a matching identity, then three
reads timed out. No model or telemetry requests were submitted. The model's
speed, concurrent fit, output quality and task usefulness remain unmeasured.

Shutdown was verified, the original entrypoint restored, and a fresh full
endpoint read matched the pre-attempt configuration. No worker or pending job
remained. The attempt marker is consumed and no paid retry was made. The
[attempt record](../../examples/two-reader-profile/outcome.json) preserves the
failure, authorization and shutdown evidence. Startup log contents were not
saved by the existing lifecycle helper; preserve those diagnostics and verify
log access before a separately authorized attempt. No implementation code was
changed in this attempt, and all eleven protected file hashes still match.
