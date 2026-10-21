# Two-reader serving diagnostic

The user asked to continue after the real experiment-mode walkthrough. Prepare
and verify the fixed-input diagnostic. A GPU budget question is pending; no
paid activation is permitted until its answer is available. Do not reuse any
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
