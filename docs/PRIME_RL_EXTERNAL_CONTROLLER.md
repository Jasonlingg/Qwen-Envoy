# Native training with an external Docker controller

October 24, 2026. The user said “ok go” after the two-update integration / conditional
ten-update pilot explanation. This launch uses the existing **$5 estimated / two-hour
owned-resource lifetime** allowance. The numerical pilot protocol remains unchanged;
its earlier `paid_authorization: false` records its preparation state. The current
launch authorization and resource IDs are separate receipts under
`out/reader-prime-integration-20261024/`.

Runpod Pods are containers; its documented image-building workaround does not
provide the nested Docker execution needed by this sandbox. We therefore keep
model-authored Python inside the unchanged local Docker reader and run only trusted
native CUDA inference/scoring/training on the GPU. No host execution fallback.
[Runpod documentation](https://docs.runpod.io/tutorials/pods/build-docker-images).

The existing launcher now accepts an optional `--controller-plan-sha256` during
preparation. Its default remains same-host Docker. External mode skips only the
GPU-host Docker-image inspection and waits for a bound controller result after
loading each policy; it cannot start the trainer without that result. Both plans
pin the same settings, source/data/prompt/reward identities, question order and
implementation hashes. Their paths may differ between machines.

`src/training/prime_controller.py` provides the file handoff. The owned GPU launcher
publishes the policy hash/version, job nonce, contracts and unchanged deadline.
The controller checks its own prepared plan, uses the original native collector
through a loopback SSH forward, and retains its persistent request/token ledger
across updates. It returns a native batch, candidate and hashes of healthy original
untrainable episodes/cleanup records. GPU admission checks the job, controller,
policy, batch, training case and complete mixed group. Native sampled-token scoring
and checkpoint/optimizer gates still precede and follow each native update.

SSH copies are deployment operations; the handoff is not a new optimizer, learner,
reward, sandbox, or model protocol. Raw controller traces remain local. Native GPU
receipts and portable adapters are copied back before temporary-resource cleanup.
Large full optimizer checkpoints remain on the temporary volume during the run;
their retention after teardown is not promised by an inference adapter export.

`scripts/watch_reader_gpu.py` runs independently of the agent, training and SSH,
under macOS `caffeinate`. It validates lifetime/cost arithmetic and deletes only
the Pod and volume recorded from this launch's create responses, with fresh read
checks before/after deletion. It also supports an explicit early cleanup marker.
It does not discover or delete older account resources. Generation reservations
remain separately bounded by the native run plans.

The cutoff requires the local computer and provider control plane to remain
available. It is not a provider-enforced dollar guarantee. Transient deletion
failures are retried; billed receipts are reported separately from projections.
The new Pod/volume are temporary and should not remain billing after the check.

Local pre-launch validation: **136 native tests**, **2,202 full-suite tests**,
16 optional skips and three existing SWIG warnings. Ruff/diff checks pass. The
external handoff test uses authored replies through actual Docker/native batching;
it does not establish GPU updates or model quality. All 29 frozen runtime files,
source/references and numerical protocol remain unchanged. Current real-run results
will be recorded separately; no held-out references were opened during preparation.
