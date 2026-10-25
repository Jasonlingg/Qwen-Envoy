# Move the reader controller off the laptop

October 25, 2026. Deployment proposal after the interrupted three-update pilot.
No new machine, paid call, training run or commit was created for this preparation.
The user's request authorizes preparing the move; it does not authorize another
paid training retry. Hosting selection, fresh prices and a combined allowance
must precede provisioning.

## Recommended small deployment

```text
Laptop: launch/check status/download results; may disconnect after acceptance
                            |
                 Linux CPU VM with Docker
                 - existing controller and native renderer
                 - model-authored Python in restricted Docker
                 - supervised orchestration and SSH forward
                 - separate resource watchdog service
                 - durable traces, receipts and adapter exports
                            |
                    Runpod GPU Pod + volume
                    - existing native inference/GRPO
                    - full optimizer checkpoints during work
                    - fixed request/token/work limits
```

The external-controller handoff already exists in
`src/training/prime_controller.py`. It binds both plans, job nonce, policy version
and hash, batch, original episodes and cleanup evidence. Moving the controller
does not require a different reward, trainer, prompt or model. Linux execution
must pass the frozen Docker contract; byte identity of source is not proof of
platform parity.

Use a real Linux VM that can launch Docker containers. Do not assume installing
the Docker CLI inside another Runpod Pod supplies a working sandbox daemon.
[Runpod's current documentation](https://docs.runpod.io/tutorials/pods/build-docker-images)
describes its image-building workaround; it is not proof that our nested runtime
sandbox works there. Verify actual restricted container execution before a GPU
rental. No generated-Python host fallback or privileged sandbox containers.

## Work required

1. Prepare one temporary CPU VM or use an explicitly identified existing VM.
   Measure controller RAM, CPU and storage needs before selecting a paid shape.
   Transfer an allowlisted checkout including current uncommitted implementation,
   pinned tokenizer/native dependencies, V5 adapter and sandbox image. Do not copy
   the entire home directory, private SSH key or credential files. Use a dedicated
   GPU-access key. Provider credentials stay outside all model containers.
2. Make the deployment-only driver use configured checkout, interpreter, run and
   key paths instead of `/Users/jasonling/...`. Reuse the existing collector and
   fatal export/cleanup wrapper; preserve the previous as-run driver and receipts.
   Prepare fresh source-bound controller/GPU plans. Never resume the failed run
   or treat the step-three portable adapter as an exact optimizer-state resume.
3. Run the trusted orchestration under a system service that survives SSH logout.
   Disable automatic training restart/retry. Keep the watchdog in a separate
   supervised service, armed before renting the GPU and tied to absolute recorded
   deadlines and owned IDs. Fatal collection triggers bounded receipt export and
   prompt owned-resource cleanup. Killing the training process must not kill its
   watchdog. Budget CPU compute, its disk and artifact retention as well as GPU
   compute/storage; do not silently reuse the previous GPU-only cost allowance.
4. Save traces, receipts and portable adapters on controller storage before GPU
   teardown. Decide explicitly whether full optimizer state is worth retaining;
   it is much larger than a portable adapter. Do not promise exact resumption
   without exporting that state. Schedule controller teardown only after durable
   output retention is verified; stopped VM disks can continue billing.
5. Prove disconnect tolerance with scripted CPU-only work first: disconnect the
   launching SSH session and confirm progress afterward. Kill the controller job
   and show the independent guard still reaches scoped cleanup. Inject export
   failure, transport failure and deadline expiry; show no training retry and no
   deletion of unrelated resources. A real cloud CPU test still requires a quoted
   allowance. Only after these gates and new approval should a paid GPU pilot run.

System services and timers are ordinary deployment plumbing, using
[systemd supervision](https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html)
and [absolute calendar timers](https://www.freedesktop.org/software/systemd/man/latest/systemd.timer.html)
where available. This is a modest deployment change, with fault testing the main
work. It does not require a cluster, a new agent framework or a model rewrite.

## Cutoff limitations and acceptance

An independently supervised watchdog on the CPU VM removes the laptop-sleep and
training-process dependency. It still depends on that VM and provider API being
available. If protection against loss of the entire controller VM is required,
arm and verify a second deadline from a separate always-on scheduler before
GPU creation. A timer on the same VM does not provide that independence.

The installed `runpodctl pod create --help` exposes no `--terminate-after` flag.
The official [v2.12.0 release](https://github.com/runpod/runpodctl/releases/tag/v2.12.0)
removed stop/terminate-after flags; the current
[official plugin guide](https://github.com/runpod/runpod-plugins-official/blob/main/plugins/runpod/skills/runpodctl/SKILL.md)
says the backend never enforced them. Older local golden paths still recommend
them and must not be used as evidence of a provider cutoff. No exact provider
dollar cap is established by this design.

Acceptance means a cloud-hosted restricted-Docker controller keeps making progress
after the laptop/SSH client disconnects; the separate cutoff remains armed; logs
and exported checkpoints can be recovered; failure and expiry delete only the
owned GPU/temporary volume with read-back proof; and CPU/storage retention costs
are explicit. Until then, the existing laptop deployment still requires power,
awake operation and an open lid. No such dependency should be left implicit.

Current evidence: external handoff and watchdog tests already exist; the previous
attempt passed 32 focused controller/watchdog/deployment tests and the 2,224-test
full suite. This preparation reviewed those paths and current provider documents;
it did not execute a cloud controller or establish these acceptance gates.
