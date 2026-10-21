# Authorized retry — model loaded; timing comparison incomplete

The user approved one fresh attempt with the repaired startup check, one A5000,
900 seconds including loading, ten model requests, nine telemetry jobs and a
$0.50 local allowance. [authorization.json](authorization.json) binds the new
plan to the previous consumed attempt. Neither marker nor the old artifacts
were reset. Actual live code is commit `971932c`; later repairs were not run
against the GPU in this attempt.

| Check | Observed result |
| --- | --- |
| Model/adapter identity | Independently verified from current-worker MCP logs |
| vLLM/fitness readiness | Both readiness events observed |
| Automated CLI log startup | Did not pass; repeated timeouts continued |
| Telemetry preflight | Completed, including required metrics and GPU memory |
| First input warmup | Worker input-integrity guard rejected it before generation |
| Second input warmup | Unattempted |
| Serial/concurrent timing rounds | None completed |
| Shutdown and entrypoint restoration | Verified; full endpoint configuration restored |

The preflight device snapshot was 20,783/24,564 MiB used. Startup logs report
15.31 GiB for model loading and 4.39 GiB available KV cache. These are snapshots
and engine reports, not measured peaks or proof of concurrent-request fit.

## Readiness handoff and limits

[startup-proof.json](startup-proof.json) retains selected independent MCP log
records, including the exact pinned identity, vLLM health and fitness success.
Direct CLI comparisons with tail and both timestamp forms sometimes returned
logs; the supervised reads continued to time out. These observations do not
establish a deterministic timestamp-format or GPU defect.

The operator verified those same-worker records with the existing identity and
readiness functions, then launched the already-authorized worker phase under
the **original activation, absolute deadline, exclusive worker marker and
request allowance**. The handoff controller imposed that same deadline on its
child. There was no second activation. The original supervisor still owned
shutdown/owned-job cleanup and entrypoint restoration. It was interrupted after
the worker exited, which explains `KeyboardInterrupt` and a null child exit code
in [lifecycle.json](lifecycle.json). [handoff.json](handoff.json) records the
deviation; [the controller snapshot](handoff-controller.py.txt) is an execution
artifact, not a reusable launcher. Do not reexecute it or reset consumed markers.

Automated startup reliability therefore remains unverified. This manual
handoff must not be reported as a successful automated end-to-end test.

## Failure and offline repairs

One telemetry job completed. One model job failed with
`profile input differs from the pinned payload`, before the hook called vLLM.
[failed-job.json](failed-job.json) preserves the actual provider error;
[result.json](result.json) is incomplete. One request reserved 8,192 tokens;
no generated answer or usage was returned. No inference was retried.

The saved inputs contain `temperature=0.0` and `top_p=1.0`. A transport that
serializes these as `0` and `1` changes the original canonical hash while
preserving the decoding values. This is a locally reproduced hashing hazard.
The rejected received input/hash was not recorded by the original hook, so
normalization is a plausible explanation, **not its proven live cause**.

Commit `43097e1` preserves the historical raw/canonical pins and adds separate
transport hashes that normalize safe integral JSON numbers. Changes to text,
booleans or decoding values still fail. Rejections now include the received
transport hash. The client also saves terminal FAILED job responses before
raising, closing the diagnostic gap observed here.

Commit `ae8a874` reads a bounded tail and retains strict worker/source/activation
time checks locally, reducing dependence on the inconsistent server-side filter.
Both repairs are locally tested; neither received another paid live attempt.

## Verification and interpretation

Before activation, 33 focused tests and the full 1,879-test suite passed. After
the offline repairs, 45 focused checks and the full 1,882-test suite passed
(80.37 seconds, no skips, three existing SWIG deprecation warnings). Ruff and
whitespace checks passed. Source/evidence hashes are in
[validation.json](validation.json).

[endpoint-after.json](endpoint-after.json) records 0/0, zero workers, no pending
jobs and the restored original configuration. The catalog maximum-window
compute estimate was $0.1725 at $0.69/hour. Actual invoice cost was not obtained;
the local budget/time guard is not a provider dollar cap.

This attempt verifies model loading and a telemetry component. It does not
measure serial/concurrent speed, supported-answer quality, reactions to new
feedback, MCP advantage or fine-tuned Qwen improvement. No human answer review
was performed. No further paid attempt is authorized by this record.
