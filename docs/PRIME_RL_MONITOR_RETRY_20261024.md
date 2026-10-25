# October 24 Qwen retry with the repaired SSH monitor

**Two real gradient updates executed; one passed full verification.** The fixed
30-minute work deadline interrupted step 2's checkpoint save. The two-update gate
therefore remains incomplete, and the ten-update pilot was not launched.

This was the retry authorized by “can we check again,” using the same $5 estimated /
two-hour resource ceiling and fresh source-bound plans. The unchanged hypothesis
was that two consecutive native Qwen updates would preserve optimizer state and
reload exactly. Success required both complete verification receipts, not just
trainer success logs. No deadline, reward, scoring threshold or selection rule was
changed after observing the run.

## What worked

The initial **full Qwen** checkpoint contained all 504 AdamW counters at **0**.
After update 1, a fresh native DCP reload found all 504 at **1**, finite optimizer
state and exact adapter equality with the export. Trainer startup matched the
sampled policy. The verified portable adapter SHA-256 is
`0e870e3110bb5f68909be5830ff8564f6e4305869aa9f386a1bb4368a3261a29`.

Cycle 2 preparation loaded that checkpoint with counter **1** and exactly the same
optimizer fingerprint, including moments. Both native sampled-token scorer gates
passed. This validates the initializer fix through the first real Qwen update
and subsequent state carryover; it is stronger evidence than the earlier tiny
CUDA fixture. The incomplete second checkpoint does not establish counter 2.

The deployment monitor recovered from **two actual SSH connection timeouts** on
read-only status requests. Each retry succeeded without restarting the experiment,
resetting budgets or replaying a model call. The captured errors say `Operation
timed out`; they do not identify why the earlier attempt's uncaptured SSH error
occurred. The launch wrapper also now includes initial status/tunnel setup in its
cleanup scope. These are deployment-only changes, not reader or trainer changes.

## Actual run

| Cycle | Selected group rewards | Native scorer tokens / mean delta / max delta | Gradient update | Full checkpoint verification |
| --- | --- | --- | --- | --- |
| 1 | [0, 1, 1, 0] | 687 / 0.004456 / 0.190100 | 26.8 seconds | Passed, counter 1 |
| 2 | [1, 0, 0, 0] | 528 / 0.003901 / 0.099404 | 21.5 seconds | Not completed: deadline during save |

Both updates had finite loss and gradient norm; peak GPU memory was **35.34 GiB**.
The scorer thresholds remained mean ≤0.03 and max ≤0.25. The persistent ledger
contains **139 generation requests / 570,969 reserved tokens**, with all requests
returned. These are reservations, not actual generated-token counts or API charges.
All **44 original episodes remained untrainable**, infrastructure-healthy and
backed by successful Docker cleanup proofs. Only the two admitted mixed groups
were copied for native training.

The collector visited 11 training questions in the fixed seed-42 order. Cycle 2
skipped nine healthy uniform-zero groups before finding its mixed group. The
[per-question group results](../examples/verifiers-port/prime-monitor-retry-20261024/question-group-results.json)
retain question identifiers and every group's scores. No validation or locked
evaluation question was sampled.

Rescoring the saved training traces with the unchanged reward produced:

- 3 reviewed answer/support matches;
- 24 answers outside the finite accepted-wording variants;
- 4 missing required reviewed support after quote clipping;
- 13 without a verified supported terminal packet.

These are **mechanical reward reasons, not semantic answer grades**. In particular,
unknown wording receiving zero does not prove an answer false. The two cycles
used different policies and questions; these observations are not a model-quality
comparison. No independent human review or held-out evaluation was performed.

## Why the gate remains incomplete

The native parent enforced its original 1,800-second deadline while the second
trainer was saving its final checkpoint. Its stopped result reports **one verified
update** and attempted steps `[1, 2]`. Actual step-2 metrics independently show
that a gradient update occurred. The postmortem found **no committed `.metadata`**
for the second DCP checkpoint. No fresh reload or portable verified step-2 export
is claimed, and the partial checkpoint must not be treated as resumable.

Cold inference/model setup, full checkpoint I/O and collecting enough mixed groups
dominated elapsed time. The roughly 48 seconds of gradient updates do not predict
the cost of this workflow. Review the reward misses and establish a feasible
prospective checkpoint/runtime budget before another paid attempt. Keep the
strict state and scoring gates. The ten-update pilot and quality comparison remain
closed until the required integration gate completes.

## Reproducibility, cleanup and checks

The run used the unchanged frozen reader/reward/protocol, original pinned V5 and
base revisions, seed 42, four trajectories per group, temperature 1, top-p 1,
1,024 action tokens, four investigator steps and assigned source windows. The
[GPU preflight](../examples/verifiers-port/prime-monitor-retry-20261024/gpu-preflight.json)
records exact package versions, all seven native source hashes and base/adapter
identities on an A100-SXM4-80GB. It uses Prime-RL 0.10.0, PyTorch 2.13.0+cu130 and
vLLM 0.31.0, with the previously pinned image digest. Model-authored Python ran
only inside the frozen local Docker sandbox. Engine load RPC/path reporting is
not tensor attestation; native scoring and checkpoint comparisons are separate
recorded checks.

The independent watchdog deleted live Pod `amuoal2bmzfp2z` and temporary 600GB
volume `nkk6nbdrz7`; fresh MCP reads returned 404 for both. All 44 local owned
Docker containers were confirmed absent. Older user resources were untouched.
Owned-resource lifetime was **38.2 minutes**, including setup and read-only failure
diagnostics. The conservative quoted $1.90/hour combined bound gives **$1.21
estimated**, before tax. Filtered provider billing returned no posted rows;
actual billed spend remains pending, not zero.

Current validation: **139 native tests passed**, with three expected CPU DCP
warnings, and four authored SSH retry tests passed. The latest full suite passed
**2,204 tests, 16 optional skips** before this retry on unchanged production
sources; it was not rerun for documentation/receipt additions. Source-bound plan,
frozen artifact hashes, receipt hashes and user-file preservation were checked.
There were no commits, uploads, promotions or additional paid retries after this
stop. The verified step-1 adapter is retained locally for diagnostics only.

Public receipts:
[summary](../examples/verifiers-port/prime-monitor-retry-20261024/summary.json),
[native result](../examples/verifiers-port/prime-monitor-retry-20261024/native-result.json),
[counter postmortem](../examples/verifiers-port/prime-monitor-retry-20261024/optimizer-counter-postmortem.json).
Operational plans, raw traces, logs, archive and retained adapter are under
`out/reader-prime-monitor-retry-20261024/`. The started plans cannot be reused to
reset budgets. This work does not establish that RL or fine-tuning improved Qwen.
