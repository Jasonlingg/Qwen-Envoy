# October 25 reader GRPO pilot: incomplete after three verified updates

Actual Qwen training ran on one A100 SXM 80GB, starting from original pinned V5.
Three of the required ten updates completed and passed fresh checkpoint reloads.
Collection for update four failed with a generation `502 ReadTimeout`. The
experiment stopped without a paid retry, candidate selection, held-out generation,
reference opening, model promotion or upload. **Retain original V5. There is no
answer-quality result from this attempt.**

The user approved the specific $8 estimated / 3h45 proposal. The source-bound
[v2 protocol](../examples/reader-rl-pilot-v2/protocol.json) predeclared ten updates,
final-step-ten selection only, then the unchanged 12-question / 24-episode paired
comparison. Its request/token limits and success rule were not changed during
the run. Partial checkpoints do not unlock evaluation.

## Training evidence

| Update | Selected group rewards | Native scoring: tokens / mean delta / maximum delta | Trainer step | Fresh reload |
| --- | --- | --- | --- | --- |
| 1 | [0, 0, 1, 1] | 587 / 0.005459 / 0.181356 | 26.3s | Passed, counter 1 |
| 2 | [1, 0, 1, 1] | 591 / 0.005168 / 0.115765 | 23.8s | Passed, counter 2 |
| 3 | [0, 0, 1, 0] | 587 / 0.007030 / 0.186038 | 25.7s | Passed, counter 3 |

The unchanged alignment thresholds are mean ≤0.03 and maximum ≤0.25. All three
reloads matched exported adapters and verified the trainer's starting policy.
A separate read-only inspection of committed real Qwen DCP checkpoints found all
504 optimizer states at **0 → 1 → 2 → 3**. The fourth trainer was never invoked.
Peak memory was approximately 35.3 GiB. Trainer step times exclude checkpoint
export/reload, process startup and most collection time.

Portable adapter SHA-256 values, rehashed locally before deletion:

- Step 1: `f5a2a18f2d32b1c70ec8bb355acb4b8740533363a6a7aae698cbdcb7552f0234`
- Step 2: `f6be04b997249a63bbe50aa3abd6ef28c4d9e0afd5734a4c87bd73af2cf7d5d5`
- Step 3: `29827073a808bbb81f0819336c2f7766024c3e97c09e4ac5272ba286f5ef553d`

They remain diagnostic artifacts in `out/reader-prime-pilot-v2-20261025/`.
The temporary full optimizer checkpoints were deleted with the volume. Portable
adapters do not preserve exact optimizer-state resumption.

The collector attempted 23 training questions once, in the fixed seeded order:
22 complete groups, three mixed groups, 92 recorded episodes, 91 healthy and one
transport failure. All original traces remained untrainable. Only copies of the
three complete mixed groups were admitted for training. The ledger contains
**297 requests: 296 returned and one failed/uncertain; 1,223,340 reserved tokens**.
Reservations are not actual generated tokens or API billing. Healthy traces had
six mechanical reward-one and 85 reward-zero outcomes; those are not semantic
answer grades. No validation/evaluation question was sampled.

## Failure and operational correction

macOS power logs record Clamshell Sleep at **13:30:00**, followed by Sleep at
**13:30:05 -0400**. The failed generation began at **13:30:05.119647**, during
collection of the fourth trajectory for the 23rd question. After a maintenance
DarkWake at 13:38:49, the collector recorded `502 {'error': 'ReadTimeout'}` at
13:39:11.689733. It discarded the unhealthy group without converting the failure
to zero reward. Request 297 is saved; no response 297 was saved.

The most likely cause is laptop sleep interrupting the local controller/SSH
transport. The native HTTP timeout was 25 seconds, while the observed wall interval
was 546.6 seconds across sleep. This is not evidence of a Qwen reasoning failure,
GPU OOM or optimizer-counter failure. Exact upstream network/server state during
sleep was not observed. Model-authored Python continued to be restricted to local
Docker; trusted inference and training ran on the GPU.

The deployment driver requested native termination and cleaned local containers,
but its failure path did not notify the resource watchdog to delete immediately.
The native parent and idle inference process still existed at evidence capture.
The laptop remained in repeated sleep/maintenance-wake cycles until full wake at
14:13:09. After saving the receipts and adapters, an explicit stop marker triggered
resource deletion at 18:14 UTC. Approximately 35 minutes elapsed between the
collector's recorded failure and deletion. This operational gap added avoidable
allocated time.

The ignored deployment driver now has a tested fatal-error wrapper: bounded
receipt export, a failure record and an unconditional stop request to the existing
owned-resource watchdog, including when export fails. Success keeps the resource
available for the already authorized evaluation. The as-run driver was preserved;
the repair was tested offline and was **not used in this failed attempt**. No
frozen reader, reward, native trainer or numerical protocol was changed.

This wrapper cannot execute while its host is asleep. Neither `caffeinate` nor a
watchdog on that same laptop establishes a sleep-independent resource deadline.
Before another paid attempt, use an always-on Docker controller and an external
resource deadline that survives controller loss. The current bound was local
control-plane automation, **not a provider-enforced dollar cap**. A new paid run
requires explicit approval; none was started here.

## Cost, cleanup and validation

Owned lifetime including setup, evidence export and cleanup was **102.91 minutes**.
The conservative $1.90/hour combined compute/storage planning bound gives **$3.26
estimated before tax**, below the approved $8 estimate. Provider billing queried
at 18:17:51 UTC had posted $1.3745998831 for this Pod (including root disk) and
$0.0583333336 for this temporary volume: **$1.4329332167 posted so far**. Posting is
incomplete; that is not the final invoice or total cost. Raw filtered receipts are
saved separately from the estimate.

The watchdog deleted owned Pod `r4bp0gb1wv49sc` and STANDARD 600GB volume
`qgcgohq7tk`. Fresh independent MCP reads returned 404 for both. All 92 recorded
local Docker containers are absent. Unrelated resources were untouched.

Reproducibility identities are bound in the saved plans and
[summary](../examples/verifiers-port/reader-pilot-v2-20261025/summary.json):
Qwen/Qwen3-8B revision `b968826d9c46dd6066d109eabc6255188de91218`, original
V5 revision `27b912a863ff914ad45baa03624d8911dc1e17fb`, source revision
`13b496d2a5359329b110e3419628de3cf791843b`, reward
`reader-short-fact-reference-v2`, seed 42, temperature/top-p 1, four samples per
group, LR 1e-6, LoRA rank 4/alpha 8, native centered GRPO/PPO clipping, no KL.
The pinned Prime-RL 0.10.0 / PyTorch 2.13.0+cu130 / vLLM 0.31.0 stack and image
were checked on the GPU. Bootstrap required the pinned image's passwordless sudo
for SSH setup because its default user is appuser; the same Pod was repaired.

Validation after the attempt: **32 focused controller/watchdog/deployment checks
passed**; **2,224 full tests passed with 17 optional skips** and three existing
SWIG deprecation warnings. Preparation had also passed 183 focused native tests.
Saved case-study reproduction remains verified as an offline integrity/recount
check, not new semantic validation. All 2,534 pre-existing snapshotted files remain
byte-identical except the intended handoff update. No files were staged or
committed. Human review and model-quality comparison remain absent.

Small receipts: [counter inspection](../examples/verifiers-port/reader-pilot-v2-20261025/optimizer-counters.json),
[per-question groups](../examples/verifiers-port/reader-pilot-v2-20261025/question-group-results.json),
[billing](../examples/verifiers-port/reader-pilot-v2-20261025/billing-receipt.json),
[power events](../examples/verifiers-port/reader-pilot-v2-20261025/local-power-events.json).
Full raw traces, plans, native logs, metrics, archive and adapters are retained
under `out/reader-prime-pilot-v2-20261025/`.
