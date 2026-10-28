# October 26 reader RL run: nine verified updates, evaluation not unlocked

Later follow-up: the user authorized a separate, bounded exploratory comparison
of saved step 9 with original V5. Its [report](READER_STEP9_EXPLORATORY_20261026.md)
found no supported-answer gain; two ambiguous references make its formal
decision inconclusive. It does not complete this original ten-update pilot.

The restarted original-V5 GRPO pilot stopped on a request timeout during rollout
collection for update ten. Nine native optimizer updates were completed and
verified, and all nine portable adapter checkpoints are saved locally. The
predeclared ten-update candidate does not exist, so the held-out comparison was
not opened or run. **Whether RL improved answer quality remains unknown.**

This is an incomplete experiment, not a negative model-quality result. Keep
original V5 as the evaluated reference; do not promote this partial checkpoint.

## Hypothesis and experiment identity

The prospective hypothesis was that ten GRPO updates would yield at least three
net additional complete, source-supported answers on twelve paper-disjoint
questions versus original V5. At most one regression, no new unsupported answers,
no execution-failure-count increase, and the protocol's token/latency tolerances
were allowed. Review would be model-assisted with independent human review null.
Training reward or token F1 alone could not pass this gate.

- Protocol: `examples/reader-rl-pilot-v2/protocol.json`, SHA-256
  `6946e67ff7433495310813c88ccc630b01ab7012a2ed44745b51fa7657964d66`.
- Base: `Qwen/Qwen3-8B`, revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Initial V5: `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5`, revision
  `27b912a863ff914ad45baa03624d8911dc1e17fb`,
  `artifacts/full/checkpoint-50`, adapter SHA-256
  `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
- QASPER revision: `13b496d2a5359329b110e3419628de3cf791843b`.
  Eighty eligible training questions, one seed-42 pass, four rollouts per group.
  The protocol lists every training and locked evaluation question/paper ID.
- Prompt SHA-256:
  `1f752247467187689092c0366ac9923b077cd7c6df0ea6969bb21acabb994e16`.
  No hint/checklist intervention. Reward: `reader-short-fact-reference-v2`.
- Native Prime-RL 0.10.0 at `9e15490dcab1d506ce3462e69b8aa5dde45c6b3a`;
  renderer commit `5d870d8eaa65114f5bef8a4cffc146fbcb418ec4`;
  Verifiers 0.4.0, Transformers 5.6.2, PyTorch 2.13.0+cu130, vLLM 0.31.0.
- One NVIDIA A100-SXM4-80GB. Rank-4 LoRA, alpha 8, dropout 0, AdamW
  learning rate 1e-6, constant scheduler, native centered GRPO/PPO clipping,
  no reference KL. Temperature/top-p 1, top-k -1, min-p 0, thinking disabled,
  at most 1,024 generated tokens per action, four investigator steps,
  8,192-token context. Seeds and native configurations are saved in the run.
- Model-authored Python ran only in the local restricted Docker sandbox.
  The GPU handled trusted inference, scoring and training. No host-REPL fallback.

## Observed training and failure

Forty complete groups were collected across forty questions; nine were mixed and
each admitted one native update. A forty-first question's group failed on its
third rollout. Across the run there were 163 episodes: 162 healthy and one
infrastructure failure. Original saved traces remained untrainable; only complete
healthy mixed groups produced separately admitted training candidates.

The request ledger contains 526 requests: 525 returned and one failed or uncertain.
It reserved 2,173,972 tokens. Returned responses account for 1,631,811 prompt and
24,384 completion tokens; the unfinished response's token count is unknown.
No tenth gradient update was attempted. The failed group received no numeric
failure reward and was not retried.

### Retrospective training-signal audit (not a new evaluation)

After the run, the implementing assistant rescored the saved rollouts from the
40 **complete** groups with the same pinned `reader-short-fact-reference-v2`
reward and reviewed training cards. This excludes the two healthy rollouts from
the incomplete forty-first group. The reason counts sum to 160:

| Reward reason | Rollouts |
| --- | ---: |
| `reviewed_answer_and_support_matched` (reward 1) | 14 |
| `answer_not_in_reviewed_variants` | 113 |
| `no_verified_supported_terminal`: no evidence / incomplete | 17 / 11 |
| `reviewed_support_missing_after_quote_clipping` | 4 |
| `additional_citation_without_reviewed_support` | 1 |

The 113 nonmatching answers are **not 113 proven reward errors**. Source-visible
spot inspection found both kinds of case. For “What data do they train the
language models on?”, `BABEL speech data` received zero because the accepted
variants say `BABEL speech corpus`; its inspected quote names that corpus, so
this is a plausible complete paraphrase for further review. For “What
categories does the dataset come from?”, `five categories` received zero while
the inspected quote lists abuse, social, anxiety, PTSD, and financial; the answer
omits the requested categories. For “What do they learn jointly?”, `person
category` received zero while the inspected passage says relations were learned
jointly. These are implementing-assistant observations on **training-only**
episodes, not independent human labels or permission to add an alias. The
underlying receipts are in the ignored
`out/reader-prime-restart-20261026/controller-training/cycle-*/rollouts/*/rollout.json`
files; the scoring entry point is `src/training/reader_data.py:score_episode`.

A focused implementing-assistant triage of five distinct training cases makes
the mixed bucket concrete. These observations are **diagnostic**, not new gold
labels:

| Question ID prefix | Saved answer | Source-visible issue to review |
| --- | --- | --- |
| `qasper_train_0cf5132a` | `fastText, SVM` | A submitted inspected span says “We used fastText and SVM”; the card accepts `fastText and SVM` but not comma wording. Candidate alias, pending independent review. |
| `qasper_train_fb56743e` | `BABEL speech data` | An inspected span names the BABEL speech corpus. Wording looks related, but the short span alone does not settle whether it specifically supports language-model training. |
| `qasper_train_d10e256f` | `0.62` | The reviewed target specifies **macro F1**; the answer omits the metric. |
| `qasper_train_1ccfd288` | `five categories` | The inspected span names all five domains, but the answer does not list them. |
| `qasper_train_281cb27c` | `person category` | The inspected span says **relations** are learned jointly; “person category” names a subgroup, not the requested object. |

This small source check suggests both reward-coverage and answer-use problems.
It cannot estimate their proportions among all 113 nonmatches. The next review
should sample that bucket by question, hide the reward label from the reviewer,
and record completeness and support separately from wording equivalence.

The practical next step is a small, source-bound review of training-only
nonmatches that separates valid paraphrases, incomplete facts, wrong facts, and
missing support. Any revised reward must be frozen and checked on distinct
reward-validation examples before another optimization run. This audit does not
change the original reward, checkpoint or evaluation result.

Request 526 began after the second passage read at 01:47:38 Toronto. Around
01:48:04 the local non-streaming budget proxy returned a 502 with `ReadTimeout`.
`src/training/prime_budget.py` bounds each upstream HTTP read to 25 seconds.
The recovered vLLM log shows a running request and approximately 18 generated
tokens/second through 01:48:00, followed by the controller's cleanup SIGTERM at
01:48:06. No sleep/wake transition was found in the failure interval, and the
recovered final engine log has no pre-failure OOM evidence.

The immediate failure is established: the proxy timed out while generation was
still active. At that observed generation rate, the allowed 1,024-token response
could take roughly 57 seconds, longer than the read timeout. A longer response is
a plausible explanation; its unfinished text, eventual length and eventual
success were not recorded. Do not interpret the shutdown `EngineDeadError` as
proof that the engine crashed before the timeout.

The user's removal of the fixed **rental** deadline was applied and verified on
both watchdogs. The experiment's request/token/work caps stayed unchanged. This
failure was the individual request timeout, not exhaustion of those caps or the
removed rental deadline.

## Preventing recurrence before the next run

The user requested this prevention plan after the failure. **These changes are
proposed, not implemented or live-tested.** The immediate repair is a small
transport change; durable resumption is a separate follow-up. Neither requires
changing the model, reward, prompt or document tools.

1. **Allow a legitimate long response to finish.** Replace the 25-second upstream
   read timeout with a proposed 120-second generation limit. At the observed
   18 tokens/second, 1,024 tokens need roughly 57 seconds before transport and
   prefill overhead. Keep short connection/write timeouts and cap each request
   by the remaining experiment time. Align the native client, local bridge,
   proxy and controller timeouts so an outer layer cannot still abort at 25
   seconds. Enforce an actual request deadline as well as per-read timeouts;
   slow response chunks must not extend a request indefinitely. Keep response
   size, request-count and token limits. A longer limit reduces this failure
   mode; it cannot guarantee that every request succeeds.
2. **Record enough to distinguish slow generation from a dead worker.** Save
   request ID, policy/checkpoint hash, start/end times, timeout/error kind and
   available token usage in the durable ledger. Capture the worker's active
   request state and relevant logs on failure. Unknown completion tokens stay
   unknown. Streaming can be considered later, but must preserve the native
   renderer/token contract and is not necessary for the first timeout repair.
3. **Make recovery explicit, with no blind retry.** A timeout can leave generation
   active upstream. Cancel or reconcile that request and verify worker state
   before further dispatch. Preserve its nonrefundable reservation. Continue
   rejecting infrastructure failures as numeric training rewards. The current
   protocol permits zero retries; a future recovery protocol must declare its
   retry/abort rules and budgets in advance, keep the sampled policy and seeds
   identifiable, and never admit an incomplete four-rollout group or replay an
   uncertain optimizer update.
4. **Back up a resumable checkpoint, not just adapter weights.** Before a future
   paid launch, choose and budget an off-Pod backup destination. Preserve the
   latest complete native checkpoint, including optimizer and scheduler state,
   plus RNG state where supported, controller question cursor, request ledger,
   checkpoint identities and source-bound configuration. Verify completeness
   and hashes before marking that update recoverable. Keep the previous verified
   backup until the new one is verified. Deduplicate pinned base weights where
   the native format permits; do not assume full native checkpoints are small.
   Exact-resume support needs a tested loader, not merely a file named checkpoint.
5. **Separate GPU shutdown from checkpoint retention.** Prepare durable backups
   during the run; do not rely on a last-minute transfer after credit, transport
   or controller loss. Stop the GPU under the agreed lifecycle policy, and
   delete temporary checkpoint storage only after the recovery copy is verified
   or an explicit retention/expiry policy applies. Any retained storage needs a
   stated cost and expiry; failed export must not silently buy unlimited storage.
   Preserve the Docker sandbox and the independent failure/disconnect guard.

Verification gates before spending again:

| Check | Required result | Status |
| --- | --- | --- |
| Local fake engine returns after more than 25 seconds | Accepted within the new bound; saved response and ledger agree; no duplicate dispatch | Pending repair |
| Fake engine exceeds the request or remaining run deadline | Bounded failure with recorded cause, reservation retained and no numeric reward | Pending repair |
| Timeout occurs while worker is still active | No automatic duplicate request; cancellation/reconciliation is explicit | Pending repair |
| Backup interruption or corrupt transfer | Previous verified backup survives; new update is not declared recoverable | Pending recovery support |
| Native checkpoint restore | Optimizer/scheduler and controller state are restored; next verified update is n+1 without a skipped or duplicate update | Pending recovery support |

The saved step-nine adapter remains usable for inference. Its deleted optimizer
state cannot be reconstructed by implementing this plan. Testing those weights
does not require repeating training, but needs an explicitly amended exploratory
evaluation and a newly approved bounded rental. Keep the current run's receipts
and its incomplete ten-update outcome unchanged.

## Preserved artifacts and cleanup

Local ignored run directory: `out/reader-prime-restart-20261026/`.

- Adapters: `controller-training/cycle-{1..9}/adapter/`.
  Step-nine SHA-256:
  `a2a773caf222aa34becbeb2f886ce553c0aefa0871783fa17efe8da0557ca810`.
- Complete portable failure export: `training-failure-gpu-receipts.tar.gz`,
  357,355,190 bytes; SHA-256
  `cc7ed032753d474ca4a84372f80b5fcec1a57655f0ee7f16db87e1f91bd77883`.
  Extraction succeeded. Recovered files are in `gpu-recovered/`.
- `incomplete-result.json` checks every local/exported adapter hash, native
  optimizer-step counter 1 through 9, prior-policy and optimizer continuity,
  batch hash, native scoring receipt, reload and startup proof.
- `training-controller-cleanup.json` confirms all 163 recorded Docker containers
  absent. Fresh provider reads in `owned-cleanup-final.json` confirm GPU Pod
  `udw64z1kqrfu7d`, temporary volume `faaj8ilakr` and watchdog Pod
  `v1vavx85538xj5` all return 404. Unrelated resources were not changed.
- Full optimizer/base checkpoints were excluded from the portable export and
  deleted with the temporary volume. Saved adapters support inference and a new
  continuation with fresh optimizer state, not exact training resumption.

The freeze observer refused to unlock a partial candidate. There are zero held-out
evaluation episodes; references remain closed, with no claim grading or Answer F1
result. No upload, promotion, commit or additional paid retry was performed.

## Cost and verification

The recorded combined rate was conservatively $1.92/hour including GPU, CPU guard
and storage. Earliest resource creation to the final all-absent read gives a
conservative lifetime cost estimate of **$4.07 before tax**. This is not an invoice.
The scoped provider billing read has posted **$2.0764532451** so far: GPU/root disk
$1.9456780585, CPU/root disk $0.0724418530 and network volume $0.0583333336.
Returned billing buckets end at 05:00 UTC, before shutdown; the final cost is not
yet posted at that initial read. `billing-receipt.json` preserves those reads.

Later October 26 billing refresh, covering the shutdown bucket through 06:00 UTC:
Runpod has posted **$3.8286607833 total** for the three owned resources. GPU compute
is $3.5643146150, GPU root disk $0.0185185170, CPU guard including its root disk
$0.1291609841, and network volume $0.1166666672. The GPU is the largest cost.
`billing-receipt-updated.json` preserves this refresh; the earlier partial receipt
and conservative estimate above remain unchanged historical observations.

Focused verification: 27 paired-evaluation/protocol tests and 22 lifecycle/fatal
cleanup tests passed separately; their combined final rerun passed all 49.
The first combined rerun exposed eleven deployment-test fixture failures because
installation had consumed the temporary input files; stable fixture snapshots
resolved those failures without changing runtime code. Source/receipt checks passed; training, reward, prompt and frozen runtime
implementation files were not changed. The preceding unchanged-code full suite
had 2,156 passes and 19 optional skips; it was not rerun for these operational
helpers and documentation.

The next bounded experiment should first repair and verify request-timeout handling.
An explicitly amended, exploratory V5-versus-saved-step-nine comparison could test
the already trained weights without repeating training. It would be a new protocol
and paid launch, not completion of the original ten-update experiment. No such
launch has been authorized or started here.
