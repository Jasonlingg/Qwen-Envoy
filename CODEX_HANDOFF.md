# Envoy / Qwen research-agent handoff

> **October 25: historical prototype cleanup completed.**
> User approved a bounded cleanup of personal-memory and map demos. Recovery and
> results: docs/ARCHIVED_PROTOTYPES.md; complete earlier source at 5240090.
> Removed37files: old personal-memory web/MCP/CLI/eval, Learning Lab/map/impact/
> Nemotron-chat product modules, three HTML interfaces and18dedicated test files.
> Trimmed memory.py/chat.py to3shared helpers with original function bodies.
> Current research/experiment MCPs, Qwen, training, evaluation and Studio retained.
> Net7879implementation lines removed (69707->61828;11.3%),3316test lines removed
> (34424->31108). Seven direct provenance tests preserve shared-boundary coverage.
> Full2141pass/17optionalskips, native160pass, focused68+7pass; Ruff/diff checks pass.
> Lower full count reflects90retired-feature tests removed and7newdirect checks.
> All29frozen runtime hashes, source-bound plan and saved-case-study recount pass;
> all data/examples/release bytes and protected legacy files unchanged. Original
> three untracked plan/spec inputs preserved. No paid work or model-quality claim.
> Cleanup is one local commit; no push. Historical commands for removed apps need
> the earlier checkout. Operational record: out/prototype-cleanup-20261025/.


> **October 25: commit authorization and unused-code review completed.**
> User explicitly authorized committing the accumulated work after an unused-code
> check, superseding earlier no-commit instructions for this work. Review:
> docs/PRECOMMIT_REVIEW_20261025.md. No unused runtime code was established;
> ReaderEnv is a dynamically loaded plugin, and offline reward candidates retain
> documented reproduction commands/tests. Withdrawn custom backend remains absent.
> Verified 160 native tests and 2224 full tests / 17 optional skips; Ruff, artifact
> checks and frozen source-bound plan pass. No paid work or cloud deployment.
> Work grouped into sandbox selection, training/evidence, and README/resume/status
> commits. Preserve the three original untracked October16 plan/spec inputs.
> Ignored out/ adapters, deployment drivers, connection files and environments
> stay local. Protected legacy files unchanged. No push authorized or performed.


> **October 25 cloud-controller move: inspected and scoped; not deployed.**
> User asked to move the controller/guard into the cloud and whether it is a big
> lift. [Deployment proposal](docs/READER_CLOUD_CONTROLLER.md): reuse existing
> external Docker controller on a real Linux CPU VM, keep native GPU training,
> supervise controller and watchdog separately, save outputs before teardown and
> prove SSH/laptop-disconnect tolerance with CPU-only scripted work first. No
> model/reward/harness rewrite. Provider/VM choice and combined CPU/GPU/storage
> price remain unresolved; an async question asks existing VM vs temporary VM.
> No machine, model call, training retry, upload or commit started. Prior $8 GPU
> pilot approval does not authorize another attempt or silently added CPU costs.
> Important provider check: installed CLI has no --terminate-after. Official
> v2.12 release/current plugin say removed because backend did not enforce it;
> older local golden paths recommending it are stale. Cloud watchdog survives
> laptop sleep, but same-VM services do not survive controller-VM loss; separate
> external deadline needed for that stronger guarantee. All implementation and
> prior run receipts remain unchanged; preparation is a design, not live proof.


> **October 25 final: paid pilot stopped after three verified updates; GPU deleted.**
> [Actual report](docs/READER_RL_PILOT_20261025.md),
> [receipts](examples/verifiers-port/reader-pilot-v2-20261025/summary.json).
> Actual native GRPO from original V5: all 504 optimizer counters 0 -> 1 -> 2 -> 3;
> each full reload/export/startup check and native token scorer passed. Update-4
> collection failed: request297 received 502 ReadTimeout, no fourth trainer run.
> macOS Clamshell Sleep at13:30:00/05 coincides with failed request13:30:05.1196;
> most likely the sleeping local Docker controller interrupted SSH/native transport.
> No GPU OOM/optimizer-failure evidence; exact upstream state unobserved.
> 23 questions attempted, 22 full groups/3mixed, 92episodes/91healthy; originals
> untrainable. 297requests (296returned/1uncertain), 1,223,340 reserved tokens.
> Held-out references remain CLOSED; no partial candidate selection or evaluation.
> KEEP ORIGINAL V5; no answer-quality/model-improvement claim, upload or paid retry.
> All three adapters and logs exported and SHA-checked under
> out/reader-prime-pilot-v2-20261025/. Step3 SHA
> 29827073a808bbb81f0819336c2f7766024c3e97c09e4ac5272ba286f5ef553d.
> Portable adapters only: full optimizer checkpoints deleted, no exact resume.
> Driver failure did not immediately request resource deletion; idle allocation
> persisted until explicit stop after full laptop wake/evidence export (~35min
> after recorded failure). A deployment-only fatal export/stop wrapper is now
> tested offline, as-run driver preserved; not used during this failed attempt.
> IMPORTANT: local watchdog/caffeinate also depend on awake laptop. Before a new
> paid run, require always-on Docker controller and sleep-independent resource
> deadline, plus new explicit authorization. No provider-enforced dollar cap.
> Owned Pod r4bp0gb1wv49sc and volume qgcgohq7tk deleted by watchdog; fresh MCP
> reads both404, all92owned Docker containers absent. Unrelated resources untouched.
> 102.91min lifetime / $3.26 conservative estimate beforetax. Provider has posted
> $1.4329332167 asof18:17:51UTC; incomplete invoice, not total actual cost.
> 32focused guard/controller/deployment checks pass; full2224pass/17optionalskips,
> three existing SWIG warnings. Prep native183 passed. Frozen sources preserved;
> of2534pre-existing files only this handoff changed. No staging/commits.
> Historical ACTIVE notices below are superseded by this final notice.


> **October 25 ACTIVE paid pilot: two of ten updates fully verified; cycle 3 collecting.**
> User approved $8 estimated / 3h45 for actual ten-update training and the 12-question
> paired comparison. No new paid retry or commit. Authorization and live files:
> out/reader-prime-pilot-v2-20261025/. Native work /workspace/reader-training on
> owned Pod r4bp0gb1wv49sc, temporary STANDARD 600GB volume qgcgohq7tk, US-WA-1.
> GPU: A100 SXM 80GB, $1.79/h; 80GB root disk; conservative combined bound $1.90/h.
> Independent caffeinate/watchdog PID 98006; deadline 1792959397.414371 UTC epoch
> (20:16:37 UTC). Local detached driver PID 1323, remote native parent PID 515.
> Neither resource is cleaned up yet. Only these owned resources may be removed.
> SSH direct 195.26.233.82:57698, key ~/.ssh/id_ed25519; connection.json is saved.
> Bootstrap initially failed because image USER appuser could not write SSH host
> keys. Same Pod repaired with passwordless sudo entrypoint; server and GPU verified.
> Pinned assets/software passed. Source-bound v2 plans use 80 eligible training
> questions, four samples/group, temp/top-p 1, seed42, LR1e-6, native GRPO/LoRA,
> exactly ten verified updates selected at step ten. No partial candidate unlock.
> Verified optimizer counters 0->1->2, native startup/export reloads pass. Scorers:
> step1 587 tokens / mean .005459 / max .181356; step2 591 / .005168 / .115765.
> First adapter SHA f5a2a18f2d32b1c70ec8bb355acb4b8740533363a6a7aae698cbdcb7552f0234;
> step2 f6be04b997249a63bbe50aa3abd6ef28c4d9e0afd5734a4c87bd73af2cf7d5d5.
> Training driver already running: DO NOT re-run it or resample. Read monitor.py,
> training-orchestration.log and controller-training/ for current progress. All
> raw episodes stay untrainable; only complete healthy mixed groups are admitted.
> References still CLOSED. Once ten updates finish: freeze candidate on GPU, copy
> session locally, explicitly open/review all 12 references BEFORE model outputs,
> run both arms through local Docker + SSH native engine, export masked claim review.
> Same frozen v2 protocol and success rule; no checkpoint search. Keep judgments
> model-assisted with human review null. Capture adapters/receipts before cleanup.
> No commits/upload/model improvement claimed. Full 2224/17 optional skips and
> native183 passed during preparation; production code unchanged since then.

> **October 25: v2 training revision and bounded pilot prepared; GPU not launched.**
> [Protocol/runbook](examples/reader-rl-pilot-v2/README.md),
> [protocol](examples/reader-rl-pilot-v2/protocol.json),
> [reward revision](examples/reader-rl-pilot-v2/reward-revision.json).
> Explicit opt-in `reader-short-fact-reference-v2`: six exact reviewed phrases
> across four training cards. The ambiguous word/sentence representation question
> is held entirely, alongside the earlier incomplete financial-data question:
> 80 eligible questions. Original bundle/keys, v1 protocol and audit receipts stay
> intact; support spans, inspection/citation gates, prompt and Docker are unchanged.
> New protocol selection binds the revision through prepared controller/GPU plans,
> Verifiers tasks/rewards and paired evaluation. Default tasksets remain v1.
> Saved 44-output replay: positives 4 -> 7, mixed 2/11 -> 4/11. On the eligible
> subset (40 episodes), positives 3 -> 6 and mixed 1/10 -> 3/10. This is fixed-output
> reward coverage, not model improvement. Originals remain untrainable.
> Ten updates from original pinned V5, final step ten only, same 12 locked papers /
> 24 paired episodes and original quality decision rule. No repeat integration run.
> Proposal: **$8 estimated / 13,500s (3h45) total owned lifetime**, training 10,800s,
> evaluation 1,800s, setup/export/cleanup 900s. Requests/tokens unchanged (training
> 1,024 / 8,388,608; evaluation 288 / 2,359,296). At most $1.92/h combined, 10%
> cost headroom. Fresh catalog A100 SXM 80GB Secure $1.79/h, US-WA-1 LOW stock;
> STANDARD volume supported. Price-read saved; stock not reserved, root-disk/price
> and availability need a final launch check. Observed-pattern estimate fits, but
> a 25% time margin does not fit 3h45: completion is not guaranteed. Stop incomplete
> if a guard trips; no partial candidate unlock or automatic extra paid retry.
> **New $8 allowance is NOT authorized.** Matching explicit authorization receipt
> is required by the updated independent watchdog; false template cannot activate
> it. Default watchdog retains old $5/two-hour limits. No paid resources, model
> calls, optimizer updates, locked reference contents, commits, uploads or promotion
> during preparation. Human review null. Existing user edits and all 29 frozen
> runtime files preserved. Source-bound old execution plans are historical and must
> be regenerated for the new implementation; old receipts were not rewritten.
> Tests: 183 native focused pass; 2,224 full pass / 17 optional skips. One new test's
> optional Verifiers version handling was fixed; it passes under pinned 0.4.0 and
> skips in the incompatible main environment. Saved case-study checker, Ruff and
> diff checks pass. Prepared local plan and helpers: out/reader-pilot-v2-prep-20261025/.
> Next decision: explicit approval of the increased budget, then fresh provisioning
> with an independent guard, one pilot and comparison, and publish the outcome.

> **October 24 follow-up: offline wording audit and cost projection complete.**
> [Report](docs/READER_REWARD_REVIEW_20261024.md),
> [audit summary](examples/verifiers-port/reader-reward-review-20261024/summary.json).
> Model-assisted review of all 22 training wording rejections: 7 complete answers
> supported in the reviewed paper source, 7 incomplete, 7 unsupported/irrelevant,
> 1 ambiguous word-versus-sentence representation target. Human review null.
> Seven proposed exact aliases tested only in memory; all existing citation gates
> retained. Three gain reward; four still fail required-span/extra-citation checks.
> Fixed 44 saved outputs: positives 4 -> 7, mixed groups 2/11 -> 4/11. This cannot
> predict policy behavior or prove model improvement. Raw episodes stay untrainable.
> No active reward/key/runtime changes, model calls, paid resources, training,
> held-out reference review, uploads or commits during this audit.
> Native timing includes checkpoint/process overhead: ten-update training plus
> setup/export and the existing 30m evaluation allowance projects 3.09h / $5.88
> at the previous $1.90/h planning rate, $7.35 with a 25% time margin. Counterfactual
> alias scenario 2.66h / $5.05, optimistic every-group-mixed case 2.38h / $4.53.
> These are rough historical extrapolations, not quotes or enforceable ceilings.
> **Do not launch the unchanged ten-update plan:** every scenario exceeds its
> one-hour training and two-hour rental limits. Original protocol/budget untouched.
> Next: review/version training-only aliases, resolve/hold the ambiguous question,
> then prepare a pilot with mutually consistent update count, timing and budget.
> Evaluation stays locked; larger paid work needs separate authorization.
> 55 focused tests pass; full 44-episode scorer replay and 29 frozen runtime hashes
> checked. Of 2,515 pre-existing files only this handoff changed; user edits intact.

> **October 24 latest: two-update native Qwen integration gate PASSED.**
> [Report](docs/PRIME_RL_TWO_UPDATE_20261024.md),
> [receipts](examples/verifiers-port/prime-two-update-20261024/summary.json).
> Two real updates and both fresh full DCP reloads passed. All 504 optimizer
> counters **0 -> 1 -> 2**, exact moment/state carryover between cycles, native
> sampled-token scorers and startup/export tensor checks passed. Native result
> verified_updates=2, no unverified attempts. Updates 24.9s / 22.0s; peak 35.34GiB.
> 139 calls / 572,022 reserved tokens, all returned; 44 healthy original untrainable
> Docker episodes over 11 training questions. Nine uniform-zero groups skipped
> before cycle-2 mixed [1,0,0,0]; cycle 1 [1,1,1,0]. Reward categories are mechanical,
> not semantic grades. No held-out outputs/reference review, human review or
> model-improvement claim. Ten-update pilot/paired evaluation not launched.
> Prospective timing override used (14,400s native ceiling later than rental
> guard); actual native work 1,783s, narrowly within old 30m in this attempt.
> Do not claim a controlled timing improvement. Source/reward/runtime unchanged.
> Portable adapters and logs: out/reader-prime-no-work-cutoff-20261024/. Final
> step-2 SHA af5db09f6391e35f2ab8782ebb8d5ca5abb75537a50c291fa3af7d0b97479995;
> retained locally for diagnostics, no promotion/upload. Full optimizer checkpoints
> deleted with temporary volume after verification/evidence export. Watchdog
> deleted Pod x40apwdixaelxi and volume ie1httce8x; fresh MCP reads both 404, all
> 44 local owned Docker containers absent. About $1.16 estimated, invoice pending.
> Current native 139 / deployment 8 tests pass; latest full 2,204 / 16 skips predates
> this retry on unchanged production sources. No commits. Next: assess sparse
> signal and a feasible larger-run budget before any ten-update/quality run.

> **October 24 next-check configuration: remove the separate 30-minute cutoff.**
> User requested “set no time limit then.” Prepared a fresh two-update plan and
> deployment driver at `out/reader-prime-no-work-cutoff-20261024/`; no new GPU or
> paid run was started by this configuration change. The native API requires a
> finite ceiling: 14,400 seconds is deliberately later than the existing rental
> guard's maximum 7,200 seconds, so it cannot stop work before that guard. Driver
> execution requires a current independent watchdog heartbeat and the unchanged
> estimated $5 rental bound. This is not an unlimited-spend authorization or a
> provider-enforced dollar cap. Requests/tokens remain 256 / 2,097,152. Eight
> transport/guard tests pass. The prospective integration timing override is
> `examples/verifiers-port/prime-next-integration-policy.json`; historical numeric
> protocol/results are untouched. Reader/reward/native trainer unchanged. Use this
> fresh driver/plan for a future launch; do not restart the prior timed-out plan.

> **October 24 latest retry: two actual Qwen updates, one fully verified.**
> [Report](docs/PRIME_RL_MONITOR_RETRY_20261024.md),
> [receipts](examples/verifiers-port/prime-monitor-retry-20261024/summary.json).
> Full-model initializer/DCP counters: all 504 start at **0**, update-1 fresh
> native reload has **1**, and cycle-2 preparation preserves the exact optimizer
> fingerprint/counter **1**. Step 1 tensor export/startup checks passed. Both native
> scorers passed. Real updates: 26.8s / 21.5s, peak 35.34GiB, finite loss/gradient.
> **Two-update gate incomplete:** unchanged 1,800s work deadline killed step-2
> checkpoint save; no committed metadata, no counter-2/full reload proof. Do not
> resume/promote that partial checkpoint. Native result verified_updates=1.
> 139 calls / 570,969 reserved tokens; 44 healthy original untrainable Docker
> episodes across 11 training questions. Cycle 2 skipped nine uniform-zero groups
> before [1,0,0,0]. Reward reasons: 24 unknown accepted-wording, 4 missing reviewed
> support, 13 no verified terminal, 3 matches; these are not semantic grades.
> Monitor recovered two live SSH connection timeouts without experiment/model
> retries. No ten-update pilot, held-out model/reference review, quality claim or
> paid retry after stop. Watchdog deleted Pod amuoal2bmzfp2z and volume nkk6nbdrz7;
> fresh MCP reads both 404; all 44 local Docker containers absent. About $1.21
> estimated this attempt; actual invoice pending. Current native 139 / transport 4
> tests pass; latest full 2,204 / 16 skips predates this retry on unchanged sources.
> Operational directory: out/reader-prime-monitor-retry-20261024/; step-1 verified
> adapter SHA 0e870e3110bb5f68909be5830ff8564f6e4305869aa9f386a1bb4368a3261a29
> retained locally for diagnostics only. No commits/upload. Next: review sparse
> reward reasons and checkpoint/runtime budget before another paid integration;
> keep strict gates. Old notices below are historical and superseded here.

> **October 24 later check: tiny CUDA counter diagnostic passed; Qwen gate incomplete.**
> [Report](docs/PRIME_RL_COUNTER_CHECK_20261024.md),
> [receipts](examples/verifiers-port/prime-counter-check-20261024/summary.json).
> Fresh corrected-code plans and exact GPU preflight passed. An SSH status read
> exited 255 during inference startup; deployment cleanup terminated the native
> parent before generation. This attempt: **0 Qwen calls / 0 attempted or verified
> updates**, no new adapter, no ten-update pilot or held-out evaluation. Underlying
> SSH disconnect detail was not captured. A separate tiny CUDA AdamW/DCP test on
> the actual A100/torch2.13 stack passed **0 -> 1 -> 2**, three checkpoint reloads
> and exact equality with fresh AdamW weights. This is not Qwen/DTensor/Prime-RL
> integration proof. Deployment-only read monitor now has bounded read retries
> and stderr receipts (four authored tests); not yet live-training verified.
> Repaired driver/tests: `out/reader-prime-counter-check-20261024/`. Do not reuse
> the started plans. Watchdog deleted live Pod qptbwk1qk67kx5 and volume 0rdp3k2d1m;
> fresh MCP reads both 404. About $0.33 estimated this attempt; actual invoice rows
> pending. No additional paid retry after this stop. **139 native / 2,204 full
> tests passed**, 16 optional skips; sources/protocol unchanged, no commits/upload.

> **October 24: integration attempt stopped; optimizer bootstrap repaired locally.**
> [Actual report](docs/PRIME_RL_INTEGRATION_20261024.md),
> [receipts](examples/verifiers-port/prime-rl-integration-20261024/summary.json).
> Real V5 -> external local Docker -> native scorer -> one actual gradient update;
> four episodes on one training question, rewards [0,1,1,0], 15 calls / 62,002
> reserved tokens. Scorer mean .00362124 / max .20048976 passed .03/.25. Update
> 26.7s, peak 35.3GiB, finite loss/gradient. **Zero verified updates**: fresh native
> DCP reload found AdamW step 2 at progress 1. Initial native DCP had already
> materialized a dummy zero-gradient AdamW step (all 504 counters at 1).
> Strict gate stopped before update two. No ten-update pilot or held-out outcomes.
> Reproduced get_state_dict's side effect on the actual GPU stack and inspected
> actual checkpoint scalar states. Local initializer fix permits only fresh
> zero-decay AdamW, verifies unchanged weights/zero moments, normalizes dummy step
> to zero before saving. No native optimizer/loss/trainer patch or weakened gate.
> Real CPU PyTorch DCP round-trip and first-real-update equality regression passes;
> corrected GPU path remains unverified. No extra paid retry after failure.
> Pod dgnq9xllg9jecw and volume sjnuc6xqu3 **deleted, fresh GETs both 404**.
> Watchdog early cleanup refused mismatched MCP/CLI timestamp string formats;
> explicit MCP fallback removed resources. Timestamp normalization/regression fixed
> locally; do not claim that automatic live deletion succeeded. Conservative
> time/rate estimate $0.95; actual billed amount pending (provider rows not posted).
> Raw receipts/failed forensic adapter in out/reader-prime-integration-20261024/;
> no promotion/Hub upload/commit. All 29 reader runtime/data/reference hashes and
> numerical protocol unchanged; held-out refs unopened. Large optimizer files were
> deleted with temporary storage after evidence export. Next: repaired bounded
> two-update gate, then a full-run projection including measured checkpoint I/O.
> Final repair validation: **139 native tests / 2,204 full passed**, 16 optional
> skips; native has three expected CPU DCP warnings, full has three existing SWIG
> warnings. Ruff/diff clean. Original edits and 29 frozen reader files/data hashes
> preserved. Real CLI post-deletion absence reads pass idempotently, not a live
> deletion test. Old source-bound launch plans are stale; prepare new ones for a
> future authorized trial. Current artifact manifest explicitly says post-failure
> local repair, with prior manifest SHA and unchanged numerical protocol hash.

> **October 24: bounded external-controller integration launched.**
> User said “ok go” after the two-update / conditional-ten-update explanation.
> Current allowance: $5 estimated, at most two hours owned GPU lifetime, one A100
> 80GB; generation/update/workload ceilings remain the unchanged protocol's.
> [External-controller runbook](docs/PRIME_RL_EXTERNAL_CONTROLLER.md).
> `prime_controller` is a hash-bound file handoff to the existing native collector;
> Python still executes only in local frozen Docker. GPU stages keep native
> scoring/training/optimizer/process cleanup. Default same-host mode is retained.
> An independent local caffeinated watchdog scopes deletion to creation receipts.
> It is not provider-enforced and needs local/control-plane availability.
> Live temporary Pod `dgnq9xllg9jecw`, volume `sjnuc6xqu3` (600GB, US-WA-1).
> Fresh reads: A100 $1.79/hour, standard volume $0.07/GB/month; projection < $4
> before tax at two hours. Current image digest was revalidated against GHCR bytes.
> Pre-launch **136 native / 2,202 full passed**, 16 skips / three existing warnings;
> Ruff/diff clean. All 29 frozen runtime and original data/reference hashes intact.
> Logs/ownership/watchdog/launch scripts: `out/reader-prime-integration-20261024/`.
> At this entry: image startup only, no new optimizer update or held-out outcome.
> Check actual latest runtime receipts before assuming completion. No Git commits.

> **October 24: paired RL evaluation collection/review wired locally.**
> User approved the next local step; no paid compute, new Qwen generation,
> optimizer update, Hub upload or Git commit. [Runbook](docs/READER_PAIRED_EVALUATION.md).
> `src.training.reader_evaluation` freezes only the complete predeclared V5-start
> ten-update pilot after checking all ten adapter/optimizer/scorer/load receipts,
> then copies hash-bound adapters/receipts into a portable controller session.
> Explicit reference export/review is gated after candidate freeze and before
> generation. Locked reference contents remain unopened in this preparation pass.
> The numerical protocol remains byte-identical (SHA `1d0038b7…05aa8d9`).
> New unscored evaluation tasks use the same native renderer/token transport and
> frozen Docker reader; ordinary training rejects evaluation-session tasks.
> Shared durable 288-request/2,359,296-token/1,800-second caps, V5-first order,
> paired seeds, no retries/replacements, persisted failures and owned cleanup.
> VF omits task state in its JSON: saved records explicitly retain Docker packets.
> Masked review export preserves raw traces/private arm mapping; claim inventory,
> requested-fact review, source bindings and original semantic/resource thresholds
> are required. Missing billed cost/startup/lifetime receipts stays inconclusive.
> Quote checks, strict reward, official F1 and fixture success do not prove support
> or model gain. Existing secondary scorers are not auto-run by this new command.
> Independent human review remains null unless an actual human supplies it.
> Loopback-only engine forwarding supports local Docker plus remote native
> inference for evaluation; native adapter loading records path/RPC/bytes identity,
> not engine tensor attestation. No actual SSH/GPU connection made this pass.
> Found/fixed a GPU-preflight blocker: Prime-RL is a namespace package with an
> empty `__file__`; both source checks now locate a real native module, retaining
> the same seven source hashes. Training plans now enforce exact seed-42 order.
> **Remaining launch gates:** consecutive training updates require Docker and CUDA
> together with the current launcher; a newly rented GPU container is not certified.
> Split training orchestration is not implemented by this evaluation work. Also
> install/verify the separate provider lifetime/cost watchdog and obtain fresh paid
> authorization. Proposed $5/two-hour allowance remains unauthorized. No paid
> allowance from the completed one-update smoke is reusable.
> Isolated native suite: **115 passed**, including authored replies through real
> Docker for both arms and native admin HTTP with explicitly mocked GPU identity.
> These fixtures are untrainable, not Qwen inference or GPU evidence. New valid
> preparation plans retain 81 training questions and the 12-paper evaluation lock.
> Logs/plans: `out/reader-paired-prep-20261024/`. Earlier edits, frozen runtime/data,
> and protocol preserved; artifact manifest records the prior manifest SHA.
> Full **2,182 passed / 16 optional skips / three existing SWIG warnings**;
> Ruff/diff clean. All 29 frozen runtime hashes and source/reference hashes remain
> intact; earlier user edits preserved. Detailed validation is in the runbook.

> **October 24: bounded RL pilot protocol and training-label review prepared.**
> User approved continuing RL; this pass prepares evaluation/success criteria
> and reviews training labels, without paid compute or commits.
> [Protocol/runbook](docs/READER_RL_PILOT.md),
> [bound artifacts](examples/reader-rl-pilot-v1/README.md).
> Follow-up implementing-assistant review covers all 82 training questions,
> accepted variants and supporting excerpts, not full papers or independent human
> review. One paper has the accepted incomplete alias `the public financial news
> dataset released by`: hold that entire training question. Frozen source/keys/
> strict reward remain unchanged; 81 questions are eligible for this pilot.
> `reviewed_training_ids()` binds every card to the attributed review; the bounded
> launch driver captures its hash and rejects reintroduced holds. The generic
> taskset still inventories 82 train questions. Regenerate older prepared plans;
> their source/review hashes are stale. No held training question becomes eval.
> Locked 12 model-evaluation references remain unopened except byte hashing.
> Proposal: two-update integration gate, then fresh original-V5 ten-update native
> pilot and same-12-paper paired evaluation. Step ten only; no checkpoint search.
> Positive pilot gate: net +3 fully supported answers, <=1 supported regression,
> zero newly unsupported questions/count increase or execution-failure increase,
> bounded tokens/latency. This is a provisional engineering signal, not statistical
> significance or broad model improvement. Human review remains null.
> $5 estimated / two-hour single-GPU budget is a **proposal, not authorization**;
> fresh pricing/storage, resource watchdog and Docker layout must be verified.
> Retained integration+training optimizer checkpoints need 464 GiB free disk
> beyond installed assets. Paired evaluation/review-pack runtime wiring is still
> pending; consecutive GPU updates remain unverified. No new paid run/Hub upload.
> Focused **34 passed**, isolated native **87 passed**, full **2,157 passed /
> 16 skipped / three existing warnings**; Ruff/diff clean. Logs and preparation
> plans: `out/reader-rl-pilot-prep-20261024/`. All 29 runtime and frozen data/
> reference hashes unchanged; earlier user edits preserved. No Git commits.

> **October 24: repeatable sequential native RL launcher implemented locally.**
> User approved implementing the repeatable path after the one-update smoke;
> this pass does not authorize more paid compute or commits.
> [Runbook](docs/PRIME_RL_REPEATABLE_RUN.md). `python -m src.training.prime_run`
> prepares a source/model/prompt/reward-bound plan, then an explicit local-GPU
> mode runs cold inference → complete mixed group in frozen Docker → native
> scoring → one native update → optimizer-resumable checkpoint verification.
> Subsequent cycles load the prior verified adapter and check optimizer-state
> continuity. Native trainer/loss/optimizer remain unchanged; generic VF training
> is still blocked and raw traces remain untrainable. Strict finite reward stays
> active; guarded F1 remains separate. Training-role questions only, seed 42;
> validation/locked evaluation not sampled. Global request/token/time/update
> limits, durable nonrefundable reservations, failure stop, process/container
> cleanup; no cloud provisioning or provider dollar guarantee. The last adapter
> is cold-loaded only if there is another cycle. Two metadata-only public Hub
> reads pin base file identities and V5 config; no weights downloaded.
> Local isolated native suite **79 passed**; budget/process/asset/optimizer-state
> fixtures **21 passed**; final full suite **2,149 passed / 16 skipped**, three
> existing SWIG warnings. Ruff clean; all 29 runtime files and frozen RL source/
> reference hashes unchanged; all prior tracked user changes preserved, index empty.
> Initial concurrent native/full run had Docker preflight
> failures before inference; isolated native rerun passes without relaxing checks.
> Logs and prepared plan: `out/reader-prime-repeatable-20261024/`. Three new
> implementation modules, one test module and two added native integration tests;
> full GPU checkpoint stages have not been executed with this new driver.
> Next gate is a separately budgeted two-update GPU run, then a predeclared
> quality comparison. No new model inference, GPU jobs, updates, Hub upload or Git
> commits. Independent human review null; no model-improvement claim.

> **October 24: guarded-F1 prototype passes authored development controls; not enabled.**
> User proposed combining F1 with checks and a shorter final answer.
> [Probe](examples/reader-guarded-f1-v1/README.md): 74 authored controls on the
> same eight reward-validation papers plus four saved training-role episodes.
> A 32-word answer cap only fixes repetition. Numeric/unit/negation guards block
> the earlier exploits; pairing counts with nearby recognized units also blocks
> the swapped threads/tweets stress case. All six declared development checks
> pass; all eight reference answers and proposed paraphrases retain their credit.
> Fixed vocabulary/proximity heuristics remain limited and were iterated on these
> cases; no independent semantic validation/generalization or quality claim.
> Prototype only lives under examples; current reward and training path unchanged.
> No model/API/GPU calls, updates or commits. Focused 47 passed; eight direct parser
> checks pass; Ruff clean; all 29 runtime/source-reference hashes unchanged.

> **October 23: gated Answer F1 tested; candidate rejected, not enabled.**
> [Report](docs/READER_F1_AUDIT.md),
> [69 per-record results](examples/reader-f1-reward-v1/results.json).
> User authorized testing the proposed F1-plus-existing-support-gate reward.
> Same eight reward-validation papers: 65 authored controls; four saved real
> training-role episodes are exposed diagnostics only. Evidence gates reject all
> 32 tiny/uninspected/extra/broad controls. Proposed paraphrases average .598,
> but upstream normalization gives `15.00 sentences` full credit against reference
> `1500 sentences`; negated references average .827. The predeclared development
> gate fails. Keep official F1 as a metric and the existing narrow finite reward
> as the current baseline; do not wire this candidate into training. The evidence
> gate checks attributed support ranges, not arbitrary-answer entailment. No
> new model/API/GPU calls or optimizer steps, no commits or locked-eval scoring.
> Optional raw-record fixture export defaults off; all 36 old semantic-audit cases
> reproduce unchanged. Focused 47 passed; full suite 2,128 passed / 16 skipped,
> three existing SWIG warnings; Ruff clean. All 29 runtime hashes and frozen
> source/reference hashes unchanged; previous user changes preserved.

> **October 23: offline F1 reward audit; no reward switch.**
> [Report](docs/READER_F1_AUDIT.md),
> [52 per-case results](examples/reader-f1-audit-v1/results.json).
> User asked whether reference matching/Evidence F1 is enough. On eight existing
> reward-validation papers, any-overlap paragraph mapping gives full credit to
> all eight one-character citations; wrong answers retain perfect evidence scores
> when their quotes stay unchanged. Official Answer F1, literal paragraph matching
> and explicitly nonofficial overlap/character metrics are kept separate. Four
> saved training-question episodes are exposed diagnostics, not evaluation.
> Evidence F1 alone is not adopted for the current answer-and-short-quote task.
> Next minimal proposal: continuous Answer F1 behind existing inspection and
> attributed-reference-support checks; still needs local validation and is not
> enabled. LLM reward stays on hold; no paid judge call is needed for that local
> comparison. Active reward/runtime/data unchanged; all 29 runtime hashes and
> frozen source manifest pass. No model/GPU calls or commits. Focused 39 passed;
> full suite 2,120 passed / 16 skipped, three existing SWIG warnings; Ruff clean.

> **October 23: QASPER semantic-reward candidate prepared; not enabled.**
> [Design/status](docs/READER_SEMANTIC_REWARD.md),
> [audit pack](examples/reader-semantic-reward-v1/README.md).
> User approved pursuing human-calibrated RLAIF and emphasized that QASPER already
> contains answers. The candidate uses all original text answer/evidence annotations,
> not the finite accepted-answer whitelist. Three separate axes cover correctness,
> support and citation relevance; uncertain/malformed verdicts never become numeric
> negative rewards. 32 authored controls on eight reward-validation papers plus four
> saved real training-role Qwen episodes are exported; exposed episodes are excluded
> from the calibration gate. Human fields remain null, judgments empty, zero API/GPU
> calls. Current reward/runtime/data unchanged, no Git commits. Focused 36 passed;
> full suite 2,117 passed / 16 skipped; final contract 8 passed, Ruff clean.
> A fresh <=$0.50 Nebius diagnostic allowance was requested via
> the pending question; do not dispatch until the user answers. The previous hosted
> judge allowance stays consumed. Independent human validation and verified live
> judge configuration remain required; training_ready stays false.

> **October 23: real native Prime-RL GPU smoke passed; resources removed.**
> [GPU report](docs/PRIME_RL_GPU_SMOKE.md),
> [saved receipts](examples/verifiers-port/prime-rl-smoke/README.md).
> User authorized Runpod and a bounded smoke, with checkpoint upload afterward.
> One A100 SXM4 80GB at $1.79/hour ran V5 generation and native training sequentially.
> Four real Docker reader episodes produced rewards [0,0,1,0], 12 branches and 553
> sampled loss tokens. Sampler/scorer delta mean .00237037, max .18167973 passed
> preset .03/.25 gates. Native centered-GRPO advantages + PPO clipped surrogate,
> one AdamW update at 1e-6: 23.33 seconds, finite loss/gradient, peak 35.34 GiB.
> All 504 adapter matrices changed and reloaded exactly; startup policy matched V5,
> original V5 SHA unchanged; portable adapter cold-loaded into vLLM.
> Private Hub adapter: `jasonlingg/qwen-envoy-v5-reader-rl-smoke-20261023`, commit
> `ef297bcc832dc949483fb1c99f83c7f2f3a4e89d`; downloaded SHA verified.
> Native packed probabilities required renderer submodule revision 5d870d8 rather
> than release 0.1.12; pin and regression checks added. First seven failed calls
> were discarded and remain counted in the total 19/64 dispatches. No tolerance,
> prompt, reward, scope or frozen runtime changes. Complete groups only; native
> trainer/optimizer unchanged. Generic production training remains blocked.
> Full suite 2,109 passed / 16 skipped / three warnings; native suite 56 passed;
> final token regression 13 passed. All 29 runtime files and user dirty files preserved.
> Pod osmoo0a2gzjqd9 and temporary volume q3egk3botx each freshly confirmed 404.
> Conservative GPU estimate $1.225 for 41.06 minutes including image pull, plus
> short-lived storage; itemized billing had not posted. No further paid run is
> authorized by this result. No quality/semantic-accuracy/model-gain claim; one
> question and narrow reviewed reference reward are engineering smoke only.
> No Git commits. Independent human review null. See
> [the local integration report](docs/PRIME_RL_LOCAL_CHECK.md) for the authored controls.

> **October 22: local RL reward/setup validation; training remains blocked.**
> User approved validating reward and training setup before an RL retry, not
> a paid/GPU launch. Scope `269c56f`, candidate/preflight `50393d7`, scientific-unit
> case preservation `c958271`. [Report](docs/READER_RL_VALIDATION.md),
> [saved per-record results](examples/reader-rl-validation/README.md).
> Separate offline candidate uses source-reviewed complete answer variants plus
> claim/support keys: 14 cards, 20 claims, 28 variants. On the unchanged 150 exposed
> records, 0/74 answerable known negatives receive positive reward; 13/14 old
> positive controls and all 28 reviewed variants pass. The rejected old positive
> claims independent test sets; its cited excerpt establishes perplexity on test
> sets, not independence. Old answer/artifact remains unchanged; narrowed variants
> pass. All 50 missing-scope records have no numeric task reward and must be
> excluded from this objective. All 12 reference-string controls score zero;
> correct unlisted paraphrases can also fail. These finite-match development
> results are partly by construction, not calibrated semantics or a Qwen gain.
> Review is implementing-assistant/model-assisted; independent human null.
> Preflight binds V5, source, exact effective prompt, Docker image, 29 unchanged
> runtime files and unchanged V1 bytes. The historical generation-defaults fix
> is present; old host-execution/old-prompt trainer remains incompatible with V2.
> Its optimizer/config regressions pass, not proof of a current V5 GPU update or
> proof that a bug caused the old held-out failure. No trainer/reward/runtime
> was changed. Report CLI exits 2 for not-ready; it never launches and cannot
> prevent someone directly invoking the historical trainer. Original data hold,
> missing separate reward validation, V2 Docker/raw-token rollout parity, genuine
> GPU update/reload proof and zero paid budget block retraining. No inference,
> GPU activation, real Qwen optimizer update, training export or reserved-pack use.
> Next: eligible training-only pool/reuse policy and separately locked reward
> validation, then the V2 training adapter and a separately budgeted signal probe.
> Focused 60 passed; final full suite 2,116 passed in 109.23s, no skips, three
> existing SWIG warnings. One focused reporting failure and the earlier green
> full run are retained. Saved scores reproduce exactly. Of 2,041 prior hashes,
> only this intentional handoff update changed; 2,040 remain unchanged. Test logs,
> preservation proof and score reproduction are in the validation bundle.


> **October 22: RL investigated offline; reward and data gates block training.**
> User authorized RL investigation, not a paid launch. Scope `05e7a24`, reward
> audit `dc67ac7`, structured-history/hold fix `ddfef59`.
> [Report and future gates](docs/READER_RL_INVESTIGATION.md),
> [150-record audit](examples/reader-rl-investigation/README.md).
> Research supports generating trajectories with outcome RL, but still needs QA
> data and reliable reward. Search-R1/R1-Searcher/s3 are relevant, not evidence
> for our Qwen3/V5 reader. s3's ~2.4k processed examples came from a ~70k eligible
> pool; this is not support for a 24-example generalization experiment.
> Existing September QASPER GRPO already failed fresh confirmation: SFT 13 passes
> versus GRPO 12, despite a small automatic-reward increase. Review was model-assisted.
> New offline audit: 24 saved demonstrations + 3 actual exposed V5 outputs + 2
> replayed corrections + 121 authored terminals, not 150 model rollouts. Legacy,
> scoped-overlap and scoped-exact scorers give full reward to 12/24/36 unacceptable
> records. Generic/false gaps, irrelevant real quotes and reversed task bindings
> demonstrate why syntax, provenance and lexical reward cannot certify support.
> Source judgments are implementing-assistant/model-assisted; independent human null.
> No model inference, GPU, optimizer update, active reward/trainer/runtime change.
> The old QASPER trainer uses host execution, old prompt and no submit_gap path;
> it must not be launched for V2. New training needs a Docker-only raw-token adapter.
> **Correction to earlier all-fresh data claim:** the old inventory missed 13
> structured historical task files, adding 96 paper / 659 question IDs. Seven
> training papers (14 trajectories) have prior prepared-pool/evaluation membership;
> this is not proof of optimizer exposure. Development papers are unaffected by
> this supplemental scan. The 24-example export is ON HOLD; HOLD.json supersedes
> historical eligibility. Old data/registry/receipts remain unchanged. The current
> data adapter uses schema-v2 structured-history coverage and rejects the old registry.
> The old selector had only 16 eligible train questions even before this fix.
> Do not promise hundreds of fresh examples by rerunning it. Explicit training-pool
> reuse or new reviewed facts needs its own data policy; preserve evaluation exclusions.
> Next is a validated task/reward specification and runtime training adapter, then
> a separately budgeted exploration-signal probe. No training launch is authorized.
> Focused 58 passed; final full suite 2093 passed in 133.22s, no skips, three SWIG
> warnings. One prior full attempt failed a loopback timing-comparability fixture;
> isolated recheck 3 passed, then full rerun passed. All failure logs retained.
> All 29 frozen runtime files unchanged. Of 2,043 prior hashes, four intended
> inventory/test/hold-notice files changed; 2,039 others, protected files and
> original user dirty files remain unchanged. No new data/model performance claim.


> **October 22: paired passage-reader pilot data prepared; zero model calls/training.**
> User approved local data preparation under scope `9640256`. Implementation
> `1062832`, Docker reporting correction `44f87b5`.
> [Saved data/replay](examples/passage-reader-pilot-data/README.md),
> [training export](data/sft/passage-reader-pilot-v1/manifest.json).
> Identifier inventory covers 213 prior workspace files, excluding 956 papers and
> 1,370 question IDs. Twelve fresh train papers yield 24 paired trajectories; six
> different official-validation papers yield 12 scoring-only development episodes.
> Thirteen draft pairs were rejected for source ambiguity/remaining support before
> any model outcomes; reasons and all four local draft rounds are retained.
> All 36 accepted demonstrations passed fixed-action Docker replay: 72 actual
> inspections, 108 local formatter turns, exact current prompt and reviewed final
> actions. Zero inference/paid calls/GPU activations/optimizer updates. The first
> attempt stopped after two successes on a transient Docker availability probe;
> saved separately, no source/model failure hidden. All 38 owned containers were
> freshly inspected absent. Frozen runtime/tools/retrieval/reward/trainer unchanged.
> Source review is implementing-assistant/model-assisted; independent human null.
> Export contains only 24 train trajectories, 72 assistant next-action targets;
> 225,496 input tokens, 3,018 supervised, max 3,729 under 8,192. Token allocation
> present-read 784 / present-terminal 1,028 / missing-read 749 / missing-terminal 457.
> Runtime prefix/masking checks pass; equal row counts do not balance token loss.
> No fine-tuned Qwen improvement is measured. Future work is the proposed single
> budgeted V5 continuation plus matched candidate/V5/base development comparison,
> then fresh evaluation before promotion. Current paid/training allowance is zero.
> Historical local inventory files are not all redistributed; strict re-export
> requires them. Saved source/replay/export artifacts have a portable hash check.
> Focused 28 passed; full 2075 passed in 113.11s, no skips, three existing SWIG
> warnings. Ruff/diff passed; all 1,747 prior file hashes and 29 frozen runtime
> files preserved, including protected dirty files and paid markers.


> **October 22: V5 training alignment audit completed; no new training run.**
> User approved auditing existing data and defining a targeted SFT pilot.
> Scope `3cc3d5b`, audit/pilot implementation `226722e`.
> [Actual audit](examples/reader-training-audit/README.md),
> [proposed narrow pilot](docs/PASSAGE_READER_SFT_PILOT.md).
> Reproduced train/validation token totals 4,812/1,162 with the cached pinned
> tokenizer and existing assistant-only preprocessing. Train is 19 answerable /
> 16 insufficient, so not a refusal-majority dataset. Across 44 conversations:
> zero final EVIDENCE spans, zero submit_gap actions, zero current prompt matches;
> all 18 insufficient examples end in cited prose. Old export whitespace
> normalization is explicitly reproduced; original observations are preserved.
> These are structural/lexical findings, not fresh semantic grades or causal proof.
> The September two-example overfit already passed; the October V6 pilot already
> failed promotion (answerable 5/17 versus V5 4/17, insufficient 2/7 versus 7/7).
> Do not repeat either as if untested. The old V6 holdout is consumed regardless
> of its unchanged historical lock text. No new reserved question pack was opened.
> Proposed next data batch pairs present/missing evidence under passages-v2;
> actual IDs, full previous-training/evaluation exclusion registry, reviewed
> demonstrations, replay and a paid budget must be frozen before any GPU job.
> Current training/inference/teacher paid allowance is zero. No training/reward
> or frozen runtime file changed, no model call/GPU activation occurred.
> Separate four-case source-visible spot check is explicitly implementing-assistant
> review: a supported definition, qualified/ambiguous manual-evaluation answer,
> and scoped gaps that do not establish whole-paper absence. No reuse approved.
> Focused 23 passed; full 2056 passed in 166.52s, no skips, three existing SWIG
> warnings. Ruff/diff passed; all 29 frozen files and 1,736 prior hashes preserved.


> **October 22: source-reviewed corrections replayed in the frozen Docker harness.**
> User approved the next offline validation step. Scope `2674209`, implementation
> `dc350a8`. [Actual replay](examples/reader-correction-replay/README.md): all three
> original hybrid packets reproduced except fresh UUIDs before either correction
> ran. Fifteen generated policy inputs matched recorded payloads; ten actual
> source-inspection steps used the pinned Docker image, network none and exact
> read-only corpus mounts. The reviewer-authored humor extraction returned its
> exact 2400..3050 citation; the reviewer-authored bias gap ended no_evidence with
> the specific unexecuted query retained. Both original read/observation prefixes
> were unchanged. All five owned containers were confirmed absent, including a
> separate fresh inspection. No model inference/tokenizer load, credential read,
> GPU activation, training, reward change or runtime baseline change occurred.
> Fixed-action replay cannot measure how Qwen chooses actions or reacts to new
> feedback. It demonstrates interface/provenance compatibility, not a Qwen gain.
> Evaluation papers/questions remain regression-only, excluded from training;
> prior draft files remain unchanged. Independent human review is null. All
> 29 frozen implementation files and 1,692 prior files/markers are unchanged.
> Focused 39 passed; full 2051 passed in 137.22s, no skips, three existing SWIG
> warnings. Ruff/diff passed. Next work is separate reviewed training data under
> paper/question exclusion rules; no training or paid inference allowance exists
> from this completed replay.


> **October 22: offline reader improvement queue added; runtime baseline frozen.**
> User approved a continuously improvable workflow after freezing the tested
> harness at `cce82c2`. Scope `e5232a3`, implementation `01615f9`.
> [Guide](docs/READER_IMPROVEMENT_LOOP.md),
> [saved example](examples/reader-improvement-loop/README.md).
> `scripts/reader_improvement.py prepare/check` binds saved source, original
> multi-turn traces and actual final model inputs to an attributed review.
> The actual hybrid smoke imports all three cases: implementing-assistant source
> review records two failures (humor extraction, structured gap reporting) and
> one passing translation control. Two final-action correction drafts pass
> provenance/structure checks, not automatic semantic grading. Independent human
> review remains null. The drafts are regression-only, excluded from training,
> and require fresh Docker replay; no corrected observations are fabricated.
> This is not an SFT export, automatic collector, self-training loop or promotion
> system. All 29 frozen implementation files and 1,680 prior files/markers were
> verified unchanged. No GPU/cloud call, training, reward or runtime change.
> Focused 55 passed; full 2042 passed in 272.17s with no skips and three existing
> SWIG warnings. Ruff/diff and deterministic artifact reproduction passed.
> A future training adapter needs separate reviewed examples and paper/question
> exclusions; a new checkpoint needs predeclared matched evaluation. Do not
> reopen retrieval tuning or claim weights alone explain every remaining failure.


> **October 22: matched hybrid-input live V5 smoke completed, partial answer gain.**
> User explicitly said “ok go test it.” Scope `1de2485`, runner `7ba4e52`, exact
> frozen input comparison `6ff9404`. [Results](examples/qasper-hybrid-v5/README.md):
> same three exposed QASPER questions, source snapshots, prompt, V5 weights,
> decoding, Docker image and four-request budget. Only scope ranges/read examples
> changed to saved plain-hybrid windows. Nine requests / 28,270 reported tokens,
> zero execution or action/submission/gap errors, six real Docker inspections,
> three mechanically verified terminal packets. Complete answerable cases under
> implementing-assistant review: historical BM25 0/2 → hybrid 1/2. Translation now
> correctly gives LSTMs/attention/dropout with supporting citations. Humor still
> denies lexical cues despite all seven reference terms in the actual model
> request and returned quotes. Bias now gives question-specific prose inability
> as evidence_found, without atomic gap/query; useful gaps 0/1 in both runs.
> Overall quality rule not met. Evidence_found/provenance checks do not establish
> answer correctness. No further retrieval tweak, training, reward change or
> paid retry followed. Source/input/semantic evidence is saved; human review null.
> Historical control, exposed examples and unguaranteed remote seed prevent a
> general/causal improvement claim; no trained-versus-base model gain measured.
> Fresh 16:35:47 UTC reads: full config restored, workers 0/0, no workers/pending
> jobs; six owned Docker containers absent. New qasper-hybrid-v5 live marker
> consumed, all older allowances unchanged. 64 focused and 2012 full tests passed;
> 804 prior hashes including protected files unchanged. Next reader work should
> target use of supplied evidence and specific structured gap reporting. No new
> paid run, base call, reserved-set use or training is included by this test.


> **October 22: CPU-only hybrid retrieval comparison completed; candidate not promoted.**
> User approved better RAG before more Qwen calls/training. Scope `54227ea`,
> implementation `e79bd99`, frozen model/data/code plan `ce63cc3`.
> [Actual comparison](examples/hybrid-passage-retrieval/README.md): 80 existing
> exposed QASPER questions, four methods, exact same two disjoint 1600-character
> windows as the live smoke. Gold stayed outside retrieval. Full-paragraph counts
> dev / exposed secondary: BM25 4/19 · 9/20; dense 5/19 · 11/20; hybrid 5/19 · 14/20;
> hybrid+reranker 5/19 · 10/20. Dev has one answerable row without gold spans;
> unanswerable rows are not retrieval successes. Preselected reranked candidate
> failed required dev gain +2, actual +1; no default or MCP tool was changed.
> Plain hybrid contains the last smoke's humor cue lists and translation details;
> reranking loses them. Do not promote an ablation using exposed secondary scores.
> All dense windows exceeded the encoder's 256-token limit; report truncation.
> Code exports exact proposed reader windows, without inspection credit or Qwen
> dispatch. Local M3 CPU only, zero paid/Qwen calls, GPU launches or Runpod API
> operations; no training/reward change or reserved-pack access. 35 focused and
> 1993 full tests passed; 774 prior hashes including protected files and consumed
> paid markers unchanged. Human review stays null. Qwen quality on these new
> windows remains untested; this is not a model-improvement or workflow win.


> **October 22: bounded QASPER passage-v2 live smoke completed, quality gate not met.**
> User explicitly requested QASPER. Scope `2064b62`, implementation `f7f11ff`,
> frozen inputs `9291514`. [Results](examples/qasper-passage-v2/README.md):
> three deterministically selected, already exposed questions; fixed full-paper
> BM25 selected two disjoint passages per paper without gold labels in model
> inputs. Dataset assignment retains QASPER/S2ORC revision rather than fake arXiv
> versions. This is an evaluation-only MCP adapter, not product QASPER ingestion.
> V5 made nine calls / 27,936 reported tokens, no inference retries or execution,
> action/submission/gap errors; three terminal packets passed provenance checks.
> Humor retrieval missed its gold paragraph, and scoped inability was accepted
> as an evidence_found candidate. Translation answered only NMT despite reading
> LSTM/attention details. The unanswerable case used terminal submit_gap but
> copied generic placeholder strings. Mechanical completion is not useful answer
> quality. Implementing-assistant claim review is saved; human review stays null.
> Final suite 1974 passed in 107.95s, no skips, three existing SWIG warnings.
> Fresh 14:32:21 UTC reads: workers 0/0, zero workers/pending jobs, full config
> restored; six owned Docker containers absent. Eleven protected files, 637 prior
> artifacts/markers and 15 atomic-gap files unchanged. One new
> `out/qasper-passage-v2-LIVE_ATTEMPT` marker consumed; no further paid calls,
> base-model run, reserved-set use, training/RL or reward change are included.
> No general fine-tuning gain or full QASPER score is claimed. Address retrieval,
> answer completeness and gap specificity separately before another live run.


> **October 22: atomic passage gaps implemented and locally verified.** User
> approved simplifying the gap workflow. Scope `802ccec`, implementation
> `ed75991`. Optional experiment MCP now uses `passages-v2`: after two distinct
> successful passage inspections, `submit_gap(missing=..., follow_up_query=...)`
> saves an unreviewed gap and ends without an accepted answer in one action.
> It launches no search or host Python. Source code remains Docker-only; source
> bounds, inspection checks and the four-request budget remain. A premature
> separate abstention can recover on turn four. The verifier checks the terminal
> trace; the missing statement and unexecuted next query survive in the note.
> Same-request retry/restart makes no new call. Legacy baseline/focused/v1 prompts
> (12 hashes), V1 preamble pin, 11 protected files and 637 prior artifacts/markers
> are unchanged. Historical comparison runners keep v1; no paid allowance reset.
> [Scripted MCP/Docker example](examples/atomic-passage-gap/README.md) uses synthetic
> responses and paper text. Focused 90 passed; standalone full 1963 passed in
> 214.82 seconds, no skips, three SWIG warnings; Ruff/diff checks passed. No
> post-change live Qwen, GPU/paid call, training/reward change or semantic/model
> improvement result is included. A bounded fresh development smoke is a separate
> next step; do not relabel old V5/base failures as new-protocol successes.

> **October 21 local / October 22 UTC: base passage comparison completed.**
> User explicitly requested base alongside V5. `6c84969` adds a pinned model arm;
> `4ad24e3` freezes matching first inputs (only model alias differs). [Actual base
> result](examples/passage-reader-base/README.md): QLoRA supported after four calls;
> AWQ incomplete with zero source reads because mixed code/SUBMIT output was
> cleaned into premature abstentions; target gap incomplete after two reads and
> abstention without a gap report. Twelve base requests, 34,646 reported tokens,
> all `envoy-base-only`, saved `adapter_applied: false`; no new V5 calls or training.
> V5's saved repaired run completed two answerable cases versus base's one, with
> nine calls / 26,302 tokens. Both failed the gap procedure. This small exposed
> diagnostic suggests better V5 protocol compliance here, not a validated general
> fine-tuning or domain-knowledge gain. Base AWQ semantic reading was not assessed.
> Preserve raw outputs alongside the policy-cleaned actions; they differ on AWQ.
> No production harness/gap/training/reward change followed this comparison.
> Focused 46 passed. Initial full run had one Docker startup failure before any
> scripted call (1928 passed) while another rehearsal ran; cause unproven. Without
> code/timeout changes, standalone Docker checks passed 4/4 and standalone full
> suite passed 1929 in 152.10 seconds. Fresh 00:54:54 UTC reads verified workers
> 0/0, no workers/jobs, full original configuration restored; six owned Docker
> containers absent. Eleven protected and 507 prior artifact hashes unchanged.
> `out/passage-reader-three-case-BASE-1` is consumed; no further paid attempt is
> included. Training remains an option after stable interfaces, reviewed targets
> and separate evaluation; it is not declared worthless or demonstrated necessary.

> **October 21: repaired-harness live rerun completed.** User said “ok rerun it”.
> One new allowance and `out/passage-reader-three-case-RERUN-1` marker were consumed;
> the first attempt remains frozen. [Actual rerun](examples/passage-reader-rerun-1/README.md):
> QLoRA 3 calls and AWQ 2 calls produced source-supported requested facts under
> implementing-assistant review. PagedAttention's missing target remained incomplete
> after 4 calls: two successful reads, premature abstention, then a gap report at
> the last step. No target p95 was invented, but no substantive partial findings
> were returned and the overall acceptance rule failed. All live reads used literal
> lengths, so arithmetic normalization itself was not exercised. Exact read examples
> also changed; no causal isolation or model-weight improvement is claimed.
> Nine actual requests, 25,784 input + 518 output = 26,302 reported tokens, no
> inference resubmissions. Full suite: 1925 passed in 160.61 seconds. Fresh reads
> at 23:19:41 UTC verified full endpoint configuration restored, workers 0/0,
> no workers/pending jobs; all six owned Docker containers were absent. No new
> training/reward work, reserved evaluation access or further paid attempt is
> authorized by this rerun. Keep the remaining gap-protocol issue separate from
> factual extraction ability. Both original product gates remain unproven.

> **October 21: real Qwen passage trial, partial success.** The combined
> experiment MCP now has an opt-in `read_experiment_passages` tool; default mode
> still makes no model calls. [Configure it explicitly](docs/PASSAGE_READER.md).
> The [three-case live attempt](examples/passage-reader-three-case/README.md)
> used ten Qwen v5 requests: AWQ returned all requested benchmark conditions
> with sufficient source support under assistant review; QLoRA and the missing
> target measurement remained incomplete. The overall rule was not met. A
> literal-only guard rejected valid read-length subtraction; `9161aab` repairs
> this with bounded AST-data normalization and exact read-call hints. That fix
> has not been live retested. Final checks: 1925 tests passed. GPU shutdown and
> full configuration restoration were verified; no allowance was reset or paid
> retry made. Preserve the frozen pre-fix plan/prompt and the negative cases.
> No general researcher, productivity advantage, human-reviewed model gain or RL
> result is established. Any new paid experiment must have its own scoped plan.

> **October 21: personal research-to-result MCP loop completed.** Separate from
> the published QASPER study, [experiment mode](docs/EXPERIMENT_MODE.md) now has a
> one-command real-stdio example connecting earlier paper research to recorded
> serving measurements, a retrospective frozen analysis and restart/resume.
> Run `python scripts/demo_experiment_mode.py --output out/my-experiment-demo`
> with a new directory. It makes no source/model calls or GPU launches. The
> [completed example](examples/experiment-mode-complete/README.md) preserves
> matching responses and same-engine overlap while keeping the speed comparison
> inconclusive. Do not reopen infrastructure or training merely to finish this
> local workflow. Qwen's separate last full research smoke still produced no
> verified evidence; serving success does not prove useful paper reading,
> fine-tuning improvement, or advantage over ordinary assistant research. New
> GPU/paid runs or RL require separately scoped work. Earlier study artifacts
> and protected uncommitted files remain unchanged.

> **October 9: QASPER attribution and official scoring added.** The visible name
> is **QASPER Agent Studio — a Qwen Envoy case study**, with Ai2/Dasigi et al.
> attribution and no claim that the questions are original. Existing repository,
> package, model and artifact identifiers are unchanged. Read
> `docs/QASPER_AGENT_STUDY.md` and `release/qasper-agent-study/README.md`.
>
> Official Answer F1, scored offline with the pinned unmodified evaluator and
> all 86 original annotations, is **19.9154 base / 30.6873 v5** on the selected
> 40 questions. The fixed answerable group worsens **29.8307 → 21.3746**;
> the unanswerable group rises **10 → 40**. Missing predictions remain zero,
> literal refusal paraphrases are not rewritten, and original inference/review
> JSON is unchanged. This is not the full QASPER leaderboard or independent
> support review. The report documents the short-target construction and
> 38/40 three-action v5 episodes as a hypothesis, not a causal training finding.
>
> The first NVIDIA/Nebius reference attempt stopped at its $2 local
> estimated-cost guard: **$1.947862**, 366 API requests, 38 terminal episodes,
> question 39 interrupted after 11 actions, question 40 unattempted. The user
> then authorized a **$25 total** cap. The continuation restarted question 39
> at step 1 because the live REPL/conversation state could not be restored;
> question 40 received its first attempt. The original partial result,
> manifest, and usage record remain unchanged under
> `release/qasper-agent-study/nebius-run/`. The amendment was recorded before
> further inference in `release/qasper-agent-study/CONTINUATION_2026_10_09.md`.
>
> The derived complete 40-row result is in
> `release/qasper-agent-study/nebius-amended-run/`: first 38 original terminal
> rows plus two continuation rows, with lineage in its manifest. Question 39
> used 15 steps without a submission; question 40 submitted `85%` after two
> steps and scored zero F1. **28/40 submitted; 12/40 missing.** The pinned
> official QASPER evaluator gives **19.02565047926817% Answer F1** in
> `release/qasper-agent-study/nemotron-official-score.json`. Cumulative usage
> is **383 API requests and $2.035422** estimated at catalog rates, not an
> invoice. The $25 cap was a local request guard. No independent answer-support
> review was performed and this is not a controlled Qwen–Nemotron comparison.
>
> Source/scoring changes are in `e99b165`; Studio changes are in `815ea5f`.
> Every recorded implementation hash from the hosted attempt matches `e99b165`
> (the run started from a dirty tree over `e82e33b`; see EXPERIMENT.md).
> The viewer adds official F1 alongside the existing provisional support scores.
> It supports separately labeled supplemental runs without merging their
> reviews. The original four Qwen rows remain alongside the amended Nemotron
> row; logprob smokes stay unscored. The original first 38 Nemotron outcomes
> and the amended final two outcomes are traceable to their source records.
>
> Earlier validation, before the continuation: **788 tests passed** after the
> partial-run viewer change; 21 focused tests and 32 release/official-score
> tests also passed. Both historical release hash sets and the standard-library
> case-study check passed. Official Qwen scores reproduced byte-identically
> with isolated standard-library Python. Opening saved runs makes no model
> calls. Hackathon live judge access, public
> video, sponsor feedback and actual submission remain outside this completed
> case-study packaging work.

> **October 9: public technical case study shipped.** The user chose to finish
> Qwen-Envoy as an inspectable training/evaluation project. The saved Studio is
> live at <https://jasonlingg.github.io/Qwen-Envoy/>. Commit `b53d70b` publishes
> `docs/TECHNICAL_CASE_STUDY.md`, original September 19 training records under
> `release/qwen-v5-training/`, the standard-library-only
> `python scripts/reproduce_case_study.py` checker, and GitHub Actions for
> artifact verification and Pages deployment. Public demo links and release
> status are updated in the following documentation commit.
>
> Verification: **755 tests passed** offline; the 14 new corruption/binding
> tests also passed in a clean Python 3.11 environment with only pytest.
> GitHub's verification and Pages workflows passed for `b53d70b`. Hosted HTML
> byte-matches the packaged snapshot. Browser QA confirmed all four run entries
> and exercised question navigation, full saved traces, JSON export, and token-likelihood
> expansion with no JavaScript errors or model/API requests. Six training
> archive checksums and three original data bindings passed. The historical
> training manifest still has a blank Git commit and a script hash that differs
> from the current script; exact original weight reproduction is not claimed.
>
> The saved inference/review JSON is unchanged: base/v5 provisional passes
> 5/40 versus 15/40, answerable passes 3/20 versus 3/20, recorded-error episodes
> 22/40 versus 1/40. Independent source/answer review remains unfinished. No
> trained-model promotion, new training, paid inference, or Runpod launch was
> performed. The public viewer is read-only and needs no credentials. It is
> not a live hosted paper-agent service or a submitted hackathon entry; live
> judge-access instructions, video, sponsor feedback, and submission remain.
> Preserve this scope rather than starting another product pivot.

> **October 8: release scope frozen and accumulated work committed.** The user
> chose to ship the current text-only EnvoyBench Studio, with simple runs,
> scores, answers, and complete tool traces. GTLM, diagram generation, new
> partial-paper tasks, and further training remain outside this release.
> README now leads with the packaged demo and September 30 provisional results,
> explains PDF/table/image limitations, and separates historical prototypes.
>
> The user explicitly authorized committing all pending project work. Changes
> are grouped into runtime, training/data, research tools, Studio/release, and
> documentation commits. Credentials, ignored corpora, checkpoints, and local
> `out/` artifacts remain outside Git; no push or paid inference was performed.
> The offline full suite passed **741 tests** after restoring the existing
> web-evidence navigation link on the three Studio pages. The standalone HTML
> has the same link and unchanged embedded run data; all 22 release checksums
> pass. Existing model-assisted scores and training artifacts are unchanged.

> **October 6: Studio finishing pass completed locally.** The user wants a simple
> runs list, scores, and single-run transcripts, not another product pivot or
> comparison dashboard. Launch `python scripts/launch_studio.py`; optionally use
> `--live-nebius` with the local `.env` key and Docker. Port 8765 serves the four
> packaged Qwen runs. `release/envoybench-v0.1/studio.html` is standalone and
> embeds all 562 September 30 turns; the two-question October 3 token diagnostic
> is separate and unscored. The 40-question model-assisted grades are unchanged.
> No demonstrated held-out supported-answer improvement is claimed.
>
> Four bounded live NVIDIA/Nebius attempts are documented in
> `release/envoybench-v0.1/LIVE_VALIDATION.md`: three Lightning failures and one
> Ultra completed protocol with a correct answer but incomplete citations.
> Do not call it a semantic-support pass or reliability estimate. All local
> original runs are in `out/envoybench/paper-workspace/runs/`; the release includes
> plans, usage, hashes, and an assistant source-review summary. No more live tests
> are needed for this finish pass. No Runpod GPU or training was launched.
>
> Focused checks: 73 policy/investigator/paper API tests plus 9 Studio/export/
> verifier tests passed; frozen artifact verification passed. UI navigation is
> Runs → Questions → Trace, with optional per-turn token likelihood. New paid
> paper runs require an explicit Run click. Demo script and Devpost draft are
> in the release directory. Public hosting, video, sponsor feedback, and actual
> submission remain pending; local completion is not hackathon submission.
> Preserve the large preexisting dirty worktree and do not blanket-commit it.

> **October 1: Studio paper workflow and ML recipe.** User authorized completing
> the live-paper integration and packaging the trained-model case study. Studio
> adds `/papers` (web link or upload/read/question/run/history/evidence/export) and `/model`
> (observed v5 QLoRA recipe, pinned identities, current limitations). See
> `benchmarks/envoybench/PAPER_READER.md`. The new API uses the existing
> Docker-only public-paper investigator; one bounded job at a time, persisted
> progress, immutable source checks, no benchmark grades for uploads. Enable it
> with `--paper-models benchmarks/envoybench/paper-models.example.json` and an
> already served v5 endpoint. No endpoint is configured in the current shell or
> `.env`; port 8000 is not serving a model. Do not describe scripted integration
> tests as a real v5 answer. No paid compute, new training, or prefix-cache
> measurement was run in this change. The original 40-question scores remain
> unchanged. Follow the reader guide for focused verification and connection.

> **September 30 current priority: EnvoyBench case study and Studio.** The user
> approved finishing the narrow base/v5 comparison and improving its diagnostics,
> rather than starting more training or another product pivot. See
> [`benchmarks/envoybench/COMPARISON_2026_09_30.md`](benchmarks/envoybench/COMPARISON_2026_09_30.md)
> and the benchmark README for current commands; the older product states below
> remain historical. The completed 40-question QASPER-derived known-paper run is
> in `out/envoybench/provisional-full-20260930/`. Separate Codex-assisted blind
> grading gives base 5/40 and v5 15/40, but **both pass only 3/20 answerable
> questions**. Source references remain unvalidated. Sonnet grading stopped at
> eight questions when API credits ran out. There is no human-reviewed promotion
> or demonstrated substantive-answer improvement.
>
> Current improvements: analytical always-refuse expectation 20/40 with zero
> useful answers; separately bound AI response-behavior labels (v5 abstains on
> eight answerable-labeled questions); three curated paired correctness/support
> diagnoses; automatic-v2 returned-error/timing fixes; future token telemetry;
> and a standalone read-only Studio export. Version-2 errors affect 22/40 base
> episodes and 1/40 v5 episodes. Base timing covers only 29/40; missing duration
> and usage are not reconstructed. Original run and quality-grade files were
> preserved. Relevant checks: **115 tests passed**, Ruff clean, actual desktop
> and mobile browser QA passed.
>
> The deterministic within-known-paper retrieval comparator has 40 prepared
> packets / 80 planned requests at
> `out/envoybench/retrieval-baseline-prepared-20260930/`, **not executed**. Its
> pre-run hypothesis and decision rule ask whether the agent loop earns its
> complexity; preparation alone is not a result. An eight-question independent
> calibration packet at `out/envoybench/calibration-20260930/review-packet.json`
> is entirely blank. Do not fill human judgments automatically. The improved
> snapshot is `out/envoybench/studio-snapshot-improved-20260930.html`; the local
> viewer uses port 8766. The temporary A40 was terminated after the original run;
> the subsequent improvements made no new GPU jobs or external model API calls.
> Keep remaining inference and independent-review work explicit; do not claim
> the prepared retrieval baseline has been measured.

> **September 28 scope correction:** the user has no substantive existing
> Obsidian vault. The first new model task is multi-step Qwen investigation of
> a frozen public AI-paper library; Nemotron/the reviewed writer creates the
> Obsidian notes. Do not keep blocking the public-library baseline on the
> one-note personal vault or describe a public-paper result as private-vault
> transfer. See
> [`docs/QWEN_RESEARCH_LIBRARY_CORPUS_AMENDMENT_20260928.md`](docs/QWEN_RESEARCH_LIBRARY_CORPUS_AMENDMENT_20260928.md).

> **September 28 implementation update:** the original-paper library now has a
> separate read-only MCP server in `scripts/research_library_mcp.py`. A real
> stdio client searched the W39 paper snapshot, opened an exact source span,
> and confirmed that unconfigured Qwen made zero model requests. The product
> Qwen investigator has an explicit `public_papers` prompt; no live Qwen call
> has been made on this path. The executable `search()` default was repaired
> from whole-document term counting to versioned passage BM25 before any new
> model comparison: first-query top-five source-ID coverage on four provisional
> development questions rose from 2/7 to 5/7 (and 7/7 at top eight). This is retrieval infrastructure,
> not Qwen improvement. See
> [`docs/RESEARCH_LIBRARY_DEV_RETRIEVAL_DIAGNOSTIC_20260928.md`](docs/RESEARCH_LIBRARY_DEV_RETRIEVAL_DIAGNOSTIC_20260928.md).
> Twelve held-out question drafts are structurally checked but remain blind,
> unreviewed, and unused for model selection. The W39 Obsidian seed is still an
> unpublished agent-authored draft awaiting a person's claim/relevance review.
> The development-only product-path runner in
> `scripts/run_research_library_qwen_dev.py` has a current no-model dry run at
> `out/research/research-library-dev-qwen-dryrun-v2-20260928/` with a pending
> reference-review template; live mode rejects that pending record before any
> Docker or model action. No base/v5 paper-library model run has occurred.
> A September 28 Runpod MCP read found zero pods and zero endpoints, plus one
> 50 GB standard network volume in EU-RO-1. A 32 GB RTX PRO 4500 was listed
> there at $0.72/hour with medium availability; this is a timestamped price
> and stock observation, not a provisioned resource. Account credit balance
> was not exposed by these MCP reads.

**Last updated:** 2026-10-08 (America/Toronto)  
**Repository:** `/Users/jasonling/Documents/GitHub/rlm-explorer`  
**Purpose:** working memory for a new Codex/Claude/other engineering agent. Read `AGENTS.md`,
`docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md`, and this file before acting.

## Current September 27 product state

- The chat-first learning-memory demo has a separate ten-note **synthetic** linked vault at
  `data/product_memory/linked_demo_vault`. `python scripts/launch_sample_demo.py` serves it on
  localhost in evidence-only mode, and normal Ctrl-C/SIGTERM cleanup removes its temporary copy.
  Its one-command HTTP smoke returned exact passages and linked notes without remote model calls.
- The browser now has a source-triggered revision loop: a captured **Result** gets a local
  prior-decision candidate, exact passages from both notes, and a dated timeline. The user can
  inspect both, investigate in chat, replace a local correction scaffold, and explicitly approve
  an append-only Obsidian note that supersedes the old decision. A normal source/plan gets no
  correction scaffold. Candidate retrieval uses links and word overlap, not semantic conflict
  detection. The optional **Draft with Nemotron** button makes one explicit model request for a
  candidate result and still requires human review. Browser flow and focused mocked-API tests
  pass; no paid model call was made for this new draft feature. See
  `docs/PERSONAL_MEMORY_LOCAL_DEMO.md`.
- A separate **real-report-derived** Qwen learning thread can be launched with
  `python scripts/launch_qwen_experiment_demo.py`. Its three linked Obsidian notes are generated
  from `reports/qasper-scale-sft-2026-09-25.json` with source hashes: retrospectively inferred
  continuation strategy, measured 15/10/4-of-40 QASPER result, and the report's rejection
  decision. They explicitly disclose model-assisted review and that this is neither base-vs-SFT
  nor personal-vault evidence. The UI now lists existing result notes for later impact review.
  An isolated integration test checks local impact, evidence-only chat, test-only approval, later
  recall, and read-only MCP source access. No trained Qwen or Nemotron call occurs in this demo;
  no actual user-approved revision is bundled. See `docs/QWEN_EXPERIMENT_MEMORY_DEMO.md`.
- Two separately triggered **live Nemotron** turns over this public thread are recorded in
  `reports/qwen_experiment_nemotron_smoke_20260927.json` and
  `reports/qwen_experiment_nemotron_smoke_20260928.json`. Both are partial after Codex source
  review: the first confused a reported rule with an inferred strategy, and the second grouped
  citations at paragraph end and omitted a factual sentence from its claims. The source note was
  clarified, and `NemotronChatClient.answer` now rejects grouped end citations across multiple
  factual sentences; a local replay of the saved second response was rejected. This checks
  citation placement only, not semantic support. No trained Qwen checkpoint was run.
- A live Nebius/Nemotron synthetic chat path answered one dated plan/correction question and
  correctly said another planned Qwen comparison had **no measured outcome**. Reports are in
  `reports/nebius_linked_demo_smoke_20260927_v3.json` and
  `reports/nebius_linked_demo_no_outcome_20260927.json`. Exact spans were checked; semantic
  support still awaits independent human review. Nemotron sometimes groups citations at the end
  of an answer. A stricter prompt caused an evidence-only fallback and was reverted.
- `scripts/personal_memory_mcp.py` is a working local, read-only stdio MCP server over a frozen
  vault snapshot. It offers `search_memory` and review-bound `get_memory_source`; it does not
  invoke Qwen or expose a weekly-digest tool. The real stdio client test passes.
- `scripts/weekly_radar_candidates.py` now persists two-topic, version-aware candidate stores
  using arXiv submission search and a bounded latest-update scan. Live September 21–27 stores in
  ignored `out/research/weekly-radar-candidates/` hold 54 tool-agent and 11 small-model-training
  candidates; these are unranked metadata, not a recall benchmark. See
  `docs/WEEKLY_RADAR_CANDIDATES.md` for coverage limits.
- `scripts/weekly_digest.py` validates a frozen paper snapshot, renders linked paper notes and a
  weekly Obsidian digest, and publishes only a selection explicitly marked `human_reviewed` with
  `--confirm`. The historical 2025-W32 and current 2026-W39 selections under `data/research/`
  are **agent-authored abstract-based drafts only**. Their rendered previews are in ignored
  `out/research/weekly-digest-*-agent-draft-preview.txt`; neither was written to the vault.
  Both preview batches have zero unresolved links. See `docs/WEEKLY_DIGEST_DEMO.md`.
- The paired personal-vault Qwen base/v5 evaluation remains **unrun**. The pinned v2 questions,
  source bundle, and runner are prepared (see `docs/PERSONAL_MEMORY_READINESS_EVAL.md`), but the
  Runpod CLI has no API credential in this workspace. **The Runpod OAuth MCP connection does
  work through Claude Code without an API key**; direct Runpod tools are not exposed in this
  Codex chat. On September 27, a read-only MCP call showed all 24 pods `EXITED`, and a START
  attempt on cached-weight pod `l9pxr8ptq4krop` failed with Runpod's
  `There are not enough free GPUs on the host machine to start this pod.` The pod remained
  `EXITED`; no new pod or GPU runtime was created. A separate Nebius-hosted Qwen3-30B
  Docker-tool smoke ran on two synthetic questions:
  the original prompt made no tool calls, a first-search instruction enabled tool calls, the
  temporal answer had incomplete cited support, and the absent-outcome answer mistook a plan for
  a measured result. See `reports/nebius_qwen30b_memory_smoke_review_20260927.md`. This is not
  the pinned 8B base/v5 comparison. Do not describe CPU lexical results, Nemotron smoke tests,
  or the hosted 30B smoke as Qwen checkpoint improvement.
- Focused product/radar/MCP/eval checks: **80 passed**, including the launcher SIGTERM test;
  Ruff passed on the touched product files. No
  training or reward code was changed in this product pass.

## One-paragraph state

Envoy is a small Qwen research worker that writes Python in a persistent REPL to search, read, and
extract evidence from AI papers or an Obsidian snapshot. A larger model such as Claude or GPT is the
orchestrator. The latest experiment continued a Qwen3-8B rank-4 QLoRA adapter on 995 QASPER-derived
code-execution trajectories. Its epoch-1 and epoch-2 checkpoints were recovered and evaluated on
the same reserved 40-question set as the starting adapter. Identity-blind model-assisted semantic
review found 15/40, 10/40, and 4/40 passes respectively; neither continuation checkpoint was
promoted. All 120 episodes are backed up locally, and the migrated RunPod pod is stopped. The next
product experiment is a reviewed development baseline on the frozen public-paper library,
followed by a separate held-out comparison only after a teachable failure is identified. Another
long training run is not yet justified. The RunPod recovery and pending-evaluation
sections below are historical records;
do not act on their old operational instructions.

Final artifacts: `out/research/qasper-scale-sft-v1/eval-20260925-decision.json`,
`out/research/qasper-scale-sft-v1/eval-20260925-failure-analysis.json`, and
`reports/qasper-scale-sft-2026-09-25.json`. A no-GPU demo with actual recorded Qwen actions is
documented in `docs/DEMO_AND_EVIDENCE.md`.

## Product goal and architectural boundary

The product is a weekly AI-research radar and personal second brain:

1. Discover and freeze relevant new AI papers.
2. Let Qwen investigate a bounded research question through executable Python tools.
3. Return a compact evidence packet with exact source spans and limitations.
4. Save durable notes and weekly digests in Obsidian.
5. Expose high-level read-only research operations through MCP to Claude, GPT, or another host.

Qwen is not meant to replace the larger reasoning model or serve as a general chat interface. Its
job is the cheap, repeatable middle layer between the host model and the paper library. It still
needs enough local judgment to reformulate a search, inspect a paper, recover from a weak result,
and abstain when evidence is insufficient.

Primary product documents:

- `docs/CODE_EXECUTION_SECOND_BRAIN.md`
- `docs/WEEKLY_RESEARCH_RADAR.md`
- `docs/QASPER_TARGETED_SFT.md`
- `docs/QASPER_FAILURE_ATTRIBUTION.md`
- `docs/QASPER_RL_EXPERIMENT.md`

## Current training data

The latest dataset is `out/research/qasper-scale-sft-v1/` (ignored by Git).

- Total conversations: **995**
- Answerable: **955**
- Previously reviewed abstentions: **40**
- Train: **796 conversations / 472 papers**
- Validation: **199 conversations / 122 papers**
- Train SHA-256: `9db1059ccaa8065dd23c0d337d3d18aa2994be75cf4fd56a6501b89036163a04`
- Validation SHA-256: `8bc3b0fad76ff0579ca1610609b7535ee1c12fc96ea97faa19ec9a67a2066dfc`
- Dataset manifest SHA-256: `96a0a9271dda5f003d682fb6e7b1067a2cbe50cfc5379e3c23646ce05effe201`
- Seed: `20260921`

The 955 answerable trajectories were constructed deterministically from QASPER train annotations.
Every trajectory was replayed mechanically; train/validation splits are paper-disjoint; reserved
papers do not overlap; identical question/action traces are deduplicated; and five deliberate
corruptions were rejected. Twenty answerable examples received a source-visible semantic spot
check by Codex and all passed. This was **model-assisted review, not independent human review**.

Audit: `out/research/qasper-annotation-scale-v1/audit-final-reviewed.json`  
Audit SHA-256: `6eaeb77293dc037c75bddd92b9d0c009e47cb4ff147d6acead1c3159fe78bc3c`

Data-building code added or changed during this work includes:

- `scripts/prepare_qasper_rl.py` (`--answerability` filter)
- `scripts/build_qasper_annotation_sft.py` (paper-level splitting and deterministic deduplication)
- `scripts/verify_qasper_synthetic_data.py` (replay, split, identity, and mutation checks)
- `scripts/assemble_qasper_scale_sft.py` (combines answerable data with reviewed abstentions)
- `tests/test_qasper_annotation_builder.py`

## Latest Qwen3-8B SFT run

The run continued from the public adapter:

`jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1`

The public repository corresponds to the earlier selected `checkpoint-150`; its
`adapter_model.safetensors` SHA-256 is:

`70143ec6933494efe60cab726e792fcc182965fd13a8561453202b41bba9bfb4`

Latest training configuration:

- Base: `Qwen/Qwen3-8B`
- Base revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Method: 4-bit NF4 QLoRA continuation
- LoRA: rank 4, alpha 8, dropout 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Format: one next-action example per assistant action, with exact inference-prefix verification
- Loss: assistant action tokens only; retrieved/tool-output text is context and receives no SFT loss
- Native Qwen3 thinking: disabled
- Maximum sequence length: 8,192 tokens
- Longest train example: 5,641 tokens
- Train actions: 3,711; validation actions: 924
- Learning rate: `5e-5`, cosine schedule, 5% warmup
- Epochs: 2
- Batch size: 1; gradient accumulation: 4
- Seed: `20260921`
- Hardware: secure RunPod NVIDIA RTX PRO 4500 Blackwell, 32 GB VRAM
- Runtime: 3h 19m 54s for 1,856 optimizer updates

The one-step smoke test succeeded before the full run:

- Smoke validation loss: `0.1317699999`
- Smoke validation token accuracy: `0.9675968`
- It completed a real update, full validation, and saved a reloadable adapter.

Full-run final metrics:

- Final validation loss: `0.0610487126`
- Final validation token accuracy: `0.9810336`
- Final train loss: `0.0538664`
- Final and checkpoint-1856 adapters are byte-identical.

These metrics show that the model learned the trajectory distribution. They do **not** establish
better autonomous research behavior.

Remote run directory:

`/workspace/envoy/out/research/qasper-scale-sft-v1/run-20260921-epoch-checkpoints`

Saved candidates and checksums:

- Epoch 1: `checkpoint-928/adapter_model.safetensors`
  - SHA-256: `7c259d36b196a7a9541d152be9f192ebb600ad1fd3845ee9ae07847c71f40b1c`
- Epoch 2: `checkpoint-1856/adapter_model.safetensors`
  - SHA-256: `6ae2fdcd33ddc18c9fd2ed04011953106268c6b735f5ce7211ba7603e7175d66`
- Final: `final/adapter_model.safetensors`
  - SHA-256: `6ae2fdcd33ddc18c9fd2ed04011953106268c6b735f5ce7211ba7603e7175d66`

The epoch-1 validation metrics still need to be extracted from its `trainer_state.json` or the full
remote training log after the pod becomes accessible.

## Historical RunPod recovery (resolved September 25)

The operational notes below describe the pre-migration state. The adapters and eval transcripts
were recovered, and target pod `1t43mrc8ko4ajp` is **EXITED**. No recovery action is pending.

Pod ID: `nhbh2q7mcnfcdi`  
Name: `envoy-qasper-scale-sft-secure-20260921`  
Cloud: secure  
Original GPU: RTX PRO 4500 Blackwell  
Rate when running: `$0.72/hour`  
Persistent mount: `/workspace`, 40 GB  
Status on 2026-09-24: **EXITED / not billing**

The pod cannot currently restart because its local volume is tied to a host whose GPU was rented by
someone else. Several API start attempts returned: `There are not enough free GPUs on the host
machine to start this pod.` At the latest check, the general RTX PRO 4500 pool had low stock and the
A40 pool was out of stock, but capacity changes frequently.

The available RunPod MCP can start, stop, inspect, create, and delete pods, but it does not expose
the console's **Automatically migrate your Pod data** action. The user's logged-in RunPod tab was
not available to browser control. The user must either:

1. click **Automatically migrate your Pod data** in the RunPod console; or
2. start the stopped pod with CPUs, allow the agent to SSH in and back up/transfer the adapters,
   then create an available GPU pod.

Do not terminate this pod. The latest two adapters have not been downloaded locally or uploaded to
Hugging Face and currently exist only on its persistent disk.

Once the pod is accessible, immediately back up both `checkpoint-928` and `checkpoint-1856` before
another stop. The user previously authorized saving the trained weights to their Hugging Face
account, but behavioral evaluation should select the promoted checkpoint. Back up both candidates
even if only one is ultimately published.

## Completed behavioral evaluation (original plan)

The controlled comparison described below was completed on September 25. The starting adapter
passed 15/40 semantic reviews, epoch 1 passed 10/40, and epoch 2 passed 4/40. Both continuation
checkpoints failed the preregistered promotion rule. The following text preserves the original
plan for provenance; it is not a to-do list.

The immediate experiment is a controlled three-way comparison:

1. Starting public adapter (`qasper-targeted-sft-v1`, checkpoint-150)
2. New epoch-1 adapter (`checkpoint-928`)
3. New epoch-2 adapter (`checkpoint-1856` / `final`)

Use the exact same question IDs, corpus, system prompt, tool interface, maximum steps, decoding
settings, and seed. The primary question is whether either continuation checkpoint improves
supported-answer quality over the starting adapter. Training loss and token accuracy are only
diagnostics.

Reserved confirmation data:

- Benchmark: `out/research/qasper-code-confirm-v1/benchmark.json`
- Benchmark questions: **40**
- Benchmark SHA-256: `23c98a28bfe2d4ed9f3e44b74f0f28f6ea7d244b265048f13ccbc4f45d6f2a7e`
- Corpus: `out/research/qasper-code-confirm-v1/corpus/`
- Manifest: `out/research/qasper-code-confirm-v1/manifest.json`
- Manifest SHA-256: `86c9e7a67056ef3d79d87bfb5fdc11780c4b36e03c67a3e734cb8a44add5eb55`
- Source split: QASPER validation; latest scale training uses QASPER train.

This confirmation set was reserved for the latest SFT comparison. Keep it distinct from the
already-consumed GRPO confirmation under `out/research/qasper-rl-v3/confirmation/`.

Before evaluation, freeze a machine-readable plan with model/checkpoint hashes, exact 40 IDs,
corpus hash, prompt hash, decoding settings, maximum steps, seed, GPU, and code revision. Run an
identity-blind semantic review of all outputs. Report at least:

- semantic pass / partial / fail;
- supported-answer rate;
- answerability/false-refusal behavior;
- citation and exact-span validity;
- required-paper recall;
- valid submissions;
- syntax/runtime errors and repeated actions;
- paired per-question changes; and
- latency, generated tokens, and estimated GPU cost.

The earlier project gate was at least 28/40 semantic passes plus a paired improvement over the
starting adapter. Do not silently relax that gate after seeing outputs. If a new gate is chosen,
record it before inference.

The training pod bundle contains `src/`, `scripts/train_sft.py`, the Python environment, base-model
cache, and training data. It likely does **not** contain the local confirmation corpus or the full
evaluation scripts. After migration/CPU recovery, upload an evaluation bundle containing the
needed current code plus `out/research/qasper-code-confirm-v1/` before running inference.

## Important prior evidence

Do not rewrite history around the newest run:

- On 50 common MuSiQue dev questions, Qwen2.5-7B base scored `0.158`, SFT scored `0.176`, and the
  available GRPO adapter scored `0.172`. The two old GRPO Hub repositories contain byte-identical
  adapters, so they do not show checkpoint progression.
- Base Qwen3-8B on the 10-question AI-paper pilot scored `0.445` outcome reward with citation
  precision/recall `0.85/0.95`. Retrieval was fairly strong; answer discipline, instruction
  retention, semantic accuracy, and abstention were the main failures.
- The earlier small targeted QASPER SFT run improved development behavior only slightly and failed
  its preregistered gate. The selected epoch-1 result was 11 pass / 5 partial / 9 fail versus the
  prior adapter's 10 / 4 / 11, with only 1/15 target failures rescued.
- Centered GRPO improved an already-consumed development set but did not replicate on its fresh
  paper-disjoint confirmation: automatic reward changed from SFT `0.3462` to GRPO `0.3531`, while
  blind semantic review favored SFT 13 pass / 2 partial / 25 fail versus GRPO 12 / 3 / 25.

The latest scale-SFT run was motivated by the conclusion that tens of trajectories were too small
and that reliable annotation-derived trajectories could be generated programmatically at scale.
Its behavioral result is still unknown.

## Reasoning / chain-of-thought decision

Current Qwen training is **not chain-of-thought training**. It learns visible, executable tool-use
trajectories: write Python, observe tool output, choose the next action, then submit a cited answer.
Qwen3 native thinking is disabled because an earlier baseline spent its token budget inside
`<think>` and never reached executable code.

Search-R1 is relevant precedent for a later experiment: it uses RL to interleave bounded reasoning
and search while masking retrieved tokens from policy and KL losses. If the new SFT behavioral
evaluation still shows planning failures, a legitimate follow-up is a controlled GRPO condition
with a short reasoning budget (for example, at most 96 tokens per action), explicit reasoning
delimiters, and code executed separately from the reasoning text.

Do not implement this as merely `enable_thinking=True`. Required changes would include:

- parser/harness support for bounded reasoning plus executable code;
- a hard reasoning-token cap and action budget;
- regression tests proving injected tool outputs receive zero policy and KL loss;
- unchanged no-reasoning control;
- a small development diagnostic before a costly run; and
- evaluation based on supported answers, not whether thoughts sound plausible.

This is optional. Because a larger host model performs global reasoning, Qwen only needs local
planning if the evaluation shows it chooses poor searches or gives up despite available evidence.

## Working-tree safety

The repository is heavily dirty with user and prior-agent work. Do not use blanket `git add`,
`git reset`, or checkout commands. Preserve unrelated changes. No commit was made for the latest
dataset-generation and training-preparation work.

Especially preserve the GRPO fixes and related documentation called out by `AGENTS.md`. Before a
commit, inspect `git status --short` and stage only explicitly reviewed paths.

This handoff file itself is newly added and is not committed unless a later agent/user explicitly
commits it.

## Recommended next actions

1. Ask the user to complete RunPod's **Automatically migrate your Pod data** action, or CPU-start
   the old pod for data recovery.
2. Verify the migrated/recovered checkpoint hashes against the values above.
3. Back up both new adapters locally or to Hugging Face before stopping or terminating anything.
4. Upload the current evaluation code and frozen 40-question confirmation corpus.
5. Record the evaluation manifest and decision rule before inference.
6. Run the starting adapter, checkpoint-928, and checkpoint-1856 under identical settings.
7. Perform blind semantic review, reveal identities, and report paired changes.
8. Stop GPU billing immediately after artifacts and logs are safely copied.
9. Decide from failure attribution whether the next intervention is data, harness/verifier work,
   bounded-reasoning GRPO, or no further training.

## Suggested prompt for the next agent

> Read `AGENTS.md`, `CODEX_HANDOFF.md`, `docs/GPT_ASTRA_IMPROVEMENT_BRIEF.md`, and
> `docs/CODE_EXECUTION_SECOND_BRAIN.md`. Preserve the dirty worktree. Recover or migrate RunPod pod
> `nhbh2q7mcnfcdi`, back up both scale-SFT adapters, then run the preregistered three-way behavioral
> evaluation on `out/research/qasper-code-confirm-v1`. Do not claim improvement from training loss.
> Keep the Qwen worker as a code-execution evidence subagent for a larger Claude/GPT host, and stop
> GPU billing as soon as all artifacts are safe.
