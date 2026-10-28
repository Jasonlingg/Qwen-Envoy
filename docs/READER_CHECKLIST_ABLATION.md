# Reader checklist experiment — prepared, not live-tested

October 25, 2026. The user authorized preparing a small hint/scaffolding test before
another RL run. This is a separate inference-only development experiment. It does
not change the frozen reader, reward, training protocol, checkpoint or held-out
evaluation. No new paid authorization, GPU rental, model generation, optimizer
update or quality result is recorded here.

Hypothesis: a generic execution checklist increases healthy mixed-reward groups
per reserved token while preserving complete source-supported answer quality.
This is a prerequisite diagnostic for deciding whether to test scaffolded RL;
it cannot establish faster convergence. Research on [hint guidance](https://arxiv.org/abs/2510.09388)
and [learning to hint](https://arxiv.org/abs/2604.00698) motivates the hypothesis,
but this experiment does not implement their full algorithms. If later training
uses hints, transfer to deployment without hints needs a separate evaluation.

## Fixed comparison

Use original pinned V5, not the interrupted pilot's step-three adapter. Select the
first six eligible training questions after the existing Python Random(42)
shuffle, before any new outputs. These are previously exposed development
questions, not a held-out test. Use four attempts per question per arm: 48 Docker
episodes, with no optimizer or TrainSink. References and reviewed aliases stay
in controller-side scoring; they are never included in model messages.

Control is the exact existing effective prompt, SHA-256
`1f752247467187689092c0366ac9923b077cd7c6df0ea6969bb21acabb994e16`.
Treatment appends only this checklist:

```text
Before acting, use this checklist internally; output only the allowed action.
1. Identify every fact requested by the question.
2. Inspect the supplied passages and locate evidence for those facts.
3. Before submitting, check that the answer covers every requested part and that
   names, numbers, units, comparisons and negations agree with the evidence.
4. Cite only inspected spans that support the answer. If the inspected scope
   lacks a requested fact, report the specific gap using the existing protocol.
```

Treatment SHA-256 is
`cff32d0ff844265f5dcd4b8164b1bce65a14bdc3b4f6d709af1e6bde01683dad`.
Both policy system text and investigator initial observation use the same prompt;
the resulting packet hashes that prompt. The existing duplication of system text
inside the initial observation is preserved in both arms. No example, new tool,
gold passage, answer-specific hint or deterministic pre-read is added.

Both arms use the pinned non-thinking native renderer, temperature/top-p 1,
top-k -1, min-p 0, 1,024 output tokens/action, four investigator steps, the same
scoped passages and frozen Docker image. Pair seeds as `42 + question_index*4 +
sample_index`; alternate which arm runs first. A standalone collector supplies
the experimental prompt locally rather than weakening production validation.
All original traces remain untrainable. Generated Python runs only in Docker
with networking disabled and read-only corpus mounts.

## Limits and launch conditions

Hard generation limits across both arms: **576 requests / 4,718,592 reserved
tokens / 1,500 collection seconds**. Twelve requests per episode allow bounded
format-correction calls; these count against the same ledger. Provider retries
are disabled. The native proxy bounds responses, refuses redirects, reserves
input plus maximum output before dispatch and closes after uncertain requests.
Infrastructure errors invalidate the experiment; they do not become zero rewards.
No restart, resampling, question substitution or automatic paid retry.

Proposal: **$2 estimated allowance / 2,700 seconds (45 minutes) total owned
lifetime**, including setup, collection, export and cleanup. A fresh October25
Runpod `get_gpu_type` read returned A100 SXM 80GB Secure at **$1.79/hour**,
global stock HIGH, CUDA13.2 capacity and US-MD-1 MEDIUM stock. No machine is
reserved. For this inference-only test, propose a 100GB temporary container disk
and no network volume or optimizer checkpoints. Official
[storage pricing](https://docs.runpod.io/pods/pricing) lists $0.10/GB/month for
container disks. Use a conservative combined bound of $1.90/hour: 45 minutes
projects $1.425 before tax, with the rest of the $2 allowance as headroom.
Refresh actual shape, storage, price and stock before launch; abort if they do
not fit. This is a planning estimate, **not a provider-enforced dollar cap or
guarantee that all 48 episodes will finish**.

Paid execution requires a new plan-hash-bound approval and independently verified
resource deadline/deletion guard; an approval JSON records that verification but
does not create or prove the guard. The CLI never provisions or deletes cloud
resources. Existing laptop supervision does not survive laptop sleep; do not
represent it as independent cloud protection. Resolve that deployment boundary
before launch. With the current local Docker controller, the laptop must remain
plugged in, awake, lid open and Docker running throughout collection. The separate
guard must still expire the rental if collection or that controller fails.

The runner checks local V5 bytes and the engine's reported adapter root, but
records `weights_verified=false`: a reported path is not tensor attestation.
Deployment must verify the pinned model/backend before interpreting results.
Export artifacts before the guard deletes temporary storage.

## Decision and source review

Report all six per-question reward vectors for both arms, uniform-zero,
uniform-one and mixed groups, total/actual/reserved tokens, request counts,
execution failures and latency. Uniform-one groups are solved questions, not
failures; fewer mixed groups can therefore accompany better answers. A reward
one is an exact reviewed-variant/citation check, not a semantic correctness proof.
The ledger is authoritative for failed or uncertain dispatch costs; partial
healthy-episode totals are explicitly labelled and cannot hide those costs.

Source-review all 48 answers and their material claims before deciding. Preserve
names, numbers, units, comparisons and negation; check completeness and support
against inspected source. Record reviewer identity/kind and notes, with human
review left null unless an independent human actually reviews. Quotation matching
establishes provenance, not semantic support. Reward-wording false negatives
remain a known limitation; do not change aliases after seeing these outputs.

Advance only as a provisional development signal if all 48 episodes are healthy,
checklist gains at least one mixed group and improves mixed groups per reserved
token, total reserved tokens are at most 1.25x control, execution failures do not
increase, supported-answer count does not regress and unsupported-answer count
does not increase. Missing source judgments, incomplete collection or uncertain
backend identity makes the decision inconclusive. Keep control otherwise; no
statistical-significance or model-improvement claim follows from six questions.
Any later RL experiment needs a new prospective protocol and budget; do not
rewrite the existing ten-update pilot or unlock its held-out references.

## Commands and verification

```bash
python3 scripts/reader_scaffold.py prepare --output out/reader-scaffold-v1-20261025/ready
out/prime-rl-local-20261023/venv/bin/python scripts/reader_scaffold.py probe \
  --output out/reader-scaffold-v1-20261025/ready
```

`prepare` writes the exact prompts, six tasks, paired schedule, source/model
identities, implementation hashes, limits and decision rule to `plan.json`.
Changing any bound input requires fresh preparation. `probe` renders all 12
initial question/arm prompts and executes one fixed two-passage/gap script in
each arm's real Docker sandbox. It makes **zero model calls**. Its passage
observations, actual payloads, packets, sandbox inspection and cleanup receipts
are stored under `probe/`; `probe.json` reports prompt lengths and identities.

An initial probe passed for both arms: two inspected passages each, matching
policy/observation/packet prompt identities and all containers absent. Initial
prompts were 2,676–2,695 tokens for control and 2,880–2,899 for checklist:
**204 extra tokens**, within the 7,168-token prompt ceiling. An offline check of
the saved prior pilot's first actual three-turn episode found identical token
IDs under this runner's native renderer on all three requests. This is limited
formatting parity, not new policy behavior or a model-quality result.

Live `run` additionally requires explicit engine/tokenizer/adapter paths and a
new authorization file with this plan's SHA-256, approved $2 estimate, actual
combined hourly rate, owned Pod identity, 2,700-second rental limit, verified
external watchdog and absolute resource deadline, and the engine's adapter path.
The earlier run's approval cannot authorize this different plan. The result stays
`collected_pending_source_review`; it never automatically claims improvement.
The initial probe's old preparation directories are retained for provenance;
use only a newly validated `ready/` plan after source changes.

Verification: full suite **2,156 passed / 19 optional skips**, with three existing
SWIG warnings. Pinned native environment: **134 focused tests passed**, including
actual loopback native-renderer/proxy requests and provider-failure no-retry
checks, with three expected CPU checkpoint warnings. New checks cover training
selection, paired seeds, unchanged control, reference-free exports, input drift,
authorization rejection and uniform/partial-group diagnostics. Ruff and whitespace
checks passed. The ready plan and offline receipts live under
`out/reader-scaffold-v1-20261025/ready/`; no GPU, paid calls or commits were made.
