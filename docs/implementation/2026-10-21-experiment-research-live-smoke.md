# Personal experiment research: failed bounded live smoke

**The live run did not produce a usable research brief.** Eight Qwen requests
used 17,505 provider-reported tokens, including one completion recovered after
shutdown and never consumed by the investigator. No completed reader packet,
verified evidence, accepted brief or final MCP result was produced. The original
watchdog stopped the investigation. Independent provider reads verified the GPU
was off, with endpoint min/max 0/0 and no workers or pending jobs.

This is a negative development result. It does not establish general Qwen
incapability, improvement from fine-tuning, or an advantage over host web research.
No new training, Nebius inference or paid grader calls were made.

## Question and predeclared decision

> What does PagedAttention change about KV-cache allocation, and what memory
> measurements would I still need before running two concurrent Qwen3-8B research
> readers? Propose the smallest useful local experiment.

Constraints were one RTX A5000 with 24 GB, BF16 Qwen3-8B with the existing v5
adapter, vLLM, and 7,168 input / 1,024 output token limits. Two concurrent readers
were a target, not a demonstrated capability.

The hypothesis was that a bounded Qwen investigator could return useful paper
evidence and a small test proposal with less repeated reading for the host. The
frozen decision required at least one relevant supported finding, accurate
applicability limits and a useful proposed test, with no critical factual error
or budget breach. **That decision rule was not met.**

## Trace and claim review

| Request | Model request and observed result |
| --- | --- |
| 1 | Search for PagedAttention/KV allocation/concurrent-reader memory. Semantic Scholar returned 429; date-sorted arXiv fallback supplied mostly unrelated metadata. |
| 2 | Identical search; duplicate rejected. |
| 3 | Shorter KV-allocation/concurrent-reader query; fallback again supplied poor matches. |
| 4 | Identical shorter search; duplicate rejected. |
| 5 | Attempted `finish` before reading any paper, with E1/E2/E3 that had never been returned. Repetitive uncertainty exhausted the 1,024-token output limit. Incomplete literal syntax was rejected before citation or semantic validation. |
| 6 | Opened arXiv 2610.28620v1, *Weight Placement Is Not Enough: Synchronization in Multi-Node CPU Decoding of Ternary LLMs*, an off-target paper. |
| 7 | Reader Python executed in Docker: `search_within` for KV-cache/PagedAttention returned `[]`. |
| 8 | Provider completed a `passage(..., 0, 500)` action. The watchdog killed the client before it consumed this response; the action was recovered afterward and was never executed. |

A fresh independent model-assisted reviewer checked the attempted claims against
the saved trace. The generic PagedAttention mechanism was **unsupported in this
run**, which does not imply the mechanism itself is false. Suggested concurrency
and sequence-length measurements were relevant but unverified. The proposed
two-reader VRAM check omitted a one-reader control, fixed serving configuration,
warm-up, stop rule, memory breakdown and latency/completion measurements. No
experiment ran. Uncertainty about target hardware was appropriate, but repetition
did not repair missing evidence. Human review remains unperformed.

The [claim review](../../examples/experiment-research-live/claim-review.json)
separates provenance, support and proposal quality. Neither parsing nor exact
quote matching grades semantic correctness. No citations from request 5 reached
the provenance validator, so this run does not demonstrate their rejection by it.

## Budget, startup and shutdown

The approved limit was one research invocation, 12 inference requests, 98,304
reserved tokens, one A5000 at a time and a 900-second absolute session window,
with no inference resubmissions. Actual reservations were 8 requests / 65,536
tokens. All eight jobs were read back COMPLETED; seven completions were captured
by the client, and one was recovered separately. Captured usage was 13,642 input
plus 1,167 output tokens; including the recovered completion it was 16,309 input
plus 1,196 output. Maximum prompt/output lengths were 3,305/1,024. Local prompt
counts agreed with reported usage. The partial checkpoint stops at five requests;
the durable request ledger plus the labeled posthoc response give full accounting.

The original window started at 05:16:22.530087 UTC on October 21, 2026 and ended
at 05:31:22.530087 UTC. The initial worker stayed `ready=0` despite pinned identity
and healthy vLLM/fitness logs. It was stopped with zero inference calls. After
verified shutdown, an independently reviewed, opt-in readiness correction allowed
a startup-only continuation within the same original deadline and unchanged
request/token allowance. No attempt marker or ledger was reset. Two activation
windows were used, with no overlap. The resumed timer was 233.807 seconds and
measured elapsed time before cleanup was 235.037 seconds. The resumed worker
passed the existing `ready=1` branch; the added running-plus-logs branch was unit
tested but was not exercised by this live continuation.

The watchdog fired and child exit was -9. Cleanup removed the owned Docker
container, read back all owned jobs and parked endpoint `q05pjrvguakieg` at 0/0.
Fresh independent reads found zero workers and zero queued/running jobs, with
no cleanup errors. A separate successful local Docker inventory confirmed the
owned container was absent; its readback is retained with the run artifacts.
The catalog price was $0.69/GPU-hour. The scoped billing read
returned no records yet: **actual spend is unknown**, not zero. This test used
request/token/time bounds, not a hard provider dollar ceiling. The consumed
attempt cannot be rerun automatically; another paid test needs fresh approval.

## Reproduction and implementation changes

The model was `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218` with
`jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@27b912a863ff914ad45baa03624d8911dc1e17fb`,
subfolder `artifacts/full/checkpoint-50`, adapter SHA-256
`7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`.
Decoding was temperature 0, top-p 1, seed 42, thinking disabled. Serving used
PyTorch 2.13.0+cu130 and vLLM 0.31.0 on A5000. Docker image:
`sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382`.

The [initial preparation](../../examples/experiment-research-live/prepared/plan.json)
and [explicit startup amendment](../../examples/experiment-research-live/startup-continuation/startup-amendment.json)
remain frozen. The live source was `2c225ae` plus the separately pinned existing
uncommitted changes in `scripts/arxiv_research_mcp.py` and
`src/product/arxiv_mcp_config.py`; it was not a pristine checkout. All 157 amended
source hashes matched before the post-live fix. Raw prompts, responses, source
snapshots and lifecycle reads are in
[the saved run](../../examples/experiment-research-live/run/research.requests.jsonl),
with [summary](../../examples/experiment-research-live/summary.json) and
[independent execution audit](../../examples/experiment-research-live/execution-review.json).

Implementation commits for this test were `29cabdb` (discovery fallback and
effective limits), `f57c6ba` (one-attempt queue/lifecycle runner), `afa942d` (frozen
approval), `2c225ae` (bounded startup continuation), and `0e7c492` (amendment record).

A separate post-live fix, `d5b5c16`, addresses a repeatedly reproduced existing
concurrent staging-lock creation failure. It creates the lock exclusively, or
opens the winner's existing lock with no-follow semantics; missing boundaries
still fail. Fork/spawn tests check the actual shared lock inode, duplicate
rejection and cleanup, with symlink and missing-boundary regressions. Independent
review found no safety blocker. This fix was not part of the live run; the frozen
plans were not rewritten to include it. All ten protected original files remain
unchanged. Final offline checks are recorded in
[live validation](../../examples/experiment-research-live/validation.json).

## What the MCP user can do and what remains

The user can ask a specific ML experiment question through
`research_experiment(question, constraints)`. The tool attempts bounded paper
searches and Docker-backed Qwen reading, retaining excerpts, gaps, usage and
traces for the host to inspect. This smoke shows that the live path dispatched
within its request/token limits and shut down compute; it did not demonstrate
useful autonomous research. The saved offline success fixture remains a
component test, not model-quality evidence.

The next focused changes should improve paper-query relevance and recovery when
the primary provider is rate-limited, and require a concise, evidence-backed
finish after reading. Check those changes offline before proposing another paid
test. A successful live brief, human claim review, a matched host-only baseline,
two-reader memory measurements and hackathon demo/submission still remain. No
model promotion or fine-tuning improvement follows from this result.
