# Five practical decisions: MCP used, no demonstrated advantage yet

All ten native answers completed, one ordinary-web and one explicitly
source-MCP-assisted answer for each question. **B actually used MCP in 5/5
sessions**: 31 source-tool calls, including paper opening and passage inspection.
The core recommendations were similar across all five pairs. Selected central
claims checked against primary sources were largely supported; this is an
unblinded assistant audit, not a complete quality score or human preference.

The assisted workflow averaged **84.54 seconds**, versus **65.40 seconds** for
ordinary web research—about 29% longer in this single development collection.
It did not establish faster or better answers. Both arms still used ordinary web
for official model/runtime documentation. Qwen workers and the server reasoner
were off: this measures source tools, not cheap-reader benefit or fine-tuning.

| Decision | Common practical recommendation | A web seconds | B assisted seconds | B MCP calls |
| --- | --- | ---: | ---: | ---: |
| Smaller reader | Screen 4B-Instruct-2507 and 1.7B; measure extraction before replacing 8B |63.69|86.32|6|
| Concurrent memory | Share one serving engine; cache grows per sequence, weights need not duplicate |58.94|81.06|6|
| Latency | Profile queue/prefill/decode/cold start before selecting an optimization |64.33|72.32|6|
| More training? | Diagnose evidence sufficiency and extraction/condition errors first |85.57|78.77|6|
| Speculative decoding | Defer priority until a target-only baseline and compatibility check |54.49|104.24|7|

A/B labels and verbatim answers are in [results/answers](results/answers).
[Per-answer receipts](results/receipts.json) retain time, tool counts, host token
telemetry, public source URLs and raw trace hashes. [Aggregates](results/aggregates.json)
are descriptive; no automated semantic judge, reward or synthetic winner was added.

## What the source check found

The [assistant source check](results/assistant-source-check.json) records the
actual checked claims, source versions, calculations and limits for every pair.
A few useful examples:

- Both memory answers correctly use the official Qwen3-8B configuration: 36 layers,
  eight KV heads and 128-dimensional heads. Two independent 6,256-token sequences
  require about 1.718 GiB of raw FP16/BF16 KV payload. This excludes weights,
  adapters and runtime allocations; it is not a measured 24GB fit.
- B's latency answer retrieved specific PagedAttention evidence illustrating that
  lower memory waste and better end-to-end serving can coexist with slower
  attention kernels. Its 20–26% kernel-overhead statement should explicitly name
  **FasterTransformer** as the comparator if reused. The core recommendation to
  profile first is unaffected.
- Both speculation answers preserve original-paper hardware/model conditions
  instead of projecting TPU results onto our Qwen GPU target. B links a newer
  vLLM 0.18.0 Qwen3 draft-model example; A uses older 0.10.2 documentation. B found
  that example with ordinary web tools, so it is not an attributable MCP win.
- Both training answers propose substantially more evaluation work: 48 versus 32
  hypothetical generations. Do **not** start those studies automatically. For
  this personal project, first inspect a few actual failed answers and their
  supplied evidence; the larger proposals do not earn priority merely by being
  detailed.

These selected checks do not establish that every assertion is correct or that
one answer is more useful to the user. [User preferences](results/user-preferences.json)
remain blank. No model supplied a human verdict.

## Friction and costs

Every B session had one relevance-search response marked `source_unavailable`
with rate-limit detail. Allfive recovered using date-bounded arXiv searches;
no question failed. These are real retrieval problems even though native RPC/tool
completion counters show zero transport/protocol errors. Both states are retained.
A used 23 native web invocations. B used 24 web invocations plus 31 MCP invocations.
Compound search/query actions are not underlying network-request counts.

Host-reported totals were 980,091 tokens for A and 1,375,404 for B, including
736,256/1,083,904 cached input tokens respectively. These aggregate repeated
native-host context across tool-mediated generation; they are not unique prompt
sizes or invoice amounts. Actual host requests and dollar charges are unknown,
not zero; cached tokens are not assumed free. Extra Qwen/Nemotron worker calls,
GPU launches, proposed ML experiments, paid judge calls and training were zero.

## What to do with this result

Keep the MCP as an optional way to open papers and inspect reusable source spans.
Use ordinary web tools for model cards and runtime documentation. This walkthrough
has not earned mandatory MCP use for every ML decision, another judging framework
or more RL. If taking one small product fix next, address the repeated discovery
rate-limit/fallback friction; no production code was changed during this test.

This is an implementation-assistant recommendation, not a measured product win.
The user still needs to judge whether any of these briefs actually changes a
current experiment decision.

## Scope and reproduction

Questions, arm order and generation instructions were frozen in commit `ff28a1c`.
Both arms used native Codex CLI 0.160.0, `gpt-6.1-sol`, medium reasoning, separate
workspaces and fresh server/cache roots. Actual inventories were checked before
each answer: A had no MCP tools; B had source-only research plus unused experiment
record tools. Native filesystem canaries checked isolation. The preset research
call target 20 and 600-second local deadline were never reached; there were no
answering retries. There is no enforceable native host token/dollar ceiling, and
a local deadline cannot recall a remote request already dispatched.

Timing runs from prompt dispatch to final answer, excluding native setup and
human correction. B was explicitly prescribed to use MCP, and its empty caches
exclude persistent-library reuse benefits. This is not natural-adoption testing,
a controlled causal latency study, a held-out benchmark or a complete Qwen run.
The five questions were project-derived assistant drafts offered for substitution;
no substitution arrived during collection. Source-audit work overlapped later
collection, adding another reason not to interpret timing as a precise causal gain.

Full native events, exposed source results and server caches remain in ignored
`out/practical-research-check-20261021/`. Export contains final answers and ordinary
receipts, not native reasoning traces. To reproduce that export without model calls:

```sh
python examples/practical-research-check/export.py \
  --source out/practical-research-check-20261021 \
  --destination /tmp/practical-research-export
```

Choose a new destination. Check answer and raw artifact hashes in receipts. The
existing saved reader/judge evaluations, active MuSiQue reward and all protected
uncommitted files were preserved. No Qwen improvement or general research
superiority follows from these five pairs.
