# One real-project experiment-mode walkthrough

Question: **What should we measure first to reduce latency for two concurrent
Qwen3-8B v5 research readers on the previously used 24 GB A5000 setup?**

This is a host-selected development question from the user's project goal.
The host used the actual combined stdio MCP, live paper discovery and existing
project observations. It is not a scripted-model success fixture, an independent
assistant comparison or a completed optimization experiment. No new reader
inference, GPU activation, training or paid judge call was made.

## What changed in the decision

We now have a specific next step: **measure the unchanged serving path before
choosing a latency optimization**. The historical server allowed two active
sequences, but only serial requests were tested. Its startup memory report is
not a two-reader load result, and the historical reader produced no evidence.

The selected old jobs had first-request provider queue delay of 87.748 seconds,
later delays of 0.128–0.146 seconds, execution times of 1.402–8.301 seconds and
13–20 output tokens. These are heterogeneous historical jobs, not controlled
warm/cold arms or a long-generation latency baseline. The cause of the first
queue delay was not independently identified.

The live search fell back from rate-limited Semantic Scholar to arXiv relevance
search. The host selected the original PagedAttention paper, third of five
candidates, opened `2309.06180v1` as HTML, and inspected exact passages. Its block
allocation mechanism does not establish a Qwen/A5000 performance result.
Official [vLLM 0.31 metrics](https://docs.vllm.ai/en/v0.31.0/usage/metrics/)
and [prefix-cache documentation](https://docs.vllm.ai/en/v0.31.0/features/automatic_prefix_caching/)
were inspected separately with ordinary web tools. Those external references
remain `reference_only` in the MCP, not source-verified paper receipts.

Local code inspection exposed a practical prerequisite: the launcher's explicit
cache-comparison branch requires vLLM 0.30.0 and one sequence; the historical
server used 0.31.0 and two. The queued client is non-streaming and does not return
engine phase timings directly. Neither code path was changed by this walkthrough.

## Saved outcome

[walkthrough.json](walkthrough.json) records **12 successful MCP calls across six
stdio sessions**. It binds the actual call responses and local record hashes:

1. Save the question/setup and discover papers.
2. Restart, resume, open the chosen paper and find passages.
3. Inspect surrounding context and the short source span.
4. Attach five unreviewed findings with ten source/reference receipts.
5. Save the researched proposal as brief revision 2.
6. Restart, list/resume and retrieve the full research note.

The note remains tied to brief revision 1; the newer proposal does not rewrite
it. The recap explicitly reported research-section truncation, and the full note
was recovered unchanged by ID. The final session called no source tools.

The record is **draft**, with no frozen plan and no new result. Budget, formal
metrics and acceptance criteria remain missing before any plan freeze or paid
dispatch. Nothing is recorded as a measured candidate improvement.

[research-note.json](research-note.json) and [researched-brief.json](researched-brief.json)
preserve the host's proposals. [claim-review.json](claim-review.json) is a small
unblinded host source/applicability review, not independent human review. The
user's usefulness judgment remains pending. Quote/offset checks prove provenance,
not semantic support or scientific validity.

## Proposed diagnostic, not yet run

Two pinned historical reader inputs are identified in
[profile-input-manifest.json](profile-input-manifest.json). Keep their model,
adapter, tokenizer, decoding and server configuration fixed. Subject to separate
authorization and instrumentation preflight, use two warmups and two measured
rounds per dispatch mode: serial/concurrent, then concurrent/serial. That is ten
total requests, reserving 7,168 input plus 1,024 output tokens each (81,920 total).

Capture pair completion time, individual completion times, actual token counts,
memory, same-engine overlap and phase/queue telemetry. Equalize and record cache
state per batch; do not misattribute incidental cache warming to concurrency.
Stop on missing instrumentation, identity mismatch, OOM, timeout or uncertain
inference; no automatic retries. A future compute/lifecycle budget is still needed.

The payloads contain historical weak/failed tool-use contexts and short outputs.
This proposed diagnostic cannot measure complete-agent behavior, responses to new
feedback, answer quality, tail latency or benefit from experiment mode. A later
ordinary-assistant versus MCP optimization comparison is separate.

## Inspect the saved record

The operator's local record is `experiment_af30ad1c9ea446ffbf18d3e1ba709087`.
Launch the same mode against the task-owned roots:

```sh
python scripts/experiment_mode_mcp.py \
  --store-root out/experiment-mode-walkthrough-20261021/records \
  --library-root out/experiment-mode-walkthrough-20261021/library \
  --artifact-root out/experiment-mode-walkthrough-20261021/exports
```

Call `resume_experiment` with that ID, or `get_experiment` with
`research_note_id: "latency_measurement_first"`. These calls are local. Raw source
text, exact profiling payloads and full MCP responses stay in ignored `out/`.
The JSON inputs here can create a fresh draft after matching source caches and
selected exports are available; new IDs/timestamps will differ.

## Verification and limits

The walkthrough exposed a task-1 discovery adapter signature mismatch. Commit
`b2f9e94` fixes positional arguments and preserves explicit date bounds, with
coverage for the real wrapped composition. Focused checks: **37 passed**. Full
offline suite with the existing Docker image: **1855 passed, zero failures/skips**,
three existing SWIG warnings, 144.47 seconds. See [validation.json](validation.json).

All eleven protected-file hashes remained unchanged. There is no new model
quality score, human review, latency win, cost advantage or fine-tuned Qwen
improvement. Extra worker calls and GPU launches were zero; ordinary host-chat
usage/cost and underlying source HTTP request counts were not measured.
