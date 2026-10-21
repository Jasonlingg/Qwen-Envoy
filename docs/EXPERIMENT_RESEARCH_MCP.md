# Personal ML experiment research MCP

Ask a specific experiment question and let a Qwen worker search for papers,
choose sources, read them in Docker, and propose a small local test. Invoke
research when needed; ordinary conversation and source-tool calls do not invoke
the model. This is a separate entry point from the saved QASPER Agent Studio.

## What the tool returns

`research_experiment(question, constraints="")` returns candidate findings linked
to exact paper excerpts, conditions and uncertainties, an unexecuted experiment
proposal, usage reservations, source versions, and both control and reader traces.
The host must check whether the cited passages support each finding. Verified
offsets prove where text came from, not whether the proposed conclusion is sound.

For example, ask:

> What paper-backed approaches could reduce inference memory for two concurrent
> Qwen3 8B readers? Distinguish model weights from KV cache and identify what
> measurements we need before choosing a method.

Supply constraints separately, such as GPU memory, quality requirements, workload
lengths, and the serving stack. The question is user supplied, not selected from
a fixed task list. Do not treat papers' measurements as results on your hardware.

## Start the server

Use Python 3.11, the repository dependencies including `mcp`, a locally cached
matching Qwen tokenizer and `transformers`, Docker, and an explicitly configured
OpenAI-compatible Qwen endpoint. The existing `scripts/serve_vllm.sh` documents
serving. No GPU runs on the MCP controller; inference runs at the endpoint.

```sh
python scripts/experiment_research_mcp.py \
  --qwen-endpoint http://127.0.0.1:8000/v1 \
  --qwen-model envoy \
  --qwen-checkpoint 'YOUR_PINNED_CHECKPOINT_AND_REVISION' \
  --sandbox-image sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382
```

This is a stdio MCP server: configure your MCP host to launch that command with
the repository as its working directory. For a remote endpoint, use HTTPS and
`ENVOY_MODEL_API_KEY` in the host's environment. Keep credentials out of arguments
and committed configuration. The script never provisions or activates compute.
The existing Runpod queue URL is not an OpenAI-compatible endpoint. This ordinary
server entry point does not manage its lifecycle. A separate one-attempt runner,
`scripts/run_experiment_research_smoke.py`, uses the existing durable Runpod queue
adapter and guarded startup/shutdown for the explicitly authorized live smoke.
Do not send ordinary server requests to a parked queue URL expecting activation.

The default tokenizer is locally cached
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`. Override `--tokenizer`
and `--tokenizer-revision` only to match the served model's actual chat template.
Startup cannot independently attest the weights behind a remote model alias.
Missing tokenizer files fail closed without downloads or remote Python code.
Startup checks the Docker image first and resolves tokenizer files to a local
directory before loading, avoiding remote tokenizer metadata heuristics.

The five existing model-free source tools remain available. The model runs only
when the host explicitly invokes `research_experiment`. Completed and partial
reports are saved under `~/.envoy/arxiv-library/experiment-runs/`; change the
library location with `--library-root`. Reports include prompts/source excerpts
and are local files. A save failure returns the evidence with `save_error`.
Saved briefs are not automatically searched, reused, or refreshed in this version.

## Execution and limits

One Qwen endpoint serves sequential control and paper-reading conversations.
There is no extra paid reasoner or parallel swarm. The controller parses only
literal `search_papers`, `read_paper`, and `finish` requests as data; it never
executes those strings as host Python. The assigned reader continues to write
document-tool Python in network-disabled Docker, with no LocalREPL fallback.
It receives IDs only after discovery. Network fetching stays in the existing
redirect-checked, size-bounded, allowlisted source transport.

| Limit per invocation | Maximum |
| --- | ---: |
| Control turns including rejected requests | 8 |
| Search calls including failures | 3 |
| Paper-read attempts including failed opens | 2 |
| Actions in each assigned-paper conversation | 6 |
| Total inference requests across control and reading | 18 |
| Input tokens per inference request | 7168 |
| Requested output tokens per inference request | 1024 |
| Reserved input plus output tokens | 147456 |

`--requests-max` and `--tokens-max` can lower the aggregate limits. Each dispatch
reserves 8192 tokens before sending, even if it fails or produces fewer tokens;
reservations are not refunded. Actual provider usage is recorded separately,
with missing usage marked unknown. Token bounds assume a matching local tokenizer
and a provider honoring `max_tokens`. They are not a hard dollar ceiling.
There are no automatic inference retries, hidden synthesis calls, or paid graders.
These are request/action bounds, not a hard whole-investigation time deadline.

Failed searches can lead to a refined query. Paper failures and exhausted budgets
retain earlier verified evidence. `brief_ready` means a structurally accepted
unreviewed brief; `incomplete` and `budget_exhausted` may still contain useful
excerpts and candidate paper answers. Neither status is an answer-quality grade.
An uncertain inference transport failure stops further inference with
`inference_failed`, retaining evidence already collected. A successfully returned
but malformed completion can receive a correction turn within the same budget.

Discovery is limited to Semantic Scholar/arXiv papers. The worker cannot inspect
GitHub repositories, Hugging Face model cards, arbitrary websites, or local
hardware. HTML/PDF extraction may miss tables, figures and equations; coverage
is recorded. Discovery caches may be stale. Two papers cannot establish a
comprehensive survey or reliable absence of a method.
When Semantic Scholar is unavailable, experiment discovery explicitly falls back
to arXiv over the full arXiv date range. The response records the provider failure
and warns that fallback results are newest first, not ranked by relevance. Each
logical search can therefore make up to two provider queries, within the existing
network bounds and throttles. This fallback does not change the older MCP servers.
Lower request/token limits appear in the effective controller prompt and the
remaining shared request count is supplied on each control turn. The live runner
also checkpoints partial progress before further control inference.

## Verify without inference charges

```sh
ENVOY_ASSIGNED_TEST_DOCKER_IMAGE=sha256:79bd8bf84bd4b7f2a32a0596cb1fd90164716c1cfca879f78829411193646382 \
python -m pytest -q \
  tests/test_experiment_budget.py tests/test_experiment_protocol.py \
  tests/test_experiment_research.py tests/test_experiment_mcp.py \
  tests/test_experiment_docker.py tests/test_experiment_tokenizer.py
```

The Docker tests use the real in-process MCP SDK and real Docker with scripted
model responses and fixture sources. They test successful search/read/finish,
unsafe-code rejection, invalid citations, query refinement, shared budgets,
partial results and report persistence. They do not evaluate live Qwen research.

The [offline validation record](../examples/experiment-research/validation.json)
records 36 passing new checks, a final full-suite result of 1737 passed with no
skips, and independent code review with its retry finding fixed. It also records
one intermittent staging-concurrency failure in an earlier full run; its isolated
rerun and the final full suite passed without changes to staging code.

## Live evaluation before a hackathon claim

Hypothesis: on a specific ML experiment question, this bounded paper investigator
can give the host useful evidence with less repeated source-reading effort.
Expected signal: applicable findings, their experimental conditions, and a
testable proposal, with no critical unsupported factual claim after source review.
The host should not need to repeat the same searches and paper reads to use it.

Start with one user-chosen development question and a separately authorized paid
budget. Record the exact question/constraints, checkpoint, decoding, tokenizer,
source snapshots, all requests, missing usage, elapsed time, startup and shutdown.
Review each decisive claim against its passages and rate the proposed experiment
for relevance. Leave human review blank until a person performs it.

Decision rule: a useful smoke needs at least one relevant supported finding,
accurately stated applicability limits, and a usable proposed test without a
critical factual error or budget breach. Otherwise keep it a prototype and
diagnose the trace. This one-question development check would not prove advantage
over host-only research. That comparison needs matched questions and access to
sources, explicit budgets, review of both outputs, and measured host rework.
It also would not prove fine-tuned Qwen improved over base Qwen.

The [October 21 live smoke](implementation/2026-10-21-experiment-research-live-smoke.md)
did not meet this rule: 8 requests and 17,505 reported tokens produced no accepted
brief or verified evidence. Poor fallback matches, repeated searches and a
truncated attempted finish preceded watchdog shutdown. The GPU was verified off.
This is a prototype with a negative live result, not a demonstrated useful
research assistant. The later validation also records a separately reviewed
staging-concurrency fix after that failure reproduced again.

For a submission, a useful live investigation and trace still need to be
demonstrated, alongside clearly labeled offline failure tests and existing saved
QASPER results. Another paid test needs fresh approval; sponsor integration
checks, video recording and submission remain separate steps.
