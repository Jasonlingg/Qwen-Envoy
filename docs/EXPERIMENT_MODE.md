# Personal ML experiment mode

Use one ongoing experiment conversation: understand the setup, research a
specific uncertainty, propose a small test, then interpret supplied results.
The assistant saves the context behind that conversation. You do not need to
fill in a form or choose a question from a predefined list.

This entry point combines the existing experiment store and model-free paper
tools. It is separate from the published QASPER Agent Studio case study.

## Try the completed workflow

With Python 3.11 and the repository's `mcp` and `vault` extras installed:

```sh
python scripts/demo_experiment_mode.py --output out/my-experiment-demo
```

Choose a new output directory. This runs the real combined MCP through two
stdio processes using committed research and measured serving artifacts. It
saves a question, attaches prior research, freezes a clearly retrospective
analysis, records measurements, saves the decision and resumes after restart.
It also finds that experiment from a related question in the restarted session,
records an explicitly agent-proposed decision to defer a latency claim, and
finds that decision thread after restart. It
checks safe retries and rejected history edits. It makes no source
network calls, model calls, GPU launches or generated-code executions.

Open `out/my-experiment-demo/RESUME.md` for the next conversation, `EXPERIMENT.md`
for the readable record, or `complete-record.json` for the full structured view.
The output includes the local `records`, selected `exports`, and all MCP call
responses. An existing output directory is refused rather than overwritten.
`decision-thread.json` contains the stated choice and hashes of the exact run
and research note used as its basis. It is not a human-approved conclusion.

The [completed example](../examples/experiment-mode-complete/README.md) links
the earlier live paper investigation and actual Qwen serving measurements.
It recovers a useful distinction: concurrent requests shared one loaded model,
while the speed comparison remains inconclusive. Its imported paper research
is checked as a saved artifact; this offline replay does not re-fetch the paper
or provide independent semantic review. The example's question is not built
into the MCP: normal use starts from your own question.

## Start locally

```sh
python scripts/experiment_mode_mcp.py \
  --store-root ~/.envoy/ml-experiments \
  --library-root ~/.envoy/arxiv-library \
  --artifact-root /absolute/path/to/selected-sanitized-run-exports
```

Register that command as a stdio server in your MCP host, using the repository's
Python environment with the `mcp` and `vault` extras installed. Omit the optional
artifact root if you only need questions and paper evidence. Existing store and
library locations can be reused; old records remain readable without migration.
Avoid registering duplicate copies of the same record/source tools in that host.

A host that accepts `mcpServers` configuration can use the following entry.
Replace the executable and repository paths with your local absolute paths;
these are configuration examples, not credentials or automatic installation.

```json
{
  "mcpServers": {
    "envoy-experiments": {
      "command": "/absolute/path/to/python",
      "args": [
        "/absolute/path/to/rlm-explorer/scripts/experiment_mode_mcp.py",
        "--store-root", "~/.envoy/ml-experiments",
        "--library-root", "~/.envoy/arxiv-library"
      ]
    }
  }
}
```

For the replayed example, use its `records`, `library`, and `exports` directories
as the three roots and pass `--artifact-root` for `exports`. Select the experiment
ID printed by the demo; listing/resuming does not require another research call.

No endpoint, tokenizer, Docker or GPU is needed for this mode. Starting it makes
no model calls and provisions nothing. Paper search/open tools make bounded
network calls when invoked; local record/resume tools make none.

For an explicit public-paper explanation, set `NEBIUS_API_KEY` in the environment
or ignored repository `.env` and add `--enable-nemotron` to the server command.
This exposes `explain_experiment_research(experiment_id, note_id)`. It sends the
selected research question, findings, uncertainties and at most five saved,
verified public-paper spans (12 KB packet maximum) to NVIDIA Nemotron 3.5
Lightning on Nebius Token Factory in one request, with at most 1800 output
tokens and no automatic retry. Use it only on notes safe to disclose to Nebius.
It does not send local run artifact values or activate a GPU. This tool incurs
Token Factory usage when invoked; source and
record tools still work without a key. Its claim-to-source ID check proves only
reference integrity. Review each claim's meaning before relying on it.

## Conversation and tools

For example:

> I want to improve throughput for this model and workload. Here is my setup
> and baseline. Help me decide what to try next.

1. The host calls `find_related_experiments` to surface prior local records by
   word overlap, then `resume_experiment(experiment_id)` for any promising ID.
   The lookup is a navigation aid, not semantic relevance. If the question is
   new, `save_experiment_brief` records it with known setup and an explicit ID.
2. It investigates an uncertainty with ordinary web tools for documentation,
   model cards and repositories, or the five paper tools here:
   `search_papers`, `get_citations`, `open_paper`, `find_evidence`, `get_passage`.
   Paper discovery uses the existing experiment-discovery fallback when Semantic
   Scholar is unavailable. No research is required on an ordinary conversation turn.
3. It calls `record_experiment_research` with the inspected brief revision,
   findings, source references, uncertainties and an optional proposed test.
   Notes are explicit host writes; source-tool calls do not save them automatically.
4. It proposes a small test, updates the draft with `save_experiment_brief`, and
   freezes the exact plan with `freeze_experiment_plan` only when you agree.
5. After a separately authorized test, it attaches supplied measurements through
   `record_experiment_result`. Resuming returns the current setup, latest research,
   latest supplied result and remaining unknowns for the next conversation.
6. For a question spanning multiple attempts, call `create_decision_thread` with
   its first experiment, then `link_decision_experiment` for later attempts.
   `record_decision` appends an adopt/reject/defer statement with explicit
   experiment, run and/or research-note IDs. Use `author_kind: "agent_proposed"`
   for the assistant's interpretation. Use `"user_reported"` only when the user
   actually states the choice; this field is caller-declared, not proof of review.
   `find_decision_threads` and `get_decision_thread` recover it next session.

These are host workflow instructions, not an automatic scheduler or an execution
engine. The host chooses tools and interprets outcomes. It does not automatically
start tests, load models, run training or spend money. A frozen plan stays immutable;
a changed plan uses the existing linked-successor mechanism.

## Research attachment

The host supplies an immutable `note_id`, a specific research question, up to six
findings, up to twenty references and at least one uncertainty. Each finding
names its references. A proposed test is text, not executable code.

```json
{
  "experiment_id": "experiment_<32 hex characters>",
  "expected_revision": 1,
  "note": {
    "note_id": "memory_investigation_1",
    "question": "What conditions does this memory method require?",
    "findings": [],
    "references": [],
    "uncertainties": ["We still need to inspect the paper's workload and baseline."],
    "proposed_test": null
  }
}
```

Reference formats and checks match [the existing experiment records](ML_EXPERIMENT_MCP.md):
version-pinned cached `paper_span`, selected JSON `local_artifact`, or an explicitly
unverified `external_reference`. Source verification creates receipts internally;
the caller cannot supply a preverified receipt or semantic-support verdict.
Paper spans verify exact source text; artifact receipts verify selected file values.
Neither proves that a finding is correct, applicable, causal or independently reviewed.

The exact brief revision and digest are recorded with every note. Later research
does not edit a frozen plan or rewrite a recorded result. An identical note ID,
content and revision retry is safe, even after the source disappears. A changed
note requires a new ID; a new note with a stale brief revision is rejected.

Qwen is optional. The combined server now supports a narrow
[passage reader](PASSAGE_READER.md) when an explicit endpoint, model and checkpoint
are configured. `read_experiment_passages` inspects host-selected source windows
and saves candidate evidence under a single-attempt note ID. It makes model calls
only when invoked and never activates compute. Semantic review remains the host's job.

The broader [experiment research worker](EXPERIMENT_RESEARCH_MCP.md) remains separate.
The host can explicitly attach a summary of its returned evidence with
`origin: "qwen_research"`; origin is a caller declaration, not model attestation.
This mode does not launch that worker, silently import saved reports, or send
experiment context to it. Private context needs authorization before external delegation.

## Resume and storage

Every experiment is linked by explicit ID; there is no global active experiment that
could mix two clients. Briefs, source-checked research notes and supplied results
use the existing locked, atomic, content-addressed store. Notes have their own
append-only index entries and do not consume brief revisions. Interrupted writes
leave the prior index usable. Source receipts remain available after source removal.

A decision thread has its own ID and immutable event chain in the same local
store. It groups existing experiment IDs without editing their plans or results.
Each stated choice snapshots the hashes of its selected brief, plan, run and/or
research note. A new choice appends an event; it does not overwrite an older
choice. The thread does not assert that a cited finding is semantically correct,
that a run was scientifically valid, or that a caller-declared user choice was
independently confirmed. Search is bounded word overlap on thread questions;
inspect the returned thread and experiments before relying on a match. Back up
the store together with its paper cache and selected artifact exports; an
automated portable export is not yet included.

`resume_experiment` returns a bounded text recap plus the current revision/hash,
frozen-plan hash, note/result counts and latest IDs. Long sections are explicitly
truncated; a long unknowns list cannot consume the entire research/result recap.
Notes written against older brief revisions are marked as such, not automatically
treated as applicable to the current setup.

`get_experiment` returns full saved fields and history summaries. Supply
`research_note_id` or `run_id` to inspect an older note or result. Resumption does
not refresh cached papers or automatically find related experiments. Stored
content is untrusted data, not instructions. Proposals stay unexecuted and
semantic support stays unreviewed.

## Validation and current scope

Tests use real MCP clients and temporary stores with fixture paper sources.
They verify persistence, evidence checks, revision binding, frozen-plan/result
preservation, decision-thread history, safe retries, concurrent notes, corruption handling and resumption
across a real stdio-server restart. They do not test live Qwen research or show
a productivity or benchmark improvement.

The [earlier real walkthrough](../examples/experiment-mode-walkthrough/README.md)
used live source discovery, paper opening and passage inspection. The
[completed loop](../examples/experiment-mode-complete/README.md) now attaches
the separately collected live measurements, preserves an inconclusive result,
and verifies recovery through a fresh stdio process. This is a completed local
research-to-result workflow, not a validated productivity or model-quality win.

The broader [Qwen research worker](EXPERIMENT_RESEARCH_MCP.md) remains a
separate explicit tool. Its last complete live research smoke returned no
verified evidence; later serving success does not change that result. The new
[passage-focused path](PASSAGE_READER.md) deliberately removes discovery and
paper navigation from the worker's job; its development evidence is recorded separately. The host
can use the paper tools directly and save findings without depending on Qwen.
No automatic GPU activation, training, recurring research or paid retry is
enabled by this workflow. A fresh native host session must load the configured
MCP tools before normal use; the demonstrated transport is an actual stdio
client, not an assertion that this chat already has the tools installed.

A future productivity comparison or model-improvement evaluation would need a
separate question set, resource allowance and decision rule. Neither is required
to use the personal MCP, and neither is claimed by these component checks.
