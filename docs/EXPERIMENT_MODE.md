# Personal ML experiment mode

Use one ongoing experiment conversation: understand the setup, research a
specific uncertainty, propose a small test, then interpret supplied results.
The assistant saves the context behind that conversation. You do not need to
fill in a form or choose a question from a predefined list.

This entry point combines the existing experiment store and model-free paper
tools. It is separate from the published QASPER Agent Studio case study.

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

No endpoint, tokenizer, Docker or GPU is needed for this mode. Starting it makes
no model calls and provisions nothing. Paper search/open tools make bounded
network calls when invoked; local record/resume tools make none.

## Conversation and tools

For example:

> I want to improve throughput for this model and workload. Here is my setup
> and baseline. Help me decide what to try next.

1. The host calls `save_experiment_brief` with the question and known setup.
   This creates an explicit experiment ID. On return, use `list_experiments`
   and `resume_experiment(experiment_id)` to retrieve the intended experiment.
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

Qwen remains optional through the separate [experiment research worker](EXPERIMENT_RESEARCH_MCP.md).
The host can explicitly attach a summary of its returned evidence with
`origin: "qwen_research"`; origin is a caller declaration, not model attestation.
This mode does not launch that worker, silently import saved reports, or send
experiment context to it. Private context needs authorization before external delegation.

## Resume and storage

Everything is linked by experiment ID; there is no global active experiment that
could mix two clients. Briefs, source-checked research notes and supplied results
use the existing locked, atomic, content-addressed store. Notes have their own
append-only index entries and do not consume brief revisions. Interrupted writes
leave the prior index usable. Source receipts remain available after source removal.

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

## Validation and next task

Tests use real MCP clients and temporary stores with fixture paper sources.
They verify persistence, evidence checks, revision binding, frozen-plan/result
preservation, safe retries, concurrent notes, corruption handling and resumption
across a real stdio-server restart. They do not test live Qwen research or show
a productivity or benchmark improvement.

The next separately approved task is one real experiment walkthrough. A later
paired optimization pilot would compare ordinary assistant use with experiment
mode using equal time/compute allowances, validated final performance and total
resource use. No benchmark execution or paid inference belongs to this task.
