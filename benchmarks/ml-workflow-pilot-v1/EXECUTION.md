# Native answer collection protocol

The user approved execution and, after an explanation of the native hosted-search
boundary, replied “sure ig” to a monitored 20-call target with reported in-flight
overshoot and a ten-minute local stop. This amendment is recorded before any
answering turn. The original questions, common instruction and reviewer rubric
are unchanged. Their raw hashes and the collector sources are bound in
[execution-manifest.json](execution-manifest.json).

## Procedure

Run `python scripts/run_native_ml_pilot.py --manifest
benchmarks/ml-workflow-pilot-v1/execution-manifest.json --output
out/ml-workflow-pilot-v1-collection` from the repository root. The output must not
exist. The collector refuses changed frozen inputs and never retries an answer.
It schedules ten fresh ephemeral native Codex threads in the balanced pair order.
Both arms use `gpt-6.1-sol`, medium reasoning, live web search and the same
developer/user instructions. Neither receives this document or the reviewer rubric.

Each thread has a new workspace outside the repository. A has no active MCP tools;
B has the five source and five experiment tools, with new empty cache/record/export
roots. Native inventory, resolved model, empty loaded instruction sources and
inside/outside file canaries must pass before an answering turn. Shell network
access is disabled, and model-generated commands inherit no credential environment.
MCP sources run separately with selected directories. Ordinary integrations for
plugins, apps, helper agents and memory are disabled equally in both arms.

The controller logs native JSON-RPC events, MCP schemas, supplied prompts,
source/tool arguments and returned content where exposed, model messages/errors,
wall time and reported token usage. Native hosted search may expose only actions,
queries/URLs and citations rather than the full retrieved text or underlying
request count. Complete built-in advertised schemas/provider request counts are
not exposed by the verified interface; do not pretend the MCP inventory is the
complete host inventory. These telemetry limits remain explicit in the report.

It counts unique `webSearch` and source-MCP item IDs, deduplicating start/completion
events. A compound native action can contain several queries; retained action
data is separate from the invocation count. Planning calls are recorded but are
not research calls. At the observed target of 20, the controller requests a turn
interruption once. It continues collecting terminal/error evidence, including any
in-flight overshoot. The model is told to finish within that target.

An independent wall timer kills the local app-server process group at 600 seconds
from prompt dispatch. A terminated local process cannot guarantee cancellation or
usage refunds for an already-dispatched remote request. Cleanup follows every
attempt. Partial messages/evidence, failures and unknown usage are retained. A
native protocol/model failure stops collection with remaining cases unattempted;
there is no custom host fallback or selective rerun. Research-target interruptions
and timeouts remain outcomes, not reasons to change the budget mid-pilot.

There are zero extra Qwen/Nemotron/reader/reasoner calls and zero GPU starts.
Host tokens are reported where supplied; actual dollars and a hard monetary/token
cap remain unknown. The ten-answer and elapsed limits apply locally. No model
training or proposed ML experiment is executed.

## Validation before answering

Seven focused controller tests pass: real parent/child process-group termination,
mixed web/MCP target monitoring with retained overshoot, lifecycle deduplication,
retention of notifications arriving before RPC replies, preservation of partial
errors and unknown final phases, rejection of reused state/inappropriate MCP
inventories, and continued waiting after an idle notification timeout.
Both native zero-turn setups passed under the actual collector settings: A=0,
B=10 active MCP tools; own canaries readable and repository canaries denied.
The repository suite passed **1,485 tests** with three existing SWIG warnings.
Ruff and `git diff --check` passed. The ten preserved file hashes are unchanged.

## Review and interpretation

After collection, generate anonymous answer files with separately randomized
display order. Preserve an operator-only mapping and keep traces/arm names out of
the initial review packet. Source citations or product-specific formatting may
still reveal an arm; reviewers must record such cues. Source checking and initial
review labor are recorded separately from the ten-minute correction budget.

No human reviewer has yet been assigned or supplied scores. Model-assisted
checks, if produced, must remain separate. All five scheduled pairs remain in
the denominator; missing/failed/time-limited answers are not silently dropped.
Quality, preference, correction effort and time-to-useful summaries remain unknown
until their required human/source reviews exist. Exact quotation provenance,
successful software tests and a populated record are not semantic quality scores.
This five-scenario development pilot cannot prove broad superiority, a training
benefit, Qwen usefulness, or improved outcomes from the proposed ML experiments.
