# Three-case Qwen passage reader attempt

One explicitly requested development attempt: give v5 relevant source windows
and inspect whether it can return useful facts and conditions through MCP.
The host selects windows from already exposed source reviews. This is an easier
source-selection task than paper discovery or full-paper navigation, not held-out
evaluation or evidence that the fine-tune improves Qwen.

Cases: QLoRA storage/computation/gradients, AWQ benchmark protocol and hardware,
and the missing PagedAttention target latency measurement. Expected review facts
are separate from model prompts. Qwen receives the question, paper metadata and
window coordinates, then must inspect text in network-disabled Docker. The
opt-in `passages-v1` mode permits only contained passage reads and citations to
actually inspected source text. Quotes establish provenance, not semantic support.

One existing A5000, 900 seconds including loading, at most twelve model calls
and 98,304 reserved tokens. The serverless catalog at 22:01:47 UTC on October 21
reports $0.69/hour, giving a maximum-window compute estimate of $0.1725 under
the $0.50 local allowance. Actual invoice cost is unknown; no provider dollar
cap exists. Earlier consumed attempts remain unchanged. No paid retry is included.

Implementation `fe3a5a5`. Before activation, 30 focused tests passed in 8.70
seconds and the full suite passed 1918 in 130.82 seconds, with no skips and three
existing SWIG warnings. Ruff and whitespace checks passed. A scripted rehearsal
used the real MCP, Docker and exact source windows, completing all three cases
with nine synthetic model responses. That rehearsal is not a Qwen quality result.

## Live result

**Partial success; the predeclared three-case rule was not met.** Ten actual
Qwen requests completed, using 26,505 input and 421 output tokens. Reservations
were 81,920 tokens; there were no provider-job failures or inference resubmissions.
All three cases reached the actual MCP tool and saved their outcomes as notes.
Identical note retries returned the same reports without dispatching inference.

| Case | Actual result | Source review |
| --- | --- | --- |
| QLoRA dtypes | Incomplete after four actions | Two valid end-minus-start lengths were rejected; both corrected reads succeeded but no answer was submitted |
| AWQ protocol | Evidence returned after two actions | Candidate correctly states prompt 4 tokens, generation 200 tokens, batch 1, median latency, RTX 4090 and Jetson Orin; its own citation supports all requested facts |
| PagedAttention target gap | Incomplete after four actions | Arithmetic rejection, an out-of-window replacement length, then premature abstention; no fabricated target latency, but no accepted gap |

The [claim review](claim-review.json) checks actual returned claims and requested
conditions. It is implementation-assistant/model-assisted review, not independent
human judgment. The stored notes retain `semantic_support: not_reviewed`.

This is one demonstrated useful extraction through **MCP → real Qwen → Docker
passage inspection → provenance checks → saved experiment note**. It is not
three successful cases, a reliable autonomous researcher or a demonstrated
advantage over the host reading the same selected passages.

## Controller repair after the run

Qwen wrote mathematically valid lengths such as `13131 - 12848`. Our literal-only
guard rejected them. That was a controller compatibility problem, not a Python
execution error or proof that Qwen could not understand the passage. An actual
out-of-range request for 504 instead of 502 characters was correctly rejected.

After saving the unchanged run, the controller was repaired locally: explicit
read-call examples now supply lengths, and bounded integer addition/subtraction
is converted from AST data into literals before Docker execution. The same
source limits and inspected-citation checks remain mandatory. Raw authored and
normalized actions are retained. No host `eval` or model-authored Python execution
was introduced. New scripted regressions, including real Docker, test the
recorded arithmetic form and reject calls, booleans, exponentiation, shifts and
out-of-bound integers. This repair has **not** been live retested; tests cannot
measure how Qwen responds to the changed feedback.

The local repair is `9161aab`. Its final checks passed 37 focused and 1925 full
tests, with no skips and three existing SWIG warnings; Ruff and staged whitespace
checks passed. A new exact-source rehearsal also completed all three cases with
nine explicitly synthetic responses. No live-model result was replaced by it.

The consumed plan describes the pre-fix implementation `fe3a5a5`; its frozen
inputs and live prompt remain unchanged. New software hashes do not rewrite it.
No paid retry, RL, reward change or new research-productivity comparison ran.

## Shutdown and artifacts

The watchdog did not fire. Supervisor time before cleanup was 184.782 seconds.
Fresh reads at 22:14:21 UTC confirm workers 0/0, zero workers/pending jobs and
the full original configuration restored. All six owned rehearsal/live Docker
containers were independently checked absent. Actual invoice cost is unknown.

`live/*.json` contains case reports and resume responses; `live/records` contains
the saved experiment records and `live/reports` the durable request/report
objects. `session.requests.jsonl` retains all actual model requests/responses.
The live transport was an actual in-process MCP client/server connected to the
remote Qwen queue and local Docker, not native host tool installation or a new
stdio deployment test. The default source-only stdio path remains covered by tests.
`outcome.json`, `claim-review.json`, `endpoint-after.json` and `validation.json`
summarize evidence and limits. The old literal-only prompt is preserved as
`live-system-prompt.txt`. No human-review field or research reward was filled.
