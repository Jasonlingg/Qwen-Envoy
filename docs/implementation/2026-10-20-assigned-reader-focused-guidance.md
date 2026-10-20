# Focused reader guidance — offline candidate, October 20, 2026

An opt-in `focused-v1` candidate now provides clearer source-reading instructions,
safe recovery feedback and a gap requirement before explicit abstention. Local
checks pass, but there has been **no new model inference, GPU activation or
training**. Whether Qwen follows the new feedback remains unmeasured.

## Diagnosis and hypothesis

The [live smoke](2026-10-20-assigned-reader-smoke-v2logs.md) supplied the correct
batch-size statement in the final model-input observation and in Qwen's own
citation. The final answer contradicted it. This was not a demonstrated transport
or text-loss problem. The other case assigned search results without displaying
them, received a grammar rejection, then abstained after reading abstract text.
It provided no gap report or next search. The source-visible
[diagnosis](../../examples/assigned-guidance-offline/diagnosis.json) records these
observations as model-assisted review; human review was not performed.

The inherited initial request repeats system instructions in the first user
message. That layout is retained to avoid introducing another variable; its
causal role in either failure is unknown.

The hypothesis is that clearer focused-reading instructions and concrete recovery
feedback may help the worker inspect relevant context and report unresolved
questions. The offline expected signal is narrower: exact baseline reproduction,
safe actionable recovery and rejection of gapless explicit abstention without
changing sandbox or request limits. Passing those checks prepares a candidate;
it cannot establish better model interpretation or answer quality.

## Implemented changes

`build_assigned_reader(..., reader_guidance="focused-v1")` selects the candidate
through the Python builder. It remains opt-in; the MCP CLI and existing defaults
continue using the baseline. No protected configuration or CLI file was edited.
Both Qwen and Nemotron receive the same effective candidate prompt in the model
policy and investigator; its exact SHA-256 is carried by the evidence packet.
The verifier binds guidance identity to the configured reader, and merged MCP
paper results retain that identity.

The prompt asks the reader to start with a displayed targeted search, inspect a
short passage around relevant hits before claiming a fact was not found, preserve
measurement conditions and ensure its answer agrees with its quoted text. These
are instructions, not mechanically verified relevance or entailment. There are
no task-specific answers, paper IDs or benchmark facts in the candidate prompt.

If the worker submits a lone `hits = tool(...)` or `result = tool(...)`, the
controller can suggest the corresponding `print(tool(...))` after the same strict
literal-call AST validation. Nothing executes during suggestion construction.
Other rejected forms get a generic safe example. The model must choose its next
action; rejected attempts still consume actions and no automatic retry/inference
is added. Matching tool results also carry a brief focused-context reminder.

For this candidate, explicit abstention requires a previously recorded gap with
what remains unestablished and a useful next search. The existing requirement for
two distinct successful document actions and source inspection still applies;
gap reports do not count toward it or launch searches. A missing report generates
recovery feedback; if the budget ends, the result remains incomplete. The verifier
also rejects candidate `no_evidence` packets without a valid gap record. Malformed
ancillary gaps still cannot discard independently verified quoted findings.

The controller cannot recognize every unsupported claim. In particular, it does
not use a negative-answer keyword heuristic to certify semantic correctness. A
wrong supported-looking answer can still pass span verification and must be
reviewed by the host or a person.

## Actual Docker replay

Run without a model endpoint or credentials:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/replay_assigned_guidance.py \
  --prepared examples/assigned-reader-smoke-prepared \
  --recorded examples/assigned-reader-smoke-v2logs \
  --output out/new-guidance-replay
```

The runner requires the pinned local Docker image and has no host execution
fallback or model transport. It first reproduces **both baseline cases** and
requires exact compared packet fields, entire trajectories and every recorded
model-input observation before it executes a candidate case.

| Saved case | Baseline replay | Candidate with the same fixed actions |
| --- | --- | --- |
| Batch-size question | Exact observations/results reproduced | Wrong recorded claim still passes provenance checking; no semantic improvement demonstrated |
| A5000/batch-8 question | Exact observations/results reproduced | Step 2 gets concrete retry feedback; step 4 gapless abstention is rejected |

In the second candidate case, the recording has no next model response. Replay
stops with `RecordedActionsExhaustedError`, represented as an error packet and
explicitly marked `recorded_actions_exhausted`. This is replay bookkeeping, not a
new model or provider failure. It does not simulate a recovery or invent a gap
report. The attempted fifth replay policy call is **not a paid inference call**.
Actual model requests and GPU activations are zero throughout.

**Fixed-action replay cannot measure how Qwen would respond to the new feedback.**
It also cannot measure how Nemotron would respond, semantic support improvement,
or fine-tuning benefit. The [replay report](../../examples/assigned-guidance-offline/replay.json)
and four replay packets preserve those distinctions.

Baseline prompt SHA-256 remains
`721d7dc8bb9c4c8bef05b53923f9daa581715b6750e206c34d78e722cac96e56`.
Candidate SHA-256 is
`c0f14e1dca80c88d477f6a96b1676c621b54ef93c8f27de94d701617ea67d7ad`.
V1 tool preamble bytes and SHA-256 remain unchanged. Cached-tokenizer checks put
the two candidate initial requests at **2,462/2,488 tokens**, versus baseline
1,938/1,964: an increase of 524 each, within the existing 7,168 input limit.
Later-turn token use and live latency are unmeasured; the existing transport
guards remain required.

## Verification and next boundary

Commits `0c7bbd0`, `59fcd70`, `e66cf05` and `9ddf36b` separate prompt identity,
recovery/gap checks, Docker replay and scripted MCP integration. Focused tests
include the actual rejected-assignment pattern, recovery, exhausted action
budgets, forged/downgraded packet metadata, gap provenance, preservation of quotes
when ancillary reports fail, and real MCP calls backed by Docker and scripted
responses. Commands, final test counts, independent review and preservation
hashes are in the [validation artifact](../../examples/assigned-guidance-offline/validation.json).
The final focused integration suite passed **102 tests** in 10.68 seconds; the
full offline suite passed **1,662**, no skips, with three existing SWIG warnings
in 83.09 seconds. Ruff and whitespace checks passed. Independent code and replay
review found no actionable issue.

All original live artifacts, frozen prompt/plan and protected files are unchanged.
The old live plan intentionally pins old implementation-file hashes; it must not
be edited or reused as authorization for candidate execution. A future live
candidate evaluation needs a newly frozen preparation and a separately approved
bounded run. The two familiar questions can serve as development regressions;
fresh questions and appropriate controls are needed before broader quality or
fine-tuning claims. No model promotion or training run is justified by this
offline result.
