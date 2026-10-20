# Native comparison preflight — October 19, 2026

The user approved the five-question comparison. Answer collection has not
started because the approved hard research-call ceiling is not enforceable with
the native controls verified so far. This is a setup finding, not an A/B result.
The [manifest](preflight.json) binds the questions, rubric and observed inventories
by raw SHA-256. Human review is unassigned and no quality verdict is recorded.

## Observed native setup

Codex CLI 0.160.0's local app-server resolved `gpt-6.1-sol`, provider `openai`,
with explicit `medium` reasoning in both ephemeral threads. No `turn/start`
request was made. A resolved model name does not prove inference entitlement;
generation, native web results and final-answer export remain untested.

| Check | Arm A | Arm B |
| --- | --- | --- |
| Active MCP tools returned by native inventory | 0 | 10: five source tools, five experiment tools |
| Research profile | Disabled | Sources only; `investigate` absent |
| Loaded instruction sources | Empty | Empty |
| Read a public canary in its workspace | Allowed | Allowed |
| Read a canary outside its workspace in this repo | Denied | Denied |
| Host generation turns | 0 | 0 |

The [A inventory](preflight/A-inventory.json) and
[B inventory](preflight/B-inventory.json) include actual tool schemas and settings.
Disabled MCP server names remain in the inventory response, with zero tools;
they were not connected. This RPC establishes MCP availability, not the complete
model-visible built-in tool inventory. Sources were not searched or fetched.
Existing sign-in was used without reading or copying credential contents.
Host dollar costs remain unknown; zero generation turns is not a complete
account-billing audit.

The first setup attempt placed workspaces under the repository, which inherited
`AGENTS.md` and the project MCP registrations. Subsequent attempts moved the
workspaces outside the repository and explicitly disabled unrelated servers.
Intermediate transport/profile configuration errors were corrected before the
archived successful checks. These zero-turn setup attempts are not answer
attempts or evidence of answer quality.

The tested named filesystem profile permits minimal platform access, the Codex
executable installation and the current workspace. It blocks the repository
canary. The MCP servers run separately with explicitly selected roots. A real
run must create new source/record/artifact roots for every B case; these probe
roots must not be reused for answering. Matching plugin, app, memory and helper
agent integrations were disabled in both probes. Remaining ordinary capabilities
must be verified before freezing an execution manifest.

## Research-call limit

The protocol says at most 20 combined research-tool invocations per answer and
requires observable/enforceable limits before execution. Official
[Codex hook documentation](https://learn.chatgpt.com/docs/hooks#tool-coverage)
states that hosted tools such as WebSearch bypass local pre/post tool hooks.
App-server's item notifications and turn interruption support monitoring, but
they do not establish a before-dispatch gate for each hosted search. Concurrent
or already-dispatched calls could exceed the count before interruption.

No supported hard native-host research-call cap was established through the
installed CLI/schema and configuration reference. Replacing Codex with the
repository's JSON host loop would change the comparison. Disabling ordinary web
search would also change the approved control. Neither workaround was used.

The user has been offered a concrete alternative: treat 20 calls as a monitored
target, interrupt at that observed count, retain any in-flight overshoot and
report it as a deviation. This requires a protocol change and is awaiting a
reply. Ten-minute termination would still require a validated controller before
answering. There is no enforced host token/dollar cap. Extra Qwen/provider
endpoint calls and GPU starts remain zero.

The recorded [probe](preflight/native_inventory_probe.py) can be rerun from the
repository root with Python; it uses the installed `codex` binary and the
repository's existing Python MCP environment. It creates temporary workspaces,
writes diagnostics under ignored `out/ml-workflow-pilot-preflight`, starts no
model turn and terminates its own app-server process groups. It is an operator
reproduction artifact, not an answer runner or a production budget controller.

## Devin Fusion comparison

The architecture was not originally derived from Devin Fusion. After the user's
question, the official [Devin Fusion](https://cognition.com/blog/devin-fusion) and
[Local Fusion](https://cognition.com/blog/local-fusion) descriptions were read.
Their frontier lead and cheaper sidekick each keep their own tools and context;
the lead delegates, reviews and can reclaim work. That resembles the optional
host-plus-Qwen-worker responsibility split here.

This project currently uses bounded per-paper workers rather than Fusion's
persistent sidekick context and dynamic routing. The source/record pilot keeps
Qwen off, so it cannot measure that delegation design or Qwen's usefulness.
Cognition's coding-task results do not establish this project's quality or cost.
No architecture was changed during benchmark preparation based on that analogy.

## Verification

The final probe passed for both arms. Artifact checks confirmed matching model
settings, the expected MCP schemas, empty instruction sources, file boundaries,
five question IDs, input hashes and local documentation links. Question/rubric
content differs from the preparation commit only in approval status. Ruff and
`git diff --check` passed. The full repository suite passed: **1,478 tests**, with
three existing SWIG dependency warnings, in 53.64 seconds. All ten pre-existing
file hashes still match the Task 6 preservation receipt.
