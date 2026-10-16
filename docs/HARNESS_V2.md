# Opt-in harness V2

V1 remains the default. Its tool preamble is unchanged, with SHA-256
`233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71`.
Select `QwenInvestigator(harness_version="v2")` to add the separate V2 layer.
This changes the investigation protocol; it does not establish model improvement.

V2 accepts common numeric argument aliases and numeric strings, supplies usage
messages, and accepts literal `submit(answer=..., citations=[...], evidence=[...])`
as the only statement in a turn. The host converts that call with an AST parser;
variables and executable expressions are rejected without evaluation. A submit
call inside a larger program receives an error from the Docker sandbox. Invalid
non-Python replies receive up to two free format re-asks per action.

Corpus discovery `search(query)` still defaults to five documents. Only
`search_within(doc_id, query)` and its paper-scoped aliases default to eight
windows. Aliases accept both `search(doc_id, query)` and
`search(query, doc_id)`. Paper-scoped tools still require document discovery in
the investigator, and successful scoped search counts as source inspection.
`read(doc_id, start, end)` returns a bounded passage; use
`read(doc_id, offset=start, length=length)` or `passage(doc_id, start, length)`
when specifying a length. Passage results remain capped at 3,000 characters.

## Reproduce the fixed-action diagnostic

Hypothesis: V2 accepts more saved protocol mistakes while V1 remains unchanged.
Expected signal: fewer execution errors for the same saved actions. Decision
rule: first reproduce every V1 observation, reward, and terminal flag exactly;
if any action differs, block the V2 comparison. Then report each V2 action's
outcome without drawing conclusions about model policy or answer quality.

```bash
python scripts/replay_harness_v2.py \
  --summary-output reports/harness-v2-replay-20261016.json
```

The frozen corpus must already exist under `benchmarks/envoybench/data`; use
the existing dataset builder if necessary. Docker and the recorded sandbox image
`sha256:9861aeb3d3277267a0d5047cf3c12e1dfe92226721c7db4821e9a9979fc7272a`
must be available. There is no host execution fallback and no model request.
Each question uses a fresh Docker container, network disabled, with the exact
corpus mounted read-only. At most four questions run concurrently.

Complete V1 questions are saved atomically to
`out/harness-v2-replay.v1-checkpoint.json`. After interruption, repeat the command
with `--resume`. Reuse requires matching run, results, corpus, image, benchmark,
and infrastructure hashes plus every saved action/observation/reward/terminal
flag. Incomplete or changed entries are rejected. V2 is always rerun.

The replay imports hash-checked, trusted environment infrastructure from git
commit `1976690`. Model-authored Python is passed only to `DockerREPL.execute()`.
The older environment is needed to reproduce the October recording's syntax
hints and **fail-open verifier setting**. That setting is preserved only for
historical reproduction: an old accepted submission does not establish evidence
integrity. Current investigation verification remains separate.

`out/harness-v2-replay.json` includes complete per-question/action observations.
The committed compact report includes every action, outcome, exact-match flag,
observation length/hash, source/model identities, and the full report hash.
It does not normalize traceback paths, line numbers, feedback, or source output.

## October 16 saved-action results

All **40 questions and 372 V1 actions matched exactly**, including observations,
rewards, and terminal flags. The V2 diagnostic then completed on all saved
actions, with no model request. Executed source is archived in commit `03c434e`;
the subsequent artifact finalizer adds hashes and historical skip categories.

| Action outcome | Recorded V1 replay | V2 diagnostic |
|---|---:|---:|
| Executed actions | 318 | 268 |
| Executed without detected execution/tool error | 224 | 248 |
| Failed executed actions, including usage feedback | 94 | 20 |
| Format deferrals requiring a new model response | 0 | 50 |
| Historical accepted submissions, not regraded | 28 | 28 |
| Historical submissions rejected by the verifier | 16 | 16 |
| Other historical verifier skips | 10 | 10 |

V1 failures were 50 SyntaxErrors, 33 TypeErrors, three NameErrors, two KeyErrors,
and six structured tool errors. V2 failures include **11 usage messages**, three
NameErrors, five KeyErrors, and one structured tool error. Usage feedback remains
a failed call. The 50 syntax mistakes became deferrals, not successful execution.
There were 24 additional actions without detected errors; three TypeErrors became
KeyErrors. No recorded action used the supported standalone literal submit form;
that conversion path is covered by focused tests.

This is a **tool/submit/format diagnostic over historical benchmark actions**,
not a full `QwenInvestigator` V2 episode replay. Historical QASPER known-paper
and verifier decisions are retained. Product discovery and inspection gates are
tested separately. The result cannot measure a model's response to new feedback.

The committed [per-question/action report](../reports/harness-v2-replay-20261016.json)
binds the complete local report, `out/harness-v2-replay.json`, to SHA-256
`585372e8fca1d67f44300db90a3ebb35d81cae1d751460b9931bb4517c236aaf`.
It also pins the converter and V2 layer hashes. The original execution output is
retained locally as `out/harness-v2-replay.raw.json`.

## Interpretation

Replay reruns the same recorded actions. It shows which actions v2 would accept
or answer with a usage line; it does not show how a model would respond to those
replies. A live rerun on the same 40 questions measures that.

The V2 arm is a tool/protocol diagnostic, not a complete counterfactual agent
run. Recorded environment rejections remain skipped, submission syntax can be
converted without regrading the answer, and format retries cannot supply a new
model action. Later fixed actions can encounter new errors after earlier V2
actions succeed and alter cumulative state. Exact quote matching proves
provenance, not semantic support. Neither this replay nor the MCP implementation
proves that fine-tuned Qwen improved.
