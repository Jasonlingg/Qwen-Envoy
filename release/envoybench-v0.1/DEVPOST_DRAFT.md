# Devpost draft — EnvoyBench Studio

**Draft, not submitted.** Proposed track: Coding and Agentic Engineering;
confirm the final track and rules before submission. A local NVIDIA/Nebius
Studio run completed. A public judge-accessible build, the public video,
sponsor feedback, and the final submission remain **pending verification**.

## Title and short description

**EnvoyBench Studio: see how a paper-reading agent reached its answer**

EnvoyBench Studio lets a developer compare agents on the same scientific-paper
questions by opening one saved model run at a time, then switching runs to
inspect the same question. Each trace shows Python tool calls, observations,
the answer, and submitted source passages. A paper reader also lets a user
bring a public PDF or text source and inspect a new run when a working model
endpoint is configured.

## The problem and the build

An aggregate score can conceal failed tool calls, context limits, a plausible
answer with the wrong citation, or a model that improves mostly by refusing.
EnvoyBench uses a bounded, Docker-isolated Python tool environment over frozen
papers. Agents can search, read, extract passages, and submit an answer with
source IDs and spans. Studio presents saved comparisons as **Runs → Questions
→ Trace**, so a reviewer can inspect the actual actions and evidence behind a
result. The current Studio lists four saved runs: the paired September 30
comparison and the paired October 3 development smoke. The static export
includes all 562 September 30 turns in full. The release includes a
reproducible split recipe and checks that bind saved runs and quoted spans to
the frozen corpus.

The September 30 case study compares base Qwen3-8B with a trained v5 adapter
on 40 QASPER-derived questions. Provisional model-assisted review recorded
5/40 passes for base and 15/40 for v5. Both passed only **3/20 answerable
questions**. V5's observed advantage was mainly abstention and execution
reliability; this is **not** a validated improvement in substantive,
source-supported answering. The paper references and model-assisted grades
have not been independently human-validated. Exact source spans establish
provenance, not semantic support.

A separate October 3 two-question development smoke shows sampled-token
logprobs inside the saved trace. It is unscored. Token likelihood is debugging
telemetry, not correctness confidence; an interrupted 40-question expansion
did not establish a useful tool-error warning signal.

## NVIDIA model and Nebius runtime

The local Studio completed run `run_3666f88fff934274b76c3c1e6b4e4017`
with NVIDIA `nvidia/Nemotron-3-Ultra-550b-a55b` through Nebius Token Factory
(`https://api.tokenfactory.nebius.com/v1`). The model investigated a public
paper using the same Python tools and submitted an answer after ten steps in
15.367 seconds. Nebius reported 42,982 prompt tokens, 1,014 completion
tokens, and 26,112 cached tokens. Its answer identified the six encoder and
decoder layers, but one evidence span pointed to sub-layer text and another
truncated the decoder claim. The trace also contained two Python syntax errors
and three duplicate-action rejections. This proves the local integration ran;
it does not prove supported-answer quality or reliable execution. The exact
question, corpus revision, and runtime evidence belong with the release's
live-validation record.

Qwen is the subject of the saved September 30 comparison. The NVIDIA model
performs a live investigation in the Studio paper reader, separate from those
recorded Qwen scores. Verify the same path in the public judge-accessible build
before submitting.

## What changed and how to try it

The existing Envoy research-worker repository gained the EnvoyBench code-tool
evaluation protocol, the candidate split and verification path, the recorded
paired comparison, Studio trace inspection, the public-paper reader, and the
separate token diagnostic. Describe and date the final substantial changes
after verifying the public Git history.

The [candidate release README](README.md) gives commands to rebuild the
QASPER-derived corpus, verify the saved artifacts, and open Studio locally
without a GPU or API key. New paper investigations require a separately
configured model endpoint. The code is MIT-licensed; QASPER-derived data
retain attribution to Dasigi et al. and the dataset's CC BY 4.0 terms.

## Limits and final submission fields

This is a candidate evaluation protocol, not a validated leaderboard. The
September 30 review is provisional, the October 3 diagnostic is unscored, and
the trained adapter has not passed the project's held-out supported-answer
quality gate. The saved viewer itself makes no model call. The live paper
reader is a trusted-operator demo, not a multi-user hosted service.

- **Public repository and commit:** [pending final public release].
- **Working URL or test build with live Nebius/NVIDIA path:** [pending public validation].
- **Public YouTube video under three minutes:** [pending].
- **Sponsor feedback:** [pending].
- **Final track, eligibility, judging availability, and Devpost submission:**
  [pending confirmation].
