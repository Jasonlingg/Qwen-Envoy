# Devpost draft — QASPER Agent Studio

**Draft, not submitted.** Proposed track: Coding and Agentic Engineering;
confirm the final track and rules before submission. A local NVIDIA/Nebius
Studio run completed. The [saved-run viewer](https://jasonlingg.github.io/Qwen-Envoy/)
is public. Judge access to the live test build, the public video, sponsor
feedback, and the final submission remain **pending verification**.

## Title and short description

**QASPER Agent Studio: see how a paper-reading agent reached its answer**

QASPER Agent Studio is an independent Qwen Envoy study adapting 40 questions
from [Ai2's QASPER (Dasigi et al., 2021)](https://aclanthology.org/2021.naacl-main.365/)
to known-paper investigation through Python tools. It lets a developer inspect
agents on those scientific-paper questions by opening one saved model run at a time, then switching runs to
inspect the same question. Each trace shows Python tool calls, observations,
the answer, and submitted source passages. A paper reader also lets a user
bring a public PDF or text source and inspect a new run when a working model
endpoint is configured.

## The problem and the build

An aggregate score can conceal failed tool calls, context limits, a plausible
answer with the wrong citation, or a model that improves mostly by refusing.
The study uses a bounded, Docker-isolated Python tool environment over frozen
papers. Agents can search, read, extract passages, and submit an answer with
source IDs and spans. Studio presents saved comparisons as **Runs → Questions
→ Trace**, so a reviewer can inspect the actual actions and evidence behind a
result. The historical release contains four saved run rows: the paired September 30
comparison and the paired October 3 development smoke. The static export
also shows a fifth, amended NVIDIA/Nebius reference row with 40 saved episodes
and official Answer F1. It includes all 562 September 30
turns in full. The release includes a
reproducible split recipe and checks that bind saved runs and quoted spans to
the frozen corpus.

The September 30 case study compares base Qwen3-8B with a trained v5 adapter
on 40 selected QASPER questions. The questions disclose the target paper, so
the task measures finding and using evidence within it rather than paper
discovery. The original questions, answers, and annotations are QASPER's;
this is an agent adaptation, not a new question benchmark or official Ai2 product.

Scoring the unchanged answers with the
[pinned official QASPER evaluator](../../benchmarks/envoybench/vendor/qasper/README.md)
gives **19.92 for base and 30.69 for v5 on the 0–100 Answer F1 scale**.
The [score artifact](../qasper-agent-study/qwen-official-score.json) uses all
86 original annotations for these 40 questions, takes each answer's best
normalized token-overlap F1 across its references, and averages over all 40.
Missing submissions score zero. On the fixed 20 source-labeled answerable
questions, F1 fell from 29.83 to 21.37; the unanswerable group rose from
10.00 to 40.00. The aggregate gain hides that trade-off. This is the official method on a selected
known-paper workload, not a full-QASPER result. It does not assess citation
support, and refusal paraphrases are not rewritten into `Unanswerable`.

Separate provisional model-assisted support review recorded
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

The historical single-paper local Studio investigation completed as
run `run_3666f88fff934274b76c3c1e6b4e4017`
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

Qwen remains the subject of the saved September 30 paired comparison.
A separate bounded 40-question NVIDIA/Nebius attempt initially stopped at its
$2 local estimated-cost ceiling (recorded estimate $1.947862). It finished 38
episodes, interrupted question 39, and did not attempt question 40. The user
then authorized a $25 **total** local cap. The continuation restarted question
39 from step 1 because its live state could not be restored, and attempted
question 40 for the first time. The [derived full result and original partial
trace](../qasper-agent-study/README.md) are both retained. The full run has
**28 submissions, 12 missing predictions, and 19.03% official QASPER Answer
F1**, with **383 requests and $2.035422** total estimated catalog-rate usage.
Question 39 used all 15 steps without submitting; question 40 submitted `85%`
at step 2, receiving zero F1. Studio displays the amended result as its fifth
row. It shares the frozen questions, papers, and tool interface, but differs
in provider, serving context, model revision policy, and runtime. Its results
are supplementary, not a controlled extension of the base-versus-v5 comparison.
Official Answer F1 does not supply the missing semantic support reviews.
Do not transfer the historical single-paper run's timing, usage, or diagnosis
to that new study. Verify the live path in the public judge-accessible build
before submitting.

## What changed and how to try it

The existing Envoy research-worker repository gained the QASPER code-tool
evaluation protocol, the candidate split and verification path, the recorded
paired comparison, Studio trace inspection, the public-paper reader, and the
separate token diagnostic. The QASPER Agent Studio update adds the original
QASPER Answer F1 metric and separate supplementary-run inspection; internal
`envoybench` package and artifact IDs remain compatible. Describe and date the final substantial changes
after verifying the public Git history.

The [study release README](../qasper-agent-study/README.md) gives commands to reproduce
official Answer F1 and open the updated Studio. The
[historical release README](README.md) gives commands to rebuild the
QASPER-derived corpus, verify the saved artifacts, and open Studio locally
without a GPU or API key. New paper investigations require a separately
configured model endpoint. The code is MIT-licensed; QASPER-derived data
retain attribution to Dasigi et al. and the dataset's CC BY 4.0 terms.
QASPER Agent Studio has no official affiliation with Ai2.

## Limits and final submission fields

This is a selected QASPER agent study, not a full-benchmark leaderboard. The
September 30 review is provisional, the October 3 diagnostic is unscored, and
the trained adapter has not passed the project's held-out supported-answer
quality gate. The saved viewer itself makes no model call. The live paper
reader is a trusted-operator demo, not a multi-user hosted service.

- **Public repository and technical release:** [Qwen-Envoy](https://github.com/Jasonlingg/Qwen-Envoy),
  [case-study commit b53d70b](https://github.com/Jasonlingg/Qwen-Envoy/commit/b53d70b).
- **Saved-run demo:** [public Studio](https://jasonlingg.github.io/Qwen-Envoy/),
  read-only; does not call Nebius or any model.
- **Working URL or test build with live Nebius/NVIDIA path:** [pending public validation].
- **Public YouTube video under three minutes:** [pending].
- **Sponsor feedback:** [pending].
- **Final track, eligibility, judging availability, and Devpost submission:**
  [pending confirmation].
