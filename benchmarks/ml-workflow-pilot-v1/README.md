# Five-question ML workflow comparison — preparation only

Status: draft questions, rubric and budget for user review. No comparison answers
collected; execution is not authorized. All scenario measurements/hardware
numbers are hypothetical inputs, not observations from this project. These are
assistant-authored development questions; they are not untouched confirmation
data, an original research benchmark, or independent gold labels.

## Question and hypothesis

Does the same Codex host produce more supported, actionable ML experiment
recommendations with the research/experiment MCP available than with its normal
web and file tools? Expected signal: more useful initial answers, fewer material
errors, or less human correction effort under matched conditions. Any advantage
must be observed rather than inferred from tool reliability or exact quotes.

The exact public questions and common instructions are in
[questions.json](questions.json). The five decisions are:

| ID | Question | Capability being examined |
| --- | --- | --- |
| MLP01 | Which two smaller open-weight models should I test for document extraction on one 24 GB GPU with two concurrent 6,000-token requests? | Discovery, source/license checks, resource applicability and a matched model evaluation |
| MLP02 | With the supplied TTFT, decoding and p95 figures, what should I measure or change first to reduce reader latency? | Bottleneck reasoning and a controlled measurement rather than a list of optimizations |
| MLP03 | With missing-evidence, wrong-answer and schema failures, should I fine-tune next or test a different fix first? | Failure diagnosis, data separation and a diagnostic that could change the decision |
| MLP04 | Does speculative decoding transfer to this single-GPU 8B workload? Compare two primary papers and implementation constraints. | Critical reading and transfer of research claims rather than repeating speedup headlines |
| MLP05 | Do incomplete saved quality/latency/cost results justify adoption under the original three-gate rule? | Honest arithmetic, matched populations and preservation of unknowns |

These are realistic test scenarios, not a claim that they reproduce the user's
actual workload distribution. Replace/revise them with user problems before
freezing if they do not represent intended use. Do not answer these questions
during preparation or use their eventual answers to tune a confirmation set.

## The two arms

| Arm | Capabilities |
| --- | --- |
| A | Normal Codex web, document, shell and file capabilities; our two MCP servers absent from its advertised inventory |
| B | Same host/settings/normal tools plus the five research source tools and five experiment tools; Qwen and server-side reasoner off |

Both receive exactly the same question, structured hypothetical input and common
instructions. Both can save ordinary files. B may choose ordinary tools and
web fallbacks; do not force MCP calls or score storage format as useful answering.
Record MCP usage so an unused integration is not described as reader effectiveness.
No Qwen is used: this tests current source/record tooling, not worker benefit or
fine-tuning. A later worker comparison needs a separate task and budget.

Use ten fresh native sessions, one answer per question per arm. Use isolated
working/output directories, empty per-case local source/record caches, identical
ordinary permissions, the same declared host model/reasoning settings and answer
format. Neither arm may see the reviewer rubric, the other answer, old Task 5
recommendations, or previous benchmark outputs. Only copy that question's public
input into its working directory; do not expose this whole benchmark/repository.
Keep ordinary capabilities equal and archive the actual advertised inventories.
Balance arm order: A→B on cases 1/3/5, B→A on 2/4. Reviewer display order is
randomized separately and identities remain hidden until grades are recorded.
Save what live searches actually returned; hidden provider caches cannot be
guaranteed cold. Complete each pair close together with the same source cutoff.

This is an answer-to-decision pilot. It does not measure executing recommended
experiments, long-term project memory, or whether any model improvement occurs.
Case 5 examines interpretation of supplied results rather than durable retrieval
from past sessions. A follow-up memory/reuse evaluation would be separately scoped.

## Exactly what is measured

The [reviewer-only rubric](reviewer-rubric.json) defines four dimensions, each
scored 0, 1 or 2: factual correctness/source support; fit to the workload;
actionable experiment/measurement; and honest uncertainty/fair comparison.
All question-specific must-haves are required. An initial pass needs at least
6/8, no zero dimension, and no major unsupported claim or binding-constraint
violation. Useful but incomplete answers are partial; unusable/major-error
answers and timeouts fail. Unfinished human/source review is unknown, not a pass.

| Measurement | Exact report |
| --- | --- |
| Primary: useful initial answers | Human-reviewed passes / all five scheduled questions for each arm; report partial/fail/review-unknown separately and the paired pass-count difference |
| Direct preference | Blinded per-question A/B win, loss, tie or unknown, with reasons; no missing pair silently removed |
| Material errors | Number of answers with a major factual/unsupported claim (0–5), plus claim-level checks; empty refusal cannot pass an answerable task |
| Diagnostic scores | Four 0–2 scores per answer, total 0–8; no aggregate score hides a failed must-have |
| Human correction effort | Timed editing/source investigation until the brief is minimally usable, up to 10 minutes per answer; unrepaired answers are censored/unfinished rather than reported as cheap fixes |
| Response time | Actual elapsed prompt-release-to-final-answer time; timeouts and reviewed partial/fail answers assigned the 600-second ceiling for the separate time-to-initially-useful summary; unfinished review makes that derived summary unknown |
| Reliability | Every timeout, unavailable source, tool error, malformed output and failed request, plus actual tool usage |
| Resource use | Observable host/model/tool requests and tokens, actual spend if exposed, and all retries/failures. Hidden subscription/token costs remain unknown; they are not zero |

A human reviewer must check material claims in source context and compare the
plans against the frozen requirements. The user can assess practical usefulness;
record reviewer expertise/limitations. Model-assisted triage remains separately
labeled and cannot fill a human verdict. Provenance checks do not establish
semantic support. Missing human/source reviews make the corresponding comparison
inconclusive. Record any formatting/source cues that might reveal an arm despite
blinding. The rubric and blinded review follow the task-specific,
human-calibrated approach in [official OpenAI evaluation guidance](
https://developers.openai.com/api/docs/guides/evaluation-best-practices).

Do not reward more citations, longer answers, fancy formatting or more calls.
Accept a justified deferral with a bounded next measurement where the scenario
is underdetermined; do not prescribe a winning model or method to reviewers.

## Proposed budget — not execution approval

| Resource | Proposed ceiling |
| --- | --- |
| Answer generation | 10 answers: five questions × two arms, one attempt each; no selective retries |
| Elapsed answering | 600 seconds per answer; 100 minutes total maximum answering time, excluding setup/review |
| External research | At most 20 web/search/open/fetch tool invocations per answer across all research tools combined; compound calls also record their underlying requests where exposed |
| Final brief | Up to 800 words plus a short source list, same instructions in both arms; this is an output contract, not an enforced token-spend cap |
| Human work | Budget at most 10 minutes correcting each answer; source/initial blind review is separate and logged, so this is not a complete estimate of review labor |
| Extra reader/reasoner API calls and GPU starts | Zero; no Qwen/Nemotron worker or additional paid model endpoint is used by this pilot |
| Codex host tokens/dollar spend | Unresolved until the actual client/account telemetry and enforcement are checked; account usage is not assumed free |

No hard native-host token or monetary cap has been established. Elapsed/tool
ceilings require a verified controller/operator stop procedure before execution;
the planning MCP's recorded budgets do not enforce them. Do not run if the
approved limits cannot be observed/enforced. There is no cost-savings claim
without complete host/tool accounting, and no quality/efficiency winner is
predeclared for this five-case diagnostic.

## Readiness and interpretation

Before execution, the user reviews these questions, workload relevance, rubric
and proposed ceilings; the operator resolves and records the exact host settings,
isolated inventories/caches, stop mechanism, source/trace export, human reviewer
and account budget. Native planning tools are not loaded in the current session;
stdio checks alone cannot be relabeled as the native A/B comparison. Freeze the
approved inputs/manifest hashes before either arm answers. Any substantive change
after seeing outputs creates a new development version and is disclosed.

Report all five pairs, including null/harmful outcomes. A five-question pilot can
expose tool-routing failures and show promising or unhelpful patterns; it cannot
establish broad superiority or statistical equivalence. A larger confirmation
study needs new questions, a predeclared decision rule and separately approved
execution/review effort. The older [three-arm source/reader protocol](
../../docs/implementation/2026-10-19-native-web-comparison.md) remains a broader
proposal, not an executed benchmark or authorization for this pilot.

Preparation created documents/input JSON only. No answering sessions, model
workers, training, benchmark execution or infrastructure were started.
