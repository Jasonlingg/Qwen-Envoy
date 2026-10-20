# Research MCP versus ordinary web research

Status: proposed three-arm evaluation protocol; no three-arm answers collected. This document
starts no infrastructure and makes no model calls. Freeze tasks, review criteria,
execution settings and a costed run manifest before collecting results.

The separately approved [five-question ML decision pilot](
../../benchmarks/ml-workflow-pilot-v1/RESULTS.md) collected ten native A/B answers
with Qwen off. None of the B sessions called MCP; human quality review remains
pending. That development collection does not execute or authorize this broader
three-arm reader study.

## Question and hypothesis

Does the same main assistant produce more useful, supported research answers
when the research MCP is available than when it uses its ordinary web tools?
Hypothesis: paper-specific retrieval and bounded reader delegation improve
evidence coverage or reduce total effort on multi-paper research. Expected
signal: better independently reviewed answers, or comparable answer quality with
lower measured end-to-end time or total cost. Either outcome must be observed;
exact quotes, tool reliability and fewer host calls alone cannot establish it.

This is a product comparison. Base-versus-fine-tuned Qwen remains a separate
model experiment. A product win cannot be attributed to fine-tuning.

## Three capability sets

| Arm | Main assistant can use | Interpretation |
| --- | --- | --- |
| A | Its ordinary web and document tools | Actual user baseline |
| B | The same tools plus model-free research MCP tools | Value of discovery, pinned text and passage retrieval |
| C | The same tools plus the full MCP with Qwen readers | Value of the complete product |

B must hide `investigate` and reader-advertising instructions from its tool
inventory. Use the implemented `--tool-profile sources`, which exposes only the
five model-free tools and rejects reader/reasoner configuration. `--reader none`
alone in the full profile still advertises an unconfigured `investigate` tool.
Archive each native session's actual advertised inventory before running;
server flags or a stdio check do not establish native-session isolation.

A versus C is the primary product comparison. B versus C diagnoses the marginal
value of readers; A versus B diagnoses the source tools. All arms use the same
main model, reasoning setting, requested answer format and ordinary tool access.
Keep native browsing/PDF facilities and normal assistant capabilities available;
do not weaken A to make delegation look useful. The optional server reasoner is
off because the main assistant synthesizes in every arm.

Run fresh native client sessions with the actual tool integrations. Let the host
choose tools, including web fallbacks in B/C. Record whether and how often each
MCP tool and Qwen were used. A C result without Qwen requests measures tool
availability, not reader effectiveness. Do not force Qwen use in the primary
product comparison; a prescribed-delegation diagnostic is a separate analysis.

The [existing five-question pilot](2026-10-17-research-workflow-pilot.md) uses
preselected papers, a frozen corpus and a custom JSON host-action protocol. It
can diagnose delegation under controlled sources. It cannot be relabeled a
native Codex/web-search comparison or used as untouched confirmation data.

## Tasks and staged execution

1. Use five development tasks to check connectivity, native tool routing, trace
   capture, budget accounting and review effort. Include a discovery question;
   stop early on transport/accounting failures. These results are developmental.
2. Freeze 20 new real user questions before any arm answers them. Suggested
   composition: five discovery, five multi-paper comparison, five detail or
   limitation checks, and five deployment decisions requiring explicit evidence
   gaps. Record scope and a date cutoff in each prompt. This is a small paired
   study, not a universal benchmark or a statistically powered equivalence test.
3. Run each question in A, B and C: 60 answers. Randomize and balance arm order;
   run the paired sessions close together. Keep histories, caches and outputs
   separate across arms. Fix time/request ceilings before execution and apply
   the same main-assistant ceilings. Worker requests count toward total cost.
4. Review all answers before revealing identities. Report every scheduled task,
   including failures and timeouts. Select any later repetitions before looking
   at outcomes; do not rerun only losses or promote development results.

Tasks should reflect the intended arXiv/text research use case, with genuinely
useful discoveries and comparisons. Do not select only questions whose answer
is already a convenient cached paragraph. Record non-arXiv, table/figure and
unavailable-source cases as separate scope/coverage stress cases; include them
in the main sample only if they represent intended user work. A targeted
arXiv result cannot establish superiority across all internet research.

Have a human freeze the question-specific requirements before inference. These
include needed subquestions, evidence scope, acceptable uncertainty and any
independently established relevant sources; they are reviewer-only. Do not
claim exhaustive discovery recall against an incomplete reference list. If
the author inspected the answers or tuned against the questions, disclose it
and reserve different questions for confirmation.

## Fairness and reproducibility

Record exact host product/version, declared model and resolved model if exposed,
reasoning/decoding settings, permissions, tool inventory, common prompt/hash,
MCP/Qwen configuration, code revision, adapter/base identities, Docker digest,
context limits, concurrency, timestamps and all ceilings. Record unavailable
revision/seed/token telemetry as unknown. A different host is a different study;
Nemotron results cannot stand in for the native Codex baseline.

Save queries, search results, source URLs/versions, observed source text and
hashes, tool errors, model requests where exposed, final answers and fallback
actions. Live search indexes change; preserve what each arm actually saw and
record access times. Do not feed A the MCP's selected papers or use C's output
as A's instructions. A later frozen-source diagnostic can isolate retrieval
effects but changes the task from live discovery.

Use clean controlled local/client library and discovery caches for the primary
first-use condition. Remote search/provider caches may be invisible and cannot
be promised clean; disclose this limitation. Share a Qwen model server only
within a declared session. Freeze the number of GPU starts, questions per
session and startup/idle/shutdown allocation before timing. For example, one
question per cold session and five questions per session are different workloads
and cannot be substituted after results. The user's accepted loading delay does
not remove it from measurement. Record cold loading and warm answering
separately. Evaluate repeat-use caching separately, with equal opportunities to
retain prior source work. The session workload is still a manifest prerequisite.

## Review and measurements

Use human reviewers with arm identities concealed and answer order randomized.
Present the same requested answer format and a neutral source map; keep traces
and provider names hidden until judgments are frozen. Preserve raw answers and
disclose remaining cues that may reveal the arm. Reviewers inspect full source
context where needed. Model-assisted review is separately labeled and cannot
fill an independent human-review field.

For every answer record useful-answer verdict (pass/partial/fail), completeness
against the frozen task requirements, appropriate uncertainty, source relevance
and freshness. Review factual claims for semantic support, correct scope and
qualifications; count major unsupported/contradicted claims separately. Check
citation provenance separately. An exact quotation can be irrelevant, and an
empty refusal can avoid false claims while failing an answerable question.

Report per-question paired wins/losses/ties, supported-useful-answer rate, major
claim errors, false refusals, source/worker failures, human correction time and
end-to-end latency. Report task groups separately so good abstention cannot
hide poor useful answering. Human correction follows a fixed protocol and is
timed separately from blind scoring. Primary latency is mean time to a useful
supported answer across all tasks, including allocated cold-start time. Charge
the fixed per-task elapsed ceiling for every non-pass or timeout; report actual
elapsed time, median and cold/warm breakdowns separately. Freeze ceilings and
allocation in the manifest so quick failed answers cannot create a speed win.
Include all startup, retries, failed
requests, host usage, Qwen GPU active/idle/cleanup time and attributable storage
in cost. Avoid double-counting overlapping readers. Subscription or hidden
host usage is unknown, not free. Missing full costs preclude a cost-savings claim.

## Proposed decision before the 20-question study

Freeze one primary route after development and before confirmation:

- **Quality route:** C produces at least four more human pass verdicts than A
  out of 20, with no increase in answers containing a major unsupported claim.
  Report time/cost tradeoffs even if quality improves.
- **Efficiency route:** C has at least as many passes as A, no increase in
  major-claim-error answers, and at least 20% lower measured total cost or mean
  time to a useful answer as defined above. Choose cost or latency before
  confirmation. Report paired
  quality regressions and task groups; observed equality in 20 cases is not
  statistical proof of non-inferiority.

These are proposed engineering continuation thresholds, not established
statistical confidence or universal superiority. Both routes also require at
least 16/20 C passes, including at least 12/15 on the three answer-seeking task
groups, and complete paired coverage/review. Freeze any task-composition change
with matching thresholds before execution. Missing measurements or reviews make
the corresponding conclusion inconclusive. Report uncertainty alongside counts;
expanding to a powered confirmation study needs a separately fixed analysis.

If B matches C's usefulness while using less time/cost, retain the MCP source
tools and improve or disable delegation by default. If A matches or beats C,
report that result and use the traces to decide whether another iteration is
worthwhile. Publish the comparison's tasks, scope, failures and limitations;
state any advantage only for the measured tasks and conditions.

## Immediate prerequisites

Resolve the previous external endpoint 403 before another paid session. The
[October 18 attempt](../../reports/research-workflow-pilot-live-attempt-20261018.json)
made zero inference requests and produced no answer comparison; its temporary
pod was deleted and absence verified. Build a separately costed manifest for
the native three-arm study, including review effort and enforced request/token
limits. The previous frozen-pilot US$5 planning approval is not a budget estimate
for these 60 answers. The separate native A/B collector now provides isolated
host sessions and trace export, but adding reader arm C, its request/token/spend
controls and the required human review remain prerequisites for this study.
