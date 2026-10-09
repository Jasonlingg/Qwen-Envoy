# Devin Fusion sidekick research and Envoy design implications

September 28, 2026. This is a design review, not an executed Envoy experiment.
Claims about Cognition's system come from its published posts; recommendations
for Envoy are explicitly labeled as inferences. The relevant local baseline is
the [personal-memory readiness screen](PERSONAL_MEMORY_READINESS_EVAL.md), and
the locked product test remains the [vault transfer protocol](TARGETED_QWEN_VAULT_TRANSFER_TEST.md).

## What Cognition actually built

[Devin Fusion](https://cognition.com/blog/devin-fusion) has a frontier **lead**
and a lower-cost **sidekick**. Both are tool-using agents with their own persistent
contexts. The lead owns the user conversation, plan, ambiguous decisions, and
final review. It hands a bounded task to the sidekick and can reclaim work that
requires better judgment. Cognition also reports lightweight classifiers for
mid-session model switching, timed to context compaction to avoid an additional
prompt-cache miss. It does not publish the full routing classifier, thresholds,
prompts, or cache-management implementation.

The later [Fusion Desktop/CLI design post](https://cognition.com/blog/local-fusion)
makes the handoff more precise: a brief carries the task, constraints, and
success criteria; the sidekick explores, acts with tools, tests, and reports.
Only briefs, results, and feedback cross the agent boundary, not entire
conversation histories. The lead reviews the output and either accepts it,
delegates a correction, or takes over. Cognition says weaker sidekicks need
**more prescriptive briefs**; they should not conduct the exploration that
determines the lead's plan. A stronger sidekick may help with that exploration.
This is a task-specific interaction contract, not simply a router choosing the
cheapest model for each prompt.

Cognition's [3,000-session analysis](https://cognition.com/blog/making-fable-cheaper-than-opus)
suggests the main savings come from a lead that delegates early with a
specification and then reviews the result, rather than independently repeating
the worker's search and edits. With the same sidekick on FrontierCode 1.1,
Fable plus sidekick scored 60.7 at $1.86 per run versus Fable alone's 60.8 at
$4.03. Short tasks and serial debugging chains were hard to delegate. These
are Cognition's coding results, not a forecast for Obsidian research.

The distinction between *cheap per task* and *small model* matters. Cognition
now recommends SWE-2 as a Fusion sidekick. Its [SWE-2 model report](https://cognition.com/blog/swe-2)
says it was post-trained from Kimi K3, a 2.8-trillion-parameter mixture of
experts, with large-scale reinforcement learning. In Cognition's comparison,
an Astra lead paired with SWE-2 scored 63.4 at $2.34 per FrontierCode task,
versus 62.0 at $2.39 with the lower-list-price Luna sidekick: fewer correction
rounds can offset a higher token price. This does **not** demonstrate that our
Qwen3-8B can carry out SWE-2's broad coding responsibilities. The June Fusion
post does not identify the exact sidekick behind its headline chart; SWE-2 is
the September product recommendation.

### What the reported numbers do and do not show

The [original Fusion chart](https://cognition.com/blog/devin-fusion) reports
63.1 at $1.35 per FrontierCode 1.1 Extended task versus Opus 5 medium's 63.6
at $3.51. The chart was updated August 7 after the June post. The September
[multi-benchmark results](https://cognition.com/blog/local-fusion) show that
cost savings sometimes come with meaningful score loss: Astra alone versus
Astra+SWE-2 was 55.6/$10.08 versus 50.0/$6.06 on Terminal-Bench 4, and
67.7/$44.36 versus 61.3/$35.51 on Vals Code Migration. On DeepSWE 1.1 it was
67.6/$7.88 versus 67.3/$4.69. Cognition's internal report that 88% of sampled
merged PRs used the automated router is evidence of adoption, not independent
quality validation. We should measure supported answers and total system cost
on our own tasks before making a savings claim.

## The closer precedent for a Qwen3-8B worker

Cognition's earlier [SWE-grep / Fast Context](https://cognition.com/blog/swe-grep)
is a better *task* analogue than SWE-2. It is a trained, multi-turn retrieval
subagent that searches a repository and returns **file IDs and line ranges** to
the main agent. Cognition chose verifiable locations instead of free-form
summaries because summaries can mislead the lead and are hard to grade. It
evaluated file/range weighted F1, favoring precision, and end-to-end latency;
its training used labeled repository queries and relevant line ranges. It
also measured downstream coding with the retrieved context. That is a relevant
precedent for a narrow evidence-finding handoff, but its reported gains do not
transfer automatically to our notes, Qwen weights, or limited budget.

**Envoy inference:** let Qwen perform bounded, multi-step `search()`, `read()`,
and `extract()` over a frozen vault snapshot and return a small packet of exact
note IDs, spans, and uncertainty. Nemotron remains the user-facing lead that
decides what question to investigate and synthesizes the answer. The host must
check each returned span against the frozen source. Exact text proves
provenance, not that the text supports a claimed conclusion; supported-answer
quality still requires a reviewed judgment. Qwen can continue to produce a
proposed answer for its own held-out evaluation, while the product passes only
verified evidence to Nemotron.

This also fits Cognition's [earlier multi-agent analysis](https://cognition.com/blog/multi-agents-working):
their reverse arrangement, a weak lead consulting a stronger “smart friend,”
hit a ceiling because the weak lead did not reliably know when or what to ask.
Nemotron should own delegation and final synthesis; Qwen should not decide
which important research decisions the stronger model may see. Separate
[controlled agent-scaling research from Google](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)
found centralized teams helped parallelizable tasks but hurt tightly sequential
ones. Parallel source searches may help, but the lead's reasoning and final
review need not be split merely to imitate Fusion's two running contexts.

## Current Envoy versus the proposed sidekick loop

| Boundary | Today | Next design, contingent on measurement |
| --- | --- | --- |
| Delegation | `ChatService.reply()` asks Nemotron for `investigate` and a query. | Lead issues an explicit brief: objective, frozen corpus hash, source scope, allowed tools, step/time limit, required answer parts, and what counts as insufficient evidence. |
| Worker | `QwenInvestigator` runs a read-only Python tool episode and returns evidence. | Keep the multi-step episode; train toward reliable discovery and selection of necessary spans, not merely valid code or citation syntax. |
| Review | The host checks exact snapshot spans and falls back to lexical retrieval on failure. | Also check whether passages cover the requested parts; let the lead ask at most one targeted follow-up or use the known lexical fallback. Human review remains the arbiter of semantic support in the experiment. |
| Context | A new policy/REPL is created and the REPL is killed for each investigation. | Consider persistent *per-session* worker context only after repeated-query workload shows a benefit. Key it to the user and corpus revision; never assume Cognition's provider-cache economics apply. |
| MCP | The present MCP surface exposes deterministic `search_memory` and `get_memory_source`. | Expose a bounded `investigate_memory` operation only after the worker beats the retrieval baseline and the read-only sandbox is production-ready. |

The current code already implements the safe core of the handoff:
[`chat.py`](../src/product/chat.py) gives Nemotron only mechanically checked
spans, and [`qwen_investigator.py`](../src/product/qwen_investigator.py) constrains
Qwen to a read-only sandbox. The missing pieces are a richer brief, an
evidence-quality feedback loop, measured routing economics, and session context
if it proves useful. Dynamic model switching and always-on GPU serving are
later optimizations, not prerequisites for a useful sidekick.

## Experiment and decision

**Hypothesis:** on a real frozen vault, a bounded Qwen search episode supplies
necessary cross-note evidence that a cheap lexical packet misses, allowing
Nemotron to produce more fully supported answers at acceptable latency and
cost. A separately trained Qwen must also outperform base Qwen on the same
held-out code-execution questions; harness improvements alone do not satisfy
the model-improvement gate.

**Expected signal:** fewer missing or irrelevant source spans, more fully
supported answer parts, fewer guessed document IDs and execution failures, and
acceptable *total* cost per supported answer. Compare (1) Nemotron with lexical
retrieval, (2) Nemotron with base Qwen, and (3) Nemotron with a trained Qwen,
holding the frozen snapshot, question IDs, decoding, prompt, hardware, seed,
and review rubric fixed. Inspect paired cases, not just aggregate scores.

**Decision rule:** use the existing [4-development/12-locked transfer protocol](TARGETED_QWEN_VAULT_TRANSFER_TEST.md)
after a prospective amendment if training starts from base Qwen instead of v5.
First obtain a substantive snapshot and human-checked references; the last
documented configured real vault has only `Welcome.md`. The nine-note synthetic fixture
is development evidence, not a held-out claim. The [matched screen](PERSONAL_MEMORY_READINESS_EVAL.md)
found provisional base Qwen 5/10 fully supported and v5 0/10, with v5 execution
errors on all ten questions; v5 stays disabled. Its BM25 top-five packet had
100% required-note recall on that fixture, so Qwen has not shown a retrieval
advantage there. If base Qwen adds no essential
evidence beyond lexical retrieval, or the lead's answers do not improve after
review, stop sidekick promotion and fix the measured data/retrieval problem
before paying for more training. If a trained candidate is pursued, apply the
registered held-out base-versus-trained supported-answer, execution, latency,
and cost gates. Do not claim Fusion-like savings from token price alone.

No GPU run or paid API call is required by this design review.
