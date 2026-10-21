# Reader learning target and reward proposal

The saved development runs suggest a learning target of **answering the requested
facts with sufficient supporting citations and correct conditions**. Across 24
saved base/v5 episodes there were zero recorded Python execution-error episodes.
Correct execution alone would not distinguish a supported answer from the
observed wrong measurement, clipped citation or transferred benchmark number.

This is retrospective development analysis. No new inference, training or scalar
reward was computed. The historical model-assisted grades remain unchanged:
base passed 2/8 answerable questions and 2/4 gaps; v5 passed 1/8 and 3/4. These
small, strict development results do not establish general model superiority.
The reserved evaluation questions and references were not opened for diagnosis,
changed or used to choose a reward. The active MuSiQue reward is unchanged.

## Failure diagnosis

The [per-episode audit](../../examples/reader-learning-diagnosis/analysis.json)
joins every saved question, candidate answer, submitted evidence, gap, action
sequence and original review with an explicitly attributed manual diagnosis.
Each diagnosis is implementing-assistant analysis, not a new pass/fail judgment
or human training label. Source slices were checked mechanically before joining.

| Observed issue | Episodes | Example and implication |
| --- | ---: | --- |
| Citation scope missing | 4 | A block-table or GPU claim is outside the submitted window; FA2's benefit sentence is clipped. Retrieve the supporting sentence before submitting. |
| Citation boundary and derivation ambiguity | 2 | QLoRA overhead numbers are correct in the complete source, but quoted windows stop before them. Resolve the derivation policy rather than calling every value fabricated. |
| Wrong measurement | 1 | V5 supplies throughput ceilings instead of requested Triton forward/backward speedups. True quoted numbers can answer the wrong question. |
| Condition swaps | 2 | Base assigns original split-K to FA2, or transfers a 4090 speedup to an A5000 target. Preserve algorithm, baseline and workload bindings. |
| Requested fact missing | 3 | Includes the wrong-measurement case and omissions of an AWQ task label or batch size. Reward coverage of requested facts. |
| Rubric scope ambiguity | 8 | Some historical partials demand unasked detail or repetition of prompt context. Resolve required-fact lists before using those grades for learning. |

These diagnostic tags overlap. They are not an error-rate estimate. A further
base episode exhausted its policy budget without a required gap report; one v5
gap omitted the target model identity in its follow-up. No uncertain provider
failure caused that base episode's incompletion. The audit preserves all 89
historical requests and all 24 episodes rather than dropping failures.

The strongest examples are source-visible. Base's FlashAttention-2 claim
contradicts the cited description of the original algorithm. Its TinyChat answer
gives 2.7–3.9x without the requested target conditions. V5's AWQ protocol answer
omits the explicitly requested batch size even though the quote contains it.
By contrast, both QLoRA dtype answers supply the three explicitly asked fields;
their historical dequantization-completeness penalty needs adjudication. This
audit does not quietly promote them to new passes.

## Proposed terminal reward

The [machine-readable proposal](../../examples/reader-learning-diagnosis/reward-proposal.json)
is a draft, not an active reward function or authorization to train.

For an answerable question, define **C** as the fraction of distinct requested
facts correctly stated and supported by the answer's own citations. The fact
list must be fixed from the question before reviewing candidate outputs.
Equivalent paraphrases count; duplicate claims do not. A useful partial answer
can receive partial credit when its asserted facts are supported.

The candidate reward is **R = C**, gated to zero if any material answer-relevant
assertion contradicts its citation, lacks support, or transfers a measurement to
incompatible conditions. Empty answers and abstention on answerable questions
receive zero. A real quote or valid program earns no separate bonus. A correct
number with an insufficient citation is a support failure, distinct from a
fabricated value in the diagnostic record. Explicit correct derivations may
count when their cited operands and conditions suffice.

For target gaps, score the completed scoped gap and requested next query separately
from answerable quality, applying the same support gate to every material
assertion. A valid target gap accompanied by a fabricated or contradicted partial
literature finding still scores zero. Source
inspection is required, but it proves access rather than universal absence.
Always abstaining still earns zero on the eight answerable cases. The proposed
initial learning scope is answerable cases, with gap behavior retained as a
separate nonregression check. Any later training mixture needs a frozen sampling
policy and abstention baseline; mixed reward alone cannot show useful answering.
This draft has no efficiency bonus; request, token and latency limits remain
separate checks.

Policy failures such as terminal format rejection or action-budget exhaustion
receive zero. Uncertain provider outcomes, infrastructure failures and corrupt
records lack a correctness label. Missing or disputed semantic judgments must
halt the affected rollout group or batch for review; selectively discarding
unfavorable samples or treating unknown labels as rewards would invalidate the
comparison. All costs, failures and coverage remain reported.

## What must happen before RL

The required semantic judgments do not yet have a calibrated automatic verifier.
Exact-span checks and the existing token-overlap MuSiQue reward cannot supply
them. The current manual diagnoses and provisional model-assisted reviews are
not ready-made RL targets.

Next, build development-only contrast cases: a supported complete answer, a
supported partial, a correct value with clipped evidence, a wrong measurement,
a condition/baseline swap, a valid paraphrase or explicit derivation, an
unsupported target result and blanket abstention. A separately reviewed judging
process must distinguish these while keeping uncertain cases visible. Human
review fields remain blank until a person performs that review.

The [judge validation plan](../../examples/reader-learning-diagnosis/judge-validation-plan.json)
predeclares two development paper IDs for judge design and two different ones
for a future judge check. Neither partition uses the reserved reader evaluation.
Both contain already-inspected development material; this is not an untouched
generalization test. Before judge calls, freeze 32 clear fixtures and their
adjudicated labels: 12 critical negatives, eight complete/paraphrased positives,
four supported partials, four valid gaps and four answerable blanket refusals.
The gate permits zero critical false acceptances, incorrect scores, unknown
judgments or missing cases. A failed check stays recorded; tuning based on it
requires fresh development check cases. Passing this small sentinel set would
justify only a limited implementation review, not a population reliability claim
or training approval. These fixtures have not yet been constructed or judged.

Before proposing paid training, record the reviewed fact labels, judge/version,
critical-error acceptance and false-rejection results, unresolved cases, sampled
successful trajectories and group reward variance, plus a fixed cost and stop
rule. One deterministic sample per model/question cannot establish whether GRPO
would get informative reward variation. A later comparison must preserve the
reserved split and compare supported answering separately from refusals.

## Reproduce the diagnostic

```sh
python examples/reader-learning-diagnosis/analyze_saved.py --output /tmp/reader-diagnosis.json
```

Choose a new output path. The script opens output exclusively, never executes
recorded Python and reads only development cases and their four source files.
It preserves the original review and adds manual tags; it is not a semantic
grader. Input hashes identify all joined packets, questions and annotations.
Independent review and final checks are retained with the diagnosis artifacts.
