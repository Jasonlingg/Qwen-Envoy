# Proposed passage-reader SFT pilot

Status: reviewed local data prepared, **not a training launch**. No new checkpoint
has been trained. [Actual data and Docker receipts](../examples/passage-reader-pilot-data/README.md)
freeze 24 training trajectories on 12 papers and 12 separate development episodes
on six papers. Source review is implementing-assistant/model-assisted, not human.
The frozen execution baseline is `cce82c2`; prompts, tools, retrieval,
source-inspection rules, four-action limit and decoding remain unchanged.
The [V5 alignment audit](../examples/reader-training-audit/README.md) motivates
this proposal. Actual question IDs, paper IDs, source hashes, exclusions and
reviewed targets
are now frozen in the [export manifest](../data/sft/passage-reader-pilot-v1/manifest.json)
and linked source bundle. A paid/training budget remains unapproved.

## What we already learned

- The [September two-example diagnostic](QWEN_LEARNING_DIAGNOSTIC.md) achieved
  six exact next actions and two correct autonomous answers after 20 updates.
  That established memorization, not generalization. Do not repeat it as if the
  model's ability to fit examples were still wholly unknown.
- The [October V6 pilot](QASPER_V6_PILOT_RESULTS.md) already tried a broader,
  protocol-aligned fresh-from-base SFT revision. Provisional answerable passes
  were 5/17 versus V5's 4/17, while insufficient-evidence passes fell from 7/7
  to 2/7. It failed promotion. More examples, tokens or protocol alignment alone
  are not an established solution. Its old holdout is consumed.
- Current V5 was trained before the `passages-v2` contract existed. Its 35
  training conversations contain 19 answerable and 16 insufficient labels;
  this is not a refusal-majority dataset. Across all 44 train/validation rows,
  there are no evidence-span submissions or terminal `submit_gap()` examples.
  All 18 insufficient examples use cited prose uncertainty. This resembles a
  current observed failure, but does not establish its cause or exclude prompt
  effects. Passing token-prefix checks within the old prompt is different from
  matching today's runtime prompt.

## Distinct learning hypothesis

Teaching the same narrow worker to condition its final action on **the supplied
evidence state** will improve extraction when facts are present while preserving
useful scoped gaps when they are missing. We are not asking it to learn literature
discovery, general orchestration or new world facts. The teacher supplies concise
observable document-tool demonstrations, not private reasoning traces or a reward.

Use a proposed bounded batch of **12 training questions on 12 papers**, each with
two source scopes: requested facts present and insufficient evidence in the
selected passages. This yields 24 three-action trajectories: two real passage
reads, then a supported answer or specific terminal gap. Use **six different
development questions/papers** with the same paired-scope design, yielding 12
development episodes. These counts are an engineering pilot size, not a claim
that this amount of data is sufficient. Evidence-state pairs from one paper stay
together; they are not independent samples for confidence calculations.

Use different questions/papers from existing evaluations, including the current
three-case regression, consumed EnvoyBench sets, learning-loop cases and old V6
holdout. Development papers must also exclude every prior fine-tuning data pool,
including larger continuations, not merely V5's 44 rows. Build and hash that full
exclusion registry before selecting cases. Do not inspect or retarget the reserved
`benchmarks/research_reader_v1` pack. Any later untouched test needs its own lock;
this small development set cannot certify a general model improvement.

Scopes retain two disjoint 1,600-character windows and the same 3,200-character
budget. They are source-grounded controlled evidence conditions, not an online
retrieval benchmark. Never fabricate source text or credit controller metadata as
inspection. An answerable paper can still have an insufficient selected scope;
record original QASPER answerability separately from scope answerability. A gap
applies only to the inspected passages, not to the whole paper or literature.

## Data gates before training

1. Freeze named questions, all paper exclusions, source revision/text hashes,
   assigned ranges, expected requested facts and scope-specific answerability.
   No evaluation answers or regression corrections enter training.
2. Produce two actual Docker reads before each terminal action. Save the real
   policy payloads and tool observations. Review each answer clause for source
   support, completeness, attribution and numeric conditions; review every gap
   and suggested query for specificity. Annotation overlap alone cannot pass.
3. Label review origin explicitly. Model-assisted source review is not independent
   human review; human fields stay null until an actual person reviews them.
   Keep rejected cases and reasons; do not replace failures after model evaluation.
4. Require clean Docker replay, valid inspected exact spans and the correct
   terminal action type. Reuse the existing verifier. Every training conversation
   must match the current effective system prompt and actual policy payloads.
   Passing replay proves interface/provenance compatibility only.
5. Run the existing per-action prefix/masking checks with the cached pinned Qwen3
   tokenizer. Supervise only next assistant actions, including terminal actions;
   prior instructions, observations and history receive no loss. Require all
   targets to fit without truncation. Record token allocation by evidence state
   and tool/terminal action, rather than assuming balanced rows balance the loss.

The current regression-only correction queue cannot export this new training set.
A separate training-data adapter must enforce the exclusion registry and review/
replay gates. Do not flip eligibility flags on saved evaluation records.

## Proposed optimizer and comparison

Continue a **separate copy of V5**, preserving base and V5 as controls. Pin the
same Qwen3-8B base revision and V5 checkpoint-50 adapter SHA from the frozen
baseline. Reuse rank 4 / alpha 8, dropout 0.05, learning rate 2e-4, microbatch 1,
accumulation 4, sequence limit 8,192 and seed 42. One epoch, maximum 24 optimizer
updates, no hyperparameter sweep or repeated attempt; 72 actions would normally
give 18 updates with accumulation four. Pin the exact trainer and GPU software
before execution and check actual initialization/save/reload identities.
This tests one targeted continuation, not the isolated effect of every setting.

Evaluate the single final candidate, V5 and base on the same 12 development
episodes with frozen prompts/scopes, temperature 0, top-p 1, thinking off,
1,024 output tokens and four actions each. Record whether the serving transport
actually enforces seed 42. No selection by development loss or repeated answer
inspection. The proposed maximum is 144 inference requests and 1,179,648 reserved
input/output tokens across all arms, with no inference retries. Do not confuse
these proposed ceilings with authorization for new paid calls.

Predeclare an initial **development signal**, not release promotion:

- At least two more complete source-supported answers out of six present-evidence
  cases than V5, and more than base; unsupported factual clauses fail an answer.
- At least five useful scoped gaps out of six insufficient-scope cases, no fewer
  than V5; no increase in false gaps on the six answerable cases.
- No increase in Python-error/protocol-failure episodes versus either control;
  complete request/token telemetry; median paired tokens and latency at most
  1.25 times V5 on matched serving hardware.

Use source-visible identity-blind review with reviewer provenance, per-question
and per-evidence-state results, and separate mechanical/semantic judgments.
If the rule fails, retain the experiment and stop; no automatic retry or model
promotion. Even a pass requires independent review and a fresh final evaluation
before claiming trained Qwen improved or replacing the deployed checkpoint.

## Current budget and readiness

**Current paid request allowance: zero. GPU activations: zero.** This document
authorizes no training, teacher API or inference job. The local data gates now
pass: 213 historical files inventoried, 956 papers and
1,370 question IDs excluded, 13 source-review pairs rejected before any model
outcomes, all 36 accepted demonstrations replayed in Docker, and all 72 training
action prefixes/masks checked with the pinned cached tokenizer. The 3,018
supervised tokens are approximately 60% present and 40% missing despite equal
trajectory counts. Development references are scoring-only, excluded from the
training export. Independent human review remains null. The registry covers
declared workspace history, not unknown remote jobs or pretraining; strict
re-export needs historical local files that are not all redistributed.
There is no new trained candidate or approved GPU/inference budget. Fixed-action
replay measures compatibility, not a model's action choices or learning.

Before any later paid attempt, freeze the final data and implementation, read
current resource/pricing information, and specify the request/token, active-time
and dollar limits with verified shutdown. A time/rate compute estimate is not a
provider invoice cap. Skip paid calls if their required hard limit cannot be
enforced. Demonstration drafting and review can proceed locally without a paid
teacher endpoint. The remaining work is data quality and a controlled model test,
not another execution-harness rewrite.
