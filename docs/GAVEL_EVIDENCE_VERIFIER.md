# GAVEL-inspired evidence-state verifier

Status: one-turn answer-quality intervention rejected; two-recovery host-routing contract passed
a five-question development probe on September 20, 2026.

## Why this experiment exists

The current Qwen research policy usually produces executable actions, but several reviewed
failures follow the same sequence: it makes one weak within-paper query, sees an irrelevant
passage, and submits `Unanswerable`. The targeted SFT continuation did not pass its behavioral
promotion gate. That makes a harness intervention the next isolated variable to test.

[GAVEL](https://arxiv.org/abs/2609.19315) separates semantic planning from failures that a
structured action model can detect and repair. Its graph can prove facts about robot actions.
Paper research has no equally reliable semantic transition model, so Envoy adopts only the
mechanically verifiable part of that design.

## Hypothesis and acceptance rule

**Hypothesis:** one structured recovery message based on observed episode state will reduce
premature abstention and repeated searches without changing Qwen's weights or giving it gold
answers.

The controlled comparison uses the same checkpoint, 25-question development diagnostic,
corpus, prompt, decoding, seed, and raw top-three retrieval as the existing baseline. Only
`--evidence-verifier` changes.

Before a live run, the verifier was applied in shadow mode to the saved trajectories. Shadow mode
can identify where feedback would have occurred, but it cannot predict Qwen's response because
the remainder of the trajectory becomes counterfactual after the first intervention. On the
current behavioral candidate (`checkpoint-150`), it would interrupt five episodes: three of the
15 target cases and two of the ten controls. The three targets are two semantic failures and one
partial answer; both controls are correct refusals after only one search.

That result invalidated the original proposed gate of six target rescues before GPU time was
spent: this conservative verifier can affect only three target cases. The live decision rule is
therefore scoped to the behavior it can change. Require a strictly better verdict on at least two
of those three target cases, retain both affected controls and at least 9/10 controls overall,
introduce no execution errors, and keep mean steps at or below 3.0. The separately locked
40-question confirmation set remains untouched until that development gate passes.

## What the verifier tracks

The episode-local state records:

- successful document-tool actions;
- structurally distinct searches and inspections;
- document IDs actually visible in tool output;
- full-document reads;
- verifier interventions already used.
- whether a premature-abstention recovery is active.

It does not read expected citations, QASPER answerability labels, reference answers, or gold
evidence. It does not judge whether a passage semantically supports a claim.

With the verifier enabled, the environment may reject an action and return structured feedback
when:

- an action repeats document-tool work already attempted;
- the policy abstains before two successful investigation actions;
- the policy continues to abstain after a premature-abstention recovery has begun;
- a non-abstaining answer has no citations;
- a cited document was never observed;
- required evidence is absent or contains invalid offsets;
- citation IDs and evidence-span document IDs disagree.

With `--escalate-after-verifier-failure`, another verified failure after the configured feedback
budget ends the small-model episode with `status: escalated`. The transcript includes the
candidate action, evidence state, and trajectory for the host model. Without that flag, the
original evaluation behavior remains: the next submission is accepted normally. Both modes
prevent an unbounded verifier loop and make the additional model-call budget explicit.

## Run the development ablation

Use the same arguments and environment variables as the recorded raw-top-three baseline, adding:

```bash
python scripts/run_eval.py \
  --evidence-verifier \
  --verifier-feedback-budget 1 \
  --questions data/research/qasper_failure_ablation_v1.json \
  --corpus out/research/qasper-code-dev-v2/corpus \
  --policy qwen_sft_policy \
  --question-only-observation \
  --no-vector-index \
  --max-steps 10 \
  --seed 42 \
  --output out/research/qasper-evidence-verifier-v1/candidate.json
```

Every transcript records `verifier_events`, and the manifest records whether the verifier was
enabled and its feedback budget. This keeps verifier runs from being mixed with the original
baseline.

## Result

The live run used `Qwen/Qwen3-8B` plus the checkpoint-150 adapter published as
`jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1`. It ran on one NVIDIA A40 with
temperature 0, seed 42, raw top-three `search_within` results, no vector index, and a ten-step
limit. The transcript, run manifest, logs, and semantic review are in
`out/research/qasper-evidence-verifier-v1/`.

| Measure | No verifier | Evidence verifier |
|---|---:|---:|
| Semantic verdicts (pass / partial / fail) | 11 / 5 / 9 | 11 / 4 / 10 |
| Supported-answer rate | 44% | 44% |
| Mean review score (0–2) | 1.08 | 1.04 |
| Mean steps | 2.24 | 2.64 |
| Execution errors | 0 | 0 |

The verifier intervened in five episodes: three target failures and two controls. Qwen repeated
the identical `search_within(...)` call after every recovery message, even though the message
explicitly asked for a different query or inspection method. Two target verdicts remained
failures; the third moved from partial to fail because the model dropped the correct `19,300`
figure and declared it absent. Both affected controls remained correct, and all ten controls
still passed.

This fails the pre-registered answer-quality gate: zero of three affected targets improved, while
the required minimum was two. The one-turn verifier is therefore rejected as an answer-quality
improvement, and the locked 40-question set remains untouched.

The result narrows the next problem. The harness can detect a weak stopping decision, but this
checkpoint does not know how to revise its investigation from generic state feedback. Any next
attempt should teach or demonstrate the recovery behavior itself: change the query terms, widen
the result count, use `scan()`, or inspect another passage. It should first be tested on the same
three affected development targets and must leave the ten controls intact. Adding more verifier
rules or running another broad fine-tune would not address the behavior observed here.

## Recovery and routing follow-up

A second post-hoc probe isolated the five episodes affected above: three answerable targets and
two correctly unanswerable controls. This is a development diagnostic, not a held-out score.
Feedback was changed from a generic instruction to concrete, answer-free Python choices:
rephrase `search_within`, widen `top_k`, use `scan`, or inspect the document with `read`.

That change fixed the control-flow failure. All five episodes replaced the repeated call with a
different investigation action. It did not fix answer quality: the three target verdicts did not
improve after one recovery. This separates two findings that should not be conflated:

- concrete harness feedback can change Qwen's search behavior;
- structurally different search does not imply a semantically correct answer.

The final routing probe allowed two recoveries. Once a premature-abstention recovery began, a
continued abstention triggered the second recovery. A third abstention ended with a structured
host handoff instead of returning the uncertain answer as a completed result.

```bash
python scripts/run_eval.py \
  --evidence-verifier \
  --verifier-feedback-budget 2 \
  --escalate-after-verifier-failure \
  --questions data/research/qasper_recovery_probe_v1.json \
  --corpus out/research/qasper-code-dev-v2/corpus \
  --policy qwen_sft_policy \
  --question-only-observation \
  --no-vector-index \
  --max-steps 10 \
  --seed 42 \
  --output out/research/qasper-recovery-routing-v3/candidate.json
```

All five episodes performed three successful document actions, received exactly two recovery
messages, and ended as `escalated`; there were no execution errors. One target's final proposal
recovered both the `19,300` collected tweets and the additional `2,500` sampled tweets, but still
contained a contradictory hedge. The handoff preserves that useful partial work for the host.

Both controls also escalated. That is the cost of using persistent abstention as the routing
signal: a structural verifier cannot distinguish a correct refusal from a mistaken one. The
routing contract passed this narrow probe because uncertain answers were safely handed off, not
because Qwen became more accurate. Before enabling it by default, measure escalation rate and
end-to-end host-model quality on a representative development set. The locked 40-question set is
still untouched.

## Boundary of the result

This is an evidence-state machine, not a paper knowledge graph and not a semantic world model.
These experiments show that deterministic episode-state verification can enforce bounded
recovery and safe routing, but does not improve scientific answer quality by itself. The harness
still cannot decide whether arbitrary scientific claims are true. The larger host model must
resolve escalations, and that end-to-end path still needs evaluation.
