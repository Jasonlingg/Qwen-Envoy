# QASPER GRPO experiment — September 20, 2026

This run asked a narrow question: after supervised fine-tuning taught Qwen3-8B the
code-execution protocol, could a small GRPO run improve its answers and evidence use?

The result was mixed but useful. The first saved GRPO checkpoint improved the automatic
held-out score from **0.4563 to 0.5233**. Continuing the same training erased the gain;
checkpoint 20 scored **0.4535**, slightly below the SFT starting point. The early checkpoint
is a candidate for semantic review, not a promoted model.

## Experiment setup

| Item | Value |
| --- | --- |
| Starting model | Qwen3-8B plus the targeted QASPER SFT adapter at checkpoint 150 |
| Training data | 500 QASPER train questions on 390 papers; 400 answerable and 100 unanswerable |
| Questions sampled by this run | 38 unique questions across 40 prompt groups |
| Rollouts | 4 per question, 160 episodes total |
| Optimization | 20 GRPO updates, batch size 2 questions, one PPO epoch per update |
| Reward | `qasper-answer-evidence-v1` |
| Reference policy | Frozen copy of the starting SFT adapter, KL beta `0.001` |
| Learning rate | `1e-6` |
| Limits | 8,192-token context, 1,024 tokens per action, 10 actions per episode |
| Sampling | Temperature 1 during training; greedy decoding during evaluation |
| Hardware | One NVIDIA L40S |
| Seed | 42 |
| Training time | 7,280 seconds, or 2 hours 1 minute |

The model never sees QASPER's answers or answerability labels in its prompt. It receives a
question and uses Python calls such as `search_within()`, `read()`, and `passage()` to inspect
the paper. The external grader compares the final answer and cited spans with QASPER's
annotations.

Before the full pilot, the SFT checkpoint passed the staged gates: it produced valid
investigations in the five-question smoke test, produced reward variation in sampled training
groups, and completed a real optimizer update with finite gradients. The policy weights
changed, the frozen reference weights did not, and the saved adapter reloaded successfully.

## Held-out checkpoint results

All candidates were evaluated greedily on the same 40 development questions: 20 answerable
and 20 unanswerable questions on papers excluded from training.

| Model | Mean reward | Supported-answer proxy | Correct abstentions | False refusals | Valid submissions | Execution-error episodes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Starting SFT | 0.4563 | 10 | 12 | 4 | 39/40 | 3 |
| GRPO checkpoint 5 | **0.5233** | **12** | **14** | **3** | 39/40 | 3 |
| GRPO checkpoint 10 | 0.4988 | 12 | 14 | **2** | **40/40** | **2** |
| GRPO checkpoint 15 | 0.4928 | 12 | 13 | 3 | 39/40 | 3 |
| GRPO checkpoint 20 | 0.4535 | 12 | 12 | **2** | **40/40** | **2** |

Checkpoint 5 beat SFT on four questions, lost on one, and tied on 35. Its mean reward gain was
0.0670, or 14.7% relative to the SFT score. It also improved abstention and did not make tool
execution less reliable.

It still missed the promotion rule declared before the run. The rule required at least four
additional supported answers; checkpoint 5 added two. A nonzero automatic reward is only a
proxy for a supported answer, so even a checkpoint that passed that rule would still need
source-based semantic review.

## What happened after checkpoint 5

Only six of the 40 questions changed reward between checkpoints 5 and 20. The later checkpoint
made two answers better, but four regressions were larger:

- Two correct `Unanswerable` decisions became unsupported answers, costing 1.0 reward each.
- A correct **Yes** answer became **No** while citing the same evidence, costing 0.8275.
- A complete ten-dataset answer dropped three datasets, costing 0.2271.
- One transfer-learning answer became more specific and gained 0.1881.
- One previously refused Vietnamese segmentation question received a partial answer and gained
  0.0776.

A seventh output changed from an invalid submission to a valid but incorrect answer, leaving its
reward at zero. This small set of changes explains nearly the entire downward curve. More updates
did not make the agent uniformly worse; they changed a few borderline behaviors, and the harmful
changes outweighed the useful ones.

## Why the later updates were unstable

The training trace shows three concrete risks.

First, the run barely sampled the 500-question pool. It used 38 unique questions: 36 appeared in
one update and two appeared twice. A batch contained only two questions, so any noisy reward had
considerable influence on that update.

Second, 13 of 40 prompt groups had identical rewards across all four rollouts. Those groups
provided no relative learning signal. Of the remaining 27 groups, eight had a reward spread of
0.15 or less.

Third, standard GRPO divided each group's centered rewards by that group's standard deviation.
For one group, rewards `[0, 0, 0.027, 0]` became normalized advantages
`[-0.5, -0.5, 1.5, -0.5]`. This gives a tiny lexical-overlap difference the same advantage scale
as a much clearer success. The reward also measures answer-token and evidence-span overlap; it
cannot tell whether a fluent claim is actually supported in the full semantic sense.

These observations do not prove that standard-deviation normalization caused every regression.
They give us a specific, testable explanation for why continuing this small run could move the
model in noisy directions.

## What this run establishes

We can say that five GRPO updates improved the automatic QASPER answer-and-evidence score on the
frozen development set while preserving tool reliability. We can also say that continuing to 20
updates removed that gain, which makes checkpoint selection and frequent evaluation essential.

We cannot yet say that GRPO produced a semantically better research agent, that checkpoint 5
generalizes to an Obsidian vault or a changed tool API, or that it is ready to replace the SFT
model. The 40-question set has now been used for model selection and is no longer a fresh final
test set.

## Reproducibility

- Base model revision: `b968826d9c46dd6066d109eabc6255188de91218`
- Starting adapter SHA-256:
  `70143ec6933494efe60cab726e792fcc182965fd13a8561453202b41bba9bfb4`
- Training benchmark SHA-256:
  `e8639f2c555d9325c9a4a12f0733b766a8435d6dce72bfced2a53ec3a8d45fc6`
- Training corpus hash:
  `233445316f342d5652744bce395c2654372b84b91efcfb12e7cc03f67dd247ae`
- Local run directory: `out/research/qasper-rl-v2/gpu-20260920/`
- Full artifact archive: `out/research/qasper-rl-v2/artifacts/envoy-qasper-grpo-20260920.tar.gz`
- Archive SHA-256:
  `ea5b88dbbdc2772bc4ec1737667e1538d6aad045f116308b16fd8704672486ee`

The raw episodes, per-update metrics, manifests, environment package list, checkpoint summaries,
and paired evaluation are retained in that run directory. The RunPod GPU is stopped.

## Next experiment

The next run is a five-update, one-variable ablation. It starts from the same SFT checkpoint and
keeps the data, reward, seed, prompt, learning rate, rollout count, limits, and frozen reference
unchanged. It replaces standard-deviation-normalized group advantages with centered, unscaled
advantages: `reward - group_mean`.

Every update will be saved and evaluated. The experiment stops after five updates. A checkpoint
is promising only if it exceeds the previous best score of 0.5233 while keeping false refusals at
three or fewer, valid submissions at 39/40 or more, and execution-error episodes at three or
fewer. Semantic review remains required before promotion.
