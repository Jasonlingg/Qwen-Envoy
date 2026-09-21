# QASPER GRPO fresh confirmation — pre-registered September 21, 2026

## Question

Does centered-GRPO checkpoint 2's development-set improvement over its SFT starting point survive
on a fresh, paper-disjoint QASPER confirmation set?

This comparison is the next required test after checkpoint 2 reached 0.5485 automatic reward on
the consumed 40-question development set, versus 0.4563 for unchanged SFT. It is a confirmation
experiment, not another training run. No optimizer updates are permitted.

## Frozen systems

| System | Adapter SHA-256 |
| --- | --- |
| SFT control | `70143ec6933494efe60cab726e792fcc182965fd13a8561453202b41bba9bfb4` |
| Centered-GRPO checkpoint 2 | `59b9b271fbde01089f02e076241cdd44d40470a6dcc454322f7c74a1369f9603` |

Both use `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`. They will use the same prompt, corpus, question order,
NF4 loading, non-thinking chat template, greedy decoding, 8,192-token context limit, 1,024-token
action limit, ten-action episode limit, and seed 42. Only the adapter differs.

## Frozen confirmation set

The selection was created before either GRPO experiment and has never been evaluated. It contains
40 QASPER validation questions, balanced 20 answerable and 20 unanswerable, across 39 target
papers. The question order from the original selection is preserved. The questions and target
papers have zero overlap with:

- the 500-question RL training pool;
- the consumed 40-question RL development set;
- the earlier 20-question QASPER test pilot; and
- the targeted SFT training and validation papers.

The old selection was converted to the richer answer-and-exact-evidence schema required by
`qasper-answer-evidence-v1`. No IDs were added or removed.

- Benchmark: `out/research/qasper-rl-v3/confirmation/benchmark.json`
- Benchmark SHA-256: `d8dd2c596914cffd4d7d027dcd4fc251f6b6a7393429945eea02dd4f016cf5c9`
- Corpus: `out/research/qasper-rl-v3/confirmation/corpus`
- Corpus hash: `8d9ae4df4a850f08dd3cde872e6c2e1fa63153b156097e48a951bf308a5205cd`
- Selection audit: `out/research/qasper-rl-v3/confirmation/selection-audit.json`

## Hypothesis and expected signal

**Hypothesis:** the early centered-GRPO update improved evidence-grounded answer selection rather
than merely fitting the development questions. On unseen papers, checkpoint 2 should produce a
higher semantic score than SFT while preserving submission and execution reliability.

The primary measure is blinded, source-based semantic review using pass, partial, and fail labels.
Pass is worth two points, partial one, and fail zero. The automatic answer/evidence reward is a
secondary reproducible measure because prior experiments showed that token and span overlap can
mis-rank semantically different answers.

## Decision rule

Checkpoint 2 is promoted as the project's best research worker only if all of these hold:

1. Its blinded semantic mean score exceeds SFT, with at least two more paired wins than losses.
2. It reaches the previously declared product target of at least 28 semantic passes out of 40.
3. Its automatic mean `qasper-answer-evidence-v1` reward exceeds SFT.
4. It has at least 39/40 protocol-valid submissions, no more than SFT plus one false refusal, and
   no more than SFT plus one execution-error episode.

If semantic quality improves but the 28-pass product target is missed, report a generalizing gain
without promoting the model as product-ready. If automatic reward improves while semantic quality
does not, treat the result as reward overfitting. Report paired bootstrap intervals for automatic
and semantic deltas, but do not require statistical significance from this 40-question set.

The review must keep system identities hidden until all verdicts are recorded. An assistant or
model review is disclosed as such and is not described as independent human evaluation.

## Resource limit

Evaluate only the two frozen adapters. Do not add base, earlier GRPO checkpoints, prompt variants,
or retrieval variants after seeing results. The GPU budget ceiling is $3. Stop the RunPod pod after
both transcript sets and adapter-independent analysis artifacts are downloaded and checksum
verified.
