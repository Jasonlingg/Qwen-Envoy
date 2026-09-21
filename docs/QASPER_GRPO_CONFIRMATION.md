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

## Result — September 21, 2026

**Decision: do not promote centered-GRPO checkpoint 2.** Its large development-set gain did not
survive this paper-disjoint confirmation. The secondary automatic reward increased by only 0.0069,
while the primary blinded semantic review slightly favored the unchanged SFT checkpoint.

| Measure | SFT | Centered-GRPO checkpoint 2 | Paired result |
| --- | ---: | ---: | ---: |
| Automatic reward | 0.3462 | 0.3531 | +0.0069; 95% bootstrap CI [-0.0169, 0.0375] |
| Semantic mean (0–2) | 0.700 | 0.675 | -0.025; 95% bootstrap CI [-0.075, 0.000] |
| Semantic pass / partial / fail | 13 / 2 / 25 | 12 / 3 / 25 | 0 GRPO wins, 1 loss, 39 ties |
| Protocol-valid submissions | 40/40 | 40/40 | tie |
| False refusals | 8 | 7 | GRPO -1 |
| Episodes containing an execution error | 3 | 3 | tie |
| Mean tool steps | 5.075 | 5.225 | GRPO +0.150 |

The automatic comparison contained one win, one loss, and 38 ties. Thirty-six of the 40 complete
answer/citation/evidence packets were byte-for-byte equivalent after parsing. The four changed
packets explain why the small automatic increase is not a semantic improvement:

- On the X-Stance question, GRPO replaced SFT's false refusal with `Yes`, which gained 0.5
  automatic reward. Its quoted span only described the BERT classifier and did not support a
  cross-lingual versus single-language evaluation, so both responses failed semantic review.
- On the OpenIE question, SFT correctly described both relation extraction and the heuristic
  selection step. GRPO omitted the selection step. This was the sole semantic difference and a
  regression from pass to partial.
- Two other packets changed wording or evidence without changing their semantic verdict.

The strict semantic review required the submitted quote to support the answer, not merely a valid
paper ID or a correct-looking answer string. The judgments were recorded with system identities
hidden and checksummed before the blind key was opened. This was a Codex assistant review, not an
independent human review.

### Promotion gates

| Pre-registered gate | Result |
| --- | --- |
| Higher semantic mean and at least two more wins than losses | **Fail** — lower mean; 0 wins and 1 loss |
| At least 28 semantic passes | **Fail** — 12 passes |
| Automatic reward above SFT | **Pass** — 0.3531 versus 0.3462 |
| At least 39 valid submissions; refusal and execution-error limits | **Pass** — 40 valid; 7 refusals; 3 error episodes |

The development set had shown +0.0922 automatic reward for this checkpoint. On unseen papers that
fell to +0.0069 with an interval spanning zero. The most defensible interpretation is that the
early GRPO update mostly preserved SFT behavior and fit a few development examples or reward
surface details; it did not produce a general research-agent improvement.

### Reproduction and artifacts

- GPU: one NVIDIA A40 on RunPod, approximately 3,514 seconds of pod uptime at $0.49/hour
  (approximately $0.48), within the $3 ceiling.
- SFT evaluator time: 1,391.5 seconds. GRPO evaluator time: 1,438.3 seconds.
- Automatic analysis: `out/research/qasper-rl-v3/confirmation-run-20260921/automatic-analysis.json`.
- Blinded review, sealed judgments, key, and score:
  `out/research/qasper-rl-v3/confirmation-run-20260921/blind-review/`.
- Blinded judgment SHA-256:
  `894f2aaea775bf98ef9011c9d35baaa69cff43cc54f8b88807d86a89c740e4c2`.
- Downloaded run archive:
  `out/research/qasper-rl-v3/artifacts/qasper-grpo-confirmation-20260921.tar.gz`.
- Archive SHA-256:
  `717aaa717f3d47e683c6b6e949e316d6b4026581a31d5377bf8223b3935204f8`.
- RunPod pod `jtzpuo7bya1jxb` was stopped after the archive checksum matched locally.
