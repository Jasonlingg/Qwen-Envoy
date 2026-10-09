# Provisional base–v5 comparison (declared before inference)

**Question.** On the frozen EnvoyBench `test_candidate` QASPER-derived split,
does the pinned v5 SFT adapter improve answer quality over its pinned Qwen3-8B
base when both use the same bounded Python tools? This is a diagnostic case
study, not a human-validated held-out benchmark result.

**Hypothesis.** The adapter may improve supported answers on some paper
questions, but prior transfer tests make a regression plausible. We will
report both directions and the paired failures, regardless of outcome.

**Frozen protocol.** Use all 40 selected question IDs in
`benchmarks/envoybench/data/test_candidate/benchmark.json`, corpus hash
`bc71958267aa4ef6216085ffc42d364423760f9f042fc1da661822e87bd10fe5`,
base `Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`, and v5
adapter `jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@27b912a863ff914ad45baa03624d8911dc1e17fb`
(weight SHA-256 `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`).
Both arms use one Runpod Secure Cloud NVIDIA A40 48 GB in CA-MTL-1, vLLM
0.30.0, seed 42, a 15-step cap, 1,024 output tokens per action, temperature
0, top-p 1, and Qwen thinking disabled. The runner records exact
implementation/configuration hashes and observed episode times. The Runpod
catalog price observed before launch was $0.49 per GPU-hour; the estimate
excludes setup and idle time and is not a bill.

**Assessment.** Check source IDs and quoted spans deterministically. Use an
identity-blind Sonnet 5 model grader on both answers for each question against
the existing QASPER answer/evidence, with the grader model, prompt hash,
verdict notes, and token counts saved. QASPER source references have not been
independently revalidated, and a model grader may be wrong. Label all answer
scores provisional. Report pass/partial/fail, paired wins/losses, McNemar's
exact p-value as a descriptive statistic, execution/no-submission episodes,
latency, and estimated serving compute cost separately. Do not apply the
human-review promotion gate to these scores.

**Decision rule.** If v5 gains at least 15 percentage points in model-graded
pass rate and adds no more than two execution-error or no-submission episodes,
the result justifies seeking independent review on a small sample and a new
untouched split. Otherwise, report the null or negative result and inspect
the saved traces. Neither outcome justifies training on these candidate
questions or claiming a validated model improvement.
