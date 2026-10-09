# Saved logprob execution diagnostic

This is an offline analysis of the October 3, 2026 generated-token logprob
[probe](LOGPROB_PROBE_20261003.md). It uses the previously used EnvoyBench `dev`
questions. It is not a new inference run, answer-quality review, calibrated
confidence measure, or model-promotion result.

## Decision declared before inspecting step outcomes

**Hypothesis:** emitted action tokens may have lower logprobs on code turns that
immediately produce a structured tool error or Python/harness error than on
code turns with clean observations. **Expected signal:** at least 80% of code
turns have usable sampled-token logprobs, and error-turn means have a lower
central tendency, with overlap expected. **Decision rule:** if coverage is below
80% or an error cannot be bound to its generating step, report coverage only.
Otherwise compare per-step mean logprobs within each model descriptively,
separately for immediate errors and clean observations. Never use this result
as a correctness threshold or compare the models' answer quality.

The [script](logprob_analysis.py) checks that each saved result matches its
manifest and that step error counts agree with EnvoyBench's existing scorer.
It excludes `SUBMIT:` turns from the error comparison. It reads the provider's
logprob summary for *emitted content tokens*, which is not necessarily aligned
with the cleaned action executed by the sandbox. It copies neither raw tokens
nor questions, actions, observations, answers, passages, or endpoints into the
sanitized [release diagnostic](../../release/envoybench-v0.1/logprob-diagnostic.json).

## Observation

All 368 base and 36 v5 generated turns in the 40-question expansion had valid
sampled-token telemetry: 11,521/11,521 and 1,575/1,575 token logprobs,
respectively. Code-turn coverage was 332/332 for base and 24/24 for v5.

| Development arm | Code turns with immediate error | Clean code turns | Median step mean logprob, error | Median, clean |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3-8B | 15 | 317 | -0.008964 | -0.011481 |
| V5 adapter | 1 | 23 | -0.102660 | -0.085225 |

The expected lower-logprob pattern **did not appear for base**: its detected
error turns had a slightly *higher* median than clean turns, and the
distributions overlap. V5 has one detected error turn, too few to interpret a
contrast. The two models' logprob scales and action lengths should not be
pooled. Within base, 13 detected error turns were runtime/harness errors and
two were structured tool-return errors. V5 had one structured tool-return
error. These mechanical categories do not say whether the final answer was
right or supported.

The 40-question runner recorded all 80 rows, but only **10 question IDs** have
submissions from both models. Base submitted 36/40; v5 submitted 11/40 after
one disconnect and 28 connection refusals following the pod stop. Those
episode-level endpoint failures have no aligned generated-action logprob and
are excluded from the comparison. The smoke's two question IDs both had
paired submissions and confirm instrumentation only; its base error group has
one step and v5 has none. No new model-quality score is derived from either
run.

**Decision:** retain logprobs as trace telemetry for developers. This saved
sample gives no evidence for a useful tool-error warning threshold and no basis
for estimating answer correctness, evidence support, or abstention confidence.
The comparison is observational and unmatched by action length, position, or
question difficulty; low logprobs can arise for many reasons besides errors.

## Reproduce and inspect provenance

From the repository root, with the original ignored `out/` artifacts present:

```bash
python -m benchmarks.envoybench.logprob_analysis \
  --smoke-dir out/envoybench/logprob-probe-20261003/smoke \
  --dev-full-dir out/envoybench/logprob-probe-20261003/dev-full \
  --output release/envoybench-v0.1/logprob-diagnostic.json
```

The JSON records source-file SHA-256 hashes; exact model and adapter revisions;
question IDs and development split; corpus/benchmark/prompt/implementation
hashes; decoding, seed, reward-version field, and hardware metadata. The
reward-version field describes legacy environment diagnostics, **not** a
research-synthesis quality score. The expansion was interrupted and has no
independent human answer review. The original saved run used vLLM 0.30.0 on a
Runpod A40 with an 8,192-token context; no GPU is needed to rerun this analysis.

The release also contains an exact copy of the two-question public-paper smoke
at `release/envoybench-v0.1/token-diagnostic-smoke/` for Studio inspection.
Its manifest SHA-256 is
`1af65b5ad303b560115f0539d155ee1ffa346eab6f4b4087833efc810d2e0c2a`;
its results SHA-256 is
`14289699d3075c4f5a93b239f54076319678fc920deff0dfe6ee76b4777f25df`.
The files are byte-identical to the ignored October 3 smoke originals. They
contain model-generated actions and public QASPER-derived text, unlike the
sanitized aggregate. A targeted scan found no API-key-like values, bearer
tokens, private home paths, or local endpoint strings in those two JSON files;
the release does not include the original model connection configuration.
