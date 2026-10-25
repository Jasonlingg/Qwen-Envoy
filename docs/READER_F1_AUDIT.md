# Offline F1 reward audit — October 23, 2026

## Question and decision rule, recorded before running

Can QASPER Evidence F1 replace the current short-fact reward without changing the
reader's task or returned evidence format? Hypothesis: it cannot by itself. The
reader returns character spans, while upstream Evidence F1 compares whole
paragraph strings; mapping any overlapping quote to a paragraph may reward
insufficient quotes. Answer-independent evidence scoring also cannot penalize a
wrong answer with the same citations.

Use only the eight existing reward-validation papers and optionally the four
already exposed training-role smoke episodes. Do not use the twelve locked
model-evaluation papers. Controls are authored fixtures with assumed inspection,
not model performance or independent human review. Keep the current reward,
frozen runtime, prompts, scopes and references unchanged. No model/API/GPU calls.

Compare the unchanged finite reward, upstream Answer F1, literal quote-string
paragraph F1, an explicitly adapted paragraph-overlap mapping, and custom
character-coverage F1. Use all original annotations separately, taking the best
score rather than their evidence union. Never silently clip reference evidence
to the available scope. Candidate quotes use the production 1,200-character cap.

Do not promote Evidence F1 alone if wrong-answer controls retain full credit or
one-character controls gain full credit under the mapping. A negative result is
a reason to keep evidence and answer signals separate, not a reason to introduce
an LLM judge automatically. This small mechanical audit cannot establish semantic
correctness, an optimal RL reward, or model improvement.

## Reproduce

```sh
python scripts/audit_reader_f1.py --output out/reader-f1-audit.json \
  --saved-rollouts out/prime-rl-smoke-20261023/rollouts
```

Omit `--saved-rollouts` for authored controls only. Output creation is exclusive.

## Actual results

[Saved per-case results](../examples/reader-f1-audit-v1/results.json) contain
48 authored controls (six per reward-validation paper) and four saved real Qwen
training-paper episodes. The original annotations remain intact, including
alternative annotations with partially or wholly unavailable evidence. No locked
model-evaluation answers were scored.

| Authored control, eight each | Answer F1 | Literal quote paragraph F1 | Overlap-mapped paragraph F1 | Character coverage F1 |
| --- | ---: | ---: | ---: | ---: |
| Reference answer and full reference quote | 1.000 | 1.000 | 1.000 | 1.000 |
| Proposed paraphrase, same quote | 0.598 | 1.000 | 1.000 | 1.000 |
| Wrong answer, same quote | 0.104 | 1.000 | 1.000 | 1.000 |
| Reference answer, one-character evidence quote | 1.000 | 0.000 | 1.000 | 0.005 |
| Reference answer, all assigned windows | 1.000 | 0.000 | 0.297 | 0.337 |
| Reference answer, unrelated first character | 1.000 | 0.000 | 0.000 | 0.000 |

Literal scoring applies upstream's paragraph-string comparison directly to the
returned quotes. The overlap mapping promotes each cited span to every source
paragraph it intersects, including non-gold paragraphs. It is deliberately tested
as a candidate adaptation; these columns are **not official QASPER Evidence F1
benchmark results**. Character coverage is a custom interval metric. None of
these numbers is a semantic-support judgment. Proposed paraphrases have not been
independently reviewed.

The four saved episodes average upstream Answer F1 0.429 and finite reward 0.250.
They are four samples of one training question, not a held-out evaluation. The
same range expressed with an en dash gets zero token F1, while two answers using
"1900s to 2010s" each get 0.571 even though only one meets the current support
contract. This demonstrates why answer and evidence checks must stay distinct;
it does not establish that any proposed reward improves training.

## Decision

Do not replace the current reward with Evidence F1 alone. Its independence from
answer wording is appropriate for paragraph retrieval, but our current terminal
also contains an answer and shorter quotes. Any-overlap mapping gives full credit
to all eight one-character controls. Exact paragraph matching instead rejects
shorter quotes. Character coverage penalizes tiny and broad citations, but also
measures quotation extent rather than semantic sufficiency.

The next minimal candidate is continuous upstream Answer F1 behind the existing
inspection and attributed-reference-support checks, with evidence metrics
reported separately. Validate that candidate on these same controls before
enabling it; partial lexical credit for wrong numbers and paraphrase sensitivity
remain limitations to disclose. This would be a short-fact proxy, not a semantic
reward or a general research-agent judge. No arbitrary answer/evidence weighting
is introduced by this audit. The LLM candidate stays on hold; no paid judge run
is needed for this next local comparison.

The active `reader-short-fact-reference-v1` reward remains unchanged. Training
readiness is false for this new proposal. No GPU, model requests, optimizer steps,
Git staging or commits occurred. All 29 frozen runtime files and the frozen
dataset/reference manifest validate unchanged. Focused tests: 39 passed; Ruff
passes. Full-suite verification is recorded in the saved audit README.

## Follow-up: gated Answer F1 candidate — protocol before running

User authorized this local test. The candidate is a separate offline function;
the active scorer and Verifiers adapter remain unchanged. The original submitted
answer is scored against all original text QASPER answers. To avoid duplicating
the evidence validation code, the old assessor checks a private copy containing
one known reference answer, with the original citations, spans and trusted
inspection receipts. The original submitted answer is never replaced in traces
or shown a reference. This isolates the existing support gate from its finite
answer-whitelist restriction; it does not add semantic support judgments.

Hypothesis: this admits partial lexical credit for proposed paraphrases while
preserving the citation boundary. Expected signal: full reference controls retain
1, proposed paraphrases gain nonzero credit, and tiny/uninspected/extra/broad
citations remain 0. Include negated references and a decimal-normalization
collision as stress cases, alongside the original wrong answers. The preliminary
development gate also requires these authored wrong answers to stay below full
credit. A failed check blocks promotion, without turning to an LLM automatically.
These are mechanical authored cases, not independent human semantic calibration.

Run using `scripts/audit_reader_f1_reward.py` with the same `--output` and optional
`--saved-rollouts` arguments as the earlier audit. Preserve that audit's saved
results. Score only existing reward-validation and exposed training-role records;
locked model-evaluation references stay unused. No model requests or GPU launch.

### Actual follow-up result: candidate rejected

[The gated-reward audit](../examples/reader-f1-reward-v1/results.json) contains
65 authored controls on the same eight reward-validation papers and four saved
real training-role episodes. All submitted actions and original inspection
receipts are retained, with per-record hashes. No new Qwen sampling occurred.

| Control | Count | Current finite reward, mean | Candidate reward, mean |
| --- | ---: | ---: | ---: |
| Reference answer and reference quote | 8 | 1.000 | 1.000 |
| Proposed paraphrase and same quote | 8 | 0.000 | 0.598 |
| Wrong answer and same quote | 8 | 0.000 | 0.104 |
| One-character quote | 8 | 0.000 | 0.000 |
| Missing inspection | 8 | 0.000 | 0.000 |
| Additional unrelated quote | 8 | 0.000 | 0.000 |
| All assigned windows | 8 | 0.000 | 0.000 |
| Negated reference answer | 8 | 0.000 | 0.827 |
| Decimal normalization collision | 1 | 0.000 | 1.000 |
| Saved real Qwen episodes, exposed training question | 4 | 0.250 | 0.143 |

The evidence gate behaves as intended on the authored controls. The candidate
also accepts partial word overlap that the finite whitelist rejects. However,
the reference `1500 sentences` and the deliberately different `15.00 sentences`
normalize to the same tokens under upstream's punctuation removal and receive
the same full credit. This fails the predeclared wrong-answer check. Negating
an answer retains high lexical overlap, another limitation the evidence gate
cannot resolve. Its ranges are tied to the reference claim, not the meaning of
an arbitrary submitted sentence. These facts should not be described as an
official scoring implementation bug: the upstream metric measures lexical
overlap, and we reused it unchanged.

**Decision: do not enable this candidate for RL.** Keep upstream Answer F1 as a
benchmark metric and preserve the existing finite short-fact reward as the
current training baseline. That baseline is restricted to reviewed answer
variants and support ranges; it still rejects some correct unlisted wording
and is not a general semantic judge. No live LLM judge is validated or required
by this failed audit. A new reward must be assessed against these counterexamples
before promotion; neither arbitrary numeric thresholds nor patching the eight
answers establishes a general solution. This pass does not authorize another
paid RL run or show model improvement.

Implementation is isolated in `src/training/reader_f1_reward.py` and the offline
`scripts/audit_reader_f1_reward.py`. It is not imported by the active Verifiers
reward. The shared fixture builder has an optional raw-record export, disabled
by default; all 36 existing semantic-audit case hashes and exports remain
unchanged. The active scorer, MuSiQue reward, policy and frozen data are unchanged.
Focused checks: **47 passed**; Ruff and whitespace checks pass. Full-suite results
are recorded in the new audit README.

## October 24 follow-up: F1 with critical-fact guards

The user proposed a hybrid with a shorter answer limit. An isolated
[guarded-F1 probe](../examples/reader-guarded-f1-v1/README.md) compares length-only,
numeric/unit/negation checks and an additional numeric-unit binding heuristic.
On 74 authored controls and the same four saved episodes, the final prototype
blocks all constructed wrong-answer/evidence/repetition controls and preserves
credit for the eight reference answers and eight proposed paraphrases. The
length limit alone leaves the short numeric, negation and swapped-binding
exploits intact. Pairing counts with their units fixes the tested swapped
threads/tweets example. All six declared development checks pass.

This supports testing a narrowly scoped hybrid, not claiming F1 can never be
used as a reward or that simple guards prove semantic correctness. The parser
has a fixed unit vocabulary, a five-word proximity heuristic, and was iterated
on these development cases. It is not integrated into the training path. The
unguarded F1 candidate remains rejected; current finite reward, official benchmark
metrics, harness and data stay unchanged. No model calls or paid training.
