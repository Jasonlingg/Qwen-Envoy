# Guarded F1 probe — October 24, 2026

## Protocol recorded before running

The user proposed combining F1 with guards and a shorter answer limit. Compare
the rejected gated-F1 candidate, a 32-whitespace-word final-answer cap, and that
cap plus numeric/unit/negation guards. The cap never applies to Python actions,
observations or the model's generation budget. All variants retain the existing
inspection and attributed-reference-support checks. No production changes.

Hypothesis: length alone cannot block the short numeric and negation exploits;
critical-value guards may block these while retaining partial paraphrase credit.
Require the original eight reference answers and proposed paraphrases to retain
credit, all prior evidence negatives to stay zero, and numeric/negation controls
to become zero. Separately report stress cases that repeat the reference or swap
the numeric quantities' associated nouns. A guard that only compares a bag of
numbers may miss those bindings. Never turn passing this small authored probe
into semantic validation, independent human review, or training authorization.

Reuse the previous 69 records: 65 authored reward-validation controls and four
exposed real training-paper episodes. Add repeated-answer controls on the eight
validation references and one swapped-count stress case. No locked model-eval
references, new Qwen sampling, judge calls, GPU, or optimizer steps.

The numeric parser is an experimental heuristic: Decimal values, conventional
comma thousands, k/m/b suffixes and scientific notation; QASPER BIBREF/FIGREF/
TABREF artifacts are ignored. Units preserve the case of a small explicit list.
Negation uses a word/contraction list. These are not general mathematical or
linguistic parsers. Candidate critical values must match one original reference
or an existing accepted short-fact variant. No new answer aliases are authored.

Reproduce from the repository root, into a new output file:

```sh
python examples/reader-guarded-f1-v1/audit.py --output out/guarded-f1.json
```

The active reward stays unchanged.

Follow-up before rerunning: if the bag-of-values guard misses swapped numeric
bindings, test a separate typed-unit variant. Recognize sentence/thread/tweet
counts in addition to the small SI-unit list, normalize singular/plural, and
pair each numeric value with the first recognized unit within the next five
words before the next number. Report the bag variant separately. This is a
narrow heuristic with a fixed vocabulary and proximity rule, not semantic
verification or a new model-output schema.

## Actual result

[Results](results.json): **78 records**, comprising 74 authored controls on the
same eight reward-validation papers and four previously exposed real training
episodes. These are development controls, not a new held-out model evaluation.

| Control | Count | Length cap only, mean reward | Number/unit/negation guard | Guard with numeric-unit bindings |
| --- | ---: | ---: | ---: | ---: |
| Reference answer | 8 | 1.000 | 1.000 | 1.000 |
| Proposed paraphrase | 8 | 0.598 | 0.598 | 0.598 |
| Earlier wrong answers | 8 | 0.104 | 0.000 | 0.000 |
| Negated reference | 8 | 0.827 | 0.000 | 0.000 |
| Decimal collision | 1 | 1.000 | 0.000 | 0.000 |
| Swapped thread/tweet quantities | 1 | 1.000 | 1.000 | 0.000 |
| Tiny/uninspected/extra/broad evidence | 32 | 0.000 | 0.000 | 0.000 |
| Repeated reference | 8 | 0.000 | 0.000 | 0.000 |
| Saved real training-role episodes | 4 | 0.143 | 0.143 | 0.143 |

The short-answer limit addresses repetition; it does not fix short wrong numbers
or negations. Guarded F1 is a workable **development candidate** for narrow
fact extraction: enforce the critical checks, then award partial lexical credit.
The additional binding heuristic blocks the swapped-count example that a bag of
numbers and units misses. All six declared control checks pass after this local
iteration; that is not independent validation or a guarantee against new exploits.

The fixed unit vocabulary and five-word proximity rule can misread unfamiliar
phrasing, quantities, scientific notation/units outside its supported syntax, or
more complex relationships. Known-unit canonicalization does not establish which
fact an arbitrary sentence asserts. Guard vocabulary was developed using these
same papers; do not report performance as generalization. A real training
integration would require versioning the parser/contract, validating its scope,
and a separately bounded pilot. This prototype is not imported by training.
Official QASPER evaluation remains unchanged; the guarded score is a custom proxy.

All eight reference controls and eight proposed paraphrases retain their previous
credit; proposed paraphrases have not been independently reviewed. The four real
episodes still concern one exposed training question, not a post-update quality
test. Their inspected source ranges are unchanged. Early diagnostic scripts and
results are preserved locally in `out/guarded-f1-first-probe/` and
`out/guarded-f1-bag-probe/`; no prior audit artifacts were overwritten.

Verification: 47 focused reward/data tests passed; eight direct parser checks
passed (comma formatting, decimal distinction, unit case, possessives, numeric
bindings, scientific notation, k suffix, negation). Ruff and whitespace checks
pass. All 29 frozen runtime files and frozen data/reference hashes validate
unchanged. Application code remains the same as the previously passing full
suite (2,128 passed / 16 skipped); no additional full rerun was needed for this
isolated example-only probe. No new model/API/GPU calls, training, staging or
commits. Human review is null and `training_ready` stays false.
