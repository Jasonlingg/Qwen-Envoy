# V5 training audit against the frozen passage-reader role

This is a read-only audit, not a new Qwen evaluation or training run. Scope
`3cc3d5b`; implementation and pilot design `226722e`. The audit checks original
V5 data/review/benchmark hashes, reproduces the historical export rule, compares
the training actions with `passages-v2`, and uses the cached pinned Qwen3 tokenizer
with the existing assistant-only per-action preprocessing. No model weights,
GPU, teacher API or endpoint were used. Training, reward and runtime files remain
unchanged.

## Actual counts

| Item | Train | Validation |
| --- | ---: | ---: |
| Conversations | 35 | 9 |
| Labeled answerable | 19 | 7 |
| Labeled insufficient evidence | 16 | 2 |
| Assistant next-action targets | 106 | 26 |
| End by action three | 27 | 7 |
| Assistant-repaired conversations | 30 | 7 |
| Final submissions with citations | 35 | 9 |
| Final submissions with evidence spans | **0** | **0** |
| Terminal `submit_gap()` actions | **0** | **0** |
| Insufficient labels ending in cited prose | **16** | **2** |
| System prompt identical to current reader | **0** | **0** |
| First tool action compatible with current passage-only form | **0** | **0** |

Train is 19 answerable versus 16 insufficient, so the data is **not a refusal
majority**. All 18 insufficient examples teach a cited prose response rather
than the current terminal gap action. None of the 44 conversations demonstrates
today's exact evidence-span submission contract. V5 predates that new contract;
this is a retargeting mismatch, not proof its original training was incorrect.
Short trajectories are not inherently wrong: today's intended two reads plus
terminal action also takes three turns.

The reproduced training total is **4,812 supervised tokens**:

| Target type | Train supervised tokens | Validation supervised tokens |
| --- | ---: | ---: |
| Tool actions | 2,397 | 715 |
| Answerable terminal actions | 1,413 | 297 |
| Insufficient terminal actions | 1,002 | 150 |
| Total | **4,812** | **1,162** |

All recorded action/input/supervised/max-length counts match the saved tokenizer
check. The training prefixes match inference under the **old** prompt; this does
not mean they match the current passage-reader prompt. The local audit used
Transformers 5.3.0 and tokenizer revision
`b968826d9c46dd6066d109eabc6255188de91218`, local files only. The current trainer
hash differs from the historical training script hash; both are recorded rather
than claiming byte-for-byte reproduction of the original optimizer run.

`audit.json` contains all per-question row counts, hashes, original final actions,
lexical uncertainty flags, action-form checks and actual masked-token allocations.
The uncertainty flags are text patterns, **not semantic refusal judgments**.
The core counts assign no fresh semantic support or completeness grades. A
separate explicitly model-assisted spot check is described below. The original
model-assisted curation is not independent human review.

An initial audit attempt incorrectly expected byte-identical raw observations;
the historical legacy exporter strips outer whitespace. The final audit reuses
that existing rule, records its normalization counts and still rejects changed
observation content. It never repairs or rewrites the saved data.

No V5 train/validation paper overlap was found. V5 question/paper identities also
have zero overlap with the exposed EnvoyBench dev/test-candidate sets, current
three-case regression and old V6 holdout lock. These are checks of those specified
sets, **not a complete exclusion registry for a future training job**. The old V6
holdout was consumed by its completed comparison despite its historical lock text.
No new reserved question pack or full holdout paper was opened.

## Small source-visible spot check

`spot-review-input.json` selects two training rows per historical label using
SHA256(`42:<question ID>`) order. This is a post-audit check of already inspected
training data, not a blind or held-out evaluation. `spot-review.json` labels the
implementing assistant as reviewer, binds the input hash and quotes actual tool
observations. Independent human review remains null.

- The monotonicity-definition answer is supported by its actual read, including
  word replacement and upward/downward contexts.
- The manual-evaluation answer qualifies its Yes as qualitative manual inspection.
  The passage supports that inspection, but not a formal scored human study;
  interpreting the question either way affects the target. Flag before reuse.
- The BioASQ ranking and entity-tracking language answers reasonably limit their
  uncertainty to inspected passages. Neither demonstrates whole-paper absence or
  proves stopping the older unrestricted search was optimal.

No row is approved for training reuse by this spot check. These cases show why
semantic review and scope labels matter; they do not mean the old dataset is
uniformly wrong or that its labels caused the new reader's failures.

## Decision informed by earlier experiments

The [two-example learning diagnostic](../../docs/QWEN_LEARNING_DIAGNOSTIC.md)
already demonstrated successful memorization and autonomous replay on the fitted
examples. The [V6 pilot](../../docs/QASPER_V6_PILOT_RESULTS.md) already tried more
protocol-aligned data: provisional answerable passes 5/17 versus V5's 4/17, but
insufficient-evidence passes 2/7 versus V5's 7/7. It failed promotion. These are
historical model-assisted results, not new or independent human measurements.

Do not repeat either as an untested idea or assume more data alone is sufficient.
The [new proposed pilot](../../docs/PASSAGE_READER_SFT_PILOT.md) targets the
current narrow role: paired evidence-present/missing scopes, explicit extraction
and structured gaps, complete source review and real Docker observations. It
requires a separately frozen dataset and exclusion registry before any training.
There is no selected training batch or new paid budget yet.

Reproduce this read-only audit using cached tokenizer files:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python scripts/audit_reader_training_alignment.py \
  --tokenize --output out/my-v5-reader-audit.json
```

The original local provenance files listed in the V5 manifest are required; this
is not a standalone fresh-clone corpus package. Omit `--tokenize` for structural
checks without tokenizer dependencies. Existing output files are never overwritten.

The audit supports testing targeted demonstration-based SFT; it does not prove
that data imbalance, prompt differences or model weights caused the failures,
or that a new adapter will improve. Current harness baseline remains `cce82c2`.

Validation: **23 focused tests passed; full suite 2056 passed in 166.52 seconds**,
no skips, three existing SWIG warnings. Ruff/diff passed. All 29 frozen runtime
implementation files and 1,736 prior file/marker hashes are unchanged. Token
counts and input hashes were independently checked when packaging this audit.
`validation.json` binds the actual output and review; `full-tests.txt` is the
standalone suite output. No new optimizer run or model inference is included.
