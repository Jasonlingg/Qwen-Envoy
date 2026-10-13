# QASPER v6 pilot: protocol-aligned SFT did not meet the promotion rule

**Run date:** October 13, 2026. **Decision:** keep v6 as an experimental
checkpoint; do not replace v5 or call it a better paper agent. The blinded,
model-assisted review found 5 supported answers out of 17 answerable questions
for v6, versus 4 for v5 and 2 for base Qwen. The [registered rule](QASPER_V6_TRAINING_PILOT.md)
required at least four more supported answers than base; v6 gained three. On
seven insufficient-evidence questions, v6 passed two versus v5's seven. These
small, provisional counts need independent human source review before any
generalization claim.

This decision is sensitive to one disputed answer. On
`qasper_test_acac0606aab83cae5d13047863c7af542d58e54c`, v5 and v6 both
answered "Yes" to whether the paper introduced the *first* graded dataset.
The blind reviewers marked both fail: the paper contrasts its new 0–10 graded
dataset with cited binary datasets, but does not establish global first-ever
priority. If a human adjudicator reasonably reads "first" only relative to
those cited datasets and passes both answers, v6 becomes 6/17 and meets the
numerical gain threshold. The current frozen review says **no promotion**;
the sensitivity means the threshold is not a robust finding.

## What changed from v5

V5 was trained on 35 mostly short, assistant-repaired conversations: 27 ended
by action three, and its saved 40-question development run submitted by action
three on 38 questions. It also used a shorter system prompt, a question-only
first observation, a ten-step tool protocol, and no required final evidence
spans. Those were plausible causes of premature stopping, not proven causes.

This pilot restarted from pinned `Qwen/Qwen3-8B` revision
`b968826d9c46dd6066d109eabc6255188de91218`, **not from v5**. It used 137
replayed trajectories under the current 15-step, evidence-required code
execution protocol. All 137 passed strict action/payload/evidence replay gates;
the paper-disjoint training and validation sets contained 108 and 29
conversations (474 and 122 assistant actions). Twenty answerable trajectories
had individual model-assisted source-visible semantic review; the other 87
answerable rows had annotation, exact-span, and replay checks but no equivalent
semantic judgment. All 30 abstentions had model-assisted source review. This
is a data-quality limitation, not a claim that every training answer is sound.

One QLoRA epoch used rank 4, alpha 8, dropout 0.05, learning rate `2e-4`, batch
size 1, gradient accumulation 4, sequence limit 8,192, seed 42, and one
Runpod A40 48 GB. The run supervised 39,868 assistant tokens; the validation
set had 9,394. The saved adapter SHA-256 is
`b9931f9576e0a876b82ccb386923b81d784ad584bf5ed969162e7487353c1e37`.
The training manifest records the exact dataset, script, and base hashes. A
one-step reload smoke produced parseable Python with an unsupported `search`
keyword argument, so that smoke was never treated as proof of executable
actions.

## Locked comparison

The holdout has 24 original QASPER **test** questions on 23 papers: 17 labeled
answerable and seven insufficient. Those papers do not overlap the SFT papers
or prior EnvoyBench paper sets. All three arms used one vLLM 0.10.2 server,
the same A40, 306-document corpus, Docker `search`/`read`/`extract` runtime,
15-step fail-closed verifier, greedy decoding, 1,024 output tokens per action,
thinking disabled, and an 8,192-token context limit. The base model may have
seen QASPER during pretraining; paper-disjointness here only prevents *our*
fine-tuning leakage.

| Outcome on 24 questions | Base | V5 | V6 |
| --- | ---: | ---: | ---: |
| Source-supported passes, answerable (of 17) | 2 | 4 | 5 |
| Appropriate responses, insufficient (of 7) | 2 | 7 | 2 |
| Total model-assisted passes (of 24) | 4 | 11 | 7 |
| Reviewed false refusals on answerable questions | 1 | 6 | 1 |
| Official QASPER answer F1 on this subset (0–100) | 19.8 | 40.5 | 45.5 |
| Submissions / verifier escalations / endpoint errors | 11 / 12 / 1 | 24 / 0 / 0 | 24 / 0 / 0 |
| Python-error episodes | 8 | 0 | 4 |
| Mean actions | 8.04 | 3.04 | 5.63 |
| Mean episode seconds | 14.98 | 5.95 | 16.74 |

The base endpoint error was a context-limit 400 after its history grew beyond
the fixed serving budget. V6 had one verifier rejection within an otherwise
submitted episode. Estimated per-episode GPU compute from saved durations and
the declared `$0.59/hour` A40 rate was about `$0.00246` base, `$0.00098` v5,
and `$0.00274` v6. These exclude startup and idle time and are not a Runpod
invoice. The new pod was stopped immediately after the comparison and read
back as `EXITED`; its separate 30 GB network volume remains allocated.

## Interpretation and next decision

The new training data reduced the most obvious v5 behavior: mean action count
rose from 3.04 to 5.63, and reviewed false refusals fell from six to one.
Its answerable F1 rose from 33.6 to 52.5, but only one additional answerable
question received a fully supported pass. On insufficient-evidence questions,
v6's reviewed passes fell from seven to two. More exploration did not reliably
produce grounded final answers or preserve v5's abstention behavior.

The paired traces show the tradeoff concretely. On
`qasper_test_a606bffed3bfeebd1b66125be580f908244e5d92`, v5 refused after
an early passage; v6 searched again and found the paper's caption-rating
counts, earning a pass. On five insufficient-evidence questions, v5's correct
abstentions became v6 answers or a partial answer. For example, v6 answered
"competitive results" when asked for an unreported pretraining gap, and
"two" from a sentence about two *components* when asked for a layer count.
These passages were real, but did not entail the submitted answers. The five
degradations had no Python-error episode, so they are source-interpretation
errors rather than endpoint failures. This supports testing an explicit
evidence-to-question check in a future training set; it does not prove that
protocol alignment caused the shift.

The code trace audit found a separate training-data weakness. Final v6 had
12 Python exception steps in four episodes: nine `IndexError`s from indexing
past a three-item `hits` list after storing a longer list in `more_hits`, and
three repeated failed exact-quote assertions. Base had 28 Python-error steps,
mostly invalid `search(..., doc_id=...)` calls; v5 had none. The preliminary
one-step adapter smoke also made that invalid `search` call, but the **final**
v6 holdout did not, so the smoke cannot stand in for the final model.

The 137 accepted v6 training conversations contain **zero** `search()` calls,
all 137 use `search_within`, and 107 index `hits` and assert that exact quote
lookup succeeds without showing error recovery. This success-only data could
teach a brittle code pattern. A next data revision should include replayed
examples that handle unsupported tool arguments, short hit lists, and quote
mismatches, while keeping a separate evidence-entailment check for answer
quality. These are testable hypotheses, not established causes of the pilot's
semantic failures.

The preregistered rule also required no increase in false refusals or Python
errors versus base, and latency/cost within twice base. V6 met those ancillary
conditions but missed the required supported-answer gain by one question.
This is a directional signal worth analyzing, **not** evidence that protocol
alignment alone solved the agent task. The 24-question sample is small, QASPER
reference labels have unresolved ambiguities, and all answer-support verdicts
were made by three blinded model-assisted reviewers rather than independent
humans. The review contains per-answer source passages and an explicit
`pass`/`partial`/`fail` rationale for audit.

Before another GPU job, review the paired v5/v6 failures and the disputed
reference labels with a human, then add source-checked examples that teach
both successful recovery *and* when evidence remains insufficient. Keep a new
paper-disjoint test batch untouched for that next decision. Do not select the
next adapter by the validation loss or this holdout's individual answers.

The [committed audit bundle](../benchmarks/envoybench/pilots/qasper-v6-20261013/README.md)
contains the 72 saved traces, blind review and identity key, model-assisted
verdicts, scores, hashes, and reproduction commands. The local artifact
directory is
`out/research/qasper-v6-pilot-20261013/`: it contains the adapter backup,
training manifest and logs, exact run traces, original blind packet, three
review parts, completed blinded review and key, automatic F1, and the
hash-verified promotion decision. The frozen question lock is
[`data/research/qasper-v6-holdout-lock.json`](../data/research/qasper-v6-holdout-lock.json).
