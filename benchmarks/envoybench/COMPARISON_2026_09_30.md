# Base Qwen3-8B versus v5: September 30, 2026

**Status: full inference and a separate Codex-assisted review complete; planned
Sonnet grading interrupted.** This is a provisional, QASPER-derived case study.
There is no completed human review or validated supported-answer improvement claim.

The exploratory review scored v5 at **15/40 passes versus base at 5/40**.
**Both passed 3/20 answerable questions.** The aggregate gain is in correct
abstentions on the 20 unanswerable questions: 12/20 versus 2/20. This result
supports investigating tool reliability and abstention, not claiming improved
substantive paper-answering ability.

An **always-refuse policy would be expected to pass 20/40** under the frozen
answerability labels and rubric, while providing **zero useful answers** and
refusing all 20 answerable questions. This is an analytical expectation, not an
executed or judged baseline. It exposes why the aggregate pass rate alone is
unsuitable for claiming useful model improvement.

Both checkpoints attempted the same 40 frozen candidate questions over 363
papers using a Docker-isolated Python environment and the same paper tools.
The [predeclared protocol](PROVISIONAL_COMPARISON_2026_09_30.md) specifies model
revisions, decoding, hardware, and the decision rule. This run evaluates the
existing v5 adapter; it does not train a new checkpoint.

Each question explicitly names its target paper and document ID. This tests
bounded **known-paper question answering**, evidence selection, and abstention;
it does not measure discovering relevant papers or open-ended multi-paper research.

## Observed execution results

| Measure | Base Qwen3-8B | v5 SFT adapter |
| --- | ---: | ---: |
| Questions attempted | 40 | 40 |
| Final answers submitted | 29/40 | 40/40 |
| Endpoint context-limit failures | 11/40 | 0/40 |
| Episodes with any recorded tool, runtime, or endpoint error | 22/40 | 1/40 |
| Episodes with structured errors returned by tools | 2/40 | 1/40 |
| Steps with runtime exceptions | 38 | 0 |
| Steps with structured tool-error returns | 3 | 1 |
| Mean trajectory steps | 11.00 | 3.05 |
| Provisional model-graded passes | 5/40 (12.5%) | 15/40 (37.5%) |
| Passes on answerable questions | 3/20 | 3/20 |
| Correct abstentions on unanswerable questions | 2/20 | 12/20 |
| Partial / fail judgments | 2 / 33 | 3 / 22 |

The v5 adapter is more reliable at using this particular tool environment in
this run. That is a useful system-level observation. It does not establish that
its answers are generally reliable, that it reasons better, or that it succeeds
on the personal-vault product task. Only three answerable responses per model
passed the rubric requiring support from the answer's own submitted evidence.

Across paired pass/partial/fail verdicts, v5 wins 14, loses four, and ties 22.
For pass versus non-pass, v5 alone passes 12 questions and base alone passes two
(exact two-sided McNemar p=0.012939, descriptive only). The grader changed after
the original protocol was declared, labels remain unvalidated, and this is one
small evaluation; this p-value is not evidence of a validated model promotion.
On the 29 questions where both models submitted, pass counts are 9 for v5 and
5 for base. Six of v5's 15 passes occur where base hit the context limit. This
post-hoc subset helps expose the setup dependence; it does not replace the
all-question result.

Automatic diagnostics version 2 includes structured error objects returned by
tools, even when the Python program itself does not raise. The original version
counted 21 affected base episodes and zero v5 episodes; it missed these returned
errors. Original inference and version-1 score artifacts remain unchanged.

The base model repeatedly used unsupported tool arguments: 32 failures from
`search(..., doc_id=...)`, three from `search_within(..., method=...)`, one
`NameError`, and two syntax errors. The 38 runtime errors occur in 15 episodes.
Five of those episodes also eventually hit the context limit; six additional
episodes hit the context limit without a runtime error. Structured tool-error
returns affect two base episodes, one outside that union, giving 22 distinct
affected base episodes. V5 has one recovered tool error, so it is no longer
described as error-free. Category counts overlap and must not be added.

## Answer behavior and separate evidence diagnoses

A fresh-context Codex reviewer classified all 80 anonymous final responses
using only response text and submission status. It did not see questions,
references, model identities, or existing quality grades. The coordinator then
bound each classification to its exact saved result. This is model-assisted
classification, not human review or new correctness grading; exact provider
model ID, decoding, and session token usage were not exported.

| Response behavior | Base | v5 |
| --- | ---: | ---: |
| Substantive responses | 27/40 | 19/40 |
| Abstentions | 2/40 | 20/40 |
| Mixed answer and abstention | 0/40 | 1/40 |
| No submission | 11/40 | 0/40 |
| Abstentions on answerable-labeled questions | 0/20 | 8/20 |

V5's greater abstention brings both useful refusals and unnecessary refusals.
The last row is conditional on the unvalidated QASPER labels. Base's zero there
is not evidence of better answering: it also has eleven failed episodes and
many unsupported substantive answers. Mixed responses remain a separate class.

The viewer separately diagnoses **answer correctness** and **support from the
submitted evidence** for three selected paired examples:

- A useful refusal versus a context-limit failure.
- Both models name the right dataset but cite an unrelated passage.
- Base identifies architectural simplifications with inadequate evidence;
  v5 misses the requested simplifications and also lacks supporting evidence.

These six component diagnoses are explicitly **unblinded, AI-assisted case
explanations**, not a representative accuracy sample. Exact diagnostic source
passages are displayed separately from the evidence the model actually submitted.
Other correctness/support component labels stay unknown. Original blind
pass/partial/fail grades are unchanged.

## Limitations that affect interpretation

- The serving script used an **8,192-token context cap**, with 1,024 tokens
  reserved for each completion. All 11 endpoint failures occurred when the
  accumulated prompt exceeded that budget. This is a failure of the evaluated
  model-plus-harness configuration, not evidence that base Qwen cannot answer
  those questions with a larger context or better context management. The
  context cap is recorded here from the serving script and actual errors; it
  was omitted from the predeclared prose and run manifest.
- The harness wrote zero duration for all 11 failed base episodes despite
  preserving their 10–14 tool steps. Its aggregate mean time and episode cost
  therefore undercount base work. The Studio shows timing coverage and omits
  these zero placeholders. Among the 29 questions both submitted, observed
  means were 14.81 seconds for base and 4.53 seconds for v5; this is a
  post-hoc, completion-conditioned diagnostic, not an all-question latency
  comparison. GPU startup and idle time are separate costs.
- Updated harness code records elapsed time on failure and retains observed
  request-token usage on future runs. These fixes cannot reconstruct missing
  historical timing or token usage. Version-2 diagnostics leave base's full
  mean duration, total duration, and estimated full episode compute cost null.
  Its observed 29-episode time is 429.44 seconds. V5's complete 40-episode time
  is 179.46 seconds. At the recorded $0.49/hour rate, episode-only compute is
  an observed base subtotal of $0.058451 and v5 total of $0.024426; these exclude
  setup/idle time and do not establish a complete cost comparison.
- Both models' submitted document IDs and extracted spans pass the exact
  existence checks. Exact source text does not establish semantic support.
  Thirteen v5 submissions contain no evidence spans. The rubric distinguishes
  valid abstentions from unsupported answers.
- The 40 source references are unreviewed QASPER annotations: 20 answerable
  and 20 unanswerable labels. The judge receives one canonical answer and
  annotated evidence, not all annotations or every full paper. The data are
  an adapted evaluation set, not newly authored benchmark questions.
- This is one run, with no inference seeds guaranteed by the remote endpoint.
  Model identities are operator declarations; the adapter bytes were checked
  against the pinned SHA-256 before serving.

## Grading method and compute status

The identity-blind Sonnet 5 judge saved eight question judgments (16 answers)
before the Anthropic API rejected the next request for insufficient account
credits. Those checkpoints are retained. They are **not** presented as a
40-question quality score. Resuming the same command with `--resume` reuses
the saved judgments. The original Sonnet-based decision remains incomplete.

To finish without more separate API credits, two Codex session agents each
reviewed 20 paired questions in fresh contexts. They saw anonymous packets and
the same rubric, without the identity key, run results, the other agent's
verdicts, or Sonnet's grades. Every question was reviewed once by this method,
not twice. The coordinator has access to model identities. The full 80-row
Codex review is kept separately; it does not mix in the eight Sonnet judgments.
Exact provider model IDs, decoding settings, and session token usage were not
exported. These limitations make this exploratory model-assisted review, not
a reproducible substitute for the predeclared provider judge or human review.

The two methods agree on 14 of the 16 answers Sonnet graded before its credits
ran out. One disagreement is fail versus partial, and one is pass versus fail.
That small, non-random prefix is a diagnostic, not a validation sample. Notes
also flag ambiguous QASPER references. None were removed from the denominator.

The +25-point aggregate pass gain exceeds the original numerical threshold,
and execution errors decreased. Because the grader changed and answerable
pass counts did not improve, **no promotion gate is claimed**. Report the
specific abstention/tool-use finding and retain the failures in the demo.

## Improvements ready without another paid run

The [deterministic retrieval comparator](RETRIEVAL_BASELINE.md) has **40 prepared
question packets and 80 planned paired answer requests**. It ranks passages only
within the disclosed target paper, then requests one answer from each matched
checkpoint. It does not use reference answers or annotations to select context.
Cached pinned-tokenizer counts reserve output space under an explicit 8,192-token
server limit. **It has not run and has no measured quality result.** Its protocol
asks whether the agent loop improves useful supported answers enough to justify
its resource use; ties do not demonstrate that benefit.

An eight-question, sixteen-answer calibration packet contains full source papers
and blank reference/answer decisions. It hides model identities and prior quality
grades, samples across answerability and prior pass outcomes, and is reproducible
with seed 42. It is awaiting independent review. Its stratified sample can expose
grader disagreements but cannot produce a population score or satisfy the full
promotion gate. Human judgments have not been invented or filled automatically.

The Studio now exports a standalone read-only HTML snapshot and its displayed
JSON. Inspecting or sharing that saved comparison needs no endpoint, GPU, or API
key. This exports local artifacts; it does not publish them anywhere.

The temporary A40 pod `ueyaz39qiz668q` was terminated after the local inference
artifacts were verified. The subsequent Runpod list returned no pods. The
observed GPU rate was $0.49/hour and the last uptime read was about 30 minutes,
or roughly $0.25 in GPU time; this is an estimate, not a billing statement.

## Inspect or resume locally

```bash
python -m benchmarks.envoybench.demo --port 8766 \
  --run-dir out/envoybench/provisional-full-20260930 \
  --review-dir out/envoybench/provisional-review-20260930 \
  --judged-review out/envoybench/codex-reviewed-20260930/review.json \
  --provisional
```

This displays all 40 paired traces, provisional verdicts and explanations,
answerability and response-behavior breakdowns, analytical baseline, selected
case diagnoses, and version-2 automatic diagnostics without new inference or
API calls. Omit both review arguments to show automatic diagnostics with blank
answer-quality metrics.

New local artifacts:

- `out/envoybench/automatic-v2-20260930/automatic.json`: corrected diagnostics
  with immutable source bindings and explicit missing timing.
- `out/envoybench/retrieval-baseline-prepared-20260930/`: packets and pre-run
  protocol, with `requests_executed: 0`.
- `out/envoybench/calibration-20260930/review-packet.json`: blank independent
  review packet, eight paired questions.
- `out/envoybench/studio-snapshot-improved-20260930.html`: standalone viewer.

Repository annotation artifacts are in `case_study_20260930.json` and
`response_behavior_20260930.json`; loaders reject mismatched result hashes and
invalid exact-source quotes. See [the README](README.md#offline-diagnostics-and-next-comparisons)
for preparation/export commands.

After Anthropic API credits are available, the original grader can resume:

```bash
python -m benchmarks.envoybench.judge \
  --review out/envoybench/provisional-review-20260930/review.json \
  --output-dir out/envoybench/provisional-judged-20260930 --resume --execute
```

Run artifacts live in `out/envoybench/provisional-full-20260930/`; the blinded
input and identity key are in `out/envoybench/provisional-review-20260930/`.
The grader's append-only checkpoint is in
`out/envoybench/provisional-judged-20260930/judgments.jsonl`. Keep the blind key
away from an independent reviewer. Human reference decisions remain untouched.

The separate Codex packets, raw verdicts, declared fallback method, and assembly
script are in `out/envoybench/codex-review-20260930/`; the validated combined
review, provenance report, and score are in `out/envoybench/codex-reviewed-20260930/`.
The scored report verifies every blind assignment against the exact saved
model result before revealing identities.
