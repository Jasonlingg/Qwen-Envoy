# EnvoyBench human review rubric

The primary result is whether an answer is useful **and supported by the paper**.
An exact character span only proves that text was copied from the frozen source;
it does not prove that the answer follows from it. Model-graded or automatic
labels must not be reported as independent human judgments.

## Before reviewing model outputs

For each candidate test question, inspect the QASPER question, source paper,
annotated answer(s), and cited evidence. Mark `reference_valid` only if the
question has a clear interpretation and the converted source text contains the
information needed to score it. Record ambiguous, missing, or contradictory
references as `unscorable` **before** opening any model outputs. Keep excluded
question IDs and reasons in the release rather than silently deleting them.

## Blind answer review

Shuffle and anonymize model answers per question. Reviewers see the question,
validated reference, submitted answer, cited paper IDs, exact cited passages,
and any stated uncertainty. They do not see model names, training stage,
automatic reward, or aggregate results until verdicts are locked.

- **Pass:** Answers all important parts correctly, with supporting source text
  for material claims and honest limits. For insufficient-evidence questions,
  correctly explains why the available paper cannot answer instead of
  inventing a result.
- **Partial:** Supplies a useful, supported part of the answer but misses an
  important requested detail or qualification. No material false or unsupported
  claim may be present.
- **Fail:** Gives a wrong or material unsupported claim, misstates the paper,
  cites irrelevant text as support, invents a source or result, falsely refuses
  an answerable question, or submits no useful answer.

Record a short reason and the relevant source passage for every `partial` or
`fail`. If the model's answer exposes a genuine ambiguity in the reference,
pause the question and adjudicate its reference *without looking at model
identity*. Do not change the rubric after seeing which model wins. The review
artifact should record reviewer identity or role, review date, rubric version,
and any adjudication. A second independent reviewer on a subset helps estimate
disagreement; report that agreement rather than treating consensus as given.

## What to report separately

Publish paired per-question verdicts and aggregate pass/partial/fail counts.
Alongside them, report valid-span rate, required-paper recall where annotated,
submission rate, execution-error episodes, repeated actions, step count, wall
time, and hardware/serving configuration. Do not combine these into an
unexplained single score. The existing MuSiQue reward is not an answer-support
verdict, and a valid quote is not a pass.
