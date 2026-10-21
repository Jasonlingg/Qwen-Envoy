# Development semantic-judge checks — fixtures only

These 32 authored counterfactual answers test whether a **future judge** can
recognize supported requested facts and reject clear evidence failures. No judge
has been selected or run. Expected labels are provisional, authored by the
implementing assistant and source-reviewed by a separate model-assisted reviewer;
they are not human gold, agent episode results or rewards for training.

The hypothesis and zero-error decision rule were declared in
[`judge-validation-plan.json`](../../examples/reader-learning-diagnosis/judge-validation-plan.json)
before construction. Only the already-inspected **development** FlashAttention-2
`2307.08691v1` and AWQ `2306.00978v6` snapshots are used. PagedAttention/QLoRA are
reserved for judge design examples. The separate reader evaluation questions,
references and paper contents were not used or changed for this task. These
public checks are not certified disjoint from model pretraining.

| Fixture group | Count | Authored expected score |
| --- | ---: | --- |
| Critical negatives: six types, two each | 12 | 0 |
| Complete supported answers, including paraphrases | 8 | 1 |
| Supported partials | 4 | 1/2, 2/3, 2/5, 5/6 |
| Scoped target gaps with useful next queries | 4 | 1 |
| Blanket refusals to answerable questions | 4 | 0 |

The six critical types are hardware/workload transfer, wrong baseline/algorithm,
true number for the wrong measurement, invented target result, unrelated valid
quote, and a valid target gap accompanied by an unsupported partial claim. The
unrelated-quote group includes one clipped quote: its answer values are true in
the complete paper, but its own quote stops before both measurements. This is a
support failure, distinct from an invented value.

`questions.json` fixes atomic requested-fact criteria from each literal question.
It adds no unasked details. For answerable cases, C is the fraction correctly
answered **with support in the candidate's own evidence**. Measurement correctness
includes its stated hardware, workload and baseline bindings. Equivalent wording
counts; repeated assertions do not. Any unsupported or contradicted material
assertion gates the expected score to zero, including optional partial findings
attached to an otherwise valid gap. Reference contexts help check correctness;
they cannot repair missing support in the candidate's own citations.

Gap cases assume completed source inspection **only as an explicit synthetic
fixture precondition**. There are no generated or executed search/read traces.
Real agent evaluation must independently establish recorded source inspection;
these static cases cannot validate that runtime requirement or the multi-step
worker. The existing `missing` / `follow_up_query` gap-report interface is kept.

`cases.json` mixes groups under neutral IDs; `labels.json` contains expected
per-fact judgments, the all-material gate, score and rationale. Source bytes,
labels, questions, human template and source-review hashes are bound in
`freeze-lock.json`. The independent review checks source meaning; the checker
below checks only file integrity, counts, spans and label arithmetic. Neither
is a human review or a successful judge-validation result.

```sh
python scripts/check_research_judge_cases.py
# Optional: a new output path; no model request is made.
python scripts/check_research_judge_cases.py --export-judge-inputs /tmp/judge-inputs.json
```

The export is **grader input**, not a reader-policy prompt. It contains the
question, fixed fact criteria, trusted full source/context, candidate and own
citations, and synthetic inspection precondition. Expected candidate scores,
per-fact verdicts, categories and rationales are excluded through nested field
allowlists. Public file separation is not access control. The exporter does not
choose a judge, prompt, decoding or budget, and never executes candidate text.

Before any judge calls, freeze the actual judge model/revision, instructions,
limits and decoding separately. Every critical error, incorrect positive/partial/
gap/refusal score, unknown judgment or missing case fails the predeclared gate.
Any tuning informed by failures retires this set into design and needs fresh
development checks. Do not repair a judge using the reserved reader evaluation.
Passing 32 deliberately clear sentinels would not establish a population error
rate, optimization safety, training approval or improved Qwen performance.

The active MuSiQue reward, historical grades and original frozen reader bundle
remain unchanged. Human-review fields are blank. No candidate-model requests,
paid API calls, GPU activation or training were performed.
