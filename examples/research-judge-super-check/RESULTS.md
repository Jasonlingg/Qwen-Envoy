# Hosted judge diagnostic: no usable judgments

The single authorized Nemotron Super request ended at its completion limit,
with **zero visible answer characters and no valid case judgments**. The response
reports `finish_reason: length`, 16,384 completion tokens and 16,384 reasoning
tokens. All 32 cases are missing/unknown. The predeclared diagnostic thresholds
failed; this did not measure semantic agreement or disagreement with the labels.

| Observed result | Value |
| --- | --- |
| Judge | `nvidia/nemotron-3-super-120b-a12b` via Nebius |
| Model requests / retries | 1 / 0 |
| Input / completion tokens | 42,746 / 16,384 |
| Provider-reported reasoning tokens | 16,384 |
| Visible JSON judgments | 0 |
| Unscored cases | 32/32 |
| Recorded elapsed time | 119.72 seconds |
| Catalog-rate cost | $0.0275694; invoice cost unknown |
| Conservative preflight maximum | $0.0933888, below $0.50 |
| Human review / training | Not performed / not authorized |

The prepared request/prompt were committed in `7257aa9` before inference, using
implementation `b611a8b`. [The receipt](run/receipt.json) binds the exact request,
manifest, fixture lock and raw response. Served model identity matches the frozen
alias; provider fingerprint is recorded. [The comparison](run/comparison.json)
retains a row for every case, with each judgment and score **null**, rather than
inventing zero rewards. Its zero observed false-acceptance/error counters are not
measured zero error rates: there were **no usable judgments**. All 32 missing and
unknown cases breach the predeclared zero-error limits.

This is a failure of the tested batch/default-reasoning/output-budget
configuration. It does not show that the model could not judge a smaller task,
that Qwen is incapable, or that fine-tuning helped. The batch size, prompt and
provider reasoning behavior were not isolated in an ablation. No hidden reasoning
was converted into verdicts, no model was substituted, and no extra paid attempt
was made. The experiment-wide allowance remains consumed, including from an
alternate output directory.

The [pre-dispatch amendment](pre-dispatch-amendment.json) already limited this
to a hosted-alias diagnostic: no immutable provider weight revision is available.
That readiness condition remains unmet independently of this response failure.
The full-context cost reservation and provider usage show this attempt stayed
within its local API-contract spending bound; it was not a provider account or
invoice lock. No Runpod resource was launched.

Before proposing another diagnostic, verify a supported reasoning control or
use smaller batches with separately frozen request/output limits. Any changes
informed by this attempt retire these cases into design and require fresh
**development-only** validation; the reserved reader evaluation remains unused.
Do not increase the budget, retry this run or launch RL automatically. No active
reward, historical quality grade, frozen fixture label or reader source changed.

Offline verification: 61 focused tests passed, followed by the final relevant
suite with **1,830 passed**, no skips and three existing SWIG deprecation warnings.
The focused tests include global one-shot consumption, failed-request retention,
response/identity/hash binding, fact-order independence, partial failure coverage,
redirect refusal and response limits. Synthetic oracle responses in those tests
check comparison mechanics only and are not live model results.

Reproduce the saved comparison without inference or overwriting its output:

```sh
python - <<'PY'
import json
from pathlib import Path
from src.eval.research_judge_score import score_saved
repo = Path.cwd()
run = repo / 'examples/research-judge-super-check/run'
actual = score_saved(run, repo / 'benchmarks/research_judge_checks_v1', repo)
assert actual == json.loads((run / 'comparison.json').read_text())
print(actual['status'], actual['metrics'])
PY
```

Preparation records in the frozen fixture pack and original judge plan retain
their historical `not_run` state. This separate dated diagnostic supplies the
actual outcome; those earlier records were not rewritten. Labels and independent
source reviews remain model-assisted, with human fields blank.
