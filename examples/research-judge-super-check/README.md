# One hosted semantic-judge diagnostic

This task checks a **hosted Nemotron Super alias** against the 32 frozen
[development fixtures](../../benchmarks/research_judge_checks_v1/README.md).
It does not run Qwen, modify a reward, train a model or use reserved reader
questions, references or paper contents. Labels remain provisional model-assisted
source judgments; human fields stay blank.

The [pre-dispatch amendment](pre-dispatch-amendment.json) preserves the original
judge plan and its zero-error score thresholds. The API does not expose an
immutable weight revision. Consequently, even 32/32 agreement can establish only
agreement for this batch and hosted alias. The original immutable-judge readiness
condition remains **unmet**; no production-verifier or training approval follows.
The batch shares context across cases and is not a test of one-case deployment.

The fixed request sends the generic frozen rubric, question/fact criteria,
candidates and their own evidence, and each of the two full source texts once.
Expected candidate scores, per-fact verdicts, categories and rationales are not
sent. All content is treated as data; candidate text is never executed. The
raw response is retained before comparison with the expected labels.

Budget: one request, n=1, zero retries or continuations, and at most 16,384
completion tokens **including reasoning**. Reserve the full provider context
of 262,144 tokens as input, avoiding assumptions about tokenizer or chat-template
length. At uncached $0.30/M input and $0.90/M output, the conservative local bound
is **$0.0933888**, below $0.50. This deliberately overcounts output if the context
limit includes it. Active model status, exact context and rates are checked
again before dispatch. The bound assumes the provider enforces its documented
billing/context/completion contract; it is not a provider-side invoice lock.

Ultra was considered and rejected **before inference**: its 1M context at $1/M
input did not provide the same spending bound without an unverified hosted
tokenizer assumption. No Ultra request was sent. The authoritative price/context
snapshot is saved in the prepared manifest. See the
[official model catalog](https://tokenfactory.nebius.com/api/public/models_info)
and [completion API schema](https://api.tokenfactory.nebius.com/openapi.json).

A fixed experiment-wide `paid-allowance-consumed.json` is written exclusively and
fsynced **before dispatch**, independent of `--directory`. A crash, timeout,
provider failure or storage error consumes the only attempt. Never remove or
reset that receipt to retry. Redirects are refused, HTTP retries disabled and
responses bounded to 2 MB. There is a 240-second per-I/O timeout and soft elapsed
check; this is not a hard absolute deadline. Credentials are read from the local
environment only for `run` and are never saved.

```sh
# Creates a NEW directory; no model inference.
python scripts/run_research_judge_check.py prepare
# Exactly one paid dispatch under the fixed experiment allowance.
python scripts/run_research_judge_check.py run
# Offline comparison, after checking request/manifest/fixture/response/model/usage bindings.
python scripts/run_research_judge_check.py score
```

Prepared request/manifest bytes and implementation hashes are frozen before
inference. Raw response, receipt and comparison remain separate. Unknown or
malformed judgments are never converted into rewards. Missing, duplicate or
invalid case rows fail the thresholds while other valid judgments remain in the
report. Fact IDs are matched independently of their order. Any diagnostic output
used to tune judge/prompt/reward retires these cases into design and requires
fresh development checks. Passing these clear sentinels would not establish a
population error rate, semantic human gold or fine-tuned Qwen improvement.
