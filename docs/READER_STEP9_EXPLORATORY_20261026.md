# Saved step 9 reader: exploratory paired evaluation (October 26, 2026)

The original ten-update RL pilot stopped during update 10 because a generation
request timed out. Nine updates and the step 9 adapter were verified and saved;
the planned step 10 evaluation never ran. This follow-up compared that saved
adapter with the original V5 adapter. It did not resume training or complete the
original pilot.

## Frozen question and decision rule

Hypothesis: step 9 would improve complete, source-supported short-fact answers
on the same twelve QASPER validation questions, without worse execution errors,
unsupported claims, latency, or token use. The initial exploratory protocol was
frozen before opening references. Source review then found two ambiguous
questions (0 and 6) before any model output. The [audited protocol](../examples/reader-step9-eval-20261026/protocol-reference-audit.json)
kept all twelve, held the candidate and reference labels fixed, marked both
ambiguous items unresolved in both arms, and forced the overall quality
decision to **inconclusive**. The other ten yield descriptive paired diagnostics,
not a validated improvement claim. The [original ten-update protocol](../examples/reader-rl-pilot-v2/protocol.json)
and its incomplete result were not rewritten.

Both arms used the pinned Qwen3-8B base revision
`b968826d9c46dd6066d109eabc6255188de91218`, the same prompt, Docker tool
environment, question order, paper scope, sampling settings (temperature 1,
seed 42 plus question index), and up to four investigator steps. V5 adapter SHA
was `7afba233aecc9fb9f1f23638514e309d139caf5c55d81af4138a05edd89d62a3`;
verified RL step 9 adapter SHA was
`a2a773caf222aa34becbeb2f886ce553c0aefa0871783fa17efe8da0557ca810`.
V5 ran first, followed by step 9 on the same A100. The shared ceilings were 24
episodes, 288 generation requests, 2,359,296 reserved tokens, and 1,800
controller work seconds. There were no generation retries.

## Results

| Measure | Original V5 | Saved RL step 9 |
| --- | ---: | ---: |
| Complete, correct, source-supported answers | 1/12 | 1/12 |
| Complete, correct, source-supported answers on clear references | 1/10 | 1/10 |
| Answers with at least one unsupported material claim | 5 | 6 |
| False refusals | 3 | 3 |
| Episodes with Python execution errors | 0 | 0 |
| Incomplete packets | 3 | 2 |
| Generation requests, all returned | 37 | 35 |
| Reserved tokens | 151,970 | 142,482 |
| Actual prompt plus completion tokens | 116,912 | 108,583 |
| Median episode time | 11.98 s | 11.31 s |
| 95th percentile episode time | 70.60 s | 27.51 s |
| Pinned upstream QASPER answer token F1, all twelve | 0.0937 | 0.0937 |
| Pinned upstream QASPER answer token F1, clear ten | 0.1125 | 0.1125 |

The token F1 figures use the pinned upstream tokenizer/metric, all original
annotations for each selected question, and zero for missing or incomplete
submissions. They are secondary overlap diagnostics. These twelve validation
questions are not a full official QASPER test leaderboard result. The latency
tail includes one 70.6-second incomplete V5 episode; one seed and twelve
questions cannot establish a durable speed difference. The 137-second V5
startup includes a boot-wrapper repair before any model request. Step 9 startup
was 45.9 seconds; startup is excluded from episode timing.

| Question | V5 | Step 9 | Review finding |
| --- | --- | --- | --- |
| Prior state of the art, paper 1905.10851 | No evidence | Unsupported UPA/PPA claim | Original label is ambiguous; neither counts correct. |
| Airbnb dataset size | Described Airbnb | Described Airbnb | Both omitted roughly 40,000 listings, which appeared in their evidence. |
| Other clustering hyperparameter | Incomplete | Incomplete | Both omitted choice of word vectors. |
| Character count | Said count unavailable | Said count unavailable | Both cited text explicitly saying 45,821 characters. |
| New corpus document count | Said count unavailable | Said count unavailable | Both omitted 53 documents. |
| Offensive-language dataset difference | Unsupported phrase | Same phrase | Both missed the target and hierarchical labels. |
| RNN variant | Incomplete | Incomplete | Question conflates proposed and baseline variants; unresolved. |
| Embedding dimension | Said size unavailable | Same answer | Both cited text explicitly saying 300 dimensions. |
| Translation language pair | German–English amid unrelated claims | Same answer | Correct pair was present, but unsupported extra claims make the answer unsafe. |
| Prior WinoGrande system | Unsupported AUC metric | Same answer | Both missed RoBERTa. |
| Meaning of industry | Incomplete | Repeated “users' industry” | Neither gave the definition: aggregate of enterprises in a field. |
| EHR dataset | MIMIC-III | MIMIC-III | Both answered and cited it correctly. |

The answer review was **model-assisted**, performed on the identity-masked pack
after both arms completed. It checked each material claim against the cited
inspected passage and checked whether the requested fact was complete.
`independent_human_review` remains `null`. Exact quotation and span matching
alone did not establish semantic support. The [comparison record](../examples/reader-step9-eval-20261026/comparison.json)
contains per-question booleans, latency, reserved tokens, and source-bound
review bindings; raw episodes and the private arm map remain in that ignored
local run directory. The formal decision is
`inconclusive_reference_ambiguity`. On the ten clear questions, neither arm
gained a supported answer; the extra unsupported step 9 answer was on an
ambiguous question.

## Runtime and cost

The first baseline engine attempt failed before generation because the trusted
inference entrypoint lacked a Python `__main__` guard for vLLM worker spawning.
That wrapper was repaired in place within the same rental; the frozen adapter,
prompt, questions, protocol, and generation settings were unchanged. The
corrected wrapper booted both sequential engines. Every generation request
returned, and the model-authored Python still ran in the local Docker sandbox.

One A100 SXM80GB pod (`iydj4t5dlaifcf`) and one independent CPU cleanup guard
(`aeijtcgn6bc6jk`) were rented. The guard and local watchdog both carried the
same scoped resource IDs. Both pods returned provider 404 after collection;
there is no remaining pod from this run. The GPU was observed for at most
842.8 seconds and the guard for at most 884.8 seconds. Quoted compute rates
were $1.79 and $0.06 per hour, respectively; the conservative combined
$1.90/hour allowance implies about **$0.47** for that observed lifetime,
including the allowance for disk charges. This is an estimate, not final
billing. Runpod's per-pod billing endpoint had posted **zero records** for
these IDs at the last read; the [billing receipt](../examples/reader-step9-eval-20261026/billing-receipt.json)
marks actual spend pending rather than $0. The absolute rental deadline was
15:23:52.808 UTC; both pods were deleted around 14:43:38 UTC.

## Interpretation and reproducibility

This run gives no evidence that the saved RL adapter improved the reader over
V5. The longer request deadline allowed one 60.2-second request to return
successfully; the earlier 25-second proxy limit would have cut it off. This
does not prove future requests cannot time out. Many errors were failures to use evidence already
inspected, so new training would need a stronger, reviewed answer-quality
signal and more data. The step 9 optimizer state was not saved, so the original
ten-update run cannot be resumed exactly from this adapter. No model was
uploaded or promoted.

The frozen protocol, source hashes, original training proof hashes, adapters,
backend receipts, 24 raw episodes, 72 request/response records, reference audit,
masked claim review, QASPER F1 diagnostics, and lifecycle receipt are in
`examples/reader-step9-eval-20261026/` and the ignored
`out/reader-step9-eval-20261026/`. Verification before launch: 2,168 tests
passed, 19 skipped in the full suite; 68 focused native tests passed with one
deselected; delayed proxy response, timeout, and native Docker paths passed.
Post-run summary validation passed. The review has no independent human
adjudication and the provider bill is still pending.
