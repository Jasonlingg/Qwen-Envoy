# Native ML workflow pilot — collected answers, human review pending

All ten native answers were collected, one attempt per question/arm. **Codex made
zero MCP calls in all five B sessions**, despite the native inventory exposing
the expected ten tools. Both arms used ordinary web research. This is a finding
about tool use under the frozen instructions; it does not establish whether the
MCP was undiscoverable, unnecessary, or less attractive for these questions.
No quality or cost advantage is established, and Qwen remained off.

## Actual collection

Both arms used native Codex CLI 0.160.0, `gpt-6.1-sol`, medium reasoning, matched
instructions and separate temporary workspaces. B had source-only research and
experiment MCPs with new empty roots; A had no active MCP tools. All ten native
inventory/settings/canary checks passed. The frozen generation inputs and source
hashes remained unchanged throughout collection. No answer was retried, edited,
or generated with a custom host fallback.

| Question | A seconds | B seconds | A/B research calls | A/B words including sources |
| --- | ---: | ---: | ---: | ---: |
| MLP01 model selection | 76.36 | 73.59 | 5 / 6 | 587 / 644 |
| MLP02 latency diagnosis | 45.36 | 54.99 | 4 / 4 | 598 / 591 |
| MLP03 fine-tuning decision | 69.50 | 65.91 | 3 / 3 | 667 / 668 |
| MLP04 speculative decoding | 60.68 | 56.63 | 4 / 5 | 732 / 672 |
| MLP05 result interpretation | 40.99 | 33.00 | 0 / 0 | 524 / 516 |

Each arm produced five classified final answers. There were zero recorded native
tool/protocol errors, timeouts, research-target interruptions, or observed
overshoot. There were **16 A / 18 B native research invocations**, all web actions,
and zero experiment-record writes. Compound action/query details remain in the
operator receipts and raw traces; invocation counts are not underlying network
request counts. The 20-call target was never reached. No Qwen/Nemotron/extra
reader endpoint was called, no GPU was started, and no proposed ML experiment ran.

Answering time sums were 292.89 seconds for A and 284.11 for B; means were 58.58
and 56.82 seconds. These measure prompt dispatch to terminal answer after native
setup, excluding review/correction. They are descriptive observations from five
stochastic pairs. Quality-normalized time remains unknown, and the observed
timing difference is not evidence of MCP efficiency. Complete end-to-end setup,
review labor, provider load and hidden caches were not controlled or accounted.

| Reported host token category | A, five answers | B, five answers |
| --- | ---: | ---: |
| Input | 680,161 | 813,474 |
| Cached input, included in input | 519,552 | 604,672 |
| Output | 10,057 | 10,104 |
| Reasoning output, included in output | 2,787 | 2,692 |
| Total input + output | 690,218 | 823,578 |

These are the native per-thread reported totals across tool-mediated generation,
not ten single API requests. Provider request counts and actual dollar charges
are unknown; cached tokens are not assumed free. There was no hard host token or
monetary cap. A local process-group deadline cannot recall an already-dispatched
remote request. The paid Nebius smoke was not part of this run.

## Review status

The [anonymous review packet](results/review/README.md) has all five pairs,
verbatim answers, questions, frozen rubric and an empty human worksheet.
Arm mapping and technical metadata are in the separate operator directory;
do not inspect them before recording initial grades. Formatting/citation cues
can still reveal identity and should be recorded. All ten human verdicts,
pair preferences, correction times and derived time-to-useful results are
**unknown**, not passes. No human review has been fabricated.

A separately labeled [assistant spotcheck](results/assistant-spotcheck.json)
examined four limited claims across MLP04/MLP05 after collection. The speculative
sampling distribution claim agrees with the [Leviathan paper abstract](
https://proceedings.mlr.press/v202/leviathan23a.html), [Chen v1 abstract](
https://arxiv.org/abs/2302.01318v1) and the numerical/implementation caveats in
[pinned vLLM documentation](
https://docs.vllm.ai/en/v0.17.1/features/speculative_decoding/). It does not verify
all table values, licensing assertions, implementation compatibility or local
performance predictions. Both result-review answers retain the hypothetical
quality gate as met and latency/all-in-cost gates as unknown, with +1 percentage
point and 24% raw reported reduction calculated correctly. Those limited checks
are unblinded assistant assessments, not independent human grades or complete
answer-support review. The two post-collection operator web calls are logged
separately from answer-generation research.

## Artifacts and limits

[collection-report.json](results/collection-report.json) retains all ten outcomes;
[aggregates.json](results/aggregates.json) contains descriptive totals. The raw
native traces, full exposed source results, actual MCP schemas, settings and
boundary checks remain locally under `out/ml-workflow-pilot-v1-collection/`.
Exported operator receipts bind those files by SHA-256, list source URLs and
record tool actions. The committed review packet does not publish raw reasoning
or unrelated account telemetry; full local traces are not bundled in Git. Keep
the local collection to reproduce the receipt checks and inspect source context.

The complete built-in advertised tool schemas and hidden provider requests are
not supplied by the verified interface. Native web result bodies were exposed
in this run and saved locally; they are not proof of semantic support. Source
cutoff compliance beyond the limited spotcheck remains part of human review.
Workspace setup and operator work during collection also limit timing comparisons.

This diagnostic used five assistant-authored hypothetical development scenarios,
one sample per arm and no Qwen. It cannot prove broad research superiority,
equivalence, memory/reuse benefit, or improved fine-tuned Qwen. Actual MCP use
would need to occur before a later comparison could evaluate its retrieval or
record-keeping contribution. No routing changes or extra runs were made after
seeing this non-use finding.

## Verification and one unresolved test failure

The final full repository suite passed **1,486 tests** with three existing SWIG
dependency warnings, in 54.96 seconds. Eight focused controller/packet tests,
Ruff and `git diff --check` passed. All ten anonymous answers match the local
native outputs byte-for-byte; raw artifact hashes, all 48 frozen generation
inputs, matched native settings/inventories/canaries and all ten pre-existing
file hashes were verified. No pilot MCP processes remained after cleanup.

An earlier final-suite attempt had 1,485 passes and one failure in
`test_duplicate_run_is_rejected_across_processes`: a child received
`FileNotFoundError` opening `.stage.lock`. The focused test then passed, all 28
staging tests passed, and the full unchanged suite passed. No staging code or
test was changed to suppress the failure; its cause is unresolved. This software
test rerun did not rerun or replace any native benchmark answer. The validation
receipt retains both the failure and the subsequent successful checks.
