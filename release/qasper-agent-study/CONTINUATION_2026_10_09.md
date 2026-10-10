# Nebius reference continuation — protocol amendment

Recorded before the continuation's first model request on October 9, 2026.

The user authorized raising the local **total catalog-rate estimated-cost ceiling
to $25** to finish the interrupted NVIDIA/Nebius reference run. This replaces
the original $2 ceiling for the continuation only. It is a local request guard,
not an account billing cap. The original partial run and its $1.947862 usage
record remain unchanged and count toward the $25 total.

**Hypothesis:** with enough budget to complete the final two questions, the
hosted Nemotron configuration will yield a measurable full-40 official QASPER
Answer F1 and complete code-execution traces. This does not predict that its
answers will be better supported than Qwen's.

**Expected signal:** question 39 and 40 both reach terminal episode outcomes;
their saved actions can be inspected; the first 38 result rows remain identical;
the combined estimated cost remains below $25 and the request count below 600.

**Procedure:** preserve `nebius-run/` as the immutable interrupted record. Carry
forward its 366 requests, 1,819,534 prompt tokens, 42,776 completion tokens,
and $1.947862 estimated spend into the continuation guard. Run only the last
two frozen questions, in order, with the same declared model, prompt, tool
environment, 15-step limit, decoding, and seed. The runner cannot restore the
live REPL or model conversation at question 39's interrupted step 11, so
question 39 restarts at step 1. Question 40 has never been attempted. No
finished question is rerun, and there is no answer-guided selection. The
question 39 restart is a **protocol change** from the original no-retry plan.

**Decision rule:** only after the two-question extension is complete and
validated against the original manifest, combine the first 38 original rows
with the two extension rows into a lineage-labeled derived 40-row artifact.
Then score literal submitted answers using the pinned official QASPER Answer F1
evaluator. If either question or the endpoint fails, retain the extension as
incomplete and report no full-run score. Do not call the result a controlled
Qwen–Nemotron comparison or a semantic citation-support grade.
