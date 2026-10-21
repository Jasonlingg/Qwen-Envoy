# Task 2: a real-project experiment-mode walkthrough

Approved scope: use the new mode to research one real project decision and save
a concrete next-test proposal. No GPU launch, paid reader, optimization run,
training or benchmark comparison is included.

Question: **What should we measure first to reduce latency for two concurrent
Qwen3-8B v5 research readers on the previously used 24 GB A5000 setup?**

The host selected this question from the user's recurring project goal; it is
not an independently supplied evaluation question. Historical startup/serial
request observations are available. They do not establish a current endpoint,
two-reader memory fit, phase latency or throughput baseline.

Hypothesis: linking actual setup observations and relevant research to a draft
will let the next session recover an actionable diagnostic without inventing
measurements or treating paper results as local performance.

Expected signal: actual stdio MCP calls inspect project-derived source facts,
discover/open relevant live papers, attach checked evidence, save a specific
test proposal, then recover that evidence and proposal after a process restart.

Decision rule: complete the walkthrough only if it preserves the historical
setup/unknowns, records at least one relevant source-backed finding with a
separate host claim/applicability review, and proposes a small reproducible
measurement with a stopping rule. Do not freeze an unapproved plan or record
unrun measurements as a result. Persistence checks do not prove productivity,
research quality or model improvement.

Initial inspection found a task-1 integration blocker: the experiment discovery
adapter accepts `search(query, *, limit)`, but the delegated source tool calls
`search(query, since, until, limit)`. The task-1 unit flow supplied a fake directly
and its real stdio flow never searched. Correct the adapter contract and test the
wrapped composition, including dated searches, before continuing the walkthrough.

## Outcome

The [saved walkthrough](../../examples/experiment-mode-walkthrough/README.md)
contains the exact host-authored inputs, selected historical observations,
claim/applicability review and twelve actual MCP call receipts. A real relevance
search returned the original paper through arXiv fallback after a primary rate
limit. The host opened/inspected it, attached five findings and saved revision 2.
After restarting the server, the full research note and proposal were recovered
unchanged without source calls. The recap correctly flagged note truncation;
the host fetched the full note by ID.

The walkthrough decision rule was met for the local research/planning flow,
with user usefulness still unreviewed. This is an unblinded host-driven example,
not an independent native assistant or an optimization/model benchmark. It
records no GPU inference, new candidate results or performance improvement.
The future diagnostic remains an unfrozen draft. Its fixed saved inputs cannot
measure adaptive policy behavior or complete research-answer quality.

The production adapter bug was fixed in `b2f9e94`; 37 focused checks passed.
The full offline suite passed 1855 tests with zero failures/skips and three
existing SWIG warnings in 144.47 seconds. All eleven protected-file hashes match
the pre-task values, including the active MuSiQue reward and original dirt.
Historical evaluations, training/reward code and reserved reader questions were
not changed. No new automated judge or benchmark framework was added.
