# QASPER passage-v2 development smoke

User authorized a bounded live test after the atomic-gap change, and explicitly
asked to use QASPER. This uses the already exposed frozen 40-question
`test_candidate` split, not an untouched test or full QASPER leaderboard.

Hypothesis: V5 can finish selected-passage answers or an atomic, scoped gap within
four requests. Expected signal: actual inspected passages, verified terminal
packets and source-supported facts or appropriately scoped missing information.
Decision rule: report each case, every action rejection, protocol completion,
retrieval coverage and claim-level support separately. All three must complete
without invented facts to pass this small smoke; no population/model-improvement
inference follows. Retrieval failure is not automatically reader failure.

Before inference, choose the first two answerable and first unanswerable questions
ordered by SHA-256 of `42:<question ID>`, keeping distinct papers. No filtering by
reference answers, evidence availability, model success or retrieval quality.
Rank fixed 1600-character windows (stride 800) over each entire target paper with
BM25Okapi defaults and lowercase word tokens. Pick the two highest-ranked
non-overlapping windows, break ties by start, and present them in source order.
If two windows cannot be selected, fail preparation rather than replace a case.
Queries are the original source questions. References and evidence labels are
never sent to Qwen. Selection uses answerability labels only for stratification.

QASPER/S2ORC text retains its dataset revision and coverage; it must not be
misrepresented as a freshly extracted/versioned arXiv document. Use a distinct
controller assignment version. This is an evaluation MCP adapter over the real
Docker-only investigator and verifier; it does not add QASPER ingestion to the
personal arXiv product or test online paper discovery.

V5 checkpoint, decoding and sandbox are pinned to the preceding passage tests.
Use `passages-v2`, max 4 actions/4 requests/32768 reserved tokens per case; shared
12-request/98304-reserved-token limit. No inference retries or replacement cases.
One existing A5000 activation; 900 seconds INCLUDING cold start, watchdog armed
before activation, restore configuration and verify zero workers/pending jobs.
One new exclusive marker; all older allowances stay consumed. Local allowance
$0.50 with a fresh serverless rate <=$0.69/hour (15-minute compute estimate
$0.1725); this is NOT an enforceable provider invoice cap. No Nebius calls.

Freeze inputs, implementation/source hashes and scripted real MCP/Docker preflight
before activation. Run focused and full relevant tests before paid activation.
Retain raw outputs, cleaned actions, source text/offsets, usage, gaps and failures.
After inference, inspect claim-level support and original annotations separately;
exact spans establish provenance only. An inspected-passage gap is not proof the
whole paper is unanswerable. Human review remains pending. No training, reward
changes, base-model calls or access to the reserved new-paper pack are included.


## Outcome

Implementation `f7f11ff`; selection and first inputs frozen in `9291514` before
activation. [Actual result](../../examples/qasper-passage-v2/README.md): nine V5
requests, 27,936 reported tokens; no execution errors, three mechanically valid
terminal packets. Useful-output gate not met: one retrieval miss with inability
submitted as an answer, one incomplete but broadly supported NMT answer, and one
terminal gap with copied placeholder content. No unsupported racial capability
or cue list was invented. Exact quote checks did not catch these quality issues.
A fresh check verified complete cloud shutdown/config restoration; six owned
Docker containers absent. Final suite: 1974 passed. No further paid attempt or
training is authorized by this completed smoke.
