# A real-project experiment-planning walkthrough

This is an October 19, 2026 development example for **QASPER Agent Studio — a
Qwen Envoy case study**. It uses the saved September 30 outputs and QASPER
(Dasigi et al.; Ai2), not a new question dataset or an official leaderboard run.
No new Qwen inference, training, paid API request, or GPU launch was made.

The host assistant inspected saved failures, used the native research MCP to
find/read papers, saved a proposed diagnostic through the real planning MCP,
and separately imported historical outcomes. The proposal remains a **draft**.
Its usefulness still needs the user's review; no intervention outcome exists.

## What the investigation found

The frozen release and corpus were checked before exporting. The existing
model-assisted review gives **3/20 answerable passes to both base and v5**.
Aggregate provisional passes are 5/40 and 15/40; refusal behavior contributes to
that difference. These are model-plus-harness observations, not proof of a
fine-tuning benefit. The independent human review is still absent.

Five v5 states were selected after seeing outcomes:

| Case | Saved behavior and reason for including it |
| --- | --- |
| Dataset / wrong span | Reads text containing the Stanford Plane Crash Dataset name, then submits offsets 15,600–16,000. The naming sentence starts at 16,027. This separates retrieved information from submitted evidence. |
| SEPT / shallow answer | Reads an abstract-level passage and submits “combines the span extractor with BERT,” missing the requested simplification details under the existing model-assisted analysis. |
| Chinese NER / false refusal | Stops after two source actions; its refusal was graded false by the existing model-assisted review. |
| Nigerian Pidgin / supported control | Its tokenization answer was provisionally judged supported. A review intervention should preserve it. |
| German verbs / refusal control | Its refusal matches the source label. Independent confirmation of answerability/absence is required before using it as a trial control. |

The exact question IDs, saved actions, bounded observation prefixes, evidence
offsets and existing review notes are in
[the selected export](exports/qwen-v5-task5-development.json). This is not a
representative sample. Prefixes omit later text; inspect each `result_pointer`
in the unchanged parent `release/envoybench-v0.1/run/results.json` for full context.
The exporter verifies the frozen corpus and extracts submitted passages; those
mechanical checks do not determine semantic support.

A fixed-BM25 one-shot comparison already exists. Its official Answer F1 is
41.52/58.86 for base/v5, with support still unreviewed. Recommending that same
comparison as new work would duplicate an existing experiment.

## Proposed next diagnostic — not approved or run

**Question:** Does an evidence-focused review help v5 more than a generic
second look, before considering further training?

**Hypothesis:** On these five saved states, explicit claim-to-evidence review
may repair at least one additional substantive answer without breaking the two
controls or adding an error episode.

| Fixed in both arms | One changed input |
| --- | --- |
| Same pinned v5 adapter, initial history/draft, corpus, tools, Docker sandbox, decoding and two-action continuation allowance | Generic review instruction versus explicit claim-to-evidence audit |

Both arms may inspect the source again and keep, revise, or refuse honestly.
Neither receives gold answers, existing grades, case labels, or grader notes.
The two instructions are pinned in [review-prompts.json](review-prompts.json).
This is a proposed reconstruction/continuation diagnostic, not a fresh full-run
comparison. Its runner has not been implemented or approved.

**Decision rule:** An identity-blind human checks answer relevance, each material
claim against its own cited text, and refusal correctness. Candidate needs at
least one additional supported substantive answer over generic review, both
control behaviors retained, and no additional error/no-submission episode.
Record whether a repair uses text already visible or newly inspected. That
pattern can guide the next investigation, but cannot isolate every cause.
Null or harmful outcomes are reported without extending the trial.

The proposed ceiling is 20 requests, 143,360 input tokens, 20,480 output tokens,
900 GPU seconds, and 1,800 elapsed seconds, including failed requests and no
automatic retries. **These are unapproved planning values.** Hardware, shared
runtime, seed support, reviewer, pricing, enforced billing limits, and cleanup
must be resolved before any execution approval. No reliable dollar cap has
been established. The current demonstration used zero inference/compute budget.

The full draft is [proposal.json](proposal.json). Improving this five-case
diagnostic would justify considering a larger separately approved study; it
would not prove Qwen fine-tuning improved, or beat ordinary assistant browsing.

## What the papers support

The [research log](research-log.json) records ten native research-MCP calls and
one host web-search call (two queries). Three versioned full-text papers were
opened; lexical passage search and exact excerpts were inspected. This was a
targeted investigation, not an exhaustive review or a SOTA claim.

- [ALCE](https://arxiv.org/abs/2305.14627v2) treats support by cited passages as a
  separate evaluation dimension. This motivates the claim-level review rubric;
  it does not validate this intervention on Qwen.
- [Self-RAG](https://arxiv.org/abs/2310.11511v1) trains a reflection policy. A
  prompt-only audit is an inference inspired by the concern, not a reproduction
  of trained Self-RAG.
- [Huang et al.](https://arxiv.org/abs/2310.01798v2) report intrinsic
  self-correction degradation for their reasoning models/tasks and discuss
  prompt confounds. This motivates the generic-review control and regression
  checks; it does not prove Qwen cannot correct source-grounded answers.

[research-claims.json](research-claims.json) contains a small claim-level review
with exact short excerpts, offsets, coverage and document hashes. It is **host
model-assisted analysis**, not independent human review. MCP receipts continue
to say `semantic_support: not_reviewed`; exact text matching proves provenance.
HTML extraction can omit figures, tables and equations.

## Actual MCP outcome

The paper connector was available natively in this session. The planning server
was registered locally but its tools were not loaded into this host's inventory,
so the host used a Python MCP stdio client with the registered command and roots.
This exercised the real server rather than direct store calls. Full native
planning-tool availability in the host remains unverified until a client reload.

The walkthrough made ten planning-MCP calls across two server processes. It
saved question → researched revision (still draft); saved/froze a separate
historical reconstruction; recorded results; restarted; retrieved both; and
retried the historical result without changing it. It never froze the proposal
or attached historical results to the proposed intervention.

| Historical gate | Supplied-data result |
| --- | --- |
| Original provisional model-assisted pass gain ≥15 points | Met: +25 percentage points; answerable passes still 3/20 in each arm |
| Original error/no-submission tolerance ≤2 extra episodes | Met: −21 episodes (22 → 1) |
| Independent human-supported gain | Unknown: no independent human review |
| Complete latency comparison | Unknown: only 29/40 base timings are positive; matched lifecycle scope is absent |
| All-in cost comparison | Unknown: loading, idle, storage, cleanup and host accounting are incomplete |

**Overall: inconclusive; retrospective.** The last three checks are additional
conservative import checks, not the original predeclared gates. The original
two proxy gates justified seeking review, not a validated promotion.

[walkthrough.json](walkthrough.json) preserves the actual local record IDs,
plan/result digests, call bindings and assessment. These IDs exist in the
operator's ignored local store; replaying creates different IDs/timestamps.
The result's `artifact_measurement_verified` label also applies to explicit
unknown values: it verifies the selected export object, not a successful measurement.

## Reproduce without model calls

Install the project's existing dependencies (`pip install -e '.[mcp,vault]'`).
The checked-in selected export and input JSON work with the existing three
versioned paper caches. Open these versions through the research MCP first if
absent: `2310.11511v1`, `2305.14627v2`, `2310.01798v2`. Opening may fetch public
source text; it calls no model. Exact document hashes must match the references;
do not silently replace changed extractions. Full copyrighted paper text is not
bundled here.

```bash
python scripts/demo_ml_experiment_workflow.py \
  --store-root out/ml-experiment-reproduction \
  --artifact-root out/ml-experiment-reproduction-exports \
  --library-root out/arxiv-mcp-live-library \
  --report out/ml-experiment-reproduction-report.json
```

Use a new report path for a new replay. Each replay creates two records, keeps
the proposal as a draft, and freezes only the explicitly historical reconstruction.
The script refuses to replace a different export or an existing report.

To independently rebuild the export, the pinned QASPER corpus must already be
available (see the existing benchmark data builder). No test downloads it:

```bash
python scripts/export_ml_experiment_case_study.py --output out/task5-export-rebuilt.json
```

Compare its bytes/hash with the bundled export. Parent paths and raw SHA-256s,
selection recipe and truncation are recorded. The exporter checks parent
bindings and corpus bytes; the MCP itself verifies only the selected export.
An export hash is not proof of faithful derivation without that separate check.

Focused verification covers export reproduction, bounded references, exact
prompt bindings and a temporary-store stdio round trip. Corpus/cache-dependent
checks explicitly skip if those inputs are absent; they never fetch in tests.
Task 6 is complete: independent assistant code review and re-review, two fixes,
1,478 passing repository tests, and read-back of these unchanged records through
the configured servers. See the [review record](
../../docs/implementation/2026-10-19-ml-experiment-release-review.md) and
[machine-readable verification](task6-validation.json). User usefulness review,
human semantic review and intervention/model performance remain unestablished.
