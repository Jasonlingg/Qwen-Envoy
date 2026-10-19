# ML experiment planning through MCP — v1 design

Date: 2026-10-19. Status: Tasks 1–2 complete; Task 3 approved and implemented.
The user's question drives the workflow. Tasks 4–6 await separate approval.

## Goal and first useful workflow

Help an ML engineer assess a proposed model, fine-tuning change, or latency
method against their actual workload, then design and review a small experiment.
The conversation remains in the host assistant. The existing research MCP
supplies papers; a small companion MCP supplies durable experiment records.

Example: “Qwen v5 still struggles with answerable questions. Inspect the saved
results, find relevant methods, and propose one experiment that could distinguish
missed evidence from premature stopping.” This question does not establish either
cause. The first delivery is a usable experiment brief, followed by result review.

V1 scope: explicit project context, selected JSON run artifacts, paper evidence,
a proposed baseline/candidate comparison, frozen criteria, and recorded outcomes.
There is no training, benchmark execution, infrastructure provisioning, automatic
GPU lifecycle, new dashboard, or Obsidian dependency in this implementation.
The public QASPER case study retains its existing name, artifacts, and limitations.

## Repository findings and reuse

| Existing component | Reuse and boundary |
| --- | --- |
| `scripts/arxiv_research_mcp.py` | Existing five source tools; optional sixth `investigate` tool. Keep this server's interface and source-only profile intact. |
| `src/product/arxiv_library.py`, `paper_discovery.py`, `evidence_search.py` | Existing discovery, version-pinned cache, and exact passage extraction. Read cached evidence when saving a brief; fetching stays with the source MCP. |
| `packet_verifier.py`, `qwen_investigator.py`, `parallel_readers.py` | Optional configured readers and verified packets. Preserve Docker execution, V1 compatibility, and per-paper failure isolation. Live Qwen usefulness remains unvalidated. |
| `src/product/learning_thread.py` | Reference for locked atomic persistence, revision checks, and separating assessments from verified spans. Its accepted-view/Obsidian semantics do not fit experiment plans directly. |
| `src/eval/research_pilot_review.py` | Reference for immutable content bindings, incomplete coverage, and separate human/model review. Its two-arm pilot schema is not a general experiment schema. |
| `src/eval/artifacts.py` | Existing reproducibility conventions. Its `content_hash` includes filenames; do not confuse it with a raw-file SHA-256. |
| `scripts/build_qwen_experiment_vault.py` | Historical report-to-note reconstruction, not an experiment runner or prospective plan store. Reuse evidence distinctions rather than its vault-specific format. |
| `release/envoybench-v0.1/`, `release/qasper-agent-study/` | Real saved traces, manifests, model-assisted reviews, and official scores for the first development example. Preserve the original files. |

Reviewed guidance includes `AGENTS.md`, `CODEX_HANDOFF.md`,
`docs/TECHNICAL_CASE_STUDY.md`, the original harness/arXiv designs and plans,
`docs/RND_RESEARCH_WORKFLOW.md`, and the historical second-brain/radar plans.
The user's current task approvals authorize this bounded experiment workflow;
historical training, vault, and deployment instructions are not new authorization.

Worktree at inspection: four existing modified files concern Docker image
selection (`docs/ARXIV_RESEARCH_MCP.md`, both arXiv/pilot scripts, and
`src/product/arxiv_mcp_config.py`). Three October 16 plan/spec files are untracked.
Leave all seven and the protected training files unchanged. Image-selection
verification and the blocked live-reader retry remain separate pending work.

## Architecture and responsibility

```text
User ↔ host assistant
         ├─ existing research MCP → discover/open/search papers; optional reader
         ├─ ordinary authorized file/web tools → inspect setup and run exports
         └─ experiment MCP → validate/save brief → freeze plan → record results
                                ↓
                      local JSON records + rendered Markdown
```

Add `scripts/ml_experiment_mcp.py` as a local stdio companion, with five tools
below and concise workflow instructions. It makes zero model/network calls.
Registering the companion and selecting input directories are explicit Task 3
work. The host researches, drafts hypotheses, and explains applicability; the
server checks structure, provenance, revision consistency, and metric arithmetic.
Qwen can supply paper evidence through the existing research server when served
and explicitly configured. The new workflow works with the source-only server.

The host first asks for missing setup details, inspects selected results, and
researches the question. It compares plausible options, explains assumptions,
then saves one testable change. Each recommendation states why it could apply,
what differs from the paper's conditions, and what observation could refute it.
Paper content and run text are data, never instructions to execute commands.

## Exact new MCP surface

| Tool and arguments | Behavior |
| --- | --- |
| `save_experiment_brief(brief, experiment_id=null, expected_revision=null)` | Create a draft, or append a draft revision with an exact prior revision. Validate and snapshot referenced evidence. Return ID, revision, digest, Markdown, and missing fields. |
| `freeze_experiment_plan(experiment_id, expected_revision)` | Check required planning fields and freeze the exact revision and evidence hashes. Return an immutable plan digest. Freezing records the plan; it grants no permission to spend or execute. |
| `get_experiment(experiment_id)` | Return the brief, revision history, frozen plan, recorded outcomes, limitations, and Markdown. |
| `list_experiments(limit=20)` | Return up to 100 recent summaries and IDs, with a truncation indicator. |
| `record_experiment_result(experiment_id, plan_sha256, result)` | Append a result bound to a frozen plan, preserve failures/unknowns, and return per-criterion comparisons. Identical repeated run IDs are idempotent; conflicting content is rejected. |

Only get/list are annotated read-only. All five are local/non-open-world.
Saving or freezing a stale revision fails without changing existing data.
A frozen brief cannot be edited; a changed plan requires a new experiment linked
by `supersedes_experiment_id`. Results never change the original criteria.

## Records and validation

Use strict Pydantic models (already a dependency), reject extra fields, and use
explicit schema versions: `ml-experiment-brief-v1`, `ml-experiment-plan-v1`, and
`ml-experiment-result-v1`. Unknown values are null plus an explanation, not zero.

**Brief:** question; goal (`model_comparison`, `fine_tuning`, or `latency`);
current setup (task, model/checkpoint/revision, software, hardware, workload);
constraints; observed failures with references; considered alternatives;
one proposed change; hypothesis; expected signal; applicability and differences;
baseline and candidate definitions; evaluation sample IDs/hash and split purpose;
controlled conditions; metric definitions; decision criteria; resource estimates,
ceilings and stop conditions; procedure; evidence references; limitations; author
kind (`user_supplied` or `agent_authored`). Recommendations are labeled inferences.

Drafts may have missing fields. Freeze requires an actionable hypothesis,
baseline/candidate, one declared change, defined evaluation population, metrics
and criteria, repeatable procedure, resource ceilings/stop rule, and an explicit
list of unknowns. Unknown hardware/revisions block affected conclusions, rather
than preventing all exploratory planning. Zero inference/compute is a valid
ceiling for saved-result analysis. Monetary estimates are not enforced spending
caps. Record whether candidate outcomes were already seen (`yes/no/unknown`),
and label historical reconstruction explicitly; a local freeze timestamp cannot
prove independent preregistration or absence of prior exposure.

**References:** distinguish `paper_span`, `local_artifact`, and
`external_reference`. Paper span input names an already cached versioned arXiv ID
and offsets; the server derives the document-byte SHA-256 and rebuilds the quote
(maximum 1,200 chars). Reject any supplied quote/hash mismatch. Store source
identity, extraction coverage, digest, and quote bytes with the brief.
Local artifacts name a JSON path under an operator-selected export directory,
raw-byte SHA-256, and JSON Pointer to the relevant value or bounded trace excerpt.
Recheck the hash and pointer, snapshot the selected content, and preserve metric
definitions, population, and reviewer kind. This binds content, not its truth.
External URLs (e.g. model cards or official implementation docs inspected by the
host) remain explicitly unverified references; this server never fetches them.
Separate source claims, observations, interpretations, and proposed tests.
Semantic support defaults to `not_reviewed`; model judgments never fill a human
review field. Human review supplied later is a declared process attestation.

**Result:** run ID; frozen plan hash; baseline/candidate identities; actual
conditions and deviations; workload/sample IDs and coverage; metrics and units;
raw values or counts/denominators where available; source artifact references;
failed/missing cases; timing/cost coverage; reviewer kind; interpretation and
limitations. Mark prospective/retrospective/unknown timing. Recorded measurements
can be caller-reported; only artifact-bound values get artifact verification.
Importing history never upgrades it to a prospective controlled experiment.

Metric definitions include ID, unit, aggregation, population, and direction.
V1 criteria support absolute `<=`/`>=`, candidate-minus-baseline `<=`/`>=`, and
relative improvement `>=`; the direction determines the relative formula.
Require finite numbers, matching units/populations/aggregation, relevant matched
conditions, and the declared evaluation coverage. A zero baseline makes relative
improvement unknown. Distinguish percentage points from percentage improvement.
Scope conclusions to supplied samples; compute no significance or causal proof.
Return each gate as `met`, `not_met`, or `unknown`. Overall: `criteria_not_met`
if any required gate fails; otherwise `inconclusive` if any is unknown; otherwise
`criteria_met`. This is arithmetic against a plan, not automatic model adoption.
Unsupported qualitative gates and missing support reviews remain unknown.

## Storage and boundaries

Configure `--store-root` (default `~/.envoy/ml-experiments`), `--artifact-root`
(optional, explicitly selected sanitized JSON exports), and `--library-root`
(the existing paper cache). Avoid a whole-home/repository artifact root.
MCP arguments accept generated record IDs and root-relative artifact paths only;
refuse traversal, symlink escapes, non-regular files, and hash mismatches.
Default bounds: 256 KiB tool payload, 2 MiB per JSON artifact, 20 references per
brief/result, 20 criteria, and 4,000 chars per excerpt. Bound artifact bytes before
parsing and tool payloads after SDK decoding, before processing or persistence;
enforce strict JSON depth/finite-number validation on disk too.
Large historical files need a selected bounded export, with parent path/hash and
selection recipe retained; the export is not the complete original trajectory.
An unchecked parent link remains a declared reference, not verified lineage.

Keep immutable revision/result objects, a locked atomic index, and content hashes;
use expected-revision checks and atomic writes so parallel calls cannot silently
overwrite plans. Store JSON as the authority and generate Markdown from it.
No client-supplied output path, arbitrary repository scan, shell execution, or
training/checkpoint modification. Corrupt records produce a bounded error;
they must not disappear silently from listings or destroy other experiments.

## First demonstration and acceptance criteria

Hypothesis: linking saved failures and literature to an explicit experiment brief
helps this engineer choose an actionable next test. Expected signal: a complete
brief with traceable observations, a matched baseline, and a decision rule the
user can follow. Decision: user reviews it as actionable or names what is missing;
an assistant-produced brief alone does not establish usefulness or time savings.

Task 5 uses September 30 Qwen artifacts as already-seen development material:
both models have 3/20 model-assisted answerable passes; official F1 has different
semantics; historical base latency is incomplete. Select traces covering missed
evidence, wrong citations, or stopping, without claiming a representative sample
or a causal explanation. Propose one future diagnostic, with no new benchmark or
reader inference. Normal host-assistant usage is separate from this zero-call server.
Use a separate retrospective record to exercise result import; do not attach
historical outcomes as results of the new proposed intervention. Missing timing
must yield an inconclusive latency/cost comparison. Synthetic passing/failing
result fixtures remain explicitly synthetic.

Acceptance: real stdio calls can save, retrieve, freeze, and review records;
restart preserves bindings; stale/tampered/oversized inputs fail safely; changing
criteria after freezing fails; unknowns cannot produce a success claim; paper
support is separate from quote matching; existing MCP profiles remain compatible.
One native-client research-to-brief demonstration and user review are required
before calling v1 useful. No Qwen improvement or ordinary-web superiority claim
follows; those require separate, approved evaluations.

## Remaining approval gates

| Task | Proposed files and verification; await approval before starting |
| --- | --- |
| 2 — records | New `src/product/ml_experiment_{models,store}.py` and focused tests: schema/unknowns, atomic persistence, revision conflicts, immutable freeze, hash binding. Split helpers as needed to keep modules small. |
| 3 — planning MCP | New `scripts/ml_experiment_mcp.py`, evidence/Markdown helpers, setup instructions and focused stdio tests: draft/save/get/list/freeze, paper/export references, directory boundaries, unchanged research tool inventories. Outcome recording follows Task 4. |
| 4 — results | New result comparison helper, result MCP tool and tests: thresholds/direction, units/coverage/conditions, missing data, retrospective records, immutable run IDs. |
| 5 — use it here | Small provenance-bound development exports and example brief; actual native-client source research and local planning, user review, retrospective result round trip. No GPU or paid reader. |
| 6 — release | Independent review, fixes, full relevant suite, README/demo instructions, reproducible example and limitations. Paid live-reader and native-web comparison work stay separately scoped. |

Each implementation task gets focused verification and a small logical commit.
Task 1 changed this design document only. Task 2 implemented the record foundation.
Task 3 adds source receipts, readable briefs, and four planning MCP tools; result
recording is deferred to Task 4. See `docs/ML_EXPERIMENT_MCP.md` for the current surface.
