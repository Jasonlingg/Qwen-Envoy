# Prepare reviewed passage-reader pilot data locally

User said “ok go” after the V5 audit and targeted data/pilot proposal. Prepare
the data gates locally; no GPU activation, paid teacher/inference call, training,
reward change or runtime harness change. Preserve the frozen `cce82c2` harness,
prior artifacts/paid markers and protected dirty files.

Hypothesis: paired evidence-present/missing demonstrations can teach the current
passages-v2 contract without teaching universal refusal or universal answering.
Expected local signal: named paper-disjoint training/development cases, explicit
scoped labels, source-reviewed targets, real two-read Docker observations,
matching runtime prompts and assistant-only token masks. This preparation cannot
measure whether Qwen learned anything. Decision rule: no training export unless
every accepted training case has bound source review and successful Docker replay;
development targets remain separate from training. Do not substitute annotation
overlap or exact quotes for semantic support review.

Build a conservative identifier-only registry of prior training candidate/conversation
files and evaluation metadata. Include large annotation/scale sets and earlier
diagnostics, not only V5. Record inventory rules, file hashes, question IDs and
paper IDs. Skip corpora, weights, secrets and the reserved research-reader pack.
Do not treat the whole indexed corpus as papers actually trained/evaluated; inspect
selected identifiers and training conversations. Refuse missing required inventory
sources. Machine-extract identifiers only from historical artifacts, without
returning answer/source content to the reviewer. Prior-data file existence is not
proof of optimizer use; conservatively exclude prepared candidates as well.

Select 12 training papers from pinned cached official QASPER train and six fresh
development papers from official validation, excluding registry papers/questions
from both. Deterministic SHA256(42:<question ID>) order, one question per paper.
Filter only for data-construction feasibility: unambiguous answerable text-only
annotations, short extractive answer/evidence and enough source length for paired
two-window scopes. Preserve eligibility/rejection counts and selection before
source review or model evaluation. No new reserved test pack is opened.

Each pair uses two actual 1,600-character source windows, total 3,200. Present
scope covers its annotated paragraph. Missing scope excludes annotated evidence
and exact answer strings, which is a candidate filter, not a semantic absence
proof; review the entire selected text before approving a gap. Same question,
original source text, and a shared distractor window across both states. Keep
original QASPER answerability separate from scoped answerability. Review may reject
bad/ambiguous cases before any model test, retaining reasons; never replace cases
after seeing candidate model outcomes.

Save annotation-assisted drafts, then record explicit implementing-assistant
source review of requested fact coverage, attribution, numeric conditions and
gap specificity. No independent human grade is generated. Replay approved targets
with the real policy formatter, investigator and verifier; recorded Python runs
only in network-none Docker with read-only source mounts. Save actual requests,
observations, evidence/gap packets, per-case receipts and owned-container cleanup.
Export only train-role conversations, built directly from actual request messages
and approved assistant actions; development references are scoring-only. Validate
per-action prefixes/masks with the pinned cached tokenizer. Use local-only data
and make zero inference requests. Run focused and full relevant tests.
