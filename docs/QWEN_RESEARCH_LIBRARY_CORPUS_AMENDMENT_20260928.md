# Prospective corpus amendment: build the research library first

Registered September 28, 2026, after the user clarified that they do not have
an existing personal Obsidian collection. This amends the source domain of the
[September 26 transfer protocol](TARGETED_QWEN_VAULT_TRANSFER_TEST.md) before
creating its four development or twelve new held-out question files, collecting
task-specific training data, or running a new candidate. The
[base-initialization amendment](TARGETED_QWEN_VAULT_TRANSFER_AMENDMENT_20260928.md)
remains in force. Earlier records are preserved rather than relabeled.

## Product and model boundary

The product starts with a **growing AI-research library**, seeded from
version-pinned public papers. It creates Obsidian paper notes and a weekly
digest only from a reviewed selection. Qwen is the bounded, read-only worker:
it writes Python using `search()`, `read()`, and `extract()` to investigate the
frozen paper library and returns source IDs, exact passages, a proposed answer,
and uncertainty. The host verifies source spans; Nemotron or another larger
assistant synthesizes the user-facing answer and draft note. The approved
writer, not Qwen's tool episode, creates files in Obsidian. Once the library
contains reviewed notes, later questions may search those notes as well.

The first model-improvement claim is therefore **public AI-paper library
investigation**, not transfer to a pre-existing private vault. A successful
public-library test cannot be described as performance on personal life notes.
The usability gate still requires a useful weekly digest in Obsidian and MCP
access to accumulated evidence.

## Preserve the experiment; change the corpus entry gate

Keep the same four development questions (`D01`–`D04`) and twelve locked
held-out questions (`H01`–`H12`), the code-execution interface, the blind
supported-answer rubric, BM25 evidence reference, base-Qwen primary control,
v5 diagnostic arm, and the [registered candidate promotion rule](TARGETED_QWEN_VAULT_TRANSFER_AMENDMENT_20260928.md).
Do not add the consumed nine-note synthetic screen, six-paper pilot, or W39
digest selection to the new held-out score.

Use the existing 20-paper development snapshot
`out/research/ai-agents-development-v1-20260912` (corpus hash
`9a9c1d750daf5647898eb18681a5f75775ba76047713316dcc5583f6ed201598`)
for draft and review of `D01`–`D04`. Its earlier questions are development
material, not untouched test items. Create a **new**, version-pinned public
paper snapshot for `H01`–`H12`; record its source selection, versions, parser,
corpus hash, document IDs, and candidate-retrieval output before model runs.
The development and held-out snapshots have separate hashes. Every model arm
sees the same held-out snapshot and identical retrieved candidates for each
held-out question. Keep new candidate-training paper families disjoint from
both development and held-out families.

The first held-out source **candidate** is
`out/research/research-library-heldout-v1`, built from
[`research_library_heldout_sources_v1.json`](../data/research/research_library_heldout_sources_v1.json)
with corpus hash
`fc41e6fe7e3aca2a38dc1888cd53afee8823fd30947ea1c96ea5f75edb6e82f5`.
It contains twelve pinned papers: eleven have extracted HTML paragraphs and
`2609.24264v1` has only its abstract because the HTML fetch returned HTTP
406. The frozen text totals about 520,000 characters (median paper about
42,500), so the worker must select passages rather than load the whole library
into one answer prompt. Its [source-family audit](../out/research/research-library-heldout-v1/source-family-audit.json)
found no base-ID or normalized-title overlap with earlier paper snapshots.
This is a source-quality candidate, **not** a reviewed or scored benchmark.
If review finds the abstract-only coverage unsuitable, create and register a
new immutable candidate before locking questions; do not edit this snapshot.

The four [development question drafts](../data/research/research_library_transfer_dev_v1.json)
and twelve [held-out question drafts](../out/research/research-library-transfer-heldout-v1/questions_draft.json)
pass structural and exact-anchor checks against their respective snapshots.
Their reference answers and answerability judgments still need independent
human review. The held-out drafts have not been used for prompt selection,
training, or model scoring.
The separate [development retrieval diagnostic](RESEARCH_LIBRARY_DEV_RETRIEVAL_DIAGNOSTIC_20260928.md)
found all seven provisional required paper references within BM25's top eight;
this measures paper discovery only, not answer support.
Before any new Qwen run, the executable environment's default `search()` was
repaired and versioned as `passage-bm25-okapi-v1`; use the same tool preamble in every
model arm and retain its SHA-256 in each run manifest. A separate public-paper
product prompt now exists, so the development baseline should use that product
path or explicitly disclose a protocol difference from the older eval harness.
The [development runbook](RESEARCH_LIBRARY_DEV_QWEN_RUNBOOK.md) describes the
current product-path dry run and the human-reference gate for live runs.

Split by source family rather than file: one arXiv/DOI paper, all its
revisions, derivative Markdown notes, and digest mentions belong to the same
split. Check base IDs, normalized titles, and near-duplicate content against
existing snapshots and training sources. Do not show held-out paper text,
question wording, reference answers, or gold spans to trajectory generation,
prompt tuning, or model selection. The evaluator may search its locked
held-out corpus at test time; that is the intended task, not training leakage.

The registered question coverage transfers directly:

| IDs | Public-library skill |
| --- | --- |
| H01–H03 | Explain a finding from one paper with its scope and exact support. |
| H04–H07 | Compare or connect two papers, citing both sides and their limits. |
| H08–H09 | Reject a plausible claim that the retrieved passages do not support. |
| H10–H11 | Handle missing, stale, or wrong-version evidence without inventing a result. |
| H12 | Apply a finding to an Envoy decision while separating published evidence from a proposed local test. |

An independent human checks each reference answer, answerability judgment,
required paper IDs, and exact source anchors against the frozen text before
the locked question files or training data are used. Agent-authored references
without that check remain provisional development material. A quote that
exists at the claimed offset is not by itself support for an answer claim.

## Stop and claim boundaries

Run base Qwen and v5 on the reviewed development questions first. Train one
bounded base-initialized, action-masked LoRA candidate only if those traces
show a recurring failure that source-checked trajectories can teach and the
remaining GPU balance can cover training **and** the locked comparison. On
the twelve held-out questions, the candidate must beat base by at least three
fully supported passes, beat v5, and satisfy the registered regression,
submission, execution, latency, and cost limits. If fixed retrieval gives the
same essential evidence more cheaply, do not promote Qwen as the default
retriever. Report negative results directly.

This amendment changes no training code or MuSiQue reward. It supersedes only
the earlier requirement that a pre-existing personal vault be the first
transfer corpus and the assumption that development and held-out questions
share one corpus hash.
