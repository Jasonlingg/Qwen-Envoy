# Offline weekly AI-paper digest writer

This is a focused, local publication step for the [weekly research radar](WEEKLY_RESEARCH_RADAR.md).
It turns an explicitly human-reviewed selection from one frozen public-paper snapshot into
Obsidian Markdown paper notes and one ISO-week digest. An `agent_authored_draft` selection may be
validated and previewed but cannot write to the vault. This step does not discover, rank,
summarize, or semantically judge papers. No model, account, network request, or scheduler is
involved.

First create a frozen snapshot of versioned public arXiv papers using
`python scripts/research.py snapshot --sources ... --output /path/to/snapshot`. Inspect its
`manifest.json` and `corpus/*.json` files. For a selected paper, copy its exact `doc_id`,
`arxiv_id`, and `source_url` from the manifest. Count `start` and `end` as Python character offsets
in that paper JSON's `text`; `end` is exclusive. The `quote` must equal `text[start:end]` exactly.

Prepare a small JSON file like this, replacing the illustrative IDs, hash, prose, and excerpt with
values from your own snapshot:

```json
{
  "schema_version": "envoy-weekly-selection-v1",
  "week": "2026-W39",
  "topic": "Evidence tools",
  "corpus_hash": "THE_SNAPSHOT_MANIFEST_CORPUS_HASH",
  "review_status": "human_reviewed",
  "papers": [
    {
      "doc_id": "arxiv_2609_09516v1",
      "arxiv_id": "2609.09516v1",
      "source_url": "https://arxiv.org/abs/2609.09516v1",
      "why_it_matters": "A concise human-reviewed connection to my current project.",
      "evidence": [
        {"start": 0, "end": 13, "quote": "EXACT EXCERPT"}
      ],
      "limitations": ["A concrete reason to read the paper before acting on this result."]
    }
  ]
}
```

The placeholder excerpt above is schematic and will fail validation until its offsets and text
match the frozen paper. `review_status` is a declaration supplied by the person preparing the
selection; the software verifies provenance and exact spans, not whether the human's interpretation
is correct. Use `"review_status": "agent_authored_draft"` while an agent prepares a source-backed
candidate list; preview will report `draft_preview_only`, and even `--confirm` refuses to publish
it. Change the status to `human_reviewed` only after the person reviews relevance, interpretations,
quotes, dates, and limitations.

Each selected paper defaults to `"role": "new_or_revised"`. The **pinned version's effective date**
(`updated` when present, otherwise `submitted`) must fall within the selected ISO week. A later
revision cannot be backdated to the paper's original submission week. An older paper can be included
only as `"role": "context"` with a short `"context_reason"`; it is visibly labeled as background in both
notes. A context paper's pinned version must predate the week. Every digest must contain at least
one paper whose pinned-version date falls inside the week. Paper notes and the digest show the
frozen submitted, updated, and pinned-version dates, so a reader can distinguish a new
publication from a revision and from older context.

Each paper may also name up to five existing notes in the selected vault:

```json
"related_notes": [
  {
    "path": "Learning/Research/Worker Decision.md",
    "reason": "This earlier decision is the baseline the paper might change."
  }
]
```

Paths must be vault-relative `.md` files;
missing notes, duplicates, traversal, symlinks, and paths outside the vault are rejected. The
paper note and weekly digest get real Obsidian links plus the stated reasons. The linked notes
themselves are not edited. Keep each path and reason at most 240 characters.

An optional selection-level `"next_measurement"` (at most 600 characters) appears under
**Next measurement to consider** in the digest. It is labeled a proposal, not a research finding;
for example: “Measure BM25 evidence recall, then compare base Qwen and v5 answers on the same
held-out questions and tool budget; record support, latency, and cost.” An optional
`"analysis_note"` (at most 500 characters) states the actual reading scope in both the weekly
note and each paper note. The digest links to exact excerpts in the paper notes without repeating
their quotations.

Preview every destination without changing the vault:

```bash
python scripts/weekly_digest.py \
  --snapshot /path/to/snapshot \
  --selection /path/to/reviewed-selection.json \
  --vault /path/to/ObsidianVault
```

To inspect the full rendered Markdown for every proposed paper note and digest, add
`--show-content`. It prints to stdout without writing vault files, including for
`agent_authored_draft` selections. It cannot be combined with `--confirm`:

```bash
python scripts/weekly_digest.py \
  --snapshot /path/to/snapshot \
  --selection /path/to/draft-selection.json \
  --vault /path/to/ObsidianVault \
  --show-content
```

After reviewing the paths, create the files explicitly:

```bash
python scripts/weekly_digest.py \
  --snapshot /path/to/snapshot \
  --selection /path/to/reviewed-selection.json \
  --vault /path/to/ObsidianVault \
  --confirm
```

The output is `AI Research/Papers/<paper-year>/<pinned-id> - <title>.md` for each selection and
`AI Research/Weekly/<ISO-week>.md` for the digest. The notes link to each other with ordinary
`[[Obsidian wikilinks]]`; Obsidian's native graph and backlinks can show the connections. A prior
week or paper note is never overwritten. A new week may add new papers, but selecting a paper that
already has a note will fail until a separate reviewed update workflow is built. The writer does
not silently replace history.

The batch is rejected before writing if the snapshot hash or paper identity differs, the same
paper is selected twice, a quote is not an exact span, the output folder escapes the chosen vault,
or an output file already exists. It records the snapshot hash and the source's extraction coverage.
The resulting digest is a bounded human selection, not a claim of comprehensive weekly discovery.

## Historical draft in this workspace

The [2025-W32 agent-authored selection](../data/research/weekly_digest_2025_w32_agent_draft.json)
replays a week in August 2025 using a six-paper snapshot frozen on September 12, 2026. Search-R1
v5 has a pinned revision date of August 5, 2025. ReAct and Adaptive-RAG are clearly marked older
context. This is a **retrospective example**, not evidence that the radar discovered these papers
at the time or a current weekly update. Its quoted spans are exact; the relevance and explanatory
claims still need human review. The local `out/` snapshot is ignored by Git, so a fresh checkout
must rebuild from the pinned IDs in `data/research/sources.json`; if the rebuilt corpus hash changes,
the selection needs new exact offsets and review before it can be used.

With the existing local snapshot, render its four proposed notes without writing to the sample
vault:

```bash
python scripts/weekly_digest.py \
  --snapshot out/research/starter-2026-09-12 \
  --selection data/research/weekly_digest_2025_w32_agent_draft.json \
  --vault data/product_memory/linked_demo_vault \
  --show-content
```

The sample selection connects each paper to an existing synthetic research note and proposes a
paired BM25/base-Qwen/v5 measurement. Its `agent_authored_draft` status prevents `--confirm` from
writing those notes. The links are retrospective comparisons to later notes, not claims that the
2025 papers caused the synthetic 2026 decisions.

## Current-week draft in this workspace

The [2026-W39 agent-authored selection](../data/research/weekly_digest_2026_w39_agent_draft.json)
uses three pinned papers from the September 21–27 candidate run. It is an **abstract-based
selection**: full-text HTML was frozen for provenance, but the interpretation did not review the
whole papers. The source IDs are in
[weekly_sources_2026_w39_draft.json](../data/research/weekly_sources_2026_w39_draft.json),
and the local snapshot hash is
`5a2c8ecc598fbc40b5bb0f9bae02b8ffc9f395f23810ce16dc5d2c46c588d536`.

```bash
python scripts/weekly_digest.py \
  --snapshot out/research/weekly-2026-W39-draft-snapshot \
  --selection data/research/weekly_digest_2026_w39_agent_draft.json \
  --vault data/product_memory/linked_demo_vault \
  --show-content
```

This prints a linked digest and three proposed paper notes without modifying the sample vault.
The selection's vault links are to **synthetic** project notes and are suggestions for review,
not observed paper-to-project outcomes. The frozen snapshot and candidate state live in ignored
`out/` paths; on a fresh checkout, rebuild the snapshot from the pinned source list. If extraction
changes its hash or quote offsets, recheck the excerpts and update the draft before previewing.
Neither this draft nor the candidate counts establish that the weekly shortlist is relevant to
the user; human review and a discovery/ranking benchmark are still required.
