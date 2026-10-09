# 2026-W39 research digest: human review bundle

**Status: agent-authored draft; not human-approved or published.** This bundle transcribes the three papers already selected in [`weekly_digest_2026_w39_agent_draft.json`](../data/research/weekly_digest_2026_w39_agent_draft.json). It neither adds candidates nor changes the selection. The proposed digest and paper notes have not been written to a live Obsidian vault.

The pinned week is **September 21–27, 2026**. The frozen public-paper extract is `out/research/weekly-2026-W39-draft-snapshot`, retrieved September 27, with corpus hash `5a2c8ecc598fbc40b5bb0f9bae02b8ffc9f395f23810ce16dc5d2c46c588d536`. All three source records have `html_paragraphs` coverage. The selection was made from abstracts and metadata; full HTML was frozen for provenance but was not reviewed end to end. Extraction can omit tables and figures or distort equations. Exact character ranges below were checked against the frozen extract. That check establishes text identity, **not** the paper's validity, the relevance of the selection, or support for Envoy-specific claims.

The draft's `related_notes` point to the **synthetic sample vault**. The configured real Obsidian vault currently contains only `Welcome.md`, so this unchanged selection cannot be previewed or published against that real vault: its related-note links do not exist there. The preview described below uses `data/product_memory/linked_demo_vault` only.

For an initially empty vault, a separate [seed draft](../data/research/weekly_digest_2026_w39_seed_agent_draft.json) keeps the same three pinned papers and eight source spans but omits those synthetic links and updates the measurement text to the paper-library task. Its offline preview against the real vault returned `draft_preview_only` and would create three paper notes plus one weekly digest under `AI Research/` **after** human review. The rendered [seed preview](../out/research/weekly-2026-W39-seed-preview.txt) is ignored by Git; neither preview wrote vault files.

## 1. The Fellowship of the Query: Learning Retrieval Actions

**Pinned source:** [arXiv 2609.28653v1](https://arxiv.org/abs/2609.28653v1) · `arxiv_2609_28653v1` · submitted September 23.

**Draft why-it-matters claim:** This is a methodological analogue for training a small worker on search and extraction actions. It suggests action-level trajectory supervision as a candidate Envoy experiment; it does not show a supported-answer gain on a personal vault.

**Exact excerpts in the frozen extract:**

> “trajectory fine-tuning can improve small language models (SLMs) as next-action controllers” — characters 203–293.
>
> “controller-only final-answer gains are not statistically clear” — characters 1307–1369.

**Limitations to preserve:** The study's models, accepted public-QA trajectories, and retrieval setting differ from Qwen3-8B executing code against Obsidian notes. Better next-action prediction or evidence recording cannot stand in for an independently reviewed answer-quality gain. The abstract explicitly leaves controller-only final-answer gains uncertain.

**Human checks:**

- Does the full paper's action protocol resemble `search()`, `read()`, and `extract()` closely enough to warrant this comparison? Note any differences in the available corpus or teacher filtering.
- Do the controller/generator swap and uncertainty intervals support the draft wording? Would a reader mistakenly infer that training the controller alone improves final answers?
- Is the suggested link to the existing synthetic worker-decision note useful, or does it imply a result Envoy has not measured?

**Reviewer decision:** Keep / revise / remove; support and wording notes: ______

## 2. Reinforcement Learning with Verifiable Rewards for Small Search Agents

**Pinned source:** [arXiv 2609.28765v1](https://arxiv.org/abs/2609.28765v1) · `arxiv_2609_28765v1` · submitted September 23.

**Draft why-it-matters claim:** A small-Qwen search-training study makes reward design a concrete variable if Envoy later revisits RL. Its MuSiQue result is neither a personal-vault baseline nor a reason to skip the current reviewed-baseline and SFT decision gates.

**Exact excerpts in the frozen extract:**

> “We train Qwen3.5-0.8B with Group Relative Policy Optimization (GRPO) and an interleaved Wikipedia-search tool on MuSiQue” — characters 536–656.
>
> “The reward shape also matters” — characters 977–1006.

**Limitations to preserve:** Qwen3.5-0.8B, Wikipedia search, MuSiQue, and public-QA exact match differ from Envoy's Qwen3-8B executable-code worker and human-reviewed supported answers. A reported reward-shape difference does not establish a good reward for personal memory or show that another paid RL run is warranted.

**Human checks:**

- Does the full paper compare reward shapes under sufficiently matched settings and seeds for the draft's narrow claim?
- Are the retrieved sources, verifier, and answer metric comparable in any useful way to Envoy? If not, should this paper remain background context only?
- Does the proposed connection to the earlier synthetic retrieval-correction note clearly read as a future hypothesis, rather than a reversal of the existing pilot?

**Reviewer decision:** Keep / revise / remove; support and wording notes: ______

## 3. Knowledge-as-Skill: A Structural Design for Autonomous Knowledge-Base Use by LLM Agents

**Pinned source:** [arXiv 2609.25991v1](https://arxiv.org/abs/2609.25991v1) · `arxiv_2609_25991v1` · submitted September 22.

**Draft why-it-matters claim:** Its navigable knowledge-base design suggests an optional way to expose Obsidian source scope and provenance to an agent. It is a design candidate for Envoy's MCP library, not evidence of a measured improvement in this project.

**Exact excerpts in the frozen extract:**

> “discovery layer centered on SKILL.md” — characters 725–761.
>
> “one index.md per directory” — characters 787–813.
>
> “YAML frontmatter for topic, type, provenance, and lifecycle” — characters 863–922.
>
> “directional cross-work evidence” — characters 1577–1608.

**Limitations to preserve:** The abstract itself describes the benchmark comparison as directional rather than controlled. It does not test Envoy's vault, its MCP tools, or whether extra indexes improve answer support enough to justify maintenance cost.

**Human checks:**

- Does the full paper show an actual navigation benefit, or mainly propose a structure? Keep any benchmark claim within its stated cross-work limits.
- Would this layout help a user who already works in Obsidian, or duplicate existing folders, links, and frontmatter? What one workflow would improve?
- Does `SKILL.md` here denote a knowledge-base discovery layer rather than a requirement to convert the entire vault?

**Reviewer decision:** Keep / revise / remove; support and wording notes: ______

## Release checks for the reviewer

- Are these **the right three papers** for this user's weekly research priorities? The frozen selection is bounded; it is not an exhaustive or independently benchmarked discovery/ranking result.
- Open each pinned source and inspect methods, result tables, and caveats before approving relevance, claims, or links to synthetic project notes. Exact quotes alone do not establish semantic support.
- Decide whether each proposed `why_it_matters`, limitation, and vault connection is accurate, useful, and phrased as a candidate rather than a demonstrated Envoy outcome.
- Replace or omit the synthetic `related_notes` when a genuine personal-vault snapshot exists; do not treat a passing sample-vault preview as real-vault readiness.
- Record decisions in a reviewed selection artifact before using the publisher. Keep `agent_authored_draft` until that review has actually happened; the current draft preview cannot be published through `--confirm`.

The offline publisher preview is documented in [`WEEKLY_DIGEST_DEMO.md`](WEEKLY_DIGEST_DEMO.md). Its preview validates the snapshot hash, pinned paper identities, dates, exact quote offsets, and existing sample-vault links. It does not evaluate research quality or write the vault.
