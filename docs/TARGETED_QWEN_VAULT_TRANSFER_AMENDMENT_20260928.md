# Prospective amendment to the Qwen vault transfer test

> The subsequent [public-paper corpus amendment](QWEN_RESEARCH_LIBRARY_CORPUS_AMENDMENT_20260928.md)
> supersedes this document's pre-existing personal-vault entry condition while
> retaining its base-Qwen initialization and promotion comparator.

Registered September 28, 2026, before creating the four development or twelve
held-out question files, collecting vault-task training trajectories, or running
a candidate model. This changes only the starting checkpoint and the paired
promotion comparator in the [September 26 protocol](TARGETED_QWEN_VAULT_TRANSFER_TEST.md).
The original protocol remains available as a historical record.

## Reason for the amendment

The [matched nine-note synthetic development screen](PERSONAL_MEMORY_READINESS_EVAL.md)
found that pinned base Qwen3-8B submitted 9/10 answers and received 5/10
provisional fully supported verdicts; QASPER v5 submitted 3/10, received 0/10,
and had execution errors on all ten episodes. These verdicts were independent
AI-assisted blind reviews, not human-reviewed real-vault results. V5 frequently
guessed document IDs rather than discovering them through `search()`. Continuing
that adapter is therefore a poor default for this new task. The local screen
does not prove base will transfer to a substantive personal vault.

## Changed treatment and controls

- Initialize the single proposed vault-task LoRA SFT candidate from
  `Qwen/Qwen3-8B` at revision
  `b968826d9c46dd6066d109eabc6255188de91218`, **without** the QASPER v5
  adapter. Training remains action-masked SFT on replayed, source-checked,
  multi-step Python tool trajectories from sources disjoint from the held-out
  vault notes. This is a proposed treatment, not authorization to begin a GPU
  run before the baseline gate.
- Run **base Qwen3-8B** as the primary control on the four development questions
  and the twelve locked held-out questions. Keep pinned **QASPER v5** as a
  diagnostic third arm under exactly the same question, corpus, prompt,
  decoding, tool, step, evidence-verifier, recovery, seed, and hardware settings.
  A fixed lexical/BM25 evidence packet remains the cheap retrieval reference;
  it is not scored as a generated answer.
- Keep the September 26 question slots, source-disjointness requirements,
  reference review, transcript preservation, blind `pass`/`partial`/`fail`
  review, and reproducibility fields unchanged. The old synthetic questions
  and model outputs remain development history and are excluded from the new
  held-out score and training set.

## Revised decision rule, fixed before collecting data

On the twelve locked questions, promote the trained candidate only if it has
**at least three more fully supported passes than base Qwen**, more full passes
than v5, no more than one paired base pass-to-fail regression, no lower
submission rate, no more execution-error episodes, and no more than twice
base's mean latency and estimated inference cost on the same hardware class.
Report paired question outcomes and exact-span/citation diagnostics. A valid
quote proves provenance but is not a semantic support verdict. This 12-case
threshold is a visible pilot effect requirement, not statistical proof; a
positive result still needs independent confirmation.

Do not train if a substantive frozen vault and human-approved references are
missing, if development outputs do not reveal a recurring teachable failure,
if fixed retrieval already provides the same essential evidence more cheaply,
or if the remaining GPU balance cannot cover both one bounded training run
**and** the locked comparison. The last documented personal `AI research` vault
had only `Welcome.md`, so the real-vault entry gate is currently unverified.
If that vault is populated, import its root with `--collection .`; the
September 26 example's `--collection 'AI Research'` is illustrative and that
subfolder does not presently exist in the configured vault.

This amendment does not change the active MuSiQue reward, the product's
read-only worker boundary, or the requirement to deliver a useful weekly
Obsidian digest and MCP query path.
