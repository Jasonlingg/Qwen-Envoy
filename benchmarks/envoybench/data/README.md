# EnvoyBench data provenance

EnvoyBench converts [AllenAI QASPER](https://huggingface.co/datasets/allenai/qasper)
paper questions and annotations into bounded, executable research-agent tasks.
QASPER is by [Dasigi et al. (2021)](https://arxiv.org/abs/2105.03011) and is
licensed [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The source
revision is pinned in [`splits.json`](splits.json), along with SHA-256 hashes of
the cached Arrow files and canonical source rows. The repo releases frozen IDs
and provenance, while `build_data.py` regenerates the paper text and benchmark
JSON locally from that pinned source.

`dev` reuses the 40 validation questions from the September 25 experiment. They
have already influenced model decisions, so this is a **development set**, not a
fresh test. It contains 20 answerable and 20 unanswerable questions from 39
target papers, with 244 papers in the searchable corpus.

`test_candidate` selects 40 questions from 40 distinct papers in QASPER's
official test split. It excludes all 53 test papers associated with the 58
previously used test-question IDs found in local project artifacts before the
split was frozen. The candidate corpus contains 363 other QASPER test papers.
Selection ranks eligible question IDs by SHA-256 of the fixed seed and ID, then
takes 20 answerable and 20 unanswerable questions from distinct papers. That
balance deliberately stresses abstention; it does not estimate QASPER's natural
answerability prevalence. Original annotation IDs, reference answers, and
evidence offsets remain attached to each generated question.

The complete official train, validation, and test paper ID sets are disjoint in
the pinned source. The candidate target papers also have zero overlap with the
44 QASPER papers found in the local v5 SFT train/validation conversations. The
local-artifact scan is a best-effort exposure audit, not proof that a pretrained
model has never seen a paper. The candidate is **unreviewed and unscored**:
QASPER annotations and exact matching evidence offsets still require independent
reference validation before reporting semantic benchmark results.

To materialize both splits:

```bash
python -m benchmarks.envoybench.build_data --output benchmarks/envoybench/data
```

This uses the pinned QASPER Arrow cache when available, or downloads the pinned
dataset revision. `data/manifest.json` records the expected generated hashes;
`dev/` and `test_candidate/` are ignored by git so source paper text is not
accidentally committed.
