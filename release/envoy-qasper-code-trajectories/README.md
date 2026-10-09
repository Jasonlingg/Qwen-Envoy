---
license: cc-by-4.0
language:
- en
pretty_name: Envoy QASPER Code-Execution Trajectory Pilot
task_categories:
- question-answering
- text-generation
size_categories:
- n<1K
tags:
- agents
- tool-use
- code-execution
- scientific-papers
- citation-grounding
- failure-analysis
configs:
- config_name: reviewed
  default: true
  data_files:
  - split: train
    path: data/reviewed.jsonl
- config_name: diagnostic
  data_files:
  - split: train
    path: data/diagnostic.jsonl
---

# Envoy QASPER Code-Execution Trajectory Pilot

This is a small, fully disclosed pilot of executable research-agent trajectories.
Claude Sonnet 5 generated Python actions against a persistent document REPL. The
Envoy pipeline executed every action and retained the real observations. An AI
coding assistant then reviewed answer support, stopping behavior, and replay.

This release is useful for studying trajectory validation and citation failures.
It is **not** a production-ready SFT dataset.

## Subsets

| Subset | Rows | Intended use |
| --- | ---: | --- |
| `reviewed` (default) | 2 | Examples that passed the disclosed assistant review |
| `diagnostic` | 24 | Every candidate, including known failures and pending reviews |

Diagnostic verdicts: 2 pass, 18 fail,
and 4 pending. The same 12 QASPER training questions were
attempted twice while developing the teacher prompt. Rows are not statistically
independent, and this pilot is not an evaluation benchmark.

Only rows with `accepted_for_sft=true` belong in supervised training. Consumers
must not interpret mechanically valid spans as semantic support. Review was
assistant-led with QASPER annotations visible; it was not human, independent, or
blind. Some diagnostic rows intentionally preserve execution errors, overly long
searches, answer leakage, incomplete evidence, or unsupported claims.

## Task and format

Each episode starts with a known-paper QASPER question. The agent writes Python
using `search_within()`, `read()`, and `passage()`, observes actual tool output,
and finishes with an answer, paper citation, and exact character offsets. The
`messages` field is the student-facing conversation. `trajectory` preserves both
the provider's raw action and the cleaned action that actually executed.

```python
from datasets import load_dataset

reviewed = load_dataset(
    "jasonlingg/envoy-qasper-code-trajectories", "reviewed", split="train"
)
diagnostic = load_dataset(
    "jasonlingg/envoy-qasper-code-trajectories", "diagnostic", split="train"
)
```

## Generation and review

- Teacher: `claude-sonnet-5`; the returned provider identity was checked on every
  response. There was no Haiku fallback.
- Source questions: QASPER's official training split; one known paper per question.
- Student protocol: multi-turn executable Python followed by exact-span submission.
- Candidate generation: 24 episodes over 12 unique questions, across a
  reference-guided attempt and an unhinted attempt.
- Review: automated assistant review with annotations visible. The revised 12
  episodes also replayed with identical executed observations and completion flags.
- No Qwen checkpoint was trained on this release before publication.

## Source data, license, and changes

This is an adaptation of [QASPER](https://huggingface.co/datasets/allenai/qasper), released by the Allen Institute
for AI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). QASPER
contains full text extracted from [S2ORC](https://github.com/allenai/s2orc), which
is made available under ODC-By 1.0. This repository contains only the questions
and short paper passages surfaced during agent execution; it does not redistribute
the complete QASPER corpus, full papers, or PDFs.

Changes made here include selecting QASPER training questions, converting papers
to stable character-offset documents, generating and executing Python tool-use
trajectories, adding exact-span submissions, and attaching replay and review
metadata. Neither Ai2, the QASPER authors, Semantic Scholar, nor the paper authors
endorse this derivative dataset.

The derived dataset is released under CC BY 4.0. Retain this attribution and cite
QASPER when redistributing it.

```bibtex
@inproceedings{dasigi-etal-2021-dataset,
  title = {A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers},
  author = {Dasigi, Pradeep and Lo, Kyle and Beltagy, Iz and Cohan, Arman and Smith, Noah A. and Gardner, Matt},
  booktitle = {Proceedings of NAACL-HLT 2021},
  year = {2021}
}
```

## Limitations

The release is extremely small, restricted to within-paper scientific QA, and
contains repeated questions across prompt variants. Three revised trajectories
remain pending rather than accepted. The proxy score uses lexical answer overlap
and span overlap; it is not a semantic judge. Source excerpts inherit extraction
artifacts from QASPER/S2ORC. Do not use the diagnostic subset as unfiltered SFT
data or claim that it improves a model without a held-out comparison.

## Reproducibility

`provenance.json` records source artifact hashes, prompt hash, QASPER revision,
counts, and provider-model totals. Generation scripts and the full experiment
record are in [Envoy](https://github.com/Jasonlingg/DocTracerRL).
