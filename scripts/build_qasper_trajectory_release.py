"""Build the public Hugging Face bundle for the September Sonnet pilot.

The default subset contains only reviewed passes. The diagnostic subset retains
all candidate attempts and their review labels for failure-analysis research.
No credentials, provider request IDs, local paths, full papers, or PDFs enter the
release. Source paper excerpts are limited to the observations already returned
inside each trajectory.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_qasper_sft_data import _to_conversation
from src.eval.hashing import sha256 as _sha256
from src.eval.sft_quality import known_paper_id

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "out/research/qasper-sonnet-expansion-20260919"
OUTPUT = ROOT / "release/envoy-qasper-code-trajectories"
QASPER_URL = "https://huggingface.co/datasets/allenai/qasper"
QASPER_REVISION = "13b496d2a5359329b110e3419628de3cf791843b"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _review_map(path: Path) -> dict[str, dict]:
    return {row["question_id"]: row for row in json.loads(path.read_text())["reviews"]}


def _sanitize(text: str) -> str:
    """Retain real failures while removing machine-specific traceback paths."""
    return text.replace(str(ROOT), "<REPO_ROOT>")


def _public_row(row: dict, attempt: str, review: dict) -> dict:
    usage = row["teacher_usage"]
    provider_models = sorted({item["model"] for item in usage})
    if provider_models != ["claude-sonnet-5"]:
        raise ValueError(f"Unexpected provider model identity: {provider_models}")
    conversation = [
        {**message, "content": _sanitize(message["content"])}
        for message in _to_conversation(row, 100_000)["messages"]
    ]
    trajectory = [{
        "step": step["step"],
        "raw_action": _sanitize(step.get("raw_action", step["action"])),
        "executed_action": _sanitize(step["action"]),
        "observation": _sanitize(step["observation"]),
        "done": step["done"],
    } for step in row["trajectory"]]
    return {
        "trajectory_id": f"{row['question_id']}:{attempt}",
        "attempt": attempt,
        "question_id": row["question_id"],
        "paper_id": known_paper_id(row["question"]),
        "question": row["question"],
        "expected_answerability": row["expected_answerability"],
        "source_split": "train",
        "source_dataset": "allenai/qasper",
        "source_dataset_revision": QASPER_REVISION,
        "teacher_model": "claude-sonnet-5",
        "teacher_reference_guidance": row["teacher_reference_guidance"],
        "student_protocol": row["student_protocol"],
        "system_prompt": conversation[0]["content"],
        "initial_observation": row["initial_observation"],
        "messages": conversation,
        "trajectory": trajectory,
        "review_verdict": review["verdict"],
        "review_notes": review["notes"],
        "reviewer_type": "assistant",
        "review_was_independent_or_blind": False,
        "replay_verified": bool(review.get("replay_verified", False)),
        "accepted_for_sft": review["verdict"] == "pass",
        "qasper_proxy_score": row["qasper_score"],
        "provider_models": provider_models,
        "generation_response_count": len(usage),
        "teacher_input_tokens": sum(item.get("input_tokens", 0) for item in usage),
        "teacher_output_tokens": sum(item.get("output_tokens", 0) for item in usage),
    }


def _dataset_card(counts: Counter, reviewed_count: int) -> str:
    return f'''---
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
| `reviewed` (default) | {reviewed_count} | Examples that passed the disclosed assistant review |
| `diagnostic` | {sum(counts.values())} | Every candidate, including known failures and pending reviews |

Diagnostic verdicts: {counts.get('pass', 0)} pass, {counts.get('fail', 0)} fail,
and {counts.get('pending', 0)} pending. The same 12 QASPER training questions were
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

This is an adaptation of [QASPER]({QASPER_URL}), released by the Allen Institute
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
@inproceedings{{dasigi-etal-2021-dataset,
  title = {{A Dataset of Information-Seeking Questions and Answers Anchored in Research Papers}},
  author = {{Dasigi, Pradeep and Lo, Kyle and Beltagy, Iz and Cohan, Arman and Smith, Noah A. and Gardner, Matt}},
  booktitle = {{Proceedings of NAACL-HLT 2021}},
  year = {{2021}}
}}
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
'''


def main() -> None:
    sources = [
        ("reference_guided", SOURCE / "trajectories.jsonl", SOURCE / "pilot-review.json"),
        ("unhinted", SOURCE / "unhinted-trajectories.jsonl", SOURCE / "unhinted-review.json"),
    ]
    public_rows = []
    inputs = []
    for attempt, trajectory_path, review_path in sources:
        reviews = _review_map(review_path)
        rows = _read_jsonl(trajectory_path)
        if {row["question_id"] for row in rows} != set(reviews):
            raise ValueError(f"Review does not cover exactly {attempt}")
        public_rows.extend(_public_row(row, attempt, reviews[row["question_id"]]) for row in rows)
        inputs.extend({"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
                      for path in (trajectory_path, review_path))

    counts = Counter(row["review_verdict"] for row in public_rows)
    reviewed = [row for row in public_rows if row["accepted_for_sft"]]
    if len(public_rows) != 24 or len(reviewed) != 2:
        raise ValueError(f"Unexpected release counts: {len(public_rows)=}, {len(reviewed)=}")
    if len({row["question_id"] for row in public_rows}) != 12:
        raise ValueError("Expected exactly 12 unique questions")

    data_dir = OUTPUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "diagnostic.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_rows)
    )
    (data_dir / "reviewed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reviewed)
    )
    (OUTPUT / "README.md").write_text(_dataset_card(counts, len(reviewed)))
    provenance = {
        "schema_version": "envoy-qasper-code-trajectories-v1",
        "release_date": "2026-09-19",
        "source_dataset": QASPER_URL,
        "source_dataset_license": "CC BY 4.0",
        "source_dataset_revision": QASPER_REVISION,
        "source_full_text_origin": "S2ORC (ODC-By 1.0)",
        "source_artifacts": inputs,
        "system_prompt_sha256": hashlib.sha256(
            public_rows[0]["system_prompt"].encode()
        ).hexdigest(),
        "candidate_rows": len(public_rows),
        "unique_questions": len({row["question_id"] for row in public_rows}),
        "reviewed_rows": len(reviewed),
        "review_counts": dict(counts),
        "provider_model_counts": dict(Counter(
            model for row in public_rows for model in row["provider_models"]
            for _ in range(row["generation_response_count"])
        )),
        "generated_by": "Claude Sonnet 5 through the Envoy pipeline",
        "review": "AI assistant with QASPER annotations visible; not human, independent, or blind",
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    serialized = "\n".join(path.read_text() for path in OUTPUT.rglob("*") if path.is_file())
    forbidden = [
        r"sk-ant-[A-Za-z0-9_-]+", r"hf_[A-Za-z0-9]+", r"ANTHROPIC_API_KEY",
        r"/Users/", r"/private/", r"request_id",
    ]
    matches = [pattern for pattern in forbidden if re.search(pattern, serialized)]
    if matches:
        raise ValueError(f"Potential private material in release: {matches}")
    print(json.dumps({"output": str(OUTPUT), **provenance}, indent=2))


if __name__ == "__main__":
    main()
