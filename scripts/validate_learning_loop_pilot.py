"""Source-check a learning-loop development pilot against its frozen paper snapshot."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.research_review import load_corpus  # noqa: E402
from src.research.benchmark import load_benchmark  # noqa: E402


def validate(benchmark_path: Path, snapshot_path: Path) -> tuple[dict, list[str]]:
    manifest = json.loads((snapshot_path / "manifest.json").read_text())
    benchmark = load_benchmark(benchmark_path, manifest)
    documents = load_corpus(snapshot_path / "corpus")
    skills: Counter[str] = Counter()
    review_lines = [
        f"# Source check: {benchmark['benchmark_id']}",
        "",
        f"Snapshot: `{snapshot_path}`",
        f"Corpus SHA-256: `{benchmark['corpus_hash']}`",
        "",
        "These are agent-authored development prompts and reference answers. "
        "Source anchors are inspectable, but a human has not approved the gold labels.",
        "",
    ]
    for question in benchmark["questions"]:
        qid = question["id"]
        skill = question.get("learning_skill")
        if not isinstance(skill, str) or not skill:
            raise ValueError(f"{qid} needs a learning_skill")
        if not isinstance(question.get("answer"), str) or not question["answer"].strip():
            raise ValueError(f"{qid} needs a reference answer")
        if not question["grader_notes"]:
            raise ValueError(f"{qid} needs grader notes")
        anchors = question.get("source_anchors")
        if not isinstance(anchors, list):
            raise ValueError(f"{qid} needs source_anchors, even when empty")
        required = set(question["required_doc_ids"])
        anchor_docs = {anchor.get("doc_id") for anchor in anchors if isinstance(anchor, dict)}
        if not required.issubset(anchor_docs):
            raise ValueError(f"{qid} has no source anchor for {sorted(required - anchor_docs)}")
        # An unanswerable numeric question can still require a source that
        # explicitly records an unfinished plan or missing result. Its anchor
        # supports the abstention, not the nonexistent numeric answer.
        skills[skill] += 1
        review_lines.extend([
            f"## {qid} · {skill}",
            "",
            f"**Question:** {question['question']}",
            "",
            f"**Reference answer:** {question['answer']}",
            "",
            "**Pass conditions:**",
            *[f"- {note}" for note in question["grader_notes"]],
            "",
            "**Source anchors:**",
        ])
        if not anchors:
            review_lines.append("- None; judge the corpus-coverage limit and abstention.")
        for anchor in anchors:
            if not isinstance(anchor, dict):
                raise ValueError(f"{qid} has a malformed source anchor")
            doc_id, needle = anchor.get("doc_id"), anchor.get("needle")
            if doc_id not in documents or not isinstance(needle, str) or not needle.strip():
                raise ValueError(f"{qid} has an invalid source anchor")
            content = documents[doc_id]["text"]
            start = content.find(needle)
            if start < 0:
                raise ValueError(f"{qid} source anchor is absent from {doc_id}: {needle!r}")
            excerpt = content[max(0, start - 90) : min(len(content), start + len(needle) + 90)]
            review_lines.append(
                f"- `{doc_id}` at {start}:{start + len(needle)}: "
                f"{json.dumps(excerpt, ensure_ascii=False)}"
            )
        review_lines.extend(["", "**Human reference check:** pending", ""])
    return {
        "benchmark_id": benchmark["benchmark_id"],
        "corpus_hash": benchmark["corpus_hash"],
        "question_count": len(benchmark["questions"]),
        "learning_skills": dict(sorted(skills.items())),
        "human_reference_review": "pending",
    }, review_lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--review-md", type=Path)
    args = parser.parse_args()
    try:
        summary, review_lines = validate(args.benchmark, args.snapshot)
        if args.review_md:
            args.review_md.parent.mkdir(parents=True, exist_ok=True)
            with args.review_md.open("w") as output:
                output.write("\n".join(review_lines) + "\n")
        print(json.dumps(summary, indent=2))
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
