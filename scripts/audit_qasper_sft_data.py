"""Audit saved SFT conversations without changing them or running model code."""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.sft_quality import directly_accesses_paper, known_paper_id


def audit(paths: list[Path], benchmark: dict) -> dict:
    questions = {q["question"]: q for q in benchmark["questions"]}
    reports, papers = {}, {}
    for path in paths:
        counts, action_counts, labels, rows = Counter(), Counter(), Counter(), []
        papers[str(path)] = set()
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            text = messages[1]["content"].removeprefix("Question: ")
            question = questions.get(text)
            doc = known_paper_id(text)
            papers[str(path)].add(doc)
            actions = [m["content"] for m in messages if m["role"] == "assistant"]
            flags = {
                "immediate_submit": len(actions) == 1 and actions[0].startswith("SUBMIT:"),
                "ten_or_more_actions": len(actions) >= 10,
                "over_ten_actions": len(actions) > 10,
                "no_direct_first_action": not directly_accesses_paper(actions[0], doc),
                "unmatched_question": question is None,
            }
            counts.update(k for k, value in flags.items() if value)
            labels[question["expected_answerability"] if question else "unknown"] += 1
            action_counts[len(actions)] += 1
            rows.append({"question_id": question["id"] if question else None, "doc_id": doc,
                         "actions": len(actions), "flags": [k for k, v in flags.items() if v]})
        reports[str(path)] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "conversations": len(rows),
            "assistant_actions": sum(r["actions"] for r in rows), "flags": dict(counts),
            "action_count_distribution": dict(sorted(action_counts.items())),
            "answerability": dict(labels), "doc_ids": sorted(papers[str(path)]), "rows": rows,
        }
    return {"files": reports, "paper_overlaps": [
        {"left": str(left), "right": str(right),
         "doc_ids": sorted(papers[str(left)] & papers[str(right)])}
        for i, left in enumerate(paths) for right in paths[i + 1:]
    ], "limitation": "Structural audit; does not establish semantic support or causation."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data, json.loads(args.benchmark.read_text()))
    report["benchmark_sha256"] = hashlib.sha256(args.benchmark.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for name, result in report["files"].items():
        print(name, result["conversations"], result["flags"], result["answerability"])
    print("Shared papers:", [len(pair["doc_ids"]) for pair in report["paper_overlaps"]])


if __name__ == "__main__":
    main()
