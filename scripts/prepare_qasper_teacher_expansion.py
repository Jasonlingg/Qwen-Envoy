"""Freeze 200 new training-paper questions and a 12-question generation check.

Uses the previously frozen annotation-rich training pool; no API calls/downloads.
Excludes existing reviewed demonstrations, all historical SFT validation papers,
and the two fitted diagnostic papers. Existing pool exclusions remain in force.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.sft_quality import known_paper_id


def main():
    source = Path("out/research/qasper-rl-v1/train-v2/benchmark.json")
    output = Path("out/research/qasper-sonnet-expansion-20260919")
    benchmark = json.loads(source.read_text())
    assert benchmark["source_split"] == "train"
    excluded = set(benchmark["excluded_doc_ids"])
    excluded.update({"qasper_1804_04225", "qasper_1909_11189"})
    inputs = [source, Path("data/sft/qasper-v5/train.jsonl")]
    inputs += sorted(Path("data/sft").glob("*/val.jsonl"))
    for path in inputs[1:]:
        for line in path.read_text().splitlines():
            row = json.loads(line)
            question = next(m["content"] for m in row["messages"] if m["role"] == "user")
            excluded.add(known_paper_id(question))
    eligible = [q for q in benchmark["questions"] if known_paper_id(q["question"]) not in excluded]
    eligible.sort(key=lambda q: q["id"])
    rng = random.Random(20260919)
    rng.shuffle(eligible)
    targets = {"boolean": 20, "abstractive": 35, "unanswerable": 40, "extractive": 105}
    selected, used = {}, set()
    for kind, count in targets.items():
        rows = []
        for q in eligible:
            doc = known_paper_id(q["question"])
            if q["answer_annotations"][0]["answer_type"] != kind or doc in used:
                continue
            rows.append(q)
            used.add(doc)
            if len(rows) == count:
                break
        if len(rows) != count:
            raise ValueError(f"Not enough unique eligible papers for {kind}: {len(rows)}/{count}")
        selected[kind] = rows
    pilot = (selected["extractive"][:6] + selected["abstractive"][:2] +
             selected["boolean"][:2] + selected["unanswerable"][:2])
    rng.shuffle(pilot)
    pilot_ids = {q["id"] for q in pilot}
    rest = [q for rows in selected.values() for q in rows if q["id"] not in pilot_ids]
    rng.shuffle(rest)
    questions = pilot + rest
    assert len(questions) == len(used) == 200
    assert all(q["source_split"] == "train" for q in questions)
    assert not used & excluded
    identity = {"selection_seed": 20260919, "selection": "one question per new paper",
                "pilot_ids": [q["id"] for q in pilot], "type_counts": targets,
                "extra_excluded_doc_ids": sorted(excluded),
                "inputs": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                           for p in inputs]}
    frozen = {**benchmark, **identity, "questions": questions,
              "question_ids": [q["id"] for q in questions],
              "counts": dict(Counter(q["expected_answerability"] for q in questions))}
    output.mkdir(parents=True, exist_ok=False)
    (output / "benchmark.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (output / "selection.json").write_text(json.dumps(identity, indent=2) + "\n")
    print(json.dumps({"output": str(output), "count": len(questions), "papers": len(used),
                      "pilot": len(pilot), "types": targets, "counts": frozen["counts"]}, indent=2))


if __name__ == "__main__":
    main()
