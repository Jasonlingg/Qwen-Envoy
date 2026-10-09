"""Assemble the audited QASPER scale set with reviewed abstention rehearsal.

The large deterministic set contains answerable questions only.  This utility
adds the already reviewed QASPER abstention conversations, then re-splits the
whole combined set by paper so validation is never contaminated by a training
paper.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

from src.eval.hashing import sha256

DOC_ID = re.compile(r"qasper_[0-9_]+")


def load_rows(path: Path, source: str) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            row["_source"] = source
            rows.append(row)
    return rows


def paper_id(row: dict) -> str:
    initial = row["messages"][1]["content"]
    match = DOC_ID.search(initial)
    if match is None:
        raise ValueError("QASPER conversation has no known paper id")
    return match.group(0)


def is_abstention(row: dict) -> bool:
    return any(
        message.get("role") == "assistant"
        and message.get("content", "").startswith("SUBMIT: Unanswerable")
        for message in row["messages"]
    )


def split_by_paper(rows: list[dict], *, seed: int, val_fraction: float) -> tuple[list[dict], list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[paper_id(row)].append(row)
    papers = sorted(groups)
    random.Random(seed).shuffle(papers)
    target = max(1, round(len(rows) * val_fraction))
    validation: list[dict] = []
    training: list[dict] = []
    for paper in papers:
        group = groups[paper]
        if len(validation) < target and abs(target - len(validation) - len(group)) <= abs(target - len(validation)):
            validation.extend(group)
        else:
            training.extend(group)
    if not validation or not training:
        raise ValueError("Need at least two paper groups")
    return training, validation


def export(rows: list[dict]) -> str:
    return "".join(json.dumps({key: value for key, value in row.items() if key != "_source"}) + "\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answerable-train", type=Path, required=True)
    parser.add_argument("--answerable-validation", type=Path, required=True)
    parser.add_argument("--abstention-train", type=Path, required=True)
    parser.add_argument("--abstention-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    args = parser.parse_args()

    answerable = load_rows(args.answerable_train, "audited_annotation")
    answerable += load_rows(args.answerable_validation, "audited_annotation")
    abstention_source = load_rows(args.abstention_train, "reviewed_abstention")
    abstention_source += load_rows(args.abstention_validation, "reviewed_abstention")
    abstentions = [row for row in abstention_source if is_abstention(row)]
    if not answerable or any(is_abstention(row) for row in answerable):
        raise ValueError("Answerable source contains an unexpected abstention")
    if not abstentions:
        raise ValueError("Abstention source contains no reviewed abstentions")

    training, validation = split_by_paper(
        answerable + abstentions, seed=args.seed, val_fraction=args.val_fraction
    )
    train_papers = {paper_id(row) for row in training}
    validation_papers = {paper_id(row) for row in validation}
    if train_papers & validation_papers:
        raise RuntimeError("Paper leakage across SFT splits")

    args.output.mkdir(parents=True, exist_ok=False)
    for name, rows in (("train", training), ("val", validation)):
        target = args.output / f"{name}.jsonl"
        target.write_text(export(rows))
    manifest = {
        "schema_version": "qasper-scale-sft-v1",
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "sources": [
            {"path": str(args.answerable_train), "sha256": sha256(args.answerable_train)},
            {"path": str(args.answerable_validation), "sha256": sha256(args.answerable_validation)},
            {"path": str(args.abstention_train), "sha256": sha256(args.abstention_train)},
            {"path": str(args.abstention_validation), "sha256": sha256(args.abstention_validation)},
        ],
        "counts": {
            "total": len(training) + len(validation),
            "answerable": len(answerable),
            "reviewed_abstentions": len(abstentions),
            "train": len(training),
            "validation": len(validation),
        },
        "papers": {"train": len(train_papers), "validation": len(validation_papers)},
        "splits": {
            "train": {"sha256": sha256(args.output / "train.jsonl")},
            "validation": {"sha256": sha256(args.output / "val.jsonl")},
        },
        "claim_boundary": (
            "The 955 answerable examples were programmatically constructed from QASPER train annotations "
            "and independently replay-audited; abstentions are from the previously reviewed annotation set."
        ),
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
