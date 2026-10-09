"""Prepare short code-execution demonstrations; export only reviewed candidates.

Without --reviews, writes candidates and a review template, NOT training files.
Re-run into a new directory with a completed review to export paper-disjoint SFT
splits. Failed/time-limited trajectories are never repaired by appending gold text.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.sft_quality import (
    known_paper_id,
    split_by_paper,
    trajectory_hash,
    trajectory_issues,
)
from src.policies.code_execution import SYSTEM_PROMPT, QASPER_SYSTEM_PROMPT


def _to_conversation(row: dict, max_chars: int) -> dict | None:
    spans = row.get("student_protocol") == "qasper-span-v1"
    prompt = QASPER_SYSTEM_PROMPT if spans else SYSTEM_PROMPT
    initial = f"Question: {row['question']}" + ("\n" if spans else "")
    if spans and row.get("initial_observation") != initial:
        raise ValueError("Span-protocol initial observation differs from inference")
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": initial},
    ]
    for step in row["trajectory"]:
        action = step["action"].strip()
        target = step.get("raw_action", action) if spans else action
        messages.append({"role": "assistant", "content": target})
        if not action.upper().startswith("SUBMIT:"):
            observation = step["observation"] if spans else step["observation"].strip()
            messages.append({"role": "user", "content": observation})
    if sum(len(m["content"]) for m in messages) > max_chars:
        return None
    return {"messages": messages}


def select_candidates(rows: list[dict], benchmark: dict, excluded: list[dict],
                      max_actions: int, max_chars: int) -> tuple[list, list]:
    if benchmark.get("source_split") != "train":
        raise ValueError("Only QASPER's official train split may supply demonstrations")
    questions = {q["id"]: q for q in benchmark["questions"]}
    excluded_ids = {q["id"] for b in excluded for q in b["questions"]}
    excluded_docs = {doc for b in excluded for doc in b.get("reserved_doc_ids", [])}
    excluded_docs.update(known_paper_id(q["question"]) for b in excluded for q in b["questions"])
    candidates, rejected, seen = [], [], set()
    for original in rows:
        row = dict(original)
        qid = row["question_id"]
        question = questions.get(qid)
        reasons = []
        if question is None or row.get("question") != question["question"]:
            reasons.append("unknown_or_mismatched_question")
        else:
            # The benchmark, not a trajectory's self-reported label, is authoritative.
            row["expected_answerability"] = question["expected_answerability"]
            if qid in excluded_ids or known_paper_id(question["question"]) in excluded_docs:
                reasons.append("reserved_question_or_paper")
            reasons.extend(trajectory_issues(row, max_actions=max_actions))
            if row.get("student_protocol") == "qasper-span-v1":
                if row.get("status") == "error" or not row.get("qasper_score", {}).get("valid"):
                    reasons.append("invalid_span_episode")
                if row.get("qasper_score", {}).get("reason") in {"false_refusal", "unsupported_answer", "missing_evidence"}:
                    reasons.append("answerability_or_evidence_failure")
            if not reasons and _to_conversation(row, max_chars) is None:
                reasons.append("too_long")
        if qid in seen:
            reasons.append("duplicate_question")
        if reasons:
            rejected.append({"question_id": qid, "reasons": reasons})
        else:
            seen.add(qid)
            candidates.append(row)
    return candidates, rejected


def reviewed_candidates(candidates: list[dict], review: dict) -> list[dict]:
    if not review.get("reviewer") or review.get("reviewer_type") not in {"human", "assistant"}:
        raise ValueError("Review must name its reviewer and disclose human or assistant review")
    entries = review.get("reviews", [])
    by_id = {item["question_id"]: item for item in entries}
    if len(by_id) != len(entries):
        raise ValueError("Duplicate review question IDs")
    accepted = []
    for row in candidates:
        item = by_id.get(row["question_id"], {})
        if item.get("verdict") != "pass":
            continue
        if item.get("trajectory_sha256") != trajectory_hash(row):
            raise ValueError(f"Stale review: {row['question_id']}")
        if not (item.get("evidence_checked") is True and item.get("stopping_checked") is True
                and item.get("replay_verified") is True and item.get("notes", "").strip()):
            raise ValueError(f"Incomplete passing review: {row['question_id']}")
        accepted.append(row)
    return accepted


def _identity(path: Path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main(
    trajectories: list[Path] = typer.Option(..., "--trajectories"),
    out_dir: Path = typer.Option(..., "--out-dir"),
    benchmark: Path = typer.Option(..., "--benchmark"),
    exclude_benchmark: list[Path] = typer.Option(..., "--exclude-benchmark",
        help="Repeat for development and consumed evaluation benchmarks; reserve every paper"),
    reviews: Path | None = typer.Option(None, "--reviews"),
    max_actions: int = typer.Option(6, min=2, max=10),
    val_fraction: float = typer.Option(0.2, min=0.01, max=0.99),
    max_chars: int = typer.Option(32000, min=1),
    seed: int = typer.Option(42),
    recover_with_gold: bool = typer.Option(False, "--recover-with-gold", hidden=True),
) -> None:
    if recover_with_gold:
        raise typer.BadParameter("Gold recovery is disabled: regenerate and review a real episode")
    rows = []
    for path in trajectories:
        text = path.read_text()
        rows.extend([json.loads(line) for line in text.splitlines() if line.strip()]
                    if path.suffix == ".jsonl" else json.loads(text))
    source = json.loads(benchmark.read_text())
    candidates, rejected = select_candidates(
        rows, source, [json.loads(p.read_text()) for p in exclude_benchmark], max_actions, max_chars
    )
    review = json.loads(reviews.read_text()) if reviews else None
    accepted = reviewed_candidates(candidates, review) if review else []
    protocols = {r.get("student_protocol", "document-v1") for r in candidates}
    if len(protocols) > 1:
        raise ValueError("Do not mix document-only and exact-span student protocols")
    splits = split_by_paper(accepted, val_fraction, seed) if review else None
    out_dir.mkdir(parents=True, exist_ok=False)
    questions = {q["id"]: q for q in source["questions"]}
    template = {
        "reviewer": "", "reviewer_type": None,
        "instructions": (
            "Replay each episode against its frozen corpus. Read the actual observations and "
            "source paper. Check every answer part, the scope of an abstention, and the earliest "
            "supported stopping point. A real quote, keyword hit, or gold label is not enough. "
            "Use pass/partial/fail; only fully checked passes are eligible for SFT."
        ),
        "reviews": [{
            "question_id": r["question_id"], "trajectory_sha256": trajectory_hash(r),
            "question": r["question"], "reference_answer": questions[r["question_id"]]["answer"],
            "grader_notes": questions[r["question_id"]].get("grader_notes", []),
            "verdict": None, "evidence_checked": False, "stopping_checked": False,
            "replay_verified": False, "notes": "",
        } for r in candidates],
    }
    (out_dir / "candidates.jsonl").write_text("".join(json.dumps(r) + "\n" for r in candidates))
    (out_dir / "review-template.json").write_text(json.dumps(template, indent=2) + "\n")
    report = {
        "status": "reviewed_export" if review else "candidates_need_replay_and_semantic_review",
        "sources": [_identity(p) for p in trajectories], "benchmark": _identity(benchmark),
        "exclusions": [_identity(p) for p in exclude_benchmark],
        "review": _identity(reviews) if reviews else None,
        "corpus_hash": source["corpus_hash"], "seed": seed, "max_actions": max_actions,
        "val_fraction": val_fraction, "max_chars": max_chars,
        "student_protocol": next(iter(protocols), None),
        "system_prompt_sha256": hashlib.sha256(
            (QASPER_SYSTEM_PROMPT if protocols == {"qasper-span-v1"} else SYSTEM_PROMPT).encode()
        ).hexdigest(),
        "candidate_count": len(candidates), "reviewed_pass_count": len(accepted),
        "candidate_answerability": dict(Counter(r["expected_answerability"] for r in candidates)),
        "rejected": rejected,
        "rejection_counts": dict(Counter(reason for r in rejected for reason in r["reasons"])),
    }
    if splits:
        report["splits"] = {}
        for name, split in zip(("train", "val"), splits):
            path = out_dir / f"{name}.jsonl"
            path.write_text("".join(
                json.dumps(_to_conversation(r, max_chars)) + "\n" for r in split
            ))
            report["splits"][name] = {
                **_identity(path), "question_ids": [r["question_id"] for r in split],
                "doc_ids": sorted({known_paper_id(r["question"]) for r in split}),
                "answerability": dict(Counter(r["expected_answerability"] for r in split)),
            }
    (out_dir / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"{len(candidates)} structural candidates; {len(accepted)} reviewed passes -> {out_dir}")
    if not review:
        print("No training files written. Complete replay and semantic review before export.")


if __name__ == "__main__":
    typer.run(main)
