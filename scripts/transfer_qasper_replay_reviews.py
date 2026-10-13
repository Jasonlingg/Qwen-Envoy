"""Transfer disclosed source-review provenance to replayed QASPER candidates.

This does not perform a new semantic review. The exporter uses ``verdict=pass``
to mean that a row is eligible for this SFT pilot; ``review_scope`` makes clear
whether that row also received an individual, source-visible semantic audit.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_qasper_envoybench_sft import _observation_payload, _read_journal
from src.eval.hashing import sha256
from src.eval.qasper_reward import parse_strict
from src.eval.sft_quality import trajectory_hash

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "out/research/qasper-synthetic-reliability-v1/accepted/candidates.jsonl"
SEMANTIC = ROOT / "out/research/qasper-synthetic-reliability-v1/semantic-judgments.json"
DEFAULT_REPLAY = ROOT / "out/research/qasper-v6-runtime-replayed-20261013"


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verify_runtime_replay_integrity(replay_dir: Path) -> dict:
    """Independently recheck the immutable journal and source/runtime equivalence."""
    manifest = json.loads((replay_dir / "manifest.json").read_text())
    candidate_path = replay_dir / "candidates.jsonl"
    if manifest.get("status") != "complete" or manifest.get("attempted_count") != 137:
        raise ValueError("Require the complete 137-row replay")
    if sha256(SOURCE) != manifest["source"]["candidates"]["sha256"]:
        raise ValueError("Original candidate hash differs from final manifest")
    if sha256(candidate_path) != manifest["candidates"]["sha256"]:
        raise ValueError("Runtime candidate hash differs from final manifest")
    source_rows = _rows(SOURCE)
    runtime_rows = _rows(candidate_path)
    journal_ref = manifest.get("journal")
    if journal_ref is None:
        raise ValueError("Runtime replay manifest has no append-only journal")
    journal_path = Path(journal_ref["path"])
    if not journal_path.is_absolute():
        journal_path = ROOT / journal_path
    if sha256(journal_path) != journal_ref["sha256"]:
        raise ValueError("Replay journal hash differs from final manifest")
    first = json.loads(journal_path.read_text().splitlines()[0])
    context = {key: value for key, value in first.items() if key != "entry_sha256"}
    source_keys = ("candidates", "manifest", "train", "val", "benchmark", "semantic_review")
    if context.get("source") != {key: manifest["source"][key] for key in source_keys}:
        raise ValueError("Journal source hashes differ from final manifest")
    if context.get("holdout_lock") != manifest.get("holdout_lock"):
        raise ValueError("Journal holdout hash differs from final manifest")
    if context.get("corpus_content_hash") != manifest["corpus"]["content_hash"]:
        raise ValueError("Journal corpus hash differs from final manifest")
    runtime_context = context.get("runtime", {})
    if runtime_context.get("protocol") != manifest["runtime_protocol"] or {
        key: value for key, value in runtime_context.items() if key != "protocol"
    } != manifest["runtime"]:
        raise ValueError("Journal runtime settings differ from final manifest")
    if context.get("implementation") != manifest.get("implementation"):
        raise ValueError("Journal code hashes differ from final manifest")
    if context.get("question_ids") != [row["question_id"] for row in source_rows]:
        raise ValueError("Journal source question order changed")
    journal_accepted, journal_rejected, _ = _read_journal(
        journal_path, context, source_rows,
    )
    if journal_accepted != runtime_rows or journal_rejected != manifest["rejections"]:
        raise ValueError("Final replay candidates differ from append-only journal")
    if len(runtime_rows) != manifest["accepted_count"]:
        raise ValueError("Final replay candidate count differs from manifest")
    by_id = {row["question_id"]: row for row in source_rows}
    if len(by_id) != len(source_rows):
        raise ValueError("Duplicate source candidate question ID")
    for row in runtime_rows:
        qid = row["question_id"]
        source = by_id[qid]
        if (row["source_trajectory_sha256"] != trajectory_hash(source)
                or len(row["trajectory"]) != len(source["trajectory"])):
            raise ValueError(f"Source trajectory identity changed: {qid}")
        for step, (before, after) in enumerate(
            zip(source["trajectory"], row["trajectory"], strict=True), start=1,
        ):
            if before["action"] != after["action"] or before["done"] != after["done"]:
                raise ValueError(f"Source action or terminal flag changed: {qid}")
            if before["done"]:
                if before["observation"] != after["observation"]:
                    raise ValueError(f"Final submission observation changed: {qid}")
            elif _observation_payload(before["observation"], step=step, budget=10) != (
                _observation_payload(after["observation"], step=step, budget=15)
            ):
                raise ValueError(f"Tool observation payload changed: {qid}")
        for field in ("predicted_answer", "predicted_citations", "predicted_evidence"):
            if source[field] != row[field]:
                raise ValueError(f"Submitted {field} changed: {qid}")
        if parse_strict(row["trajectory"][-1]["action"]) is None:
            raise ValueError(f"Final submission does not strictly parse: {qid}")
    return {
        "candidate_count": len(runtime_rows),
        "journal_sha256": journal_ref["sha256"],
        "source_candidates_sha256": sha256(SOURCE),
        "runtime_candidates_sha256": sha256(candidate_path),
    }


def build_reviews(replay_dir: Path) -> dict:
    """Bind every surviving runtime row to its original row and review scope."""
    verify_runtime_replay_integrity(replay_dir)
    manifest_path = replay_dir / "manifest.json"
    candidate_path = replay_dir / "candidates.jsonl"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("attempted_count") != 137:
        raise ValueError("Require a complete 137-row runtime replay, not a smoke run")
    if manifest.get("runtime_protocol") != "envoybench-v1":
        raise ValueError("Unexpected runtime protocol")
    if sha256(candidate_path) != manifest["candidates"]["sha256"]:
        raise ValueError("Runtime candidates no longer match replay manifest")
    if sha256(SOURCE) != manifest["source"]["candidates"]["sha256"]:
        raise ValueError("Original candidate source no longer matches replay manifest")
    if sha256(SEMANTIC) != manifest["source"]["semantic_review"]["sha256"]:
        raise ValueError("Semantic judgments no longer match replay manifest")

    source_rows = {row["question_id"]: row for row in _rows(SOURCE)}
    semantic_doc = json.loads(SEMANTIC.read_text())
    semantic = {item["question_id"]: item for item in semantic_doc["reviews"]}
    if len(semantic) != len(semantic_doc["reviews"]):
        raise ValueError("Duplicate semantic-review question ID")
    if semantic_doc["reviewer_type"] != manifest["source"]["reviewer_type"]:
        raise ValueError("Source reviewer type changed")

    runtime_rows = _rows(candidate_path)
    if len(runtime_rows) != manifest["accepted_count"]:
        raise ValueError("Runtime candidate count differs from manifest")
    if len({row["question_id"] for row in runtime_rows}) != len(runtime_rows):
        raise ValueError("Duplicate runtime question ID")
    reviews = []
    for row in runtime_rows:
        qid = row["question_id"]
        source = source_rows[qid]
        original_hash = trajectory_hash(source)
        if row["source_trajectory_sha256"] != original_hash:
            raise ValueError(f"Parent trajectory changed: {qid}")
        if source["review"]["trajectory_sha256"] != original_hash:
            raise ValueError(f"Parent review is stale: {qid}")
        if row["source_review"] != source["review"]:
            raise ValueError(f"Parent review metadata changed: {qid}")
        if row["review"]["trajectory_sha256"] != trajectory_hash(row):
            raise ValueError(f"Replay review is stale: {qid}")
        if row.get("runtime_protocol") != "envoybench-v1":
            raise ValueError(f"Unexpected row runtime protocol: {qid}")
        for field in ("replay_verified", "evidence_checked", "stopping_checked"):
            if source["review"].get(field) is not True or row["review"].get(field) is not True:
                raise ValueError(f"Missing inherited review check {field}: {qid}")

        individual = semantic.get(qid)
        if individual is not None:
            if individual["trajectory_sha256"] != original_hash:
                raise ValueError(f"Individual semantic review is stale: {qid}")
            if individual["verdict"] != "pass":
                raise ValueError(f"Individually rejected row was migrated: {qid}")
            scope = "individual_source_visible_model_assisted_semantic_audit"
            notes = (
                "Source-visible model-assisted semantic audit passed: "
                + individual["notes"]
                + " Exact tool payloads were re-executed under envoybench-v1; "
                "this is not an independent human review."
            )
        else:
            if row["expected_answerability"] != "sufficient":
                raise ValueError(f"Insufficient-label row lacks individual source review: {qid}")
            scope = "annotation_exact_span_and_runtime_replay_only"
            notes = (
                "The source QASPER answer/evidence annotation, exact-span check, "
                "and fresh tool replay passed. No individual source-visible semantic "
                "review was recorded for this answerable row; these checks do not "
                "establish that every answer claim is supported."
            )
        reviews.append({
            "question_id": qid,
            "trajectory_sha256": trajectory_hash(row),
            "source_trajectory_sha256": original_hash,
            "verdict": "pass",
            "verdict_scope": "eligible_for_sft_pilot_not_semantic_certification",
            "review_scope": scope,
            "individual_source_visible_semantic_review": individual is not None,
            "source_semantic_review_verdict": individual["verdict"] if individual else None,
            "source_semantic_review_notes": individual["notes"] if individual else None,
            "evidence_checked": True,
            "stopping_checked": True,
            "replay_verified": True,
            "notes": notes,
        })

    answerability = Counter(row["expected_answerability"] for row in runtime_rows)
    review_scopes = Counter(item["review_scope"] for item in reviews)
    if any(
        source_rows[item["question_id"]]["expected_answerability"] == "insufficient"
        and not item["individual_source_visible_semantic_review"]
        for item in reviews
    ):
        raise ValueError("Some abstentions were not individually reviewed")
    return {
        "schema_version": "qasper-envoybench-transferred-reviews-v1",
        "reviewer": semantic_doc["reviewer"],
        "reviewer_type": "assistant",  # Strict exporter vocabulary; see source_reviewer_type.
        "source_reviewer_type": semantic_doc["reviewer_type"],
        "review_method": "provenance_transfer_after_exact_runtime_replay_no_new_semantic_review",
        "review_scope_note": (
            "All rows inherit annotation/span/replay checks. Individual source-visible "
            "model-assisted semantic notes exist only where review_scope names that audit. "
            "There was no independent human review; a pass here means pilot training "
            "eligibility, not proof of answer correctness."
        ),
        "inputs": {
            "runtime_manifest_sha256": sha256(manifest_path),
            "runtime_candidates_sha256": sha256(candidate_path),
            "source_candidates_sha256": sha256(SOURCE),
            "source_semantic_judgments_sha256": sha256(SEMANTIC),
        },
        "counts": {
            "reviews": len(reviews),
            "answerability": dict(answerability),
            "review_scopes": dict(review_scopes),
        },
        "reviews": reviews,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, default=DEFAULT_REPLAY)
    args = parser.parse_args()
    output = args.replay_dir / "reviews.json"
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing transferred review: {output}")
    result = build_reviews(args.replay_dir)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(f"{result['counts']['reviews']} hash-bound transferred reviews -> {output}")


if __name__ == "__main__":
    main()
