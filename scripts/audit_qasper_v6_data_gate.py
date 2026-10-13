"""Audit replayed QASPER SFT rows before spending GPU time.

This checks the frozen split and exact training messages, then tokenizes every
next-action target with the same masking path as ``train_sft.py``. It does not
judge whether a quoted passage semantically supports the answer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_sft import (
    BASE_MODEL,
    BASE_REVISION,
    _expand_per_action,
    _tokenize_per_action,
)
from scripts.transfer_qasper_replay_reviews import build_reviews
from src.env.document_env import initial_observation
from src.eval.artifacts import content_hash
from src.eval.sft_quality import known_paper_id, trajectory_hash, trajectory_issues
from src.policies.code_execution import QASPER_SYSTEM_PROMPT
from src.policies.qwen3_chat_template import patch_tokenizer_for_assistant_masking

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_structure(
    replay_dir: Path, holdout_lock_path: Path
) -> tuple[dict, dict[str, list[dict]]]:
    manifest = json.loads((replay_dir / "manifest.json").read_text())
    holdout = json.loads(holdout_lock_path.read_text())
    if manifest.get("schema_version") != "qasper-envoybench-sft-replay-v1":
        raise ValueError("Unexpected replay manifest schema")
    if manifest.get("status") != "complete" or manifest.get("rejected_count"):
        raise ValueError("Full exact replay did not complete without rejections")
    if manifest.get("runtime_protocol") != "envoybench-v1":
        raise ValueError("The replay does not use the EnvoyBench protocol")
    journal_path = ROOT / manifest["journal"]["path"]
    if _sha256(journal_path) != manifest["journal"]["sha256"]:
        raise ValueError("The replay journal changed after finalization")
    runtime = manifest["runtime"]
    required_runtime = {
        "max_steps": 15,
        "use_docker": True,
        "require_evidence": True,
        "include_preamble": True,
        "evidence_verifier": True,
        "verifier_feedback_budget": 1,
        "escalate_after_verifier_failure": True,
    }
    if any(runtime.get(key) != value for key, value in required_runtime.items()):
        raise ValueError("Replay runtime differs from the frozen evaluation runtime")
    system_hash = hashlib.sha256(QASPER_SYSTEM_PROMPT.encode()).hexdigest()
    if runtime.get("system_prompt_sha256") != system_hash:
        raise ValueError("Replay system prompt differs from evaluation")
    if manifest["holdout_lock"]["sha256"] != _sha256(holdout_lock_path):
        raise ValueError("The holdout lock changed after replay")
    benchmark_path = ROOT / holdout["benchmark_path"]
    if _sha256(benchmark_path) != holdout["benchmark_sha256"]:
        raise ValueError("The frozen holdout benchmark bytes changed")
    benchmark = json.loads(benchmark_path.read_text())
    if [q["id"] for q in benchmark["questions"]] != holdout["question_ids"]:
        raise ValueError("The frozen holdout question IDs changed")
    if (
        benchmark["corpus_hash"] != holdout["corpus_hash"]
        or content_hash(ROOT / holdout["corpus_path"]) != holdout["corpus_hash"]
    ):
        raise ValueError("The frozen holdout corpus changed")

    candidates_path = replay_dir / "candidates.jsonl"
    if _sha256(candidates_path) != manifest["candidates"]["sha256"]:
        raise ValueError("Candidate bytes differ from the replay manifest")
    candidates = _jsonl(candidates_path)
    if len(candidates) != manifest["accepted_count"] or len(candidates) < 100:
        raise ValueError("Too few replayed trajectories")
    by_id = {row["question_id"]: row for row in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate training question IDs")
    answerability = Counter(row["expected_answerability"] for row in candidates)
    if answerability["sufficient"] < 80 or answerability["insufficient"] < 20:
        raise ValueError("The reviewed answerability strata are too small")
    review_path = replay_dir / "reviews.json"
    reviews = json.loads(review_path.read_text())
    if reviews != build_reviews(replay_dir):
        raise ValueError("Transferred review provenance changed after its source audit")
    if any(
        item["review_scope"] != "individual_source_visible_model_assisted_semantic_audit"
        for item in reviews["reviews"]
        if by_id[item["question_id"]]["expected_answerability"] == "insufficient"
    ):
        raise ValueError("An abstention lacks individual source-visible review")
    held_papers = set(holdout["reserved_doc_ids"])
    splits: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()
    split_papers: dict[str, set[str]] = {}
    for name in ("train", "val"):
        entry = manifest["splits"][name]
        path = replay_dir / f"{name}.jsonl"
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"{name} split bytes differ from the replay manifest")
        rows = _jsonl(path)
        ids = entry["question_ids"]
        if len(rows) != len(ids) or len(ids) != entry["count"]:
            raise ValueError(f"{name} row count differs from the replay manifest")
        if seen_ids.intersection(ids):
            raise ValueError("Training and validation question IDs overlap")
        seen_ids.update(ids)
        papers = set(entry["paper_ids"])
        split_papers[name] = papers
        if papers & held_papers:
            raise ValueError(f"{name} papers overlap the frozen holdout")
        for qid, conversation in zip(ids, rows, strict=True):
            candidate = by_id[qid]
            if known_paper_id(candidate["question"]) not in papers:
                raise ValueError(f"{name}: paper not declared for {qid}")
            messages = conversation["messages"]
            expected = [
                {"role": "system", "content": QASPER_SYSTEM_PROMPT},
                {"role": "user", "content": initial_observation(
                    candidate["question"], require_evidence=True, include_preamble=True,
                )},
            ]
            for step in candidate["trajectory"]:
                if step.get("raw_action", step["action"]) != step["action"]:
                    raise ValueError(f"{name}: the raw and executed actions differ for {qid}")
                expected.append({"role": "assistant", "content": step["action"]})
                if not step["done"]:
                    expected.append({"role": "user", "content": step["observation"]})
            if messages != expected:
                raise ValueError(f"{name}: SFT messages differ from replay for {qid}")
            review = candidate.get("review", {})
            if review.get("trajectory_sha256") != trajectory_hash(candidate):
                raise ValueError(f"{name}: stale trajectory review for {qid}")
            if not all(review.get(key) is True for key in
                       ("replay_verified", "evidence_checked", "stopping_checked")):
                raise ValueError(f"{name}: incomplete source review for {qid}")
            if candidate.get("runtime_protocol") != "envoybench-v1":
                raise ValueError(f"{name}: mismatched protocol for {qid}")
            if trajectory_issues(candidate):
                raise ValueError(f"{name}: structural trajectory error for {qid}")
        splits[name] = rows
    if seen_ids != set(by_id):
        raise ValueError("Replayed candidates and SFT splits have different question IDs")
    if split_papers["train"] & split_papers["val"]:
        raise ValueError("Training and validation papers overlap")
    return {
        "schema_version": "qasper-v6-data-gate-v1",
        "status": "structural_pass",
        "replay_manifest_sha256": _sha256(replay_dir / "manifest.json"),
        "replay_journal_sha256": _sha256(journal_path),
        "candidate_sha256": _sha256(candidates_path),
        "reviews_sha256": _sha256(review_path),
        "holdout_lock_sha256": _sha256(holdout_lock_path),
        "holdout_benchmark_sha256": _sha256(benchmark_path),
        "system_prompt_sha256": system_hash,
        "counts": {"candidates": len(candidates), **answerability},
        "splits": {
            name: {"rows": len(rows), "sha256": _sha256(replay_dir / f"{name}.jsonl")}
            for name, rows in splits.items()
        },
        "claim_boundary": (
            "This proves replay/data integrity and target masking, not source-to-claim "
            "semantic support or model improvement."
        ),
    }, splits


def audit_tokens(splits: dict[str, list[dict]]) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, revision=BASE_REVISION, trust_remote_code=True,
    )
    patch_tokenizer_for_assistant_masking(tokenizer)
    report = {"model": BASE_MODEL, "revision": BASE_REVISION, "max_seq_len": 8192}
    for name, rows in splits.items():
        actions = _expand_per_action(rows)
        _, stats = _tokenize_per_action(tokenizer, actions, 8192, name)
        report[name] = stats
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--holdout-lock", type=Path,
                        default=ROOT / "data/research/qasper-v6-holdout-lock.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    report, splits = audit_structure(args.replay_dir, args.holdout_lock)
    if not args.structure_only:
        report["tokenization"] = audit_tokens(splits)
        report["status"] = "full_pass"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))
    print(report["status"])


if __name__ == "__main__":
    main()
