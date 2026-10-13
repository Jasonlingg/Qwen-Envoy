"""Review transfer must expose semantic-review gaps instead of hiding them."""

import json

import pytest

from scripts import transfer_qasper_replay_reviews as transfer
from scripts.replay_qasper_envoybench_sft import _append_journal
from src.eval.hashing import sha256
from src.eval.sft_quality import trajectory_hash


def _fixture(tmp_path, monkeypatch):
    source_path = tmp_path / "source.jsonl"
    semantic_path = tmp_path / "semantic.json"
    replay_dir = tmp_path / "replay"
    replay_dir.mkdir()
    monkeypatch.setattr(transfer, "SOURCE", source_path)
    monkeypatch.setattr(transfer, "SEMANTIC", semantic_path)

    source_rows, runtime_rows, semantic_reviews = [], [], []
    for qid, answerability, individual in (
        ("reviewed-answer", "sufficient", True),
        ("annotation-only-answer", "sufficient", False),
        ("reviewed-abstention", "insufficient", True),
    ):
        source = {
            "question_id": qid, "question": f"Question {qid}",
            "expected_answerability": answerability,
            "trajectory": [
                {"action": "print('paper')",
                 "observation": "real bytes\n\n[Step 1/10/10]", "done": False},
                {"action": "SUBMIT: A CITATIONS: [\"paper\"] "
                           "EVIDENCE: [{\"doc_id\": \"paper\", \"quote\": \"real bytes\"}]",
                 "observation": "Submitted. Reward: 1.000", "done": True},
            ],
            "predicted_answer": "A", "predicted_citations": ["paper"],
            "predicted_evidence": [{"doc_id": "paper", "quote": "real bytes"}],
        }
        parent_hash = trajectory_hash(source)
        source["review"] = {
            "trajectory_sha256": parent_hash, "replay_verified": True,
            "evidence_checked": True, "stopping_checked": True,
        }
        runtime = {**source, "trajectory": [
            {**source["trajectory"][0], "observation": "real bytes\n\n[Step 1/15/15]"},
            source["trajectory"][1],
        ], "runtime_protocol": "envoybench-v1",
            "source_trajectory_sha256": parent_hash, "source_review": source["review"]}
        runtime["review"] = {
            **source["review"], "trajectory_sha256": trajectory_hash(runtime),
        }
        source_rows.append(source)
        runtime_rows.append(runtime)
        if individual:
            semantic_reviews.append({
                "question_id": qid, "trajectory_sha256": parent_hash,
                "verdict": "pass", "notes": f"Inspected {qid} in source paper.",
            })

    source_path.write_text("".join(json.dumps(row) + "\n" for row in source_rows))
    candidate_path = replay_dir / "candidates.jsonl"
    candidate_path.write_text("".join(json.dumps(row) + "\n" for row in runtime_rows))
    reviewer_type = "source-visible model-assisted semantic audit; not an independent human review"
    semantic_doc = {
        "reviewer": "Assistant reviewer", "reviewer_type": reviewer_type,
        "reviews": semantic_reviews,
    }
    semantic_path.write_text(json.dumps(semantic_doc))
    manifest = {
        "status": "complete", "attempted_count": 137, "accepted_count": 3,
        "runtime_protocol": "envoybench-v1",
        "rejections": [],
        "candidates": {"sha256": sha256(candidate_path)},
        "source": {
            "candidates": {"sha256": sha256(source_path)},
            "manifest": {"sha256": "source-manifest"},
            "train": {"sha256": "source-train"},
            "val": {"sha256": "source-val"},
            "benchmark": {"sha256": "source-benchmark"},
            "semantic_review": {"sha256": sha256(semantic_path)},
            "reviewer_type": reviewer_type,
        },
        "holdout_lock": {"sha256": "holdout"},
        "corpus": {"content_hash": "corpus"},
        "runtime": {"system_prompt_sha256": "runtime-prompt", "max_steps": 15},
        "implementation": {"document_env": "code-hash"},
    }
    context = {
        "type": "header", "schema_version": "qasper-envoybench-replay-journal-v1",
        "source": {key: manifest["source"][key] for key in
                   ("candidates", "manifest", "train", "val", "benchmark", "semantic_review")},
        "holdout_lock": manifest["holdout_lock"],
        "corpus_content_hash": "corpus",
        "runtime": {"protocol": "envoybench-v1", **manifest["runtime"]},
        "implementation": manifest["implementation"],
        "question_ids": [row["question_id"] for row in source_rows],
    }
    journal = tmp_path / "replay.journal.jsonl"
    journal.touch()
    previous = _append_journal(journal, context, None)
    for index, (source, runtime) in enumerate(zip(source_rows, runtime_rows, strict=True)):
        previous = _append_journal(journal, {
            "type": "accepted", "source_index": index,
            "question_id": source["question_id"],
            "source_trajectory_sha256": trajectory_hash(source),
            "candidate": runtime,
        }, previous)
    manifest["journal"] = {"path": str(journal), "sha256": sha256(journal)}
    (replay_dir / "manifest.json").write_text(json.dumps(manifest))
    return replay_dir, semantic_path, semantic_doc, manifest


def test_transfer_discloses_which_answer_lacks_individual_semantic_review(
    tmp_path, monkeypatch,
) -> None:
    replay_dir, _, _, _ = _fixture(tmp_path, monkeypatch)
    review = transfer.build_reviews(replay_dir)
    by_id = {item["question_id"]: item for item in review["reviews"]}
    assert review["reviewer_type"] == "assistant"
    assert review["counts"]["review_scopes"] == {
        "individual_source_visible_model_assisted_semantic_audit": 2,
        "annotation_exact_span_and_runtime_replay_only": 1,
    }
    assert by_id["annotation-only-answer"]["individual_source_visible_semantic_review"] is False
    assert (
        "No individual source-visible semantic review"
        in by_id["annotation-only-answer"]["notes"]
    )
    assert by_id["reviewed-abstention"]["source_semantic_review_notes"]
    assert all(item["verdict_scope"] == "eligible_for_sft_pilot_not_semantic_certification"
               for item in review["reviews"])


def test_transfer_refuses_unreviewed_abstention(tmp_path, monkeypatch) -> None:
    replay_dir, semantic_path, semantic_doc, manifest = _fixture(tmp_path, monkeypatch)
    semantic_doc["reviews"] = [
        item for item in semantic_doc["reviews"]
        if item["question_id"] != "reviewed-abstention"
    ]
    semantic_path.write_text(json.dumps(semantic_doc))
    manifest["source"]["semantic_review"]["sha256"] = sha256(semantic_path)
    (replay_dir / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(transfer, "verify_runtime_replay_integrity", lambda _path: {})
    with pytest.raises(ValueError, match="Insufficient-label row lacks individual source review"):
        transfer.build_reviews(replay_dir)


def test_transfer_rejects_rewritten_journal_even_if_manifest_hash_is_updated(
    tmp_path, monkeypatch,
) -> None:
    replay_dir, _, _, manifest = _fixture(tmp_path, monkeypatch)
    journal = tmp_path / "replay.journal.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    records[1]["candidate"]["trajectory"][0]["observation"] = "fabricated"
    journal.write_text("".join(json.dumps(item) + "\n" for item in records))
    manifest["journal"]["sha256"] = sha256(journal)
    (replay_dir / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Tampered replay journal entry"):
        transfer.verify_runtime_replay_integrity(replay_dir)
