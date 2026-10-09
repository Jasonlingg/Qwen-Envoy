"""CPU-only gates for the four-question public-paper product-path runner."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_research_library_qwen_dev as runner
from src.env.tools import SEARCH_PROTOCOL_VERSION, TOOL_PREAMBLE
from src.eval.artifacts import content_hash


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@pytest.fixture
def frozen_dev(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    corpus = snapshot / "corpus"
    corpus.mkdir(parents=True)
    paper_ids = [f"paper_{index:02}" for index in range(20)]
    for index, paper_id in enumerate(paper_ids):
        _json(corpus / f"{paper_id}.json", {
            "doc_id": paper_id, "title": f"Paper {index}",
            "text": f"Paper {index} reports a measured result.\n",
            "metadata": {"source_url": f"https://example.org/{index}",
                         "coverage": "html_paragraphs", "arxiv_id": f"2609.{index:05}v1"},
        })
    corpus_hash = content_hash(corpus)
    _json(snapshot / "manifest.json", {
        "schema_version": "research-snapshot-v1",
        "parser_version": "test-paper-parser-v1",
        "corpus_hash": corpus_hash,
        "papers": [{"doc_id": paper_id} for paper_id in paper_ids],
    })
    benchmark = tmp_path / "dev.json"
    _json(benchmark, {
        "schema_version": "research-benchmark-v1",
        "benchmark_id": runner.BENCHMARK_ID,
        "corpus_hash": corpus_hash,
        "reserved_doc_ids": paper_ids,
        "questions": [
            {
                "id": question_id, "split": "pilot_evaluation",
                "question": f"What did paper {index} report?",
                "answer": "A measured result.",
                "learning_skill": f"skill_{index}",
                "expected_answerability": "sufficient",
                "required_doc_ids": [paper_ids[index]],
                "minimum_distinct_sources": 1,
                "grader_notes": ["Answer with source support."],
                "source_anchors": [{
                    "doc_id": paper_ids[index],
                    "needle": f"Paper {index} reports a measured result.",
                }],
            }
            for index, question_id in enumerate(runner.QUESTION_IDS)
        ],
    })
    monkeypatch.setattr(runner, "CORPUS_HASH", corpus_hash)
    return snapshot, benchmark, corpus_hash


def _approved_record(path: Path, benchmark: Path, corpus_hash: str) -> None:
    record = runner._pending_review(runner._sha256(benchmark), corpus_hash)
    record["status"] = "approved"
    record["reviewer"] = {
        "kind": "human", "name": "Fixture Reviewer",
        "reviewed_at": "2026-09-28T12:00:00+00:00",
    }
    for question in record["questions"]:
        for field in (
            "reference_answer", "answerability", "required_paper_ids",
            "anchors_support_reference",
        ):
            question[field] = "approved"
    _json(path, record)


def test_dry_run_validates_sources_without_docker_or_model(frozen_dev, tmp_path,
                                                          monkeypatch):
    snapshot, benchmark, corpus_hash = frozen_dev
    monkeypatch.setattr(runner, "_worker", lambda *args: pytest.fail("model constructed"))
    monkeypatch.setattr(
        runner.PersistentREPL, "_docker_available",
        lambda image: pytest.fail("Docker checked in dry-run"),
    )
    output = tmp_path / "dry-run"

    manifest = runner.run(
        output=output, dry_run=True, snapshot=snapshot, benchmark_path=benchmark,
    )

    assert manifest["status"] == "dry_run_ready"
    assert manifest["model_requests_attempted"] == 0
    assert manifest["question_ids"] == list(runner.QUESTION_IDS)
    assert manifest["corpus_hash"] == corpus_hash
    assert (output / "source-check.md").is_file()
    review = json.loads((output / "reference-review-template.json").read_text())
    assert review["status"] == "pending"
    assert review["benchmark_sha256"] == runner._sha256(benchmark)
    assert all(item["anchors_support_reference"] == "pending"
               for item in review["questions"])
    with pytest.raises(FileExistsError):
        runner.run(output=output, dry_run=True, snapshot=snapshot, benchmark_path=benchmark)


def test_live_mode_rejects_pending_or_mismatched_reference_review_before_model(
    frozen_dev, tmp_path, monkeypatch,
):
    snapshot, benchmark, corpus_hash = frozen_dev
    monkeypatch.setattr(runner, "_worker", lambda *args: pytest.fail("model constructed"))
    output = tmp_path / "live"
    pending = tmp_path / "pending.json"
    _json(pending, runner._pending_review(runner._sha256(benchmark), corpus_hash))
    args = dict(
        output=output, dry_run=False, snapshot=snapshot, benchmark_path=benchmark,
        endpoint="http://127.0.0.1:8000/v1", model="qwen", checkpoint="pinned-base",
        hardware="test-gpu", run_label="base", review_record=pending,
    )
    with pytest.raises(ValueError, match="not approved"):
        runner.run(**args)
    assert not output.exists()

    approved = tmp_path / "approved.json"
    _approved_record(approved, benchmark, corpus_hash)
    record = json.loads(approved.read_text())
    record["benchmark_sha256"] = "0" * 64
    _json(approved, record)
    args["review_record"] = approved
    with pytest.raises(ValueError, match="exact benchmark bytes"):
        runner.run(**args)
    assert not output.exists()


def test_approved_development_run_saves_product_packets_and_manifest(
    frozen_dev, tmp_path, monkeypatch,
):
    snapshot, benchmark, corpus_hash = frozen_dev
    review = tmp_path / "approved.json"
    _approved_record(review, benchmark, corpus_hash)
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda image: True)
    calls: list[str] = []

    class FakeWorker:
        def investigate(self, question: str, selected_snapshot: Path) -> dict:
            assert selected_snapshot == snapshot
            calls.append(question)
            return {
                "status": "no_evidence", "execution": "docker_only",
                "corpus_hash": corpus_hash,
                "source_domain": "public_papers",
                "system_prompt_sha256": hashlib.sha256(
                    runner.PUBLIC_PAPER_SYSTEM_PROMPT.encode()
                ).hexdigest(),
                "tool_search_version": SEARCH_PROTOCOL_VERSION,
                "tool_preamble_sha256": hashlib.sha256(TOOL_PREAMBLE.encode()).hexdigest(),
                "model_identity": {"checkpoint": "pinned-base", "served_model": "qwen"},
                "model_requests_attempted": 3,
                "document_tool_steps": 2, "inspection_steps": 1,
                "candidate_claims": [], "evidence": [],
                "trajectory": [{"action": "print(search('paper'))"}],
            }

    monkeypatch.setattr(runner, "_worker", lambda *args: FakeWorker())
    output = tmp_path / "live"
    manifest = runner.run(
        output=output, dry_run=False, snapshot=snapshot, benchmark_path=benchmark,
        endpoint="http://127.0.0.1:8000/v1", model="qwen", checkpoint="pinned-base",
        hardware="test-gpu", run_label="base", review_record=review,
    )

    assert manifest["status"] == "packets_saved_semantic_review_pending"
    assert manifest["reference_review_sha256"] == runner._sha256(review)
    assert manifest["completed_question_ids"] == list(runner.QUESTION_IDS)
    assert manifest["model_requests_attempted"] == 12
    assert calls == [item["question"] for item in json.loads(benchmark.read_text())["questions"]]
    for question_id in runner.QUESTION_IDS:
        episode = json.loads((output / "packets" / f"{question_id}.json").read_text())
        assert episode["mechanical_checks"]["verified_handoff"] is True
        assert episode["semantic_review"] == "pending"
        assert "answer" not in episode
        assert "source_anchors" not in episode


def test_live_mode_stops_and_preserves_failed_packet(frozen_dev, tmp_path, monkeypatch):
    snapshot, benchmark, corpus_hash = frozen_dev
    review = tmp_path / "approved.json"
    _approved_record(review, benchmark, corpus_hash)
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda image: True)
    calls = []

    class FailedWorker:
        def investigate(self, question: str, selected_snapshot: Path) -> dict:
            calls.append(question)
            return {"status": "incomplete", "execution": "docker_only",
                    "corpus_hash": corpus_hash, "model_requests_attempted": 15,
                    "document_tool_steps": 2, "inspection_steps": 1,
                    "trajectory": [{"action": "print(search('paper'))"}],
                    "evidence": []}

    monkeypatch.setattr(runner, "_worker", lambda *args: FailedWorker())
    output = tmp_path / "failed"
    result = runner.run(
        output=output, dry_run=False, snapshot=snapshot, benchmark_path=benchmark,
        endpoint="http://127.0.0.1:8000/v1", model="qwen", checkpoint="pinned-base",
        hardware="test-gpu", run_label="base", review_record=review,
    )
    assert result["status"] == "stopped_on_mechanical_failure"
    assert result["completed_question_ids"] == ["D01"]
    assert len(calls) == 1
    assert json.loads((output / "packets/D01.json").read_text())["packet"]["trajectory"]
