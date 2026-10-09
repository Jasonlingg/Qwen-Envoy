"""Mechanical smoke gates; these tests do not run or impersonate live Qwen."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts.personal_memory_eval import freeze
from scripts.smoke_qwen_memory_worker import _cases, _check_packet, run_smoke
from src.research.agent import load_snapshot


@pytest.fixture
def snapshot(tmp_path):
    target = tmp_path / "snapshot"
    freeze(target)
    return target


def test_smoke_rejects_a_claimed_answer_without_model_or_tools(snapshot):
    _, cases = _cases(snapshot)
    question = cases[1]["question"]
    packet = {
        "status": "evidence_found",
        "execution": "docker_only",
        "model_requests_attempted": 0,
        "document_tool_steps": 0,
        "inspection_steps": 0,
        "evidence": [],
    }

    checks, handoff = _check_packet(snapshot, question, packet, temporal=True)

    assert not checks["model_called"]
    assert not checks["document_tool_used"]
    assert not checks["mechanical_pass"]
    assert handoff is None


def test_smoke_checks_frozen_evidence_at_product_handoff(snapshot):
    _, cases = _cases(snapshot)
    _, docs = load_snapshot(snapshot)
    doc_id, document = next(iter(docs.items()))
    quote = "20 - 2x"
    if quote not in document["text"]:
        doc_id, document = next(
            (key, doc) for key, doc in docs.items() if quote in doc["text"]
        )
    start = document["text"].index(quote)
    packet = {
        "status": "evidence_found",
        "execution": "docker_only",
        "model_requests_attempted": 3,
        "document_tool_steps": 2,
        "inspection_steps": 1,
        "corpus_hash": json.loads((snapshot / "manifest.json").read_text())["corpus_hash"],
        "evidence": [{
            "evidence_id": "E1", "doc_id": doc_id,
            "start": start, "end": start + len(quote), "quote": quote,
        }],
    }

    checks, handoff = _check_packet(snapshot, cases[1]["question"], packet, temporal=True)

    assert checks["mechanical_pass"]
    assert handoff["evidence"][0]["quote"] == quote
    assert handoff["evidence"][0]["source_path"]


def test_smoke_persists_failure_and_skips_second_question(snapshot, tmp_path, monkeypatch):
    calls = []

    class UnavailableWorker:
        model_identity = {"checkpoint": "test-only", "seed": 42}
        max_steps = 12
        image = "rlm-sandbox"

        @staticmethod
        def policy_factory():
            return SimpleNamespace(system_prompt="test prompt")

        @staticmethod
        def investigate(question, _snapshot):
            calls.append(question)
            return {
                "status": "unavailable", "execution": "docker_only",
                "model_requests_attempted": 0, "trajectory": [],
            }

    monkeypatch.setattr(
        "scripts.smoke_qwen_memory_worker.make_qwen_investigator",
        lambda *_args: UnavailableWorker(),
    )
    output = tmp_path / "smoke.json"
    result = run_smoke(
        endpoint="http://127.0.0.1:8000/v1", model="test-only",
        checkpoint="test-only", snapshot=snapshot, output=output,
    )

    assert result["status"] == "failed"
    assert result["stopped_after_question_id"] == "pm03_life_missing_outcome"
    assert len(calls) == 1
    assert json.loads(output.read_text())["episodes"][0]["packet"]["status"] == "unavailable"
    with pytest.raises(FileExistsError):
        run_smoke(
            endpoint="http://127.0.0.1:8000/v1", model="test-only",
            checkpoint="test-only", snapshot=snapshot, output=output,
        )
