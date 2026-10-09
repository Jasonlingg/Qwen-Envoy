"""The Nemotron handoff must contain source text, not evaluation answers."""

import json
import os

import pytest

from scripts import explain_code_exec_nebius as cli
from src.research.code_exec_packet import build_code_exec_packet


def _fixture(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    text = "Alpha reports a measured result. Beta is a different result."
    (corpus / "paper_a.json").write_text(json.dumps({
        "doc_id": "paper_a", "title": "A test paper", "text": text,
        "metadata": {"source_url": "https://example.test/paper", "source_kind": "paper"},
    }))
    row = {
        "question_id": "q1",
        "question": "What does Alpha report?",
        "status": "completed",
        "predicted_answer": "Alpha reports a measured result.",
        "predicted_citations": ["paper_a"],
        "predicted_evidence": [{"doc_id": "paper_a", "start": 0, "end": 32}],
        "answer": "GOLD_ANSWER_MUST_NOT_LEAK",
        "expected_citations": ["GOLD_DOC_MUST_NOT_LEAK"],
        "grader_notes": ["GOLD_GRADER_MUST_NOT_LEAK"],
        "reward": 1.0,
    }
    return corpus, row, text


def test_packet_materializes_exact_quote_and_excludes_question_gold(tmp_path):
    corpus, row, text = _fixture(tmp_path)
    packet = build_code_exec_packet(row, corpus)

    assert packet["retriever_protocol"] == "code-execution"
    assert packet["evidence"][0]["evidence_id"] == "E1"
    assert packet["evidence"][0]["quote"] == text[:32]
    assert packet["candidate_claims"] == [{
        "text": row["predicted_answer"], "evidence_ids": ["E1"],
    }]
    assert len(packet["corpus_hash"]) == 64
    assert "not semantic support" in packet["warning"]
    encoded = json.dumps(packet)
    assert "GOLD_ANSWER_MUST_NOT_LEAK" not in encoded
    assert "GOLD_DOC_MUST_NOT_LEAK" not in encoded
    assert "GOLD_GRADER_MUST_NOT_LEAK" not in encoded
    assert "reward" not in packet


@pytest.mark.parametrize("span", [
    {"doc_id": "paper_a", "start": 0, "end": 999},
    {"doc_id": "paper_a", "start": True, "end": 10},
    {"doc_id": "paper_a", "start": 0, "end": 10, "quote": "made up"},
    {"doc_id": "../outside", "start": 0, "end": 10},
])
def test_packet_rejects_unverifiable_offsets_and_documents(tmp_path, span):
    corpus, row, _ = _fixture(tmp_path)
    row["predicted_evidence"] = [span]
    with pytest.raises(ValueError):
        build_code_exec_packet(row, corpus)


def test_packet_caps_spans_and_quote_lengths(tmp_path):
    corpus, row, _ = _fixture(tmp_path)
    long_text = "x" * 12_000
    (corpus / "paper_a.json").write_text(json.dumps({
        "doc_id": "paper_a", "text": long_text, "metadata": {},
    }))
    row["predicted_evidence"] = [
        {"doc_id": "paper_a", "start": i * 1_500, "end": (i + 1) * 1_500}
        for i in range(6)
    ]
    packet = build_code_exec_packet(row, corpus)
    assert len(packet["evidence"]) == 5
    assert len(packet["evidence"][0]["quote"]) == 1_200
    assert packet["evidence"][0]["end"] == 1_200
    assert packet["evidence"][0]["quote_truncated"] is True
    assert packet["omitted_predicted_spans"] == 1


def test_packet_rejects_duplicate_spans(tmp_path):
    corpus, row, _ = _fixture(tmp_path)
    row["predicted_evidence"] = [row["predicted_evidence"][0]] * 2
    with pytest.raises(ValueError, match="duplicates"):
        build_code_exec_packet(row, corpus)


def test_packet_only_cli_runs_without_key_or_model_call(tmp_path, monkeypatch):
    corpus, row, _ = _fixture(tmp_path)
    transcript = tmp_path / "run.json"
    output = tmp_path / "packet.json"
    transcript.write_text(json.dumps([row]))
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(cli.EndpointExplainer, "explain", lambda *_: pytest.fail("model called"))

    assert cli.main([
        "--transcript", str(transcript), "--question-id", "q1",
        "--corpus", str(corpus), "--output", str(output), "--packet-only",
    ]) == 0
    assert json.loads(output.read_text())["question_id"] == "q1"


def test_live_explanation_refuses_to_call_without_nebius_key(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setattr(cli, "EndpointExplainer", lambda **_: pytest.fail("model initialized"))
    with pytest.raises(ValueError, match="NEBIUS_API_KEY"):
        cli.explain_with_nebius({"evidence": []})


def test_live_adapter_uses_nebius_key_and_restores_previous_explainer_key(
    tmp_path, monkeypatch,
):
    corpus, row, _ = _fixture(tmp_path)
    packet = build_code_exec_packet(row, corpus)
    monkeypatch.setenv("NEBIUS_API_KEY", "test-nebius-key")
    monkeypatch.setenv("EXPLAINER_API_KEY", "previous-key")

    def fake_explain(_self, received_packet):
        assert received_packet == packet
        assert os.environ["EXPLAINER_API_KEY"] == "test-nebius-key"
        return json.dumps({
            "answer": "A limited result [E1].",
            "claims": [{"text": "A limited result", "evidence_ids": ["E1"]}],
            "limitations": ["Support still needs human review."],
        })

    monkeypatch.setattr(cli.EndpointExplainer, "explain", fake_explain)
    result = cli.explain_with_nebius(packet)
    assert result["status"] == "submitted"
    assert result["checks"]["semantic_support"] == "not_reviewed"
    assert result["explainer"]["model"] == cli.DEFAULT_MODEL
    assert os.environ["EXPLAINER_API_KEY"] == "previous-key"
    assert "A limited result" in cli.explanation_markdown(result)
