"""QASPER provenance and fixed retrieval cannot be confused with reference support."""
import copy
import hashlib

import pytest

from src.eval.qasper_passage_smoke import retrieve, select_questions
from src.product.paper_assignment import QASPER_ASSIGNMENT_VERSION, paper_assignment


def document():
    revision = "a" * 40
    manifest = {"corpus_hash": "b" * 64, "source_dataset": "allenai/qasper",
                "source_revision": revision}
    doc = {"title": "Frozen paper", "metadata": {"qasper_paper_id": "1808.10059",
        "source_dataset": "allenai/qasper", "source_revision": revision,
        "coverage": "qasper_s2orc_text", "source_kind": "research_paper"}}
    return manifest, {"qasper_1808_10059": doc}


def test_qasper_assignment_keeps_dataset_identity_not_fake_arxiv_version():
    manifest, docs = document()
    result = paper_assignment(manifest, docs)
    assert result["version"] == QASPER_ASSIGNMENT_VERSION
    assert result["source_revision"] == manifest["source_revision"]
    assert result["counts_as_source_inspection"] is False
    assert "arxiv_id" not in result


@pytest.mark.parametrize("field,value", [
    ("qasper_paper_id", "another"), ("source_dataset", "arxiv"),
    ("source_revision", "c" * 40), ("source_kind", "arxiv_paper")])
def test_qasper_assignment_rejects_mismatched_identity(field, value):
    manifest, docs = document()
    docs["qasper_1808_10059"]["metadata"][field] = value
    with pytest.raises(ValueError, match="provenance"):
        paper_assignment(manifest, docs)


def test_qasper_requires_revision_in_manifest_and_document():
    manifest, docs = document()
    del manifest["source_revision"]
    with pytest.raises(ValueError, match="provenance"):
        paper_assignment(manifest, docs)
    manifest, docs = document()
    del docs["qasper_1808_10059"]["metadata"]["source_revision"]
    with pytest.raises(ValueError, match="provenance"):
        paper_assignment(manifest, docs)


def test_selection_is_deterministic_stratified_and_reference_blind():
    questions = [{"id": str(i), "source_paper_id": str(i),
        "expected_answerability": "sufficient" if i < 6 else "insufficient"}
        for i in range(12)]
    actual = select_questions(list(reversed(questions)))
    ordered = sorted(questions, key=lambda q: hashlib.sha256(
        ("42:" + q["id"]).encode()).hexdigest())
    assert actual == [q for q in ordered if q["expected_answerability"] == "sufficient"][:2] + [
        q for q in ordered if q["expected_answerability"] == "insufficient"][:1]
    altered = copy.deepcopy(questions)
    for q in altered:
        q.update(answer="changed", gold_evidence=[{"start": 0}], reference_answers=["new"])
    assert [q["id"] for q in select_questions(altered)] == [q["id"] for q in actual]


def test_selection_never_substitutes_an_answerability_stratum():
    with pytest.raises(ValueError):
        select_questions([{"id": str(i), "source_paper_id": "same",
            "expected_answerability": "sufficient"} for i in range(4)])


def test_retrieval_scans_end_of_full_paper_and_uses_disjoint_bounded_windows():
    text = ("background " * 1000) + ("needle latency " * 100)
    selected, ranking = retrieve(text, "needle latency")
    assert ranking[0]["end"] > 10000
    assert ranking[0]["start"] in [w["start"] for w in selected]
    assert len(selected) == 2 and selected[0]["end"] <= selected[1]["start"]
    assert all(w["end"] - w["start"] == 1600 for w in selected)
    assert max(w["end"] for w in ranking) == len(text)
    assert retrieve(text, "needle latency") == (selected, ranking)


def test_short_paper_fails_without_replacement():
    with pytest.raises(ValueError):
        retrieve("tiny", "q")


def test_prepare_uses_source_paper_for_unanswerable_rows_and_keeps_gold_out(tmp_path, monkeypatch):
    import json

    import scripts.try_qasper_passages as runner
    from src.eval.artifacts import content_hash

    corpus = tmp_path / "original"
    corpus.mkdir()
    questions = []
    for i, label in enumerate(["sufficient", "sufficient", "insufficient"]):
        paper = f"1808.1005{i}"
        doc_id = "qasper_" + paper.replace(".", "_")
        doc = {"doc_id": doc_id, "title": "Dataset paper", "text": "word " * 1000,
            "sections": [{"section": "Body", "start": 0, "end": 5000}],
            "metadata": {"source_dataset": "allenai/qasper", "qasper_paper_id": paper,
                "coverage": "qasper_s2orc_text", "source_kind": "research_paper"}}
        (corpus / (doc_id + ".json")).write_text(json.dumps(doc))
        questions.append({"id": f"question_{i}", "source_paper_id": paper,
            "expected_answerability": label, "source_question": "what word?",
            "required_doc_ids": [doc_id] if label == "sufficient" else [],
            "reference_answers": ["SECRET GOLD"], "gold_evidence": ["SECRET GOLD"]})
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(json.dumps({"questions": questions}))
    snapshot = {"source_dataset": "allenai/qasper", "source_revision": "a" * 40,
                "source_split": "test", "corpus_hash": content_hash(corpus)}
    monkeypatch.setattr(runner, "load_split", lambda *a: (
        {"questions": questions}, snapshot, benchmark, corpus, {}))
    monkeypatch.setattr(runner.PersistentREPL, "_docker_available", lambda *a: True)
    before = content_hash(corpus)
    output = tmp_path / "ready"
    runner.prepare(output)
    cases = json.loads((output / "cases.json").read_text())
    assert len(cases) == 3
    assert "SECRET GOLD" not in (output / "cases.json").read_text()
    for case in cases:
        manifest = json.loads((output / case["snapshot"] / "manifest.json").read_text())
        doc = json.loads(next((output / case["snapshot"] / "corpus").glob("*.json")).read_text())
        assert paper_assignment(manifest, {doc["doc_id"]: doc})["source_revision"] == "a" * 40
    assert content_hash(corpus) == before
    assert "SECRET GOLD" in (output / "review-input.json").read_text()
    assert runner.check(output)["limits"]["model_requests"] == 12
