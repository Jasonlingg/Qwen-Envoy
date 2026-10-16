import pytest

from src.product.parallel_readers import investigate_papers
from tests.arxiv_fakes import FakeInvestigator, article_page, make_library

QUOTE_A = "Eviction saves 40 percent of memory"
QUOTE_B = "We compress key-value caches."


def _library(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()},
                                         "2609.54321v1": {}})
    library.open("2609.12345v2")
    library.open("2609.54321v1")
    return library


def test_each_paper_gets_its_own_reader_and_evidence_is_merged_in_order(tmp_path):
    library = _library(tmp_path)
    reader = FakeInvestigator({"2609.12345v2": QUOTE_A, "2609.54321v1": QUOTE_B})
    merged = investigate_papers(reader, library, "How much memory?",
                                ["2609.54321v1", "2609.12345v2"])
    assert sorted(reader.calls) == ["2609.12345v2", "2609.54321v1"]
    assert merged["status"] == "evidence_found"
    assert [e["arxiv_id"] for e in merged["evidence"]] == ["2609.54321v1", "2609.12345v2"]
    assert [e["evidence_id"] for e in merged["evidence"]] == ["E1", "E2"]
    assert len(merged["candidate_answers"]) == 2


def test_one_unverifiable_packet_does_not_sink_the_others(tmp_path):
    library = _library(tmp_path)
    reader = FakeInvestigator({"2609.12345v2": QUOTE_A, "2609.54321v1": "TAMPER"})
    merged = investigate_papers(reader, library, "q", ["2609.12345v2", "2609.54321v1"])
    by_paper = {p["arxiv_id"]: p for p in merged["papers"]}
    assert by_paper["2609.54321v1"]["status"] == "unverified"
    assert merged["status"] == "evidence_found" and len(merged["evidence"]) == 1


def test_all_no_evidence(tmp_path):
    merged = investigate_papers(FakeInvestigator({}), _library(tmp_path), "q",
                                ["2609.12345v2", "2609.54321v1"])
    assert merged["status"] == "no_evidence" and merged["evidence"] == []


def test_input_limits(tmp_path):
    library = _library(tmp_path)
    with pytest.raises(ValueError):
        investigate_papers(FakeInvestigator({}), library, "", ["2609.12345v2"])
    with pytest.raises(ValueError):
        investigate_papers(FakeInvestigator({}), library, "q", ["2609.12345v2"] * 6)


def test_reader_exception_retains_other_paper_evidence(tmp_path):
    library = _library(tmp_path)
    class Reader(FakeInvestigator):
        def investigate(self, question, snapshot):
            from src.research.agent import load_snapshot
            _, docs = load_snapshot(snapshot)
            if next(iter(docs.values()))['metadata']['arxiv_id'] == '2609.54321v1':
                raise TimeoutError('reader timed out')
            return super().investigate(question, snapshot)
    result = investigate_papers(Reader({'2609.12345v2': QUOTE_A}), library, 'q',
                                ['2609.12345v2', '2609.54321v1'])
    assert result['status'] == 'evidence_found' and len(result['evidence']) == 1
    assert result['papers'][1]['status'] == 'error'
    assert 'TimeoutError' in result['papers'][1]['error']


def test_readers_actually_run_concurrently(tmp_path):
    import threading
    barrier = threading.Barrier(2)
    class Reader(FakeInvestigator):
        def investigate(self, question, snapshot):
            barrier.wait(timeout=5)
            return super().investigate(question, snapshot)
    result = investigate_papers(Reader({'2609.12345v2': QUOTE_A, '2609.54321v1': QUOTE_B}),
                                _library(tmp_path), 'q', ['2609.12345v2', '2609.54321v1'],
                                max_workers=2)
    assert len(result['evidence']) == 2


def test_snapshot_failure_is_isolated(tmp_path, monkeypatch):
    library = _library(tmp_path)
    original = library.view
    def view(ids):
        if ids == ['2609.54321v1']:
            raise OSError('snapshot unavailable')
        return original(ids)
    monkeypatch.setattr(library, 'view', view)
    result = investigate_papers(FakeInvestigator({'2609.12345v2': QUOTE_A}), library, 'q',
                                ['2609.12345v2', '2609.54321v1'])
    assert len(result['evidence']) == 1 and result['papers'][1]['status'] == 'error'
