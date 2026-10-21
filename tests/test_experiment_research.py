"""Controlled end-to-end decisions: discovery, reading, refinement and partial evidence."""
import pytest

from src.product.net import SourceUnavailable
from tests.arxiv_fakes import article_page, make_library
from tests.experiment_fakes import FINISH, researcher
from tests.test_arxiv_research_mcp import FakeDiscovery

SEARCH = 'search_papers("cache memory")'
READ = 'read_paper("2609.12345", "Memory savings and setup?")'


def library(tmp_path):
    return make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})[0]


def test_complete_loop_shared_budget_and_provenance(tmp_path):
    worker, transport = researcher(library(tmp_path), FakeDiscovery(),
        [SEARCH, READ, "fixture inspect", "fixture submit", FINISH])
    result = worker.run("Reduce memory?", "Preserve quality on our GPU.")
    assert result["status"] == "brief_ready"
    assert result["budget"]["requests_dispatched"] == len(transport.calls) == 5
    assert result["budget"]["tokens_reserved"] == 5 * 8192
    assert result["brief"]["findings"][0]["evidence_ids"] == ["E1"]
    assert result["evidence"][0]["arxiv_id"] == "2609.12345v2"
    assert result["reader_runs"][0]["paper_assignment"]["counts_as_source_inspection"] is False
    assert result["semantic_support"] == "not_reviewed"
    assert "Preserve quality" in transport.calls[0]["messages"][1]["content"]


def test_discovery_cannot_be_bypassed_or_python_executed(tmp_path):
    lib, fetcher = make_library(tmp_path, {})
    worker, transport = researcher(lib, FakeDiscovery(), [READ, '__import__("os").getcwd()',
        'finish(findings=[], uncertainties=["Need sources"], proposed_test={})'])
    result = worker.run("Memory?")
    assert result["status"] == "no_evidence" and not fetcher.calls
    assert "exact ID" in result["trajectory"][0]["observation"]
    assert "literal" in result["trajectory"][1]["observation"]
    assert len(transport.calls) == 3


def test_search_failure_counts_and_query_can_be_refined(tmp_path):
    class Discovery(FakeDiscovery):
        def search(self, query, **kwargs):
            if query == "cache memory":
                raise SourceUnavailable("semantic_scholar", "rate limited")
            return super().search(query, **kwargs)
    worker, _ = researcher(library(tmp_path), Discovery(),
        [SEARCH, 'search_papers("memory eviction")', READ, "inspect", "submit", FINISH])
    result = worker.run("Memory?")
    assert result["status"] == "brief_ready" and len(result["searches"]) == 2
    assert "rate limited" in result["searches"][0]["error"]


def test_empty_screened_discovery_is_visible_to_controller_and_saved_trace(tmp_path):
    class Discovery(FakeDiscovery):
        def search(self, query, **kwargs):
            return {"source": "arxiv", "status": "no_candidates", "papers": [],
                    "coverage_note": "Refine query; not evidence that no papers exist."}
    worker, transport = researcher(library(tmp_path), Discovery(), [SEARCH,
        'finish(findings=[], uncertainties=["Discovery gap"], proposed_test={})'])
    result = worker.run("Memory?")
    assert result["status"] == "no_evidence" and result["evidence"] == []
    assert result["searches"][0]["status"] == "no_candidates"
    feedback = transport.calls[1]["messages"][-1]["content"]
    assert '"status": "no_candidates"' in feedback and "not evidence" in feedback


@pytest.mark.parametrize("ending", [TimeoutError("outcome unknown"), FINISH])
def test_partial_evidence_survives_budget_or_inference_failure(tmp_path, ending):
    limit = 4 if ending == FINISH else 18
    worker, transport = researcher(library(tmp_path), FakeDiscovery(),
        [SEARCH, READ, "inspect", "submit", ending], requests_max=limit)
    result = worker.run("Memory?")
    assert result["evidence"] and result["brief"] is None
    if limit == 4:
        assert result["status"] == "budget_exhausted" and len(transport.calls) == 4
    else:
        assert "outcome unknown" in result["error"] and len(transport.calls) == 5


def test_failed_second_paper_keeps_first_papers_evidence(tmp_path):
    class Discovery(FakeDiscovery):
        def search(self, query, **kwargs):
            return {"papers": [{"arxiv_id": "2609.12345"}, {"arxiv_id": "2609.54321"}]}
    worker, _ = researcher(library(tmp_path), Discovery(),
        [SEARCH, READ, "inspect", "submit", 'read_paper("2609.54321", "Compare")', FINISH])
    result = worker.run("Memory?")
    assert result["status"] == "brief_ready" and len(result["evidence"]) == 1
    assert result["papers"][1]["status"] == "source_unavailable"
    assert result["paper_read_attempts"] == 2


def test_rejected_finish_can_be_corrected(tmp_path):
    worker, _ = researcher(library(tmp_path), FakeDiscovery(),
        [SEARCH, READ, "inspect", "submit", FINISH.replace('"E1"', '"E99"'), FINISH])
    result = worker.run("Memory?")
    assert result["status"] == "brief_ready" and len(result["trajectory"]) == 4
    assert "returned evidence" in result["trajectory"][-2]["observation"]


def test_three_search_limit_includes_failures(tmp_path):
    worker, transport = researcher(library(tmp_path), FakeDiscovery(fail=True),
        [f'search_papers("query {i}")' for i in range(8)])
    result = worker.run("Memory?")
    assert len(result["searches"]) == 3 and len(transport.calls) == 8
    assert result["status"] == "incomplete"


def test_transport_value_error_stops_without_a_second_dispatch(tmp_path):
    worker, transport = researcher(library(tmp_path), FakeDiscovery(),
        [ValueError("invalid provider JSON"), FINISH])
    result = worker.run("Memory?")
    assert result["status"] == "inference_failed" and len(transport.calls) == 1
    assert result["budget"]["transport_failed"]


def test_uncertain_reader_dispatch_stops_control_but_keeps_prior_evidence(tmp_path):
    class Discovery(FakeDiscovery):
        def search(self, query, **kwargs):
            return {"papers": [{"arxiv_id": "2609.12345"}, {"arxiv_id": "2609.54321"}]}
    lib, _ = make_library(tmp_path, {
        "2609.12345v2": {"html": article_page()}, "2609.54321v1": {"html": article_page()}})
    worker, transport = researcher(lib, Discovery(), [SEARCH, READ, "inspect", "submit",
        'read_paper("2609.54321", "Compare")', ValueError("invalid provider JSON"), FINISH])
    result = worker.run("Memory?")
    assert result["status"] == "inference_failed" and len(transport.calls) == 6
    assert result["evidence"] and result["papers"][1]["status"] == "error"


def test_returned_malformed_completion_can_be_corrected_within_budget(tmp_path):
    worker, transport = researcher(library(tmp_path), FakeDiscovery(), [{"choices": []},
        'finish(findings=[], uncertainties=["Need evidence"], proposed_test={})'])
    result = worker.run("Memory?")
    assert result["status"] == "no_evidence" and len(transport.calls) == 2
    assert not result["budget"]["transport_failed"]
