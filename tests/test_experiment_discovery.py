"""A failed provider remains visible when relevance fallback succeeds."""

import pytest

from src.product.experiment_discovery import ExperimentDiscovery
from src.product.net import SourceUnavailable


def test_fallback_retains_source_and_coverage():
    calls = []
    class Discovery:
        def search(self, query, **kwargs):
            calls.append((query, kwargs))
            raise SourceUnavailable("semantic_scholar", "rate limited")
        def search_arxiv_relevance(self, query, **kwargs):
            calls.append((query, kwargs))
            return {"source": "arxiv", "papers": [], "status": "no_candidates",
                    "coverage_note": "Lexical screening is not semantic support."}
    result = ExperimentDiscovery(Discovery()).search("cache")
    assert result["source"] == "arxiv" and result["fallback"]["from"] == "semantic_scholar"
    assert result["status"] == "no_candidates" and "not semantic" in result["coverage_note"]
    assert calls == [("cache", {"limit": 5}), ("cache", {"limit": 5})]


def test_failed_fallback_returns_source_failure():
    class Discovery:
        def search(self, query, **kwargs):
            raise SourceUnavailable("semantic_scholar", "down")
        def search_arxiv_relevance(self, query, **kwargs):
            raise SourceUnavailable("arxiv", "down")
    with pytest.raises(SourceUnavailable, match="arxiv"):
        ExperimentDiscovery(Discovery()).search("cache")


def test_primary_success_does_not_call_fallback():
    class Discovery:
        def search(self, query, **kwargs):
            return {"source": "semantic_scholar", "papers": []}
        def search_arxiv_relevance(self, *args, **kwargs):
            pytest.fail("fallback on successful primary search")
    assert ExperimentDiscovery(Discovery()).search("cache")["source"] == "semantic_scholar"


@pytest.mark.parametrize("since,until", [("2023-01-01", "2023-12-31"),
                                       ("2023-01-01", None), (None, "2023-12-31")])
def test_positional_source_contract_preserves_date_bounds_without_relevance_fallback(since, until):
    calls = []
    class Discovery:
        def search(self, query, since, until, limit):
            calls.append((query, since, until, limit))
            return {"source": "arxiv", "mode": "recent", "papers": []}
        def search_arxiv_relevance(self, *args, **kwargs):
            pytest.fail("date-bounded queries must not become undated")
    result = ExperimentDiscovery(Discovery()).search("cache", since, until, 7)
    assert result["mode"] == "recent" and calls == [("cache", since, until, 7)]


def test_positional_undated_source_contract_keeps_fallback_and_limit():
    calls = []
    class Discovery:
        def search(self, query, *, limit):
            calls.append(("primary", query, limit))
            raise SourceUnavailable("semantic_scholar", "rate limited")
        def search_arxiv_relevance(self, query, *, limit):
            calls.append(("fallback", query, limit))
            return {"source": "arxiv", "papers": []}
    result = ExperimentDiscovery(Discovery()).search("cache", None, None, 9)
    assert result["fallback"]["from"] == "semantic_scholar"
    assert calls == [("primary", "cache", 9), ("fallback", "cache", 9)]


def test_date_source_failure_is_not_replaced_by_undated_results():
    class Discovery:
        def search(self, *args):
            raise SourceUnavailable("arxiv", "down")
        def search_arxiv_relevance(self, *args, **kwargs):
            pytest.fail("must preserve the date-specific source failure")
    with pytest.raises(SourceUnavailable, match="arxiv"):
        ExperimentDiscovery(Discovery()).search("cache", "2023-01-01", None, 5)
