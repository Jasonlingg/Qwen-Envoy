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
