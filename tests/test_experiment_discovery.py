"""A failed provider remains visible when date-sorted arXiv fallback succeeds."""
from datetime import date

import pytest

from src.product.experiment_discovery import ExperimentDiscovery
from src.product.net import SourceUnavailable


def test_fallback_retains_source_and_coverage():
    calls = []
    class Discovery:
        def search(self, query, **kwargs):
            calls.append((query, kwargs))
            if "since" not in kwargs:
                raise SourceUnavailable("semantic_scholar", "rate limited")
            return {"source": "arxiv", "papers": []}
    result = ExperimentDiscovery(Discovery(), today=lambda: date(2026, 10, 21)).search("cache")
    assert result["source"] == "arxiv" and result["fallback"]["from"] == "semantic_scholar"
    assert "date-sorted" in result["coverage_note"]
    assert calls[1][1] == {"since": "1991-01-01", "until": "2026-10-21", "limit": 5}


def test_failed_fallback_returns_source_failure():
    class Discovery:
        def search(self, query, **kwargs):
            raise SourceUnavailable("arxiv" if "since" in kwargs else "semantic_scholar", "down")
    with pytest.raises(SourceUnavailable, match="arxiv"):
        ExperimentDiscovery(Discovery()).search("cache")
