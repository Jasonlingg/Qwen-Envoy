"""Explicit relevance fallback for experiment research, retaining source attribution."""

from src.product.net import SourceUnavailable


class ExperimentDiscovery:
    def __init__(self, discovery):
        self.discovery = discovery

    def search(self, query, since=None, until=None, limit=5):
        # The source MCP passes these four arguments positionally. Date-bounded
        # searches already use arXiv; never replace them with an undated fallback.
        if since is not None or until is not None:
            return self.discovery.search(query, since, until, limit)
        try:
            return self.discovery.search(query, limit=limit)
        except SourceUnavailable as exc:
            # The same arXiv allowlist, size bounds and process-wide throttle apply.
            result = self.discovery.search_arxiv_relevance(query, limit=limit)
            return {**result, "fallback": {"from": exc.source, "detail": exc.detail}}

    def citations(self, *args, **kwargs):
        return self.discovery.citations(*args, **kwargs)
