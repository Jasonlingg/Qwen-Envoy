"""Explicit provider fallback for bounded experiment research, retaining attribution."""
from datetime import date

from src.product.net import SourceUnavailable


class ExperimentDiscovery:
    def __init__(self, discovery, *, today=date.today):
        self.discovery, self.today = discovery, today

    def search(self, query, *, limit=5):
        try:
            return self.discovery.search(query, limit=limit)
        except SourceUnavailable as exc:
            # The same arXiv allowlist, size bounds and process-wide throttle apply.
            # Use the full arXiv era, not a recent-only window that hides foundational work.
            result = self.discovery.search(query, since="1991-01-01",
                                           until=self.today().isoformat(), limit=limit)
            return {**result, "fallback": {"from": exc.source, "detail": exc.detail},
                    "coverage_note": "arXiv date-sorted fallback; incomplete coverage"}

    def citations(self, *args, **kwargs):
        return self.discovery.citations(*args, **kwargs)
