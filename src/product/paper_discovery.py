"""Cached relevance, dated arXiv discovery, and citation traversal."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import urlencode
from uuid import uuid4

from src.product.arxiv_ids import validate_arxiv_id
from src.product.arxiv_relevance import search_relevance
from src.product.net import HttpGet, SourceUnavailable, Throttle, http_get
from src.research.papers import discover

S2_API = "https://api.semanticscholar.org/graph/v1"
FIELDS = "title,abstract,year,citationCount,externalIds,url"


def _paper(record: dict) -> dict:
    if not isinstance(record, dict):
        raise ValueError("paper is not an object")
    ids = record.get("externalIds")
    if ids is not None and not isinstance(ids, dict):
        raise ValueError("externalIds is not an object")
    arxiv_id = (ids or {}).get("ArXiv")
    try:
        validate_arxiv_id(arxiv_id)
    except ValueError:
        arxiv_id = None
    for key in ("title", "abstract", "url"):
        if record.get(key) is not None and not isinstance(record[key], str):
            raise ValueError(f"invalid {key}")
    for key in ("year", "citationCount"):
        value = record.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError(f"invalid {key}")
    return {
        "arxiv_id": arxiv_id, "title": record.get("title") or "",
        "abstract": record.get("abstract"), "year": record.get("year"),
        "citation_count": record.get("citationCount"), "openable": arxiv_id is not None,
        "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else record.get("url"),
    }


def _date(value: str | None, default: date) -> date:
    if value is None:
        return default
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value:
            return parsed
    except (TypeError, ValueError):
        pass
    raise ValueError("dates must use YYYY-MM-DD")


class PaperDiscovery:
    def __init__(self, cache_dir: Path, *, http: HttpGet = http_get, api_key: str | None = None,
                 s2_throttle: Throttle | None = None, arxiv_throttle: Throttle | None = None,
                 today: Callable[[], date] = date.today) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.http, self.today = http, today
        self.api_key = api_key if api_key is not None else os.environ.get(
            "SEMANTIC_SCHOLAR_API_KEY")
        self.s2_throttle, self.arxiv_throttle = s2_throttle, arxiv_throttle
        self._lock = threading.Lock()

    def _get(self, source: str, url: str, headers: dict) -> bytes:
        # Default transport owns the process-wide throttle, including redirects.
        # Optional caller throttles apply only to explicitly injected transports.
        throttle = self.arxiv_throttle if source == "arxiv" else self.s2_throttle
        if self.http is not http_get and throttle is not None:
            throttle.wait()
        return self.http(url, headers)

    def _cached(self, source: str, key: object, build: Callable[[], dict]) -> dict:
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{source}-{digest}.json"
        with self._lock:
            try:
                cached = json.loads(path.read_text())
                if isinstance(cached, dict) and isinstance(cached.get("papers"), list):
                    return cached
            except (OSError, ValueError):
                pass
            try:
                result = build()
            except (ValueError, TypeError, KeyError, ET.ParseError, AttributeError) as exc:
                raise SourceUnavailable(source, "invalid response") from exc
            temporary = path.with_suffix(f".{uuid4().hex}.tmp")
            try:
                temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            return result

    def _s2(self, path: str, params: dict) -> list[dict]:
        url = f"{S2_API}{path}?{urlencode(params)}"
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        payload = json.loads(self._get("semantic_scholar", url, headers))
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise ValueError("missing result data")
        return payload["data"]

    def search(self, query: str, since: str | None = None, until: str | None = None,
               limit: int = 10) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must contain 1..500 characters")
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("limit must be 1..50")
        query = query.strip()
        if since is not None or until is not None:
            return self._recent(query, since, until, limit)

        def build():
            records = self._s2("/paper/search", {"query": query, "limit": limit, "fields": FIELDS})
            return {"source": "semantic_scholar", "mode": "relevance",
                    "papers": [_paper(record) for record in records[:limit]],
                    "coverage_note": "Relevance-ranked titles and abstracts; not exhaustive."}

        return self._cached("semantic_scholar", ["search", query, limit], build)

    def _recent(self, query: str, since: str | None, until: str | None, limit: int) -> dict:
        end = _date(until, self.today())
        start = _date(since, end - timedelta(days=365))
        if start > end:
            raise ValueError("Require since <= until")

        def build():
            def fetch(url):
                body = self._get("arxiv", url, {})
                if ET.fromstring(body).tag != "{http://www.w3.org/2005/Atom}feed":
                    raise ValueError("response is not an Atom feed")
                return body

            found = discover(query, start.isoformat(), end.isoformat(), limit=limit,
                             downloader=fetch)
            papers = []
            for item in found["papers"][:limit]:
                identifier = validate_arxiv_id(item["arxiv_id"])
                papers.append({"arxiv_id": identifier, "title": item["title"],
                               "abstract": item["abstract"], "citation_count": None,
                               "year": int(item["submitted"][:4]) if item["submitted"] else None,
                               "url": f"https://arxiv.org/abs/{identifier}", "openable": True})
            return {"source": "arxiv", "mode": "recent", "since": start.isoformat(),
                    "until": end.isoformat(), "papers": papers,
                    "coverage_note": found["coverage_note"]}

        return self._cached("arxiv", [query, start.isoformat(), end.isoformat(), limit], build)

    def search_arxiv_relevance(self, query: str, *, limit: int = 5) -> dict:
        """Opt-in plain-text fallback; leaves existing recency searches unchanged."""
        return search_relevance(query, limit, get=self._get, cached=self._cached)

    def citations(self, arxiv_id: str, direction: str = "references", limit: int = 20) -> dict:
        validate_arxiv_id(arxiv_id)
        if direction not in {"references", "cited_by"}:
            raise ValueError("direction must be references or cited_by")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        endpoint = "references" if direction == "references" else "citations"

        def build():
            records = self._s2(f"/paper/arXiv:{arxiv_id.split('v')[0]}/{endpoint}",
                               {"fields": FIELDS, "limit": limit})
            key, papers = "citedPaper" if direction == "references" else "citingPaper", []
            for row in records[:limit]:
                if not isinstance(row, dict):
                    raise ValueError("citation is not an object")
                if row.get(key) is not None:
                    papers.append(_paper(row[key]))
            return {"arxiv_id": arxiv_id, "direction": direction, "papers": papers}

        return self._cached("semantic_scholar", ["citations", arxiv_id, direction, limit], build)
