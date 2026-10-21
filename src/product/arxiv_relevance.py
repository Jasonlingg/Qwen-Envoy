"""Bounded plain-text arXiv relevance queries; metadata matching is not support."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from urllib.parse import urlencode

from src.product.arxiv_ids import validate_arxiv_id

VERSION = "plain-relevance-v2"
CANDIDATES_MAX = 50
STOP_WORDS = frozenset("""a an and andnot are as at be by can could do does for from
    how i in is it of on or our should that the their these this to use using was
    we what when where which with would you your""".split())


def _words(value):
    return re.findall(r"[^\W_]+", value.casefold())


def query_terms(query):
    if not isinstance(query, str) or not query.strip() or len(query) > 500:
        raise ValueError("query must contain 1..500 characters")
    terms = list(dict.fromkeys(word for word in _words(query)
                               if word not in STOP_WORDS and len(word) >= 2))
    if not terms:
        raise ValueError("query needs searchable words of at least two characters")
    return terms


def _field(term):
    # Terms contain only letters/digits; user Boolean syntax is never interpolated.
    return f'(ti:"{term}" OR abs:"{term}")'


def build_query(terms):
    # Joined adjacent words cover compound names such as Flash Attention/FlashAttention
    # without a question-specific alias dictionary or hardcoded paper IDs.
    joined = [left + right for left, right in zip(terms, terms[1:])]
    return " OR ".join(_field(term) for term in dict.fromkeys([*terms, *joined]))


def _tokens(value):
    words = _words(value)
    return set(words) | {a + b for a, b in zip(words, words[1:])}


def _matches(terms, value):
    available = _tokens(value)
    matched = {term for term in terms if term in available}
    for left, right in zip(terms, terms[1:]):
        if left + right in available:
            matched.update((left, right))
    return matched


def _papers(body):
    root = ET.fromstring(body)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    if root.tag != "{http://www.w3.org/2005/Atom}feed":
        raise ValueError("response is not an Atom feed")
    papers, seen, legacy = [], set(), []
    for entry in root.findall("a:entry", ns)[:CANDIDATES_MAX]:
        url = entry.findtext("a:id", "", ns)
        identifier = re.fullmatch(r"https?://arxiv\.org/abs/(.+)", url)
        if identifier is None:
            raise ValueError("invalid arXiv entry URL")
        identifier = identifier[1]
        old = re.fullmatch(r"[a-z][a-z-]*(?:\.[A-Z]{2})?/(\d{2})(\d{2})\d{3}"
                           r"(?:v[1-9]\d*)?", identifier)
        if old and 1 <= int(old[2]) <= 12:
            legacy.append(identifier)  # reader supports modern IDs only; retain other matches
            continue
        identifier = validate_arxiv_id(identifier)
        title = " ".join(entry.findtext("a:title", "", ns).split())
        abstract = " ".join(entry.findtext("a:summary", "", ns).split())
        published = entry.findtext("a:published", "", ns)
        if not title:
            raise ValueError("paper title is missing")
        base = identifier.split("v")[0]
        if base in seen:
            continue
        seen.add(base)
        papers.append({"arxiv_id": identifier, "title": title, "abstract": abstract,
                       "citation_count": None, "year": int(published[:4]) if published else None,
                       "url": f"https://arxiv.org/abs/{identifier}", "openable": True})
    return papers, legacy


def search_relevance(query, limit, *, get, cached):
    terms = query_terms(query)
    if type(limit) is not int or not 1 <= limit <= 50:
        raise ValueError("limit must be 1..50")

    def build():
        url = "https://export.arxiv.org/api/query?" + urlencode({
            "search_query": build_query(terms), "sortBy": "relevance",
            "sortOrder": "descending", "start": 0, "max_results": CANDIDATES_MAX})
        candidates, legacy = _papers(get("arxiv", url, {}))
        ranked = []
        for paper in candidates:
            matched = _matches(terms, paper["title"] + " " + paper["abstract"])
            # Reject incidental single-word matches for a multi-word question.
            # This is a lexical screen, never a semantic relevance guarantee.
            if len(matched) >= min(2, len(terms)):
                leading_phrase = int(len(terms) > 1
                                     and "".join(terms[:2]) in _tokens(paper["title"]))
                ranked.append((leading_phrase, len(matched),
                               len(_matches(terms, paper["title"])), paper))
        ranked.sort(key=lambda item: item[:3], reverse=True)  # provider order breaks ties
        papers = [paper for _, _, _, paper in ranked[:limit]]
        return {"source": "arxiv", "mode": "relevance", "status": "candidates"
                if papers else "no_candidates", "query": query, "query_terms": terms,
                "query_url": url, "papers": papers,
                "candidates_examined": len(candidates) + len(legacy),
                "unsupported_legacy_ids": legacy,
                "candidates_after_screen": len(ranked),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "coverage_note": "arXiv relevance fallback: up to 50 title/abstract candidates, "
                "prioritizing the leading two-word phrase/compound in titles, then term "
                "coverage, title coverage and provider rank. Unsupported legacy IDs skipped "
                "and listed separately; the reader accepts modern IDs only. "
                "Lexical matches are not semantic support; search is not exhaustive. "
                "No candidates means refine the query or report the discovery gap."}

    # Separate namespace prevents old date-sorted cache entries from being reused.
    return cached("arxiv", [VERSION, query, limit], build)
