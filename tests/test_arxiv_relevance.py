"""Exercise fallback retrieval through the real query, transport and cache wiring."""
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from xml.sax.saxutils import escape

import pytest

from src.product.experiment_discovery import ExperimentDiscovery
from src.product.net import SourceUnavailable
from src.product.paper_discovery import PaperDiscovery


def feed(papers):
    entries = "".join(f'<entry><id>http://arxiv.org/abs/{p[0]}</id>'
                      f'<title>{escape(p[1])}</title><summary>{escape(p[2])}</summary>'
                      '<published>2023-09-12T00:00:00Z</published></entry>' for p in papers)
    return f'<feed xmlns="http://www.w3.org/2005/Atom">{entries}</feed>'.encode()


def fallback(tmp_path, body):
    calls = []
    def http(url, headers):
        calls.append(url)
        if urlsplit(url).hostname == "api.semanticscholar.org":
            raise SourceUnavailable("semantic_scholar", "rate limited")
        return body
    store = PaperDiscovery(tmp_path / "cache", http=http)
    return store, ExperimentDiscovery(store), calls


def test_recorded_rate_limit_uses_fielded_relevance_and_screened_candidates(tmp_path):
    original = Path("examples/experiment-research-live/run/library/discovery-cache/"
                    "arxiv-b3cc732bcccd58aee52fd005128274724be54e06943bc35974292f3719e6927c.json")
    old = json.loads(original.read_text())["papers"]
    papers = [(p["arxiv_id"], p["title"], p["abstract"]) for p in old]
    papers += [("2309.06180v1", "Efficient Memory Management with PagedAttention",
                "PagedAttention manages KV cache allocation and memory fragmentation.")]
    _, discovery, calls = fallback(tmp_path, feed(papers))
    result = discovery.search(
        "paged attention kv cache allocation concurrent readers memory measurements")
    assert result["fallback"] == {"from": "semantic_scholar", "detail": "rate limited"}
    assert [p["arxiv_id"] for p in result["papers"]] == ["2309.06180v1"]
    assert result["mode"] == "relevance" and result["candidates_examined"] == 6
    params = parse_qs(urlsplit(calls[1]).query)
    assert params["sortBy"] == ["relevance"] and params["max_results"] == ["50"]
    assert 'ti:"pagedattention"' in params["search_query"][0]
    assert 'abs:"allocation"' in params["search_query"][0]
    assert "submittedDate" not in params["search_query"][0]
    assert len(calls) == 2  # one primary and one fallback; no automatic resubmission


def test_general_compound_names_and_rank_order_are_not_paper_specific(tmp_path):
    _, discovery, _ = fallback(tmp_path, feed([
        ("2309.00001v1", "Attention", "Attention computation."),
        ("2309.00002v1", "Fast Method", "FlashAttention supports exact attention."),
        ("2309.00003v1", "FlashAttention exact attention", "IO aware computation."),
    ]))
    papers = discovery.search("Flash Attention exact attention IO-aware", limit=2)["papers"]
    assert [p["arxiv_id"] for p in papers] == ["2309.00003v1", "2309.00002v1"]


def test_unrelated_single_word_matches_return_actionable_empty_result(tmp_path):
    _, discovery, _ = fallback(tmp_path, feed([
        ("2309.00001v1", "Memory for robots", "Embodied control."),
        ("2309.00002v1", "Quantum phase transitions", "Matter at low temperatures."),
    ]))
    result = discovery.search("zzunknownmethod memory")
    assert result["status"] == "no_candidates" and result["papers"] == []
    assert "refine" in result["coverage_note"] and "not exhaustive" in result["coverage_note"]


def test_relevance_cache_never_reuses_recent_cache_and_keeps_attribution(tmp_path):
    store, discovery, calls = fallback(tmp_path, feed([
        ("2309.00001v1", "Cache memory", "Memory allocation.")]))
    store.search("cache memory", since="2026-10-01", until="2026-10-21")
    result = discovery.search("cache memory")
    assert result["mode"] == "relevance" and len(calls) == 3
    assert discovery.search("cache memory") == result
    assert len(calls) == 4  # primary failure repeated; cached fallback needs no HTTP


@pytest.mark.parametrize("body", [b"broken", b"<html/>", feed([("error", "Error", "Bad query")])])
def test_bad_feed_is_source_unavailable(tmp_path, body):
    store, _, _ = fallback(tmp_path, body)
    with pytest.raises(SourceUnavailable, match="invalid response"):
        store.search_arxiv_relevance("cache memory")


@pytest.mark.parametrize("query,limit", [("", 5), ("and or", 5), ("q"*501, 5),
    ("cache", 0), ("cache", 51), ("cache", True)])
def test_invalid_relevance_input_precedes_network(tmp_path, query, limit):
    store, _, calls = fallback(tmp_path, b"")
    with pytest.raises(ValueError):
        store.search_arxiv_relevance(query, limit=limit)
    assert calls == []


def test_plain_text_cannot_inject_arxiv_boolean_syntax(tmp_path):
    store, _, calls = fallback(tmp_path, feed([]))
    store.search_arxiv_relevance('cache") OR cat:physics ANDNOT abs:memory')
    query = parse_qs(urlsplit(calls[0]).query)["search_query"][0]
    assert 'cat:physics' not in query and 'ANDNOT' not in query
    assert 'ti:"physics"' in query and 'abs:"memory"' in query


def test_topic_phrase_in_title_precedes_incidental_extra_word_matches(tmp_path):
    _, discovery, _ = fallback(tmp_path, feed([
        ("2309.00001v1", "Memory management", "Alpha Beta cache allocation measurements."),
        ("2309.00002v1", "AlphaBeta memory management", "AlphaBeta cache allocation."),
    ]))
    result = discovery.search("alpha beta cache allocation measurements", limit=1)
    assert result["papers"][0]["arxiv_id"] == "2309.00002v1"


def test_supported_candidates_survive_unsupported_legacy_paper(tmp_path):
    store, _, _ = fallback(tmp_path, feed([
        ("hep-th/9901001v1", "Cache memory", "An old paper."),
        ("2309.00002v1", "Cache memory", "Memory allocation."),
    ]))
    result = store.search_arxiv_relevance("cache memory")
    assert [p["arxiv_id"] for p in result["papers"]] == ["2309.00002v1"]
    assert result["unsupported_legacy_ids"] == ["hep-th/9901001v1"]
    assert result["candidates_examined"] == 2


def test_legacy_only_results_are_an_explicit_coverage_gap(tmp_path):
    store, _, _ = fallback(tmp_path, feed([("math/0309136", "Cache memory", "Older work.")]))
    result = store.search_arxiv_relevance("cache memory")
    assert result["status"] == "no_candidates" and result["papers"] == []
    assert result["unsupported_legacy_ids"] == ["math/0309136"]


def test_malformed_legacy_identifier_still_fails_closed(tmp_path):
    store, _, _ = fallback(tmp_path, feed([("hep-th/9913001v1", "Cache memory", "Bad month.")]))
    with pytest.raises(SourceUnavailable, match="invalid response"):
        store.search_arxiv_relevance("cache memory")
