import json
from datetime import date

import pytest

from src.product.net import SourceUnavailable, Throttle
from src.product.paper_discovery import PaperDiscovery

S2 = {"data": [
    {"title": "KV Cache Compression", "abstract": "We compress.", "year": 2026,
     "citationCount": 12, "externalIds": {"ArXiv": "2609.12345"}, "url": "https://s2/p1"},
    {"title": "Journal paper", "abstract": None, "year": 2025, "citationCount": 3,
     "externalIds": {"DOI": "10.1/x"}, "url": "https://s2/p2"},
]}
ATOM = ('<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        '<id>http://arxiv.org/abs/2610.00001v1</id><title>Recent Cache Paper</title>'
        '<summary>New results.</summary><published>2026-10-10T00:00:00Z</published>'
        '<updated>2026-10-10T00:00:00Z</updated></entry></feed>')


class FakeHttp:
    def __init__(self, respond):
        self.respond, self.calls = respond, []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        reply = self.respond(url)
        if isinstance(reply, Exception):
            raise reply
        return reply


def discovery(tmp_path, respond=None, **kwargs):
    http = FakeHttp(respond or (lambda url: json.dumps(S2).encode()))
    result = PaperDiscovery(tmp_path / "cache", http=http, s2_throttle=Throttle(0),
                            arxiv_throttle=Throttle(0), **kwargs)
    return result, http


def test_relevance_search_openability_key_and_cache(tmp_path):
    store, http = discovery(tmp_path, api_key="private")
    result = store.search("kv cache compression", limit=2)
    assert (result["source"], result["mode"]) == ("semantic_scholar", "relevance")
    assert [paper["openable"] for paper in result["papers"]] == [True, False]
    assert result["papers"][0]["url"] == "https://arxiv.org/abs/2609.12345"
    assert http.calls[0][1] == {"x-api-key": "private"}
    assert store.search("kv cache compression", limit=2) == result
    assert len(http.calls) == 1
    assert "private" not in next((tmp_path / "cache").iterdir()).read_text()


def test_cached_results_serve_when_source_fails(tmp_path):
    store, _ = discovery(tmp_path)
    store.search("cache")
    store.http = FakeHttp(lambda url: SourceUnavailable("semantic_scholar", "rate limited"))
    assert store.search("cache")["papers"]
    with pytest.raises(SourceUnavailable):
        store.search("different")


def test_recency_is_cached_and_uses_effective_dates_in_key(tmp_path):
    store, http = discovery(tmp_path, lambda url: ATOM.encode(),
                            today=lambda: date(2026, 10, 16))
    result = store.search("cache", since="2026-10-01")
    assert (result["source"], result["mode"], result["until"]) == ("arxiv", "recent", "2026-10-16")
    assert result["papers"][0]["arxiv_id"] == "2610.00001v1"
    assert http.calls[0][0].startswith("https://export.arxiv.org/api/query?")
    assert store.search("cache", since="2026-10-01") == result
    assert len(http.calls) == 1
    store.search("cache", since="2026-10-02")
    assert len(http.calls) == 2


def test_citations_both_directions_are_cached(tmp_path):
    def response(url):
        key = "citedPaper" if "/references" in url else "citingPaper"
        return json.dumps({"data": [{key: S2["data"][0]}]}).encode()

    store, http = discovery(tmp_path, response)
    assert store.citations("2609.99999v2")["papers"][0]["openable"]
    assert store.citations("2609.99999v2", "cited_by")["papers"]
    store.citations("2609.99999v2")
    assert len(http.calls) == 2
    assert "/paper/arXiv:2609.99999/references" in http.calls[0][0]


def test_invalid_external_arxiv_ids_are_not_openable(tmp_path):
    payload = {"data": [{"title": "Paper", "externalIds": {"ArXiv": "bad/../id"}}]}
    store, _ = discovery(tmp_path, lambda url: json.dumps(payload).encode())
    paper = store.search("cache")["papers"][0]
    assert paper["arxiv_id"] is None and paper["openable"] is False


@pytest.mark.parametrize("payload", [
    b"not json", b"[]", b"{}", b'{"data":{}}', b'{"data":[1]}',
    b'{"data":[{"title":5}]}', b'{"data":[{"externalIds":[]}]}',
    b'{"data":[{"year":true}]}', b'{"data":[{"citationCount":-1}]}',
])
def test_malformed_provider_data_has_source_error(tmp_path, payload):
    store, _ = discovery(tmp_path, lambda url: payload)
    with pytest.raises(SourceUnavailable, match="invalid response") as info:
        store.search("cache")
    assert info.value.source == "semantic_scholar"


@pytest.mark.parametrize("payload", [b"broken", b"<feed>", ATOM.replace("2610", "2613").encode()])
def test_malformed_recency_response_has_source_error(tmp_path, payload):
    store, _ = discovery(tmp_path, lambda url: payload)
    with pytest.raises(SourceUnavailable, match="invalid response") as info:
        store.search("cache", since="2026-10-01", until="2026-10-16")
    assert info.value.source == "arxiv"


@pytest.mark.parametrize("kwargs", [
    {"query": ""}, {"query": "x" * 501}, {"query": "q", "limit": 0},
    {"query": "q", "limit": 51}, {"query": "q", "limit": True},
    {"query": "q", "since": "20261001"}, {"query": "q", "since": ""},
    {"query": "q", "since": "2026-10-17", "until": "2026-10-01"},
])
def test_input_validation_precedes_network(tmp_path, kwargs):
    store, http = discovery(tmp_path)
    with pytest.raises(ValueError):
        store.search(**kwargs)
    assert not http.calls


def test_citation_validation_precedes_network(tmp_path):
    store, http = discovery(tmp_path)
    for args in [("bad/id",), ("2609.12345", "sideways"), ("2609.12345", "references", 101)]:
        with pytest.raises(ValueError):
            store.citations(*args)
    assert not http.calls


def test_corrupted_cache_is_refetched(tmp_path):
    store, http = discovery(tmp_path)
    store.search("cache")
    next((tmp_path / "cache").iterdir()).write_text("broken")
    assert store.search("cache")["papers"]
    assert len(http.calls) == 2
