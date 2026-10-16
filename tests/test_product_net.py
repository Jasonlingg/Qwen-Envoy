import io
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

import src.product.net as net


class Response:
    def __init__(self, body=b"ok", headers=None, url="https://arxiv.org/test"):
        self.body, self.headers, self.url = body, headers or {}, url
        self.read_sizes = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def geturl(self):
        return self.url

    def read(self, size):
        self.read_sizes.append(size)
        return self.body[:size]


class Opener:
    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def open(self, request, timeout):
        self.calls.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def redirect(url, target, code=302):
    return HTTPError(url, code, "redirect", {"Location": target}, io.BytesIO())


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(net, "ARXIV_THROTTLE", net.Throttle(0))
    monkeypatch.setattr(net, "S2_THROTTLE", net.Throttle(0))


def install(monkeypatch, *replies):
    opener = Opener(*replies)
    monkeypatch.setattr(net, "build_opener", lambda *handlers: opener)
    return opener


def test_throttle_spaces_calls():
    now, slept = [100.0], []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    throttle = net.Throttle(3.1, clock=lambda: now[0], sleep=sleep)
    throttle.wait()
    now[0] += 1
    throttle.wait()
    assert slept == pytest.approx([2.1])


@pytest.mark.parametrize("url", [
    "https://example.com/paper", "http://arxiv.org/abs/2609.12345",
    "https://user:pass@arxiv.org/x", "https://arxiv.org:444/x",
    "https://arxiv.org.example.com/x", "https://arxiv.org./x", "https://arxiv.org/\nx",
])
def test_refuses_unsafe_urls_before_network(monkeypatch, url):
    opener = install(monkeypatch)
    with pytest.raises(ValueError, match="non-allowlisted"):
        net.http_get(url)
    assert not opener.calls


def test_refuses_redirect_before_following(monkeypatch):
    opener = install(monkeypatch, redirect("https://arxiv.org/test", "https://evil.test/pdf"))
    with pytest.raises(ValueError, match="non-allowlisted"):
        net.http_get("https://arxiv.org/test")
    assert len(opener.calls) == 1
    assert net._NoRedirect().redirect_request(Request("https://arxiv.org"), None,
                                             302, "redirect", {}, "/x") is None


def test_each_redirect_uses_process_throttle_and_drops_private_headers(monkeypatch):
    waits = []

    class Counter:
        def wait(self):
            waits.append("arxiv")

    monkeypatch.setattr(net, "ARXIV_THROTTLE", Counter())
    opener = install(monkeypatch, redirect("https://arxiv.org/test", "https://export.arxiv.org/x"),
                     Response(url="https://export.arxiv.org/x"))
    assert net.http_get("https://arxiv.org/test", {"x-api-key": "private"}) == b"ok"
    assert waits == ["arxiv", "arxiv"]
    assert "X-api-key" not in opener.calls[1].headers
    assert opener.calls[1].headers["Accept-encoding"] == "identity"


def test_size_cap_bounds_read_and_content_length(monkeypatch):
    response = Response(body=b"123456")
    install(monkeypatch, response)
    with pytest.raises(net.SourceUnavailable, match="size limit"):
        net.http_get("https://arxiv.org/test", max_bytes=5)
    assert response.read_sizes == [6]
    for length in ("6", "-1", "invalid"):
        response = Response(headers={"Content-Length": length})
        install(monkeypatch, response)
        with pytest.raises(net.SourceUnavailable):
            net.http_get("https://arxiv.org/test", max_bytes=5)
        assert response.read_sizes == []


def test_redirect_hop_limit(monkeypatch):
    opener = install(monkeypatch, *(redirect("https://arxiv.org/test", "/test")
                                   for _ in range(net.MAX_REDIRECTS + 1)))
    with pytest.raises(net.SourceUnavailable, match="redirect limit"):
        net.http_get("https://arxiv.org/test")
    assert len(opener.calls) == net.MAX_REDIRECTS + 1


@pytest.mark.parametrize("error, detail", [
    (HTTPError("https://api.semanticscholar.org/x", 429, "limited", {}, None), "rate limited"),
    (HTTPError("https://api.semanticscholar.org/x", 503, "unavailable", {}, None), "HTTP 503"),
    (URLError("offline"), "unreachable"),
])
def test_errors_are_source_unavailable(monkeypatch, error, detail):
    install(monkeypatch, error)
    with pytest.raises(net.SourceUnavailable, match=detail) as info:
        net.http_get("https://api.semanticscholar.org/x")
    assert info.value.source == "semantic_scholar"


def test_pdf_uses_same_redirect_guard_and_five_mb_bound(monkeypatch):
    opener = install(monkeypatch, redirect("https://arxiv.org/pdf/2609.12345", "http://arxiv.org/p"))
    with pytest.raises(ValueError, match="non-allowlisted"):
        net.http_get("https://arxiv.org/pdf/2609.12345", max_bytes=5_000_000)
    assert len(opener.calls) == 1
