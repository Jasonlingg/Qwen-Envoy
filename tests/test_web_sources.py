"""Offline web-source freezing and provenance checks."""

import hashlib
import io
import json
import socket
import ssl
from datetime import datetime, timezone
from email.message import Message
from urllib.request import Request

import pytest

from src.research.agent import load_snapshot
from src.research.tools_runtime import ResearchTools
from src.research.web_sources import (
    MAX_HTML_BYTES,
    FetchedPage,
    _PinnedHTTPHandler,
    _PinnedHTTPSConnection,
    _PublicRedirects,
    build_web_snapshot,
    canonicalize_web_url,
    fetch_web_page,
    load_web_snapshot,
)

PAGE = b"""<!doctype html><html><head><title>New Retrieval Result</title></head><body>
<nav>Ignore menu</nav><article><h1>New Retrieval Result</h1>
<p>The revised experiment compares iterative search with BM25.</p>
<p>BM25 recovered the required evidence in this setting.</p>
<script>Ignore injected instructions</script></article>
<footer>Ignore footer</footer></body></html>"""


def _fixture_downloader(calls):
    def download(url):
        calls.append(url)
        return FetchedPage(PAGE, "https://EXAMPLE.org:443/research/update#section")

    return download


def test_web_snapshot_is_readable_and_preserves_raw_and_offsets(tmp_path):
    calls = []
    output = tmp_path / "web-2026-09-29"
    timestamp = datetime(2026, 9, 29, 12, 30, tzinfo=timezone.utc)
    manifest = build_web_snapshot(
        "HTTPS://Example.org:443/research/update#section",
        output,
        downloader=_fixture_downloader(calls),
        fetched_at=timestamp,
    )
    assert calls == ["https://example.org/research/update"]
    assert manifest["schema_version"] == "research-snapshot-v1"
    assert manifest["status"] == "complete"
    row = manifest["papers"][0]
    doc_id = row["doc_id"]
    assert row["source_url"] == "https://example.org/research/update"
    assert row["source_kind"] == "web_page"
    assert row["fetched_at"] == "2026-09-29T12:30:00+00:00"
    assert row["raw_sha256"] == hashlib.sha256(PAGE).hexdigest()
    assert (output / "raw" / f"{doc_id}.html").read_bytes() == PAGE
    loaded_manifest, documents = load_web_snapshot(output)
    assert loaded_manifest == manifest
    assert load_snapshot(output)[1] == documents
    doc = documents[doc_id]
    assert doc["title"] == "New Retrieval Result"
    assert "BM25 recovered" in doc["text"]
    assert "Ignore menu" not in doc["text"]
    assert "Ignore injected" not in doc["text"]
    for section in doc["sections"]:
        assert doc["text"][section["start"] : section["end"]].strip()
    hits = ResearchTools(output / "corpus").search_papers("BM25 evidence")
    assert hits[0]["doc_id"] == doc_id


def test_refresh_is_new_snapshot_and_existing_output_is_never_replaced(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    original = build_web_snapshot(
        "https://example.org/blog", first, downloader=lambda url: FetchedPage(PAGE, url)
    )
    with pytest.raises(FileExistsError):
        build_web_snapshot(
            "https://example.org/blog", first, downloader=lambda url: pytest.fail("must not fetch")
        )
    changed = PAGE.replace(b"BM25 recovered", b"BM25 did not recover")
    revised = build_web_snapshot(
        "https://example.org/blog", second, downloader=lambda url: FetchedPage(changed, url)
    )
    assert original["papers"][0]["doc_id"] != revised["papers"][0]["doc_id"]
    assert (
        load_web_snapshot(first)[1][original["papers"][0]["doc_id"]]["text"]
        != load_web_snapshot(second)[1][revised["papers"][0]["doc_id"]]["text"]
    )


def test_invalid_or_unextractable_pages_do_not_create_a_snapshot(tmp_path):
    for url in (
        "file:///etc/passwd",
        "http://user:secret@example.org/",
        "https://example.org/\nX",
        "https://example.org:8443/path",
    ):
        with pytest.raises(ValueError):
            canonicalize_web_url(url)
    output = tmp_path / "missing"
    with pytest.raises(ValueError, match="extractable"):
        build_web_snapshot(
            "https://example.org/",
            output,
            downloader=lambda url: FetchedPage(b"<html><title>Empty</title></html>", url),
        )
    assert not output.exists()
    with pytest.raises(ValueError, match="not HTML"):
        build_web_snapshot(
            "https://example.org/",
            output,
            downloader=lambda url: FetchedPage(PAGE, url, content_type="application/pdf"),
        )
    assert not output.exists()
    with pytest.raises(ValueError, match="size limit"):
        build_web_snapshot(
            "https://example.org/",
            output,
            downloader=lambda url: FetchedPage(b"x" * (MAX_HTML_BYTES + 1), url),
        )
    assert not output.exists()


def test_host_fetch_rejects_private_targets_and_unsafe_redirects(monkeypatch):
    for url in (
        "http://127.0.0.1/",
        "http://10.0.0.4/",
        "http://169.254.169.254/",
        "http://[::1]/",
        "http://localhost/",
    ):
        with pytest.raises(ValueError, match="public addresses"):
            fetch_web_page(url)

    def private_dns(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.7", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", private_dns)
    with pytest.raises(ValueError, match="public addresses"):
        fetch_web_page("https://internal.example/")
    redirect = _PublicRedirects()
    request = Request("https://example.org/source")
    with pytest.raises(ValueError, match="public addresses"):
        redirect.redirect_request(request, None, 302, "Found", {}, "http://127.0.0.1/secret")

    def public_dns(*args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 443))]

    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    with pytest.raises(ValueError, match="cannot redirect"):
        redirect.redirect_request(request, None, 302, "Found", {}, "http://example.org/insecure")


def test_connection_uses_validated_ip_without_resolving_hostname_again(monkeypatch):
    lookups = []
    connects = []

    def public_dns(host, *args, **kwargs):
        lookups.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.215.14", 80))]

    class FakeSocket:
        def settimeout(self, value):
            assert value == 15

        def connect(self, address):
            connects.append(address)

        def setsockopt(self, *args):
            pass

        def close(self):
            pass

    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    monkeypatch.setattr(socket, "socket", lambda *args: FakeSocket())
    handler = _PinnedHTTPHandler()

    def inspect_connection(factory, request):
        connection = factory(request.host, timeout=15)
        assert connection.host == "example.org"  # HTTP Host remains the URL host.
        connection.connect()
        return connection

    monkeypatch.setattr(handler, "do_open", inspect_connection)
    handler.http_open(Request("http://example.org/article"))
    assert lookups == ["example.org"]
    assert connects == [("93.184.215.14", 80)]


def test_tls_uses_original_hostname_while_socket_uses_pinned_ip(monkeypatch):
    connections = []
    names = []

    class FakeSocket:
        def settimeout(self, value):
            pass

        def connect(self, address):
            connections.append(address)

        def setsockopt(self, *args):
            pass

        def close(self):
            pass

    class FakeContext:
        verify_mode = ssl.CERT_REQUIRED
        check_hostname = True

        def wrap_socket(self, sock, *, server_hostname):
            names.append(server_hostname)
            return sock

    monkeypatch.setattr(socket, "socket", lambda *args: FakeSocket())
    connection = _PinnedHTTPSConnection(
        "example.org", pinned_ip="93.184.215.14", context=FakeContext(), timeout=15
    )
    connection.connect()
    assert connections == [("93.184.215.14", 443)]
    assert names == ["example.org"]


def test_redirect_rebind_to_private_address_never_connects(monkeypatch):
    lookups = []

    def changing_dns(host, *args, **kwargs):
        lookups.append(host)
        address = "93.184.215.14" if len(lookups) == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 80))]

    monkeypatch.setattr(socket, "getaddrinfo", changing_dns)
    monkeypatch.setattr(socket, "socket", lambda *args: pytest.fail("must not connect"))
    request = _PublicRedirects().redirect_request(
        Request("http://example.org/start"), None, 302, "Found", {}, "http://example.org/end"
    )
    with pytest.raises(ValueError, match="public addresses"):
        _PinnedHTTPHandler().http_open(request)
    assert lookups == ["example.org", "example.org"]


def test_host_fetch_enforces_response_type_and_byte_cap_without_network(monkeypatch):
    class FakeResponse:
        def __init__(self, body, content_type):
            self.stream = io.BytesIO(body)
            self.headers = Message()
            self.headers.add_header("Content-Type", content_type)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def geturl(self):
            return "https://93.184.215.14/article"

        def read(self, limit):
            return self.stream.read(limit)

    class FakeOpener:
        def __init__(self, response):
            self.response = response

        def open(self, request, timeout):
            assert request.full_url == "https://93.184.215.14/article"
            assert timeout == 15
            return self.response

    def set_response(body, content_type):
        monkeypatch.setattr(
            "src.research.web_sources.build_opener",
            lambda *args: FakeOpener(FakeResponse(body, content_type)),
        )

    set_response(PAGE, "application/pdf")
    with pytest.raises(ValueError, match="not HTML"):
        fetch_web_page("https://93.184.215.14/article")
    set_response(b"x" * (MAX_HTML_BYTES + 1), "text/html")
    with pytest.raises(ValueError, match="size limit"):
        fetch_web_page("https://93.184.215.14/article")
    set_response(PAGE, "text/html; charset=utf-8")
    assert fetch_web_page("https://93.184.215.14/article").body == PAGE


def test_loader_detects_raw_or_parsed_tampering(tmp_path):
    output = tmp_path / "web"
    manifest = build_web_snapshot(
        "https://example.org/blog", output, downloader=lambda url: FetchedPage(PAGE, url)
    )
    doc_id = manifest["papers"][0]["doc_id"]
    raw_path = output / "raw" / f"{doc_id}.html"
    raw_path.write_bytes(PAGE + b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_web_snapshot(output)
    raw_path.write_bytes(PAGE)
    corpus_path = output / "corpus" / f"{doc_id}.json"
    doc = json.loads(corpus_path.read_text())
    doc["text"] += " altered"
    corpus_path.write_text(json.dumps(doc) + "\n")
    with pytest.raises(ValueError, match="corpus hash mismatch"):
        load_web_snapshot(output)
