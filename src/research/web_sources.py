"""Freeze a user-selected HTML page before a research worker can inspect it.

Fetching is a host-side operation. The resulting snapshot uses the existing
``research-snapshot-v1`` document shape, but the paper-only harness must opt in
to web sources explicitly; this module does not give a model network access.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import socket
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from src.eval.artifacts import content_hash
from src.research.agent import load_snapshot

PARSER_VERSION = "web-html-v1"
MAX_HTML_BYTES = 2_000_000
MAX_PDF_BYTES = 5_000_000
_BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"}
_HIDDEN = {
    "script",
    "style",
    "noscript",
    "svg",
    "template",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
}


@dataclass(frozen=True)
class FetchedPage:
    """The exact response bytes and resolved URL returned by the host fetcher."""

    body: bytes
    final_url: str
    content_type: str = "text/html"
    charset: str = "utf-8"


def canonicalize_web_url(url: str) -> str:
    """Normalize a source identity without trusting page-declared canonicals."""
    if not isinstance(url, str) or not url or url != url.strip():
        raise ValueError("URL must be a non-empty absolute web URL")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise ValueError("URL contains control characters")
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    if scheme not in {"http", "https"} or not parts.hostname:
        raise ValueError("Only absolute HTTP(S) URLs are supported")
    if parts.username is not None or parts.password is not None:
        raise ValueError("Credentials in source URLs are not supported")
    try:
        host = parts.hostname.encode("idna").decode("ascii").lower().rstrip(".")
        port = parts.port
    except (UnicodeError, ValueError) as exc:
        raise ValueError("Invalid source URL host or port") from exc
    if not host:
        raise ValueError("Source URL has no host")
    try:
        ipaddress.IPv6Address(host)
        host = f"[{host}]"
    except ipaddress.AddressValueError:
        pass
    default_port = 443 if scheme == "https" else 80
    if port not in (None, default_port):
        raise ValueError("Only the default HTTP(S) ports are supported")
    authority = host
    return urlunsplit((scheme, authority, parts.path or "/", parts.query, ""))


def _public_endpoint(url: str) -> tuple[str, str]:
    """Return a normalized URL and one validated address for its next connection."""
    normalized = canonicalize_web_url(url)
    parts = urlsplit(normalized)
    host = parts.hostname
    assert host is not None
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        raise ValueError("Source URL must resolve only to public addresses")
    try:
        addresses = [ipaddress.ip_address(host)]
    except ValueError:
        try:
            answers = socket.getaddrinfo(
                host,
                parts.port or (443 if parts.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise ValueError("Could not resolve source URL host") from exc
        addresses = [ipaddress.ip_address(answer[4][0]) for answer in answers]
    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("Source URL must resolve only to public addresses")
    # Prefer IPv4 when both families are offered; never fall back to an
    # address that was not in this fully validated DNS response.
    chosen = next((address for address in addresses if address.version == 4), addresses[0])
    return normalized, str(chosen)


def _require_public_host(url: str) -> str:
    return _public_endpoint(url)[0]


class _PinnedConnection:
    """Use the checked numeric IP for the socket, retaining the URL hostname."""

    def __init__(self, host: str, *, pinned_ip: str, **kwargs):
        super().__init__(host, **kwargs)
        self._pinned_ip = pinned_ip
        # HTTPConnection sets an instance-level _create_connection in its
        # constructor. Replace it so its normal connect/TLS path stays intact.
        self._create_connection = self._connect_to_pinned_ip

    def _connect_to_pinned_ip(self, address, timeout, source_address):
        family = (
            socket.AF_INET6
            if ipaddress.ip_address(self._pinned_ip).version == 6
            else socket.AF_INET
        )
        connection = socket.socket(family, socket.SOCK_STREAM)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                connection.settimeout(timeout)
            if source_address is not None:
                connection.bind(source_address)
            connection.connect((self._pinned_ip, address[1]))
            return connection
        except BaseException:
            connection.close()
            raise


class _PinnedHTTPConnection(_PinnedConnection, http.client.HTTPConnection):
    pass


class _PinnedHTTPSConnection(_PinnedConnection, http.client.HTTPSConnection):
    pass


class _PinnedHTTPHandler(HTTPHandler):
    def http_open(self, request):
        _, pinned_ip = _public_endpoint(request.full_url)
        return self.do_open(
            lambda host, **kwargs: _PinnedHTTPConnection(host, pinned_ip=pinned_ip, **kwargs),
            request,
        )


class _PinnedHTTPSHandler(HTTPSHandler):
    def https_open(self, request):
        _, pinned_ip = _public_endpoint(request.full_url)
        return self.do_open(
            lambda host, **kwargs: _PinnedHTTPSConnection(host, pinned_ip=pinned_ip, **kwargs),
            request,
            context=self._context,
            check_hostname=self._check_hostname,
        )


class _PublicRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        checked = _require_public_host(urljoin(request.full_url, newurl))
        if urlsplit(request.full_url).scheme == "https" and urlsplit(checked).scheme != "https":
            raise ValueError("HTTPS source cannot redirect to HTTP")
        return super().redirect_request(request, fp, code, msg, headers, checked)


def _fetch_public_resource(
    url: str, *, content_types: set[str], max_bytes: int, accept: str
) -> FetchedPage:
    """Fetch one bounded public response on the host, never in Qwen code.

    Each connection is pinned to the public IP from its validation lookup;
    the URL hostname remains the TLS verification name. A public deployment
    should also use network-level egress restrictions as defense in depth.
    """
    checked = _require_public_host(url)
    opener = build_opener(
        ProxyHandler({}), _PublicRedirects(), _PinnedHTTPHandler(), _PinnedHTTPSHandler()
    )
    request = Request(
        checked,
        headers={
            "User-Agent": "Envoy-research/0.1",
            "Accept": accept,
            "Accept-Encoding": "identity",
        },
    )
    with opener.open(request, timeout=15) as response:
        final_url = _require_public_host(response.geturl())
        content_type = response.headers.get_content_type()
        if content_type not in content_types:
            raise ValueError(
                "Source response is not HTML"
                if accept.startswith("text/html")
                else "Source response is not a PDF"
            )
        body = response.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError("Source exceeds the size limit")
        charset = response.headers.get_content_charset() or "utf-8"
    return FetchedPage(body=body, final_url=final_url, content_type=content_type, charset=charset)


def fetch_web_page(url: str) -> FetchedPage:
    """Fetch one public HTML page with pinned public IPs and a 2 MB bound."""
    return _fetch_public_resource(
        url,
        content_types={"text/html", "application/xhtml+xml"},
        max_bytes=MAX_HTML_BYTES,
        accept="text/html,application/xhtml+xml",
    )


def fetch_public_pdf(url: str) -> FetchedPage:
    """Fetch one public PDF with the same redirect/IP checks as HTML sources."""
    result = _fetch_public_resource(
        url,
        content_types={"application/pdf", "application/x-pdf", "application/octet-stream"},
        max_bytes=MAX_PDF_BYTES,
        accept="application/pdf",
    )
    if not result.body.startswith(b"%PDF-"):
        raise ValueError("Source response is not a PDF")
    return result


class _VisibleHTML(HTMLParser):
    """Conservative extraction of visible headings and text blocks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.title_parts: list[str] = []
        self.blocks: list[tuple[str, str, str]] = []
        self.current_tag: str | None = None
        self.current_parts: list[str] = []
        self.current_scope = "body"

    def _scope(self) -> str:
        if "article" in self.stack:
            return "article"
        if "main" in self.stack:
            return "main"
        return "body"

    def _finish(self) -> None:
        if self.current_tag is None:
            return
        value = " ".join("".join(self.current_parts).split())
        if value:
            self.blocks.append((self.current_scope, self.current_tag, value))
        self.current_tag = None
        self.current_parts = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _BLOCKS and self.current_tag is not None:
            self._finish()
        self.stack.append(tag)
        if tag in _BLOCKS and not set(self.stack).intersection(_HIDDEN):
            self.current_tag = tag
            self.current_scope = self._scope()
            self.current_parts = []
        elif tag == "br" and self.current_tag is not None:
            self.current_parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "br" and self.current_tag is not None:
            self.current_parts.append(" ")

    def handle_data(self, data: str) -> None:
        if "title" in self.stack:
            self.title_parts.append(data)
        if self.current_tag is not None and not set(self.stack).intersection(_HIDDEN):
            self.current_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == self.current_tag:
            self._finish()
        if tag in self.stack:
            # HTML in the wild is sometimes unbalanced; discard nested tags
            # through the matching close instead of retaining a hidden scope.
            index = len(self.stack) - 1 - self.stack[::-1].index(tag)
            del self.stack[index:]


def extract_html_text(body: bytes, charset: str = "utf-8") -> tuple[str, str, list[dict], bool]:
    """Return title, canonical snapshot text, offset sections, decode-loss flag."""
    try:
        decoded = body.decode(charset)
        decode_loss = False
    except (LookupError, UnicodeDecodeError):
        decoded = body.decode("utf-8", errors="replace")
        decode_loss = True
    parser = _VisibleHTML()
    parser.feed(decoded)
    parser.close()
    parser._finish()
    preferred = next(
        (
            scope
            for scope in ("article", "main", "body")
            if any(block[0] == scope for block in parser.blocks)
        ),
        "body",
    )
    blocks = [block for block in parser.blocks if block[0] == preferred]
    title = " ".join("".join(parser.title_parts).split())
    if not title:
        title = next((value for _, tag, value in blocks if tag == "h1"), "")
    if not title or not blocks:
        raise ValueError("HTML page has no extractable title or text blocks")
    text, sections, heading = "", [], title
    for _, tag, value in blocks:
        if text:
            text += "\n\n"
        start = len(text)
        text += value
        if tag.startswith("h"):
            heading = value
        sections.append({"section": heading, "start": start, "end": len(text)})
    return title, text, sections, decode_loss


def build_web_snapshot(
    url: str,
    output: Path,
    *,
    downloader: Callable[[str], FetchedPage] = fetch_web_page,
    fetched_at: datetime | None = None,
) -> dict:
    """Create a new immutable single-page snapshot in the research schema.

    ``downloader`` is injectable for offline replay. Fetch/parse completes
    before the exclusive output directory is made, so invalid pages leave no
    misleading partial snapshot. Existing output is never replaced.
    """
    requested_url = canonicalize_web_url(url)
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    page = downloader(requested_url)
    if not isinstance(page, FetchedPage) or not isinstance(page.body, bytes):
        raise TypeError("downloader must return FetchedPage with bytes body")
    if len(page.body) > MAX_HTML_BYTES:
        raise ValueError("HTML source exceeds the size limit")
    if page.content_type not in {"text/html", "application/xhtml+xml"}:
        raise ValueError("Source response is not HTML")
    canonical_url = canonicalize_web_url(page.final_url)
    title, text, sections, decode_loss = extract_html_text(page.body, page.charset)
    timestamp = fetched_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("fetched_at must include a timezone")
    fetched_iso = timestamp.astimezone(timezone.utc).isoformat()
    raw_hash = hashlib.sha256(page.body).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc_id = (
        "web_"
        + hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:16]
        + "_"
        + raw_hash[:16]
    )
    metadata = {
        "source_url": canonical_url,
        "canonical_url": canonical_url,
        "requested_url": requested_url,
        "source_kind": "web_page",
        "submitted": "unknown",
        "fetched_at": fetched_iso,
        "source_revision": raw_hash,
        "source_sha256": raw_hash,
        "text_sha256": text_hash,
        "content_type": page.content_type,
        "charset": page.charset,
        "coverage": "visible_html_blocks_partial" if decode_loss else "visible_html_blocks",
        "parser_version": PARSER_VERSION,
        "authority_note": "A web page's text and publication claims are unverified source content.",
    }
    document = {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "sections": sections,
        "metadata": metadata,
    }
    output.mkdir(parents=True, exist_ok=False)
    corpus_dir, raw_dir = output / "corpus", output / "raw"
    corpus_dir.mkdir()
    raw_dir.mkdir()
    corpus_path = corpus_dir / f"{doc_id}.json"
    raw_path = raw_dir / f"{doc_id}.html"
    corpus_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raw_path.write_bytes(page.body)
    manifest = {
        "schema_version": "research-snapshot-v1",
        "parser_version": PARSER_VERSION,
        "retrieved_at": fetched_iso,
        "requested_urls": [requested_url],
        "papers": [
            {
                "doc_id": doc_id,
                "title": title,
                **metadata,
                "sha256": content_hash(corpus_path),
                "raw_sha256": raw_hash,
            }
        ],
        "failures": [],
        "coverage_note": "Selected web page, not an exhaustive or live web search. "
        "HTML extraction omits images, tables, scripts, and linked pages.",
        "status": "complete",
        "corpus_hash": content_hash(corpus_dir),
        "raw_sources_hash": raw_hash,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_web_snapshot(snapshot: Path) -> tuple[dict, dict[str, dict]]:
    """Check both frozen response bytes and parsed corpus before use."""
    snapshot = Path(snapshot)
    manifest, docs = load_snapshot(snapshot)
    rows = manifest.get("papers")
    if (
        manifest.get("schema_version") != "research-snapshot-v1"
        or manifest.get("parser_version") != PARSER_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(rows, list)
        or len(rows) != 1
        or len(docs) != 1
    ):
        raise ValueError("Expected one complete web snapshot")
    row = rows[0]
    doc_id = row.get("doc_id")
    if not isinstance(doc_id, str) or doc_id not in docs:
        raise ValueError("Web snapshot document identity mismatch")
    corpus_path = snapshot / "corpus" / f"{doc_id}.json"
    raw_path = snapshot / "raw" / f"{doc_id}.html"
    if not corpus_path.is_file() or not raw_path.is_file():
        raise ValueError("Web snapshot is missing raw or parsed source")
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    doc = docs[doc_id]
    metadata = doc.get("metadata", {})
    expected_id = (
        "web_"
        + hashlib.sha256(row["source_url"].encode("utf-8")).hexdigest()[:16]
        + "_"
        + raw_hash[:16]
    )
    if (
        doc_id != expected_id
        or content_hash(corpus_path) != row.get("sha256")
        or raw_hash != row.get("raw_sha256")
        or raw_hash != manifest.get("raw_sources_hash")
        or raw_hash != metadata.get("source_sha256")
        or hashlib.sha256(doc["text"].encode("utf-8")).hexdigest() != metadata.get("text_sha256")
        or metadata.get("source_url") != row.get("source_url")
        or metadata.get("source_kind") != "web_page"
    ):
        raise ValueError("Web snapshot raw or parsed source hash mismatch")
    return manifest, docs
