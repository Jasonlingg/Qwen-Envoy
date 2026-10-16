"""Bounded, redirect-safe HTTPS reads with process-wide provider throttles."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from http.client import HTTPException
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

ALLOWED_HOSTS = frozenset({"arxiv.org", "export.arxiv.org", "api.semanticscholar.org"})
USER_AGENT = "Envoy-arxiv-research-mcp/0.1"
MAX_RESPONSE_BYTES = 5_000_000
MAX_REDIRECTS = 5
HttpGet = Callable[[str, dict[str, str]], bytes]


class SourceUnavailable(RuntimeError):  # noqa: N818 - public source-status API
    """A bounded source request failed; tools should return the source and detail."""

    def __init__(self, source: str, detail: str) -> None:
        super().__init__(f"{source}: {detail}")
        self.source, self.detail = source, detail


class Throttle:
    """Space request starts across threads; one instance serves each provider."""

    def __init__(self, min_interval: float, clock=time.monotonic, sleep=time.sleep) -> None:
        self.min_interval, self._clock, self._sleep = min_interval, clock, sleep
        self._lock = threading.Lock()
        self._last: float | None = None

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            if self._last is not None:
                delay = self._last + self.min_interval - now
                if delay > 0:
                    self._sleep(delay)
            self._last = self._clock()


# All default discovery, metadata, HTML, and PDF reads share these exact objects.
ARXIV_THROTTLE = Throttle(3.1)
S2_THROTTLE = Throttle(1.1)


def _validate_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        valid = (isinstance(url, str) and url == url.strip()
                 and not any(ord(char) < 32 or ord(char) == 127 for char in url)
                 and parts.scheme == "https" and parts.hostname in ALLOWED_HOSTS
                 and parts.username is None and parts.password is None
                 and parts.port in (None, 443))
    except (TypeError, ValueError, AttributeError):
        valid = False
    if not valid:
        raise ValueError(f"refusing to fetch non-allowlisted URL: {url!r}")
    return url


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        # Return the redirect as HTTPError so the caller validates and throttles
        # the next hop itself, before any connection to its target is attempted.
        return None


def _read(response, source: str, max_bytes: int) -> bytes:
    _validate_url(response.geturl())
    declared = response.headers.get("Content-Length")
    if declared is not None:
        try:
            length = int(declared)
            if length < 0:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise SourceUnavailable(source, "invalid Content-Length") from exc
        if length > max_bytes:
            raise SourceUnavailable(source, "response exceeds size limit")
    body = response.read(max_bytes + 1)
    if len(body) > max_bytes:
        raise SourceUnavailable(source, "response exceeds size limit")
    return body


def http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 30.0,
             *, max_bytes: int = 2_000_000) -> bytes:
    """Read a bounded HTTPS body; allowlist and throttle every redirect including PDFs."""
    current = _validate_url(url)
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_RESPONSE_BYTES:
        raise ValueError(f"max_bytes must be 1..{MAX_RESPONSE_BYTES}")
    request_headers = {**(headers or {}), "User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    for hop in range(MAX_REDIRECTS + 1):
        parts = urlsplit(current)
        source = "semantic_scholar" if parts.hostname == "api.semanticscholar.org" else "arxiv"
        (S2_THROTTLE if source == "semantic_scholar" else ARXIV_THROTTLE).wait()
        try:
            request = Request(current, headers=request_headers)
            with opener.open(request, timeout=timeout) as response:
                return _read(response, source, max_bytes)
        except HTTPError as exc:
            target = exc.headers.get("Location") if exc.headers is not None else None
            code = exc.code
            exc.close()
            if code not in {301, 302, 303, 307, 308} or not target:
                detail = "rate limited" if code == 429 else f"HTTP {code}"
                raise SourceUnavailable(source, detail) from exc
            if hop == MAX_REDIRECTS:
                raise SourceUnavailable(source, "redirect limit exceeded") from exc
            following = _validate_url(urljoin(current, target))
            if urlsplit(following).hostname != parts.hostname:
                request_headers = {key: value for key, value in request_headers.items()
                                   if key.lower() in {"user-agent", "accept", "accept-encoding"}}
            current = following
        except (URLError, OSError, HTTPException) as exc:
            raise SourceUnavailable(source, f"unreachable: {exc}") from exc
    raise AssertionError("redirect loop should always return or fail")
