"""Offline arXiv and reader stand-ins shared by the arXiv MCP tests."""

from __future__ import annotations

import hashlib
import threading
from html import escape

from src.env.tools import SEARCH_PROTOCOL_VERSION
from src.product.arxiv_library import ArxivLibrary
from src.product.net import SourceUnavailable, Throttle
from src.product.packet_verifier import TOOL_PREAMBLE_SHA256
from src.research.agent import load_snapshot


def abs_page(version: str, title: str = "KV Cache Compression",
             abstract: str = "We compress key-value caches.") -> bytes:
    return (f'<meta property="og:url" content="https://arxiv.org/abs/{version}">'
            f'<meta name="citation_title" content="{escape(title, quote=True)}">'
            f'<meta name="citation_abstract" content="{escape(abstract, quote=True)}">'
            '<meta name="citation_author" content="Lee, A.">'
            '<meta name="citation_date" content="2026/09/20">').encode()


def article_page() -> bytes:
    intro = "".join(f"<p>Paragraph {i} studies the cache eviction policy.</p>" for i in range(5))
    return (f"<article><h2>Introduction</h2>{intro}<h2>Results</h2>"
            "<p>Eviction saves 40 percent of memory at equal accuracy.</p></article>").encode()


def tiny_pdf(lines: list[str]) -> bytes:
    lines = [line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
             for line in lines]
    content = ("BT /F1 12 Tf 72 720 Td "
               + " ".join(f"({line}) Tj 0 -16 Td" for line in lines) + " ET")
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = b"%PDF-1.4\n", []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets).encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    return out


class FakeArxiv:
    """papers: version -> {html: bytes | None, pdf: bytes | None}."""

    def __init__(self, papers: dict[str, dict]) -> None:
        self.papers = papers
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def fetch(self, url: str) -> bytes:
        with self._lock:
            self.calls.append(url)
        for version in sorted(self.papers, key=lambda item: int(item.split("v")[1]), reverse=True):
            paper = self.papers[version]
            if url in (f"https://arxiv.org/abs/{version.split('v')[0]}",
                       f"https://arxiv.org/abs/{version}"):
                return abs_page(version, paper.get("title", "KV Cache Compression"),
                                paper.get("abstract", "We compress key-value caches."))
            if url == f"https://arxiv.org/html/{version}" and paper.get("html") is not None:
                return paper["html"]
        raise SourceUnavailable("arxiv", "HTTP 404")

    def fetch_pdf(self, url: str) -> bytes:
        with self._lock:
            self.calls.append(url)
        for version, paper in self.papers.items():
            if url == f"https://arxiv.org/pdf/{version}" and paper.get("pdf") is not None:
                return paper["pdf"]
        raise SourceUnavailable("arxiv", "HTTP 404")


def make_library(tmp_path, papers: dict[str, dict]) -> tuple[ArxivLibrary, FakeArxiv]:
    fake = FakeArxiv(papers)
    library = ArxivLibrary(tmp_path / "library", fetch=fake.fetch, fetch_pdf=fake.fetch_pdf,
                           throttle=Throttle(0))
    return library, fake


class FakeInvestigator:
    """Returns a packet quoting quotes[arxiv_id]; TAMPER returns a bad quote."""

    system_prompt = "fake reader prompt"

    def __init__(self, quotes: dict[str, str]) -> None:
        self.quotes = quotes
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def investigate(self, question: str, snapshot) -> dict:
        manifest, docs = load_snapshot(snapshot)
        (doc_id, doc), = docs.items()
        arxiv_id = doc["metadata"]["arxiv_id"]
        with self._lock:
            self.calls.append(arxiv_id)
        base = {
            "corpus_hash": manifest["corpus_hash"], "source_domain": "public_papers",
            "system_prompt_sha256": hashlib.sha256(self.system_prompt.encode()).hexdigest(),
            "tool_search_version": SEARCH_PROTOCOL_VERSION,
            "tool_preamble_sha256": TOOL_PREAMBLE_SHA256,
            "trajectory": [], "model_requests_attempted": 2,
        }
        quote = self.quotes.get(arxiv_id)
        if quote is None:
            return {**base, "status": "no_evidence", "evidence": [], "candidate_claims": []}
        if quote == "TAMPER":
            span = {"doc_id": doc_id, "start": 0, "end": 5, "quote": "wrong"}
        else:
            start = doc["text"].index(quote)
            span = {"doc_id": doc_id, "start": start, "end": start + len(quote), "quote": quote}
        return {**base, "status": "evidence_found", "evidence": [span],
                "candidate_claims": [{"text": f"Answer from {arxiv_id}"}]}
