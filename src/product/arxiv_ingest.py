"""Build bounded arXiv documents: HTML paragraphs, PDF text, then abstract only."""

from __future__ import annotations

import re
from collections.abc import Callable

from src.product.arxiv_ids import validate_arxiv_id
from src.research.file_sources import MAX_FILE_BYTES, MAX_TEXT_CHARS, _pdf_document
from src.research.papers import PaperTextParser, make_document, parse_metadata

Fetch = Callable[[str], bytes]
MAX_HTML_BYTES = 2_000_000
_HEADING = re.compile(
    r"^(?:[1-9][0-9]*(?:\.[0-9]+)*\.?\s+[A-Z][^\n]{1,100}|"
    r"Abstract|Introduction|Background|Related Work|Methods?|Results|Discussion|"
    r"Conclusions?|References|[A-Z][A-Z :\-]{3,100})$", re.M)


def _bounded(fetch: Fetch, url: str, limit: int) -> bytes:
    raw = fetch(url)
    if not isinstance(raw, bytes):
        raise ValueError("arXiv source must be bytes")
    if not raw or len(raw) > limit:
        raise ValueError(f"arXiv source exceeds its {limit} byte limit or is empty")
    return raw


def _document(metadata: dict, paragraphs: list[dict], coverage: str) -> dict:
    document = make_document(metadata, paragraphs, coverage)
    if len(document["text"]) > MAX_TEXT_CHARS:
        raise ValueError("Extracted text exceeds the 250,000 character limit")
    return document


def _pdf_paragraphs(text: str) -> list[dict]:
    """Keep extracted heading lines in the text; heuristics are an outline only."""
    starts = [(match.start(), match[0]) for match in _HEADING.finditer(text)]
    if not starts or starts[0][0] != 0:
        starts.insert(0, (0, "PDF text"))
    return [{"section": heading, "text": text[start:end]}
            for (start, heading), end in zip(starts, [s[0] for s in starts[1:]] + [len(text)])
            if text[start:end].strip()]


def ingest_paper(arxiv_id: str, fetch: Fetch, fetch_pdf: Fetch) -> tuple[dict, dict]:
    """The metadata page must load and pin the exact requested version, if given."""
    arxiv_id = validate_arxiv_id(arxiv_id)
    page = _bounded(fetch, f"https://arxiv.org/abs/{arxiv_id}", MAX_HTML_BYTES)
    metadata = parse_metadata(page.decode("utf-8"))
    version = validate_arxiv_id(metadata["arxiv_id"])
    if "v" not in version or arxiv_id.split("v")[0] != version.split("v")[0] or (
            "v" in arxiv_id and arxiv_id != version):
        raise ValueError(f"Requested {arxiv_id}, arXiv returned {version}")
    abstract = _document(metadata, [], "abstract_only")
    notes: dict = {"html_error": None, "pdf_error": None}
    try:
        parser = PaperTextParser()
        html = _bounded(fetch, f"https://arxiv.org/html/{version}", MAX_HTML_BYTES)
        parser.feed(html.decode("utf-8"))
        if len(parser.paragraphs) >= 5:
            return _document(metadata, parser.paragraphs, "html_paragraphs"), notes
        notes["html_error"] = "fewer than 5 article paragraphs"
    except Exception as exc:
        notes["html_error"] = f"{type(exc).__name__}: {exc}"
    try:
        pdf = _bounded(fetch_pdf, f"https://arxiv.org/pdf/{version}", MAX_FILE_BYTES)
        text, *_ = _pdf_document(pdf)
        return _document(metadata, _pdf_paragraphs(text), "pdf_text"), notes
    except Exception as exc:
        notes["pdf_error"] = f"{type(exc).__name__}: {exc}"
    return abstract, notes
