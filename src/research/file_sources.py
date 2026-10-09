"""Freeze one locally uploaded document with verifiable raw and parsed bytes.

Uploaded files are unreviewed sources. Their text can be quoted from this
snapshot, but importing them does not add them to the reviewed Obsidian vault.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from src.eval.artifacts import content_hash
from src.research.agent import load_snapshot
from src.research.web_sources import canonicalize_web_url

PARSER_VERSION = "uploaded-file-v1"
MAX_FILE_BYTES = 5_000_000
MAX_PDF_PAGES = 100
MAX_TEXT_CHARS = 250_000
_CONTENT_TYPES = {
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".pdf": "application/pdf",
}


def _filename(value: str) -> tuple[str, str]:
    """Keep a display name only; never use client path components for storage."""
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > 255
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or any(ord(char) < 32 or ord(char) == 127 for char in value)
    ):
        raise ValueError("File name must be a plain basename without control characters")
    extension = Path(value).suffix.lower()
    if extension not in _CONTENT_TYPES:
        raise ValueError("Only Markdown (.md), text (.txt), and PDF (.pdf) files are supported")
    return value, extension


def _text_document(
    raw: bytes, extension: str
) -> tuple[str, list[dict], str, list[int], str | None]:
    if b"\x00" in raw or raw.startswith(b"%PDF-"):
        raise ValueError("Text attachment contains binary data")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Text attachment must be UTF-8") from exc
    if not text.strip():
        raise ValueError("File has no extractable text")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("Extracted text exceeds the 250,000 character limit")
    if extension == ".txt":
        sections = [{"section": "Document", "start": 0, "end": len(text)}]
        return text, sections, "plain_utf8_text", [], None

    sections: list[dict] = []
    start, heading = 0, "Document"
    for match in re.finditer(r"^#{1,6} (.+)$", text, re.M):
        if match.start() > start:
            sections.append({"section": heading, "start": start, "end": match.start()})
        start, heading = match.start(), match[1]
    if len(text) > start:
        sections.append({"section": heading, "start": start, "end": len(text)})
    return text, sections, "markdown_utf8_text", [], None


def _pdf_document(raw: bytes) -> tuple[str, list[dict], str, list[int], str]:
    if not raw.startswith(b"%PDF-"):
        raise ValueError("PDF attachment is missing a PDF signature")
    try:
        import pypdf
    except ImportError as exc:
        raise ValueError("PDF attachment needs pypdf; install the project vault extra") from exc
    try:
        reader = pypdf.PdfReader(io.BytesIO(raw))
        pages = reader.pages
        if len(pages) > MAX_PDF_PAGES:
            raise ValueError("PDF exceeds the 100 page limit")
        text, sections, empty_pages = "", [], []
        for number, page in enumerate(pages, 1):
            page_text = page.extract_text() or ""
            if not page_text.strip():
                empty_pages.append(number)
                continue
            if text:
                text += "\n\n"
            start = len(text)
            text += page_text
            if len(text) > MAX_TEXT_CHARS:
                raise ValueError("Extracted text exceeds the 250,000 character limit")
            sections.append(
                {"section": f"Page {number}", "page": number, "start": start, "end": len(text)}
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("PDF attachment could not be read") from exc
    if not text.strip():
        raise ValueError("PDF has no extractable text; scanned PDFs require OCR")
    return text, sections, "pdf_text_no_ocr", empty_pages, pypdf.__version__


def build_file_snapshot(
    filename: str,
    body: bytes,
    output: Path,
    uploaded_at: datetime | None = None,
    *,
    source_url: str | None = None,
    requested_url: str | None = None,
) -> dict:
    """Create one immutable file snapshot outside the vault after validation."""
    name, extension = _filename(filename)
    if (source_url is None) != (requested_url is None):
        raise ValueError("Fetched sources need both requested and final URLs")
    if source_url is not None:
        if extension != ".pdf":
            raise ValueError("Fetched file source must be a PDF")
        source_url = canonicalize_web_url(source_url)
        requested_url = canonicalize_web_url(requested_url)
        if not source_url.startswith("https://") or not requested_url.startswith("https://"):
            raise ValueError("Fetched PDF URLs must use HTTPS")
    if not isinstance(body, bytes):
        raise TypeError("File body must be bytes")
    if not body:
        raise ValueError("File is empty")
    if len(body) > MAX_FILE_BYTES:
        raise ValueError("File exceeds the 5 MB size limit")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    timestamp = uploaded_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("uploaded_at must include a timezone")
    uploaded_iso = timestamp.astimezone(timezone.utc).isoformat()

    if extension == ".pdf":
        text, sections, coverage, empty_pages, pdf_version = _pdf_document(body)
    else:
        text, sections, coverage, empty_pages, pdf_version = _text_document(body, extension)
    raw_hash = hashlib.sha256(body).hexdigest()
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    doc_id = "file_" + hashlib.sha256((name + "\0" + raw_hash).encode("utf-8")).hexdigest()[:24]
    locator = source_url or f"file-upload://{doc_id}"
    metadata = {
        "source_url": locator,
        "source_kind": "fetched_pdf" if source_url is not None else "uploaded_file",
        "filename": name,
        "source_revision": raw_hash,
        "source_sha256": raw_hash,
        "text_sha256": text_hash,
        "uploaded_at": uploaded_iso,
        "submitted": "unknown",
        "coverage": coverage,
        "content_type": _CONTENT_TYPES[extension],
        "parser_version": PARSER_VERSION,
        "empty_pages": empty_pages,
        "pypdf_version": pdf_version,
        "authority_note": (
            "A fetched PDF is unreviewed; its URL does not establish authorship or accuracy."
            if source_url is not None
            else "An attached file is unreviewed; its type does not establish "
            "authorship or accuracy."
        ),
    }
    if source_url is not None:
        metadata.update(requested_url=requested_url, fetched_at=uploaded_iso)
    document = {
        "doc_id": doc_id,
        "title": Path(name).stem,
        "text": text,
        "sections": sections,
        "metadata": metadata,
    }
    output.mkdir(parents=True, exist_ok=False)
    corpus_dir, raw_dir = output / "corpus", output / "raw"
    corpus_dir.mkdir()
    raw_dir.mkdir()
    corpus_path = corpus_dir / f"{doc_id}.json"
    raw_path = raw_dir / f"{doc_id}{extension}"
    corpus_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    raw_path.write_bytes(body)
    manifest = {
        "schema_version": "research-snapshot-v1",
        "parser_version": PARSER_VERSION,
        "retrieved_at": uploaded_iso,
        "papers": [
            {
                "doc_id": doc_id,
                "title": document["title"],
                **metadata,
                "sha256": content_hash(corpus_path),
                "raw_sha256": raw_hash,
            }
        ],
        "failures": [],
        "coverage_note": (
            ("One fetched PDF. " if source_url is not None else "One user-attached file. ")
            + "Text extraction is partial for PDFs: images, tables, equations, and scanned "
            "pages may be missing; no OCR."
        ),
        "status": "complete",
        "corpus_hash": content_hash(corpus_dir),
        "raw_sources_hash": raw_hash,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def load_file_snapshot(snapshot: Path) -> tuple[dict, dict[str, dict]]:
    """Recheck frozen original bytes and parsed text before showing or quoting."""
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
        raise ValueError("Expected one complete file snapshot")
    row = rows[0]
    doc_id = row.get("doc_id")
    if not isinstance(doc_id, str) or doc_id not in docs:
        raise ValueError("File snapshot document identity mismatch")
    doc = docs[doc_id]
    metadata = doc.get("metadata", {})
    try:
        name, extension = _filename(row["filename"])
    except (KeyError, ValueError) as exc:
        raise ValueError("File snapshot filename is invalid") from exc
    corpus_path = snapshot / "corpus" / f"{doc_id}.json"
    raw_path = snapshot / "raw" / f"{doc_id}{extension}"
    if (
        not corpus_path.is_file()
        or corpus_path.is_symlink()
        or not raw_path.is_file()
        or raw_path.is_symlink()
    ):
        raise ValueError("File snapshot is missing raw or parsed source")
    raw = raw_path.read_bytes()
    raw_hash = hashlib.sha256(raw).hexdigest()
    expected_id = (
        "file_" + hashlib.sha256((name + "\0" + raw_hash).encode("utf-8")).hexdigest()[:24]
    )
    source_kind = metadata.get("source_kind")
    if source_kind == "fetched_pdf":
        try:
            final_url = canonicalize_web_url(metadata["source_url"])
            requested = canonicalize_web_url(metadata["requested_url"])
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError("Fetched PDF provenance is invalid") from exc
        provenance_valid = (
            extension == ".pdf"
            and final_url.startswith("https://")
            and requested.startswith("https://")
            and final_url == metadata["source_url"]
            and requested == row.get("requested_url")
            and metadata.get("fetched_at") == row.get("fetched_at")
        )
    else:
        provenance_valid = (
            source_kind == "uploaded_file"
            and metadata.get("source_url") == f"file-upload://{doc_id}"
        )
    if (
        len(raw) > MAX_FILE_BYTES
        or doc_id != expected_id
        or content_hash(corpus_path) != row.get("sha256")
        or raw_hash != row.get("raw_sha256")
        or raw_hash != manifest.get("raw_sources_hash")
        or raw_hash != metadata.get("source_sha256")
        or raw_hash != metadata.get("source_revision")
        or metadata.get("filename") != name
        or metadata.get("source_url") != row.get("source_url")
        or not provenance_valid
        or metadata.get("parser_version") != PARSER_VERSION
        or metadata.get("content_type") != _CONTENT_TYPES[extension]
        or metadata.get("uploaded_at") != row.get("uploaded_at")
        or hashlib.sha256(doc.get("text", "").encode("utf-8")).hexdigest()
        != metadata.get("text_sha256")
        or len(doc.get("text", "")) > MAX_TEXT_CHARS
    ):
        raise ValueError("File snapshot raw or parsed source hash mismatch")
    return manifest, docs
