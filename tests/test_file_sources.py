"""An attached file remains a bounded, tamper-checked, unreviewed source."""

import hashlib
import io
import json
from datetime import datetime, timezone

import pytest

from src.research.file_sources import (
    MAX_FILE_BYTES,
    MAX_TEXT_CHARS,
    PARSER_VERSION,
    build_file_snapshot,
    load_file_snapshot,
)
from src.research.tools_runtime import ResearchTools


def test_markdown_snapshot_preserves_original_and_exact_extracted_spans(tmp_path):
    raw = b"# Retrieval result\n\nBM25 recovered the required evidence.\n"
    target = tmp_path / "source"
    uploaded = datetime(2026, 9, 29, 16, 30, tzinfo=timezone.utc)
    manifest = build_file_snapshot("Retrieval result.md", raw, target, uploaded_at=uploaded)
    row = manifest["papers"][0]
    doc_id = row["doc_id"]
    assert manifest["schema_version"] == "research-snapshot-v1"
    assert manifest["parser_version"] == PARSER_VERSION
    assert manifest["status"] == "complete"
    assert row["source_kind"] == "uploaded_file"
    assert row["filename"] == "Retrieval result.md"
    assert row["uploaded_at"] == "2026-09-29T16:30:00+00:00"
    assert row["source_revision"] == hashlib.sha256(raw).hexdigest()
    assert (target / "raw" / f"{doc_id}.md").read_bytes() == raw
    loaded, docs = load_file_snapshot(target)
    assert loaded == manifest
    doc = docs[doc_id]
    assert doc["text"] == raw.decode()
    assert doc["sections"][-1]["section"] == "Retrieval result"
    hit = ResearchTools(target / "corpus").search_paper(doc_id, "BM25 evidence")[0]
    assert hit["quote"] == doc["text"][hit["start"]:hit["end"]]
    assert hit["source_url"] == f"file-upload://{doc_id}"
    with pytest.raises(FileExistsError):
        build_file_snapshot("Retrieval result.md", raw, target)


def test_text_attachment_rejects_invalid_names_types_and_contents_before_writing(tmp_path):
    invalid = [
        ("../notes.md", b"text", "basename"),
        ("folder\\notes.txt", b"text", "basename"),
        ("notes.html", b"text", "Only Markdown"),
        ("notes.txt", b"", "empty"),
        ("notes.txt", b"\xff", "UTF-8"),
        ("notes.txt", b"some\x00text", "binary"),
        ("notes.txt", b"a" * (MAX_FILE_BYTES + 1), "5 MB"),
        ("notes.txt", ("a" * (MAX_TEXT_CHARS + 1)).encode(), "250,000"),
        ("paper.pdf", b"not a PDF", "signature"),
    ]
    for index, (name, data, message) in enumerate(invalid):
        target = tmp_path / f"invalid-{index}"
        with pytest.raises(ValueError, match=message):
            build_file_snapshot(name, data, target)
        assert not target.exists()


def test_original_or_parsed_file_tampering_invalidates_the_snapshot(tmp_path):
    original = b"A documented finding.\n"
    target = tmp_path / "source"
    manifest = build_file_snapshot("finding.txt", original, target)
    doc_id = manifest["papers"][0]["doc_id"]
    raw_path = target / "raw" / f"{doc_id}.txt"
    raw_path.write_bytes(b"A different finding.\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_file_snapshot(target)
    raw_path.write_bytes(original)
    corpus_path = target / "corpus" / f"{doc_id}.json"
    doc = json.loads(corpus_path.read_text())
    doc["text"] = "A different finding.\n"
    corpus_path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="hash mismatch"):
        load_file_snapshot(target)


def _pdf_bytes(*, text: bool = True, pages: int = 2) -> bytes:
    pypdf = pytest.importorskip("pypdf")
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = pypdf.PdfWriter()
    for index in range(pages):
        page = writer.add_blank_page(width=300, height=300)
        if index == 0 and text:
            font = DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            })
            page[NameObject("/Resources")] = DictionaryObject({
                NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
            })
            stream = DecodedStreamObject()
            stream.set_data(b"BT /F1 12 Tf 20 200 Td (Evidence is on page one.) Tj ET")
            page[NameObject("/Contents")] = stream
    result = io.BytesIO()
    writer.write(result)
    return result.getvalue()


def test_pdf_snapshot_keeps_page_number_and_rejects_scans_and_overlong_files(tmp_path):
    raw = _pdf_bytes()
    target = tmp_path / "paper"
    manifest = build_file_snapshot("Study.pdf", raw, target)
    doc_id = manifest["papers"][0]["doc_id"]
    _, docs = load_file_snapshot(target)
    doc = docs[doc_id]
    assert doc["metadata"]["coverage"] == "pdf_text_no_ocr"
    assert doc["metadata"]["empty_pages"] == [2]
    hit = ResearchTools(target / "corpus").search_paper(doc_id, "Evidence page one")[0]
    assert hit["pages"] == [1]
    assert "Evidence is on page one." in hit["quote"]

    scanned_target = tmp_path / "scan"
    with pytest.raises(ValueError, match="OCR"):
        build_file_snapshot("scan.pdf", _pdf_bytes(text=False), scanned_target)
    assert not scanned_target.exists()
    overlong_target = tmp_path / "overlong"
    with pytest.raises(ValueError, match="100 page"):
        build_file_snapshot("long.pdf", _pdf_bytes(pages=101), overlong_target)
    assert not overlong_target.exists()
