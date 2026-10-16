"""Version pinning, bounded extraction, immutable caches and reproducible views."""

import io
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import src.product.arxiv_library as store
from src.product.arxiv_ingest import ingest_paper
from src.product.arxiv_library import ArxivLibrary
from src.product.net import SourceUnavailable
from src.research.agent import load_snapshot
from tests.arxiv_fakes import abs_page, article_page, make_library, tiny_pdf

VERSION = "2609.12345v2"


def test_html_paper_is_ingested_with_outline(tmp_path):
    library, fake = make_library(tmp_path, {VERSION: {"html": article_page()}})
    summary = library.open("2609.12345")
    assert (summary["arxiv_id"], summary["coverage"]) == (VERSION, "html_paragraphs")
    assert [s["section"] for s in summary["outline"]] == ["Abstract", "Introduction", "Results"]
    results = next(s for s in summary["outline"] if s["section"] == "Results")
    assert library.document(VERSION)["text"][results["start"]:].startswith("Eviction saves")
    assert fake.calls == ["https://arxiv.org/abs/2609.12345", f"https://arxiv.org/html/{VERSION}"]


def test_reopening_uses_newest_cached_version_without_network(tmp_path):
    library, fake = make_library(tmp_path, {VERSION: {"html": article_page()}, "2609.12345v10": {}})
    library.open(VERSION)
    first = (library.papers_dir / VERSION / "document.json").read_bytes()
    library.open("2609.12345v10")
    calls = len(fake.calls)
    assert library.open(VERSION)["arxiv_id"] == VERSION
    assert library.open("2609.12345")["arxiv_id"] == "2609.12345v10"
    assert len(fake.calls) == calls
    assert (library.papers_dir / VERSION / "document.json").read_bytes() == first


@pytest.mark.parametrize("html", [None, b"<article><p>Only one paragraph.</p></article>"])
def test_pdf_fallback_and_heading_outline(tmp_path, html):
    pytest.importorskip("pypdf")
    pdf = tiny_pdf(["1 Introduction", "We study caches.", "2 Results", "Eviction saves memory."])
    library, fake = make_library(tmp_path, {VERSION: {"html": html, "pdf": pdf}})
    summary = library.open(VERSION)
    assert summary["coverage"] == "pdf_text"
    section = next(s for s in summary["outline"] if s["section"] == "2 Results")
    assert library.document(VERSION)["text"][section["start"]:].startswith("2 Results")
    assert fake.calls[-1] == f"https://arxiv.org/pdf/{VERSION}"


@pytest.mark.parametrize("pdf", [None, b"not a PDF", b"%PDF-1.4\n" + b" " * 5_000_000])
def test_failed_or_oversized_pdf_keeps_abstract(tmp_path, pdf):
    library, _ = make_library(tmp_path, {VERSION: {"pdf": pdf}})
    assert library.open(VERSION)["coverage"] == "abstract_only"
    assert library.document(VERSION)["text"] == "We compress key-value caches."
    notes = json.loads((library.papers_dir / VERSION / "notes.json").read_text())
    assert notes["html_error"] and notes["pdf_error"]


def test_pdf_page_and_text_limits(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    writer, stream = pypdf.PdfWriter(), io.BytesIO()
    for _ in range(101):
        writer.add_blank_page(width=20, height=20)
    writer.write(stream)
    for index, pdf in enumerate([stream.getvalue(), tiny_pdf(["x" * 250_001])]):
        library, _ = make_library(tmp_path / str(index), {VERSION: {"pdf": pdf}})
        assert library.open(VERSION)["coverage"] == "abstract_only"
        notes = json.loads((library.papers_dir / VERSION / "notes.json").read_text())
        assert "limit" in notes["pdf_error"]


def test_oversized_html_falls_back(tmp_path):
    library, _ = make_library(tmp_path, {VERSION: {"html": b" " * 2_000_001}})
    assert library.open(VERSION)["coverage"] == "abstract_only"


def test_unknown_paper_and_bad_ids(tmp_path):
    library, _ = make_library(tmp_path, {})
    with pytest.raises(SourceUnavailable):
        library.open("2609.00000")
    for invalid in ["10.1145/123", "2600.12345", "2609.12345v0", "２６０９.12345", "../2609.12345"]:
        with pytest.raises(ValueError, match="arXiv ID"):
            library.open(invalid)
    with pytest.raises(ValueError, match="not open"):
        library.document(VERSION)


@pytest.mark.parametrize("received", ["2609.54321v2", "2609.12345v3", "2600.12345v2",
                                      "2609.12345v0"])
def test_metadata_must_pin_requested_modern_version(received):
    with pytest.raises(SourceUnavailable, match="arxiv: invalid response"):
        ingest_paper(VERSION, lambda _: abs_page(received), lambda _: b"")


@pytest.mark.parametrize("response", [b"", b"\xff", b"<html>Service unavailable</html>",
                                      abs_page("2609.12345"), b" " * 2_000_001])
def test_bad_metadata_response_is_source_unavailable(response):
    with pytest.raises(SourceUnavailable) as info:
        ingest_paper(VERSION, lambda _: response, lambda _: b"")
    assert (info.value.source, info.value.detail) == ("arxiv", "invalid response")


def test_caller_bad_id_is_rejected_before_fetch():
    def unexpected_fetch(_):
        pytest.fail("Invalid caller ID must not reach the source")
    with pytest.raises(ValueError, match="arXiv ID"):
        ingest_paper("invalid", unexpected_fetch, unexpected_fetch)


def test_view_is_a_valid_deterministic_snapshot(tmp_path):
    library, _ = make_library(tmp_path, {VERSION: {"html": article_page()}, "2609.54321v1": {}})
    library.open(VERSION)
    library.open("2609.54321v1")
    first = library.view(["2609.54321v1", "2609.12345"])
    assert first == library.view([VERSION, "2609.54321v1", VERSION])
    manifest, docs = load_snapshot(first)
    assert {p["arxiv_id"] for p in manifest["papers"]} == {VERSION, "2609.54321v1"}
    assert set(docs) == {"arxiv_2609_12345v2", "arxiv_2609_54321v1"}
    assert library.view([VERSION]).name != first.name


def test_concurrent_opens_across_instances_ingest_once(tmp_path):
    library, fake = make_library(tmp_path, {VERSION: {"html": article_page()}})
    other = ArxivLibrary(library.root, fetch=fake.fetch, fetch_pdf=fake.fetch_pdf)
    with ThreadPoolExecutor(4) as pool:
        summaries = list(pool.map(lambda lib: lib.open("2609.12345"), [library, other] * 2))
    assert {s["arxiv_id"] for s in summaries} == {VERSION}
    assert fake.calls.count("https://arxiv.org/abs/2609.12345") == 1


@pytest.mark.parametrize("corruption", ["text", "notes", "missing", "symlink"])
def test_corrupt_paper_is_rejected_without_network(tmp_path, corruption):
    library, fake = make_library(tmp_path, {VERSION: {"html": article_page()}})
    library.open(VERSION)
    path = library.papers_dir / VERSION / "document.json"
    calls = len(fake.calls)
    if corruption == "text":
        path.write_text(path.read_text().replace("Eviction", "Changed"))
    elif corruption == "notes":
        path.with_name("notes.json").write_text("{}")
    else:
        raw = path.read_bytes()
        path.unlink()
        if corruption == "symlink":
            outside = tmp_path / "outside.json"
            outside.write_bytes(raw)
            path.symlink_to(outside)
    with pytest.raises(ValueError, match="Corrupt arXiv cache"):
        library.open(VERSION)
    assert len(fake.calls) == calls


def test_corrupt_view_and_rename_errors_are_reported(tmp_path, monkeypatch):
    library, _ = make_library(tmp_path, {VERSION: {"html": article_page()}})
    library.open(VERSION)
    view = library.view([VERSION])
    (view / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="Corrupt arXiv view"):
        library.view([VERSION])
    def blocked(*_):
        raise PermissionError("blocked")
    monkeypatch.setattr(store.Path, "rename", blocked)
    library.views_dir = library.root / "other-views"
    library.views_dir.mkdir()
    with pytest.raises(PermissionError, match="blocked"):
        library.view([VERSION])
    assert not list(library.views_dir.iterdir())


def test_default_pdf_uses_bounded_shared_transport_without_extra_wait(tmp_path, monkeypatch):
    calls = []
    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        return abs_page(VERSION) if "/abs/" in url else b"unavailable"
    class NoWait:
        def wait(self):
            pytest.fail("Default HTTP transport owns throttling")
    monkeypatch.setattr(store, "http_get", fetch)
    library = ArxivLibrary(tmp_path, throttle=NoWait())
    assert library.open(VERSION)["coverage"] == "abstract_only"
    assert calls[-1] == (f"https://arxiv.org/pdf/{VERSION}", {"max_bytes": 5_000_000})
