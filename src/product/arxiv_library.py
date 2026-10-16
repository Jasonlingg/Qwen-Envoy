"""Immutable arXiv cache; unversioned opens reuse the newest cached version."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from src.eval.artifacts import content_hash
from src.product.arxiv_ids import validate_arxiv_id
from src.product.arxiv_ingest import Fetch, ingest_paper
from src.product.net import Throttle, http_get
from src.research.file_sources import MAX_FILE_BYTES, MAX_TEXT_CHARS
from src.research.papers import PARSER_VERSION

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_GUARD = threading.Lock()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ArxivLibrary:
    def __init__(self, root: Path, *, fetch: Fetch | None = None, fetch_pdf: Fetch | None = None,
                 throttle: Throttle | None = None) -> None:
        self.root = Path(root).expanduser().resolve()
        self.papers_dir, self.views_dir = self.root / "papers", self.root / "views"
        self.papers_dir.mkdir(parents=True, exist_ok=True)
        self.views_dir.mkdir(parents=True, exist_ok=True)
        # Real requests, including redirects, are paced centrally by net.http_get.
        def injected(call: Fetch) -> Fetch:
            def wrapped(url: str) -> bytes:
                if throttle is not None:
                    throttle.wait()
                return call(url)
            return wrapped
        self._fetch = injected(fetch) if fetch is not None else http_get
        self._fetch_pdf = (injected(fetch_pdf) if fetch_pdf is not None
                           else lambda url: http_get(url, max_bytes=MAX_FILE_BYTES))
        with _LOCKS_GUARD:
            self._lock = _LOCKS.setdefault(self.root, threading.RLock())

    def cached_version(self, arxiv_id: str) -> str | None:
        arxiv_id = validate_arxiv_id(arxiv_id)
        if "v" in arxiv_id:
            return arxiv_id if (self.papers_dir / arxiv_id).exists() else None
        versions = []
        for path in self.papers_dir.glob(f"{arxiv_id}v*"):
            try:
                versions.append(validate_arxiv_id(path.name))
            except ValueError:
                continue
        return max(versions, key=lambda item: int(item.split("v")[1])) if versions else None

    def open(self, arxiv_id: str) -> dict:
        with self._lock:  # Shared by library instances; one ingest per cache root.
            version = self.cached_version(arxiv_id)
            if version is None:
                document, notes = ingest_paper(arxiv_id, self._fetch, self._fetch_pdf)
                version = document["metadata"]["arxiv_id"]
                self._store(version, document, notes)
            return self.summary(version)

    def _store(self, version: str, document: dict, notes: dict) -> None:
        target = self.papers_dir / version
        if target.exists():
            self.document(version)
            return
        staging = self.papers_dir / f".{version}.{uuid4().hex}.tmp"
        staging.mkdir()
        try:
            path = staging / "document.json"
            path.write_text(_json(document), encoding="utf-8")
            (staging / "notes.json").write_text(_json({**notes, "sha256": content_hash(path),
                "retrieved_at": datetime.now(timezone.utc).isoformat()}), encoding="utf-8")
            staging.rename(target)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def document(self, arxiv_id: str) -> dict:
        version = self.cached_version(arxiv_id)
        if version is None:
            raise ValueError(f"paper {arxiv_id} is not open; call open_paper first")
        folder = self.papers_dir / version
        path, notes = folder / "document.json", folder / "notes.json"
        try:
            if folder.is_symlink() or any(p.is_symlink() or not p.is_file() for p in (path, notes)):
                raise ValueError("missing or linked cache files")
            if path.stat().st_size > MAX_FILE_BYTES * 4 or notes.stat().st_size > 100_000:
                raise ValueError("oversized cache files")
            document = json.loads(path.read_text(encoding="utf-8"))
            if content_hash(path) != json.loads(notes.read_text(encoding="utf-8"))["sha256"]:
                raise ValueError("document hash mismatch")
            metadata, text, sections = document["metadata"], document["text"], document["sections"]
            expected_id = "arxiv_" + version.replace(".", "_")
            if (metadata["arxiv_id"] != version or document["doc_id"] != expected_id
                    or metadata["source_url"] != f"https://arxiv.org/abs/{version}"
                    or metadata["parser_version"] != PARSER_VERSION
                    or metadata["coverage"] not in {"html_paragraphs", "pdf_text", "abstract_only"}
                    or not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS
                    or not isinstance(sections, list) or not sections):
                raise ValueError("invalid document identity or text")
            last = 0
            for section in sections:
                start, end = section["start"], section["end"]
                if (not isinstance(section["section"], str) or type(start) is not int
                        or type(end) is not int or not last <= start < end <= len(text)):
                    raise ValueError("invalid section offsets")
                last = end
            return document
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Corrupt arXiv cache for {version}: {exc}") from exc

    def summary(self, arxiv_id: str) -> dict:
        document = self.document(arxiv_id)
        metadata, outline = document["metadata"], []
        for section in document["sections"]:
            if not outline or outline[-1]["section"] != section["section"]:
                outline.append({"section": section["section"], "start": section["start"]})
        return {"arxiv_id": metadata["arxiv_id"], "doc_id": document["doc_id"],
                "title": document["title"], "authors": metadata.get("authors", []),
                "submitted": metadata.get("submitted"), "source_url": metadata["source_url"],
                "coverage": metadata["coverage"], "total_chars": len(document["text"]),
                "outline": outline[:200]}

    def view(self, arxiv_ids: list[str]) -> Path:
        with self._lock:
            documents = {doc["metadata"]["arxiv_id"]: doc
                         for doc in (self.document(identifier) for identifier in arxiv_ids)}
            if not documents:
                raise ValueError("choose at least one opened paper")
            versions = sorted(documents)
            target = self.views_dir / hashlib.sha256("\n".join(versions).encode()).hexdigest()[:24]
            staging = self.views_dir / f".{target.name}.{uuid4().hex}.tmp"
            corpus = staging / "corpus"
            corpus.mkdir(parents=True)
            try:
                papers = []
                for version in versions:
                    document = documents[version]
                    path = corpus / f"{document['doc_id']}.json"
                    path.write_text(_json(document), encoding="utf-8")
                    papers.append({"doc_id": document["doc_id"], "arxiv_id": version,
                                   "title": document["title"],
                                   "source_url": document["metadata"]["source_url"],
                                   "coverage": document["metadata"]["coverage"]})
                manifest = {"schema_version": "research-snapshot-v1", "status": "complete",
                            "parser_version": PARSER_VERSION, "papers": papers, "failures": [],
                            "corpus_hash": content_hash(corpus)}
                (staging / "manifest.json").write_text(_json(manifest), encoding="utf-8")
                if target.exists():
                    if (target.is_symlink() or (target / "corpus").is_symlink()
                            or (target / "manifest.json").is_symlink()
                            or any(p.is_symlink() for p in (target / "corpus").rglob("*"))
                            or json.loads((target / "manifest.json").read_text()) != manifest
                            or content_hash(target / "corpus") != manifest["corpus_hash"]):
                        raise ValueError("Corrupt arXiv view cache")
                else:
                    staging.rename(target)  # Filesystem failures must reach the caller.
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
            return target
