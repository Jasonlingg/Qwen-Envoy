"""Resolve experiment references into persistent provenance receipts; never call models."""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.product.ml_experiment_io import digest, encode
from src.product.ml_experiment_receipts import EvidenceReceipt, reference_hash
from src.product.ml_experiment_sources import read_json, select_pointer
from src.research.file_sources import MAX_TEXT_CHARS
from src.research.papers import PARSER_VERSION


def _checked_paper(document, version):
    try:
        metadata, text, sections = document["metadata"], document["text"], document["sections"]
        if (metadata["arxiv_id"] != version
                or document["doc_id"] != "arxiv_" + version.replace(".", "_")
                or metadata["source_url"] != f"https://arxiv.org/abs/{version}"
                or metadata["parser_version"] != PARSER_VERSION
                or metadata["coverage"] not in {"html_paragraphs", "pdf_text", "abstract_only"}
                or not isinstance(document["title"], str)
                or not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT_CHARS
                or not isinstance(sections, list) or not sections):
            raise ValueError("Invalid cached paper identity or text")
        last = 0
        for section in sections:
            start, end = section["start"], section["end"]
            if (not isinstance(section["section"], str) or type(start) is not int
                    or type(end) is not int or not last <= start < end <= len(text)):
                raise ValueError("Invalid cached paper section offsets")
            last = end
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed cached paper") from exc


class EvidenceResolver:
    def __init__(self, *, library_root: Path, artifact_root: Path | None = None):
        self.library_root = Path(library_root).expanduser().absolute()
        self.artifact_root = Path(artifact_root).expanduser().absolute() if artifact_root else None

    def __call__(self, references) -> list[EvidenceReceipt]:
        receipts = []
        for ref in references:
            if ref.kind == "external_reference":
                receipts.append(EvidenceReceipt(reference_id=ref.reference_id,
                    reference_sha256=reference_hash(ref), kind=ref.kind,
                    verification="reference_only", source_sha256=None, value=None,
                    metadata={"url": ref.url, "description": ref.description}))
                continue
            raw, value, metadata = (self._paper(ref) if ref.kind == "paper_span"
                                    else self._artifact(ref))
            if len(encode(value).decode("utf-8")) > 4_000:
                raise ValueError("Selected evidence exceeds 4000 characters; narrow the pointer")
            receipts.append(EvidenceReceipt(reference_id=ref.reference_id,
                reference_sha256=reference_hash(ref), kind=ref.kind,
                verification=("paper_span_verified" if ref.kind == "paper_span"
                              else "artifact_value_verified"), source_sha256=digest(raw),
                value=value, metadata=metadata))
        return receipts

    def _artifact(self, ref):
        if self.artifact_root is None:
            raise ValueError("Local artifacts require an operator-selected --artifact-root")
        raw, source = read_json(self.artifact_root, ref.relative_path)
        if digest(raw) != ref.sha256:
            raise ValueError("Artifact SHA-256 differs from the selected export")
        value = select_pointer(source, ref.json_pointer)
        return raw, value, {"relative_path": ref.relative_path, "json_pointer": ref.json_pointer}

    def _paper(self, ref):
        prefix = f"papers/{ref.arxiv_id}"
        # Read through directory descriptors first; even a cache-parent symlink is refused.
        raw, document = read_json(self.library_root, f"{prefix}/document.json")
        _, notes = read_json(self.library_root, f"{prefix}/notes.json")
        # The existing cache uses a name-prefixed file hash, not a raw-byte digest.
        expected = hashlib.sha256(b"document.json\0" + raw + b"\0").hexdigest()
        if not isinstance(notes, dict) or notes.get("sha256") != expected:
            raise ValueError("Cached paper integrity check failed")
        if ref.document_sha256 is not None and digest(raw) != ref.document_sha256:
            raise ValueError("Paper document SHA-256 differs from the supplied hash")
        # Validate the bytes already read; constructing a live library could create cache dirs.
        _checked_paper(document, ref.arxiv_id)
        if ref.end > len(document["text"]):
            raise ValueError("Paper evidence offsets exceed the stored text")
        quote = document["text"][ref.start:ref.end]
        if ref.quote is not None and ref.quote != quote:
            raise ValueError("Paper quote differs from the stored text")
        metadata = document["metadata"]
        return raw, quote, {"arxiv_id": ref.arxiv_id, "start": ref.start, "end": ref.end,
                            "coverage": metadata["coverage"], "source_url": metadata["source_url"],
                            "title": document["title"]}
