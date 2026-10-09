"""Local, approval-gated learning memory over immutable vault snapshots.

The lexical review path is a product baseline. It does not run or impersonate
Qwen or Nemotron. Saved code-execution transcripts can be reviewed separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

import yaml

from src.research.agent import load_snapshot
from src.research.code_exec_packet import build_code_exec_packet
from src.research.tools_runtime import ResearchTools
from src.research.vault import import_vault

CAPTURE_KINDS = {"source", "attempt", "result", "lesson", "correction", "decision"}
APPROVAL_KINDS = {"lesson", "correction", "decision"}
REVIEW_VERSION = "personal-memory-review-v1"
_EVIDENCE_ID = re.compile(r"E[1-5]\Z")
_QUERY_STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "by", "did", "do", "does", "for",
    "from", "how", "in", "is", "my", "of", "on", "or", "the", "to", "was",
    "were", "what", "when", "where", "which", "who", "why", "with",
}


def _vault_root(vault: Path) -> Path:
    root = vault.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("vault must be an existing directory")
    return root


def _date(value: str | None) -> str | None:
    if value is None:
        return None
    return date.fromisoformat(value).isoformat()


def _new_note_path(vault: Path) -> Path:
    target = vault / "Learning Memory" / f"{datetime.now(timezone.utc):%Y%m%d}-{uuid4().hex}.md"
    if not target.resolve().is_relative_to(vault):
        raise ValueError("Learning Memory path escapes the vault")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.resolve().is_relative_to(vault):
        raise ValueError("Learning Memory path escapes the vault")
    return target


def _write_note(vault: Path, metadata: dict, title: str, body: list[str]) -> Path:
    if not title.strip() or not body:
        raise ValueError("title and body are required")
    target = _new_note_path(vault)
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    content = f"---\n{frontmatter}\n---\n\n# {title.strip()}\n\n" + "\n".join(body).rstrip() + "\n"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return target


def capture_note(
    vault: Path, *, title: str, text: str, kind: str = "source",
    effective_date: str | None = None,
) -> Path:
    """Append user-supplied material; no model call or preexisting file is modified."""
    root = _vault_root(vault)
    if kind not in CAPTURE_KINDS:
        raise ValueError(f"kind must be one of {sorted(CAPTURE_KINDS)}")
    if not text.strip():
        raise ValueError("text must be non-empty")
    metadata = {
        "memory_id": f"memory_{uuid4().hex}",
        "kind": kind,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "effective_date": _date(effective_date),
        "review_status": "user_captured",
        "origin": "user_supplied",
    }
    return _write_note(root, metadata, title, [text.strip(), ""])


def freeze_vault(vault: Path, output: Path, collection: str = ".") -> dict:
    """Reuse the opt-in importer; each call makes a new immutable snapshot."""
    return import_vault(vault, collection, output)


def _snapshot(snapshot: Path, vault: Path | None = None) -> tuple[dict, dict]:
    manifest, docs = load_snapshot(snapshot)
    if vault is not None and manifest.get("vault_root") != str(_vault_root(vault)):
        raise ValueError("snapshot belongs to a different vault")
    return manifest, docs


def _record_context(document: dict) -> dict:
    """Surface dated memory fields without changing the existing importer schema."""
    match = re.match(r"\A---\n(.*?)\n---(?:\n|$)", document["text"], re.S)
    if match is None:
        return {}
    try:
        frontmatter = yaml.safe_load(match[1]) or {}
    except yaml.YAMLError:
        return {}
    if not isinstance(frontmatter, dict):
        return {}
    fields = (
        "record_id", "memory_id", "kind", "captured_at", "effective_date",
        "status", "review_status", "supersedes", "supersedes_doc_id", "derived_from",
    )
    return {field: str(frontmatter[field]) for field in fields
            if field in frontmatter and frontmatter[field] is not None}


def _navigation_only(quote: str) -> bool:
    """Do not offer a list of Obsidian links as factual evidence."""
    lines = [line.strip() for line in quote.splitlines() if line.strip()]
    return bool(lines) and all(
        line == "## Linked history" or re.fullmatch(r"-\s*\[\[[^\]\n]+\]\]", line)
        for line in lines
    )


def inspect_evidence(snapshot: Path, query: str, top_k: int = 5) -> dict:
    """Return exact passages from a lexical baseline for human review."""
    if not query.strip():
        raise ValueError("query must be non-empty")
    if not 1 <= top_k <= 5:
        raise ValueError("top_k must be 1..5")
    manifest, docs = _snapshot(snapshot)
    tools = ResearchTools(snapshot / "corpus")
    terms = set(re.findall(r"\w+", query.casefold())) - _QUERY_STOPWORDS
    required_terms = 2 if len(terms) >= 3 else 1
    hits = []
    # Discovery can favor an ID in YAML front matter over the actual note.
    # Re-rank within each discovered document and keep only a body passage
    # with enough query coverage to avoid generic-word distractors.
    for discovered in tools.search_papers(query, top_k=10):
        document = docs[discovered["doc_id"]]
        frontmatter = re.match(r"\A---\n.*?\n---(?:\n|$)", document["text"], re.S)
        body_start = frontmatter.end() if frontmatter else 0
        for passage in tools.search_paper(discovered["doc_id"], query, top_k=10):
            if passage["start"] < body_start or _navigation_only(passage["quote"]):
                continue
            matched = terms & set(re.findall(r"\w+", passage["quote"].casefold()))
            if len(matched) >= required_terms:
                hits.append(passage)
                break
    evidence = []
    for index, hit in enumerate(hits[:top_k], 1):
        evidence.append({
            "evidence_id": f"E{index}",
            "doc_id": hit["doc_id"],
            "title": hit["title"],
            "start": hit["start"],
            "end": hit["end"],
            "quote": hit["quote"],
            "source_path": hit.get("source_path"),
            "source_kind": hit.get("source_kind"),
            "record": _record_context(docs[hit["doc_id"]]),
            "score": hit["score"],
        })
    return {
        "schema_version": REVIEW_VERSION,
        "retriever": "lexical_paragraph_baseline",
        "status": "evidence_found" if evidence else "no_evidence",
        "query": query,
        "corpus_hash": manifest["corpus_hash"],
        "evidence": evidence,
        "uncertainty": (None if evidence else
                        "No lexical match in this selected snapshot; this does not prove the "
                        "answer is unknown outside it."),
        "warning": "Search matches and exact quotes prove provenance, not semantic support.",
    }


def review_code_exec_transcript(
    snapshot: Path, transcript: Path, question_id: str | None = None,
) -> dict:
    """Verify cited spans from a saved real run; do not execute model-authored code."""
    manifest, docs = _snapshot(snapshot)
    payload = json.loads(transcript.read_text())
    rows = payload if isinstance(payload, list) else [payload]
    if question_id is not None:
        rows = [row for row in rows if isinstance(row, dict)
                and row.get("question_id") == question_id]
    if len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValueError("select exactly one transcript row")
    row = rows[0]
    if row.get("status") != "completed":
        raise ValueError("a completed code-execution row is required")
    if not row.get("predicted_evidence"):
        answer = row.get("predicted_answer", "")
        abstained = isinstance(answer, str) and answer.strip().lower() in {
            "unanswerable", "unknown", "insufficient evidence",
        }
        return {
            "schema_version": REVIEW_VERSION,
            "retriever": "code_execution_transcript",
            "status": "no_evidence" if abstained else "unsupported_answer",
            "query": row.get("question"),
            "candidate_answer": answer,
            "corpus_hash": manifest["corpus_hash"],
            "evidence": [],
            "transcript_source": str(transcript),
            "policy_name": row.get("policy_name"),
            "uncertainty": (
                "The transcript supplies no source passages. "
                "Do not treat its answer as supported."
            ),
        }
    packet = build_code_exec_packet(row, snapshot / "corpus")
    if packet["corpus_hash"] != manifest["corpus_hash"]:
        raise ValueError("transcript packet and snapshot differ")
    packet["transcript_source"] = str(transcript)
    packet["model_identity_note"] = (
        "Model identity is reported by the transcript, not attested here."
    )
    packet["policy_name"] = rows[0].get("policy_name")
    for item in packet["evidence"]:
        item["record"] = _record_context(docs[item["doc_id"]])
    return packet


def _validated_sources(review: dict, docs: dict, selected: list[str]) -> list[dict]:
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select one or more distinct evidence IDs")
    by_id = {item.get("evidence_id"): item for item in review.get("evidence", [])
             if isinstance(item, dict)}
    if len(by_id) != len(review.get("evidence", [])):
        raise ValueError("review has duplicate or malformed evidence")
    result = []
    for evidence_id in selected:
        if not _EVIDENCE_ID.fullmatch(evidence_id) or evidence_id not in by_id:
            raise ValueError(f"unknown evidence ID: {evidence_id}")
        item = by_id[evidence_id]
        doc = docs.get(item.get("doc_id"))
        start, end = item.get("start"), item.get("end")
        if (doc is None or type(start) is not int or type(end) is not int
                or not 0 <= start < end <= len(doc["text"])
                or doc["text"][start:end] != item.get("quote")):
            raise ValueError(f"evidence {evidence_id} does not match the frozen snapshot")
        result.append({**item, "title": doc["title"],
                       "source_path": doc["metadata"].get("source_path")})
    return result


def approve_note(
    vault: Path, snapshot: Path, review: dict, *, title: str, text: str,
    kind: str, evidence_ids: list[str], effective_date: str | None = None,
    supersedes: str | None = None,
) -> Path:
    """Append a user-approved revision after rechecking the selected source bytes."""
    root = _vault_root(vault)
    manifest, docs = _snapshot(snapshot, root)
    if kind not in APPROVAL_KINDS:
        raise ValueError(f"kind must be one of {sorted(APPROVAL_KINDS)}")
    if not title.strip():
        raise ValueError("title must be non-empty")
    if not text.strip():
        raise ValueError("approved text must be non-empty")
    if review.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("review and snapshot corpus hashes differ")
    sources = _validated_sources(review, docs, evidence_ids)
    if kind == "correction" and not supersedes:
        raise ValueError("a correction must name the superseded document ID")
    if supersedes is not None and supersedes not in docs:
        matches = [doc_id for doc_id, doc in docs.items()
                   if _record_context(doc).get("record_id") == supersedes]
        if len(matches) != 1:
            raise ValueError("superseded document or record ID is absent from the snapshot")
        supersedes = matches[0]
    metadata = {
        "memory_id": f"memory_{uuid4().hex}",
        "kind": kind,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "effective_date": _date(effective_date),
        "review_status": "user_approved",
        "draft_origin": "user_provided_or_edited",
        "evidence_status": "exact_spans_verified_semantic_support_not_automated",
        "source_snapshot": manifest["corpus_hash"],
        "supersedes_doc_id": supersedes,
    }
    target = _new_note_path(root)
    lines = [text.strip(), "", "## Evidence reviewed", "",
             "> Exact quotations were checked against a frozen snapshot. The user approved this "
             "note; source support is not automatically judged.", ""]
    for item in sources:
        source_path = item.get("source_path")
        if source_path:
            source_file = (root / source_path).resolve()
            if not source_file.is_relative_to(root):
                raise ValueError("evidence source path escapes the vault")
            link = quote(Path(os.path.relpath(source_file, target.parent)).as_posix(),
                         safe="/")
            heading = f"[{item['title']}]({link})"
        else:
            heading = item["title"]
        lines.extend([f"### {item['evidence_id']} · {heading}", "",
                      f"Snapshot offsets: {item['start']}–{item['end']}", "",
                      *["> " + line for line in item["quote"].splitlines()], ""])
        source_metadata = docs[item["doc_id"]]["metadata"]
        if source_path and (not source_file.is_file() or
                            hashlib.sha256(source_file.read_bytes()).hexdigest()
                            != source_metadata.get("source_sha256")):
            lines.extend(["Source changed or moved after the snapshot; the quotation above "
                          "is from the frozen copy.", ""])
    if supersedes is not None:
        previous = docs[supersedes]
        path = previous["metadata"].get("source_path")
        if path:
            link = quote(Path(os.path.relpath(root / path, target.parent)).as_posix(), safe="/")
            lines.extend(["## Supersedes", "", f"[{previous['title']}]({link})", ""])
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    content = f"---\n{frontmatter}\n---\n\n# {title.strip()}\n\n" + "\n".join(lines).rstrip() + "\n"
    with target.open("x", encoding="utf-8") as stream:
        stream.write(content)
    return target
