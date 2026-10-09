"""Write a human-selected, evidence-checked weekly digest into an Obsidian vault.

This is an offline publisher, not paper discovery, ranking, or semantic review.
The selection declares human review; exact spans and source identity are checked
against one immutable public-paper snapshot before any vault file is created.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path, PurePosixPath

import yaml

from src.eval.artifacts import content_hash
from src.research.agent import load_snapshot
from src.research.papers import ARXIV_ID

SELECTION_VERSION = "envoy-weekly-selection-v1"
_WEEK = re.compile(r"(\d{4})-W(\d{2})\Z")
_ROOT_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]*\Z")


@dataclass(frozen=True)
class DigestPlan:
    vault: Path
    week: str
    corpus_hash: str
    review_status: str
    files: tuple[tuple[Path, str], ...]

    def summary(self, status: str) -> dict:
        return {
            "status": status,
            "week": self.week,
            "corpus_hash": self.corpus_hash,
            "review_status": self.review_status,
            "files": [str(path.relative_to(self.vault)) for path, _ in self.files],
        }


def _nonempty(value: object, field: str, *, max_chars: int = 1200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise ValueError(f"{field} must be nonempty text of at most {max_chars} characters")
    return value.strip()


def _inline(value: str) -> str:
    """Show supplied prose as text, without introducing user-authored links."""
    collapsed = " ".join(value.split())
    return re.sub(r"([\\`*_\[\]<>])", r"\\\1", collapsed)


def _wiki_alias(value: str) -> str:
    """Keep a public-source title from breaking a generated wikilink alias."""
    return " ".join(value.split()).translate(str.maketrans("", "", "[]|\\"))[:160] or "Paper"


def _fenced_quote(quote: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", quote)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{quote}\n{fence}"


def _week(value: object) -> tuple[str, date, date]:
    if not isinstance(value, str) or (match := _WEEK.fullmatch(value)) is None:
        raise ValueError("week must be an ISO week such as 2026-W39")
    year, number = map(int, match.groups())
    try:
        first = date.fromisocalendar(year, number, 1)
    except ValueError as exc:
        raise ValueError("week is not a valid ISO week") from exc
    return value, first, first + timedelta(days=6)


def _relative_root(value: str) -> PurePosixPath:
    if (not isinstance(value, str) or not value or value.startswith("/")
            or "\\" in value or "\x00" in value):
        raise ValueError("output root must be a relative vault folder")
    parts = value.split("/")
    if any(not _ROOT_PART.fullmatch(part) for part in parts):
        raise ValueError("output root contains an unsafe path component")
    return PurePosixPath(*parts)


def _slug(title: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", title).strip("-")[:56].strip("-") or "paper"


def _within_vault(vault: Path, path: Path) -> Path:
    if not path.resolve().is_relative_to(vault):
        raise ValueError("output path escapes the selected vault")
    return path


def _related_notes(value: object, vault: Path) -> list[dict]:
    """Resolve optional, existing vault-local Markdown links without editing them."""
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 5:
        raise ValueError("related_notes must contain at most five vault notes")
    related = []
    seen_files: set[tuple[int, int]] = set()
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("related note must be a path/reason object")
        raw = _nonempty(item.get("path"), "related note path", max_chars=240)
        reason = _nonempty(item.get("reason"), "related note reason", max_chars=240)
        parts = raw.split("/")
        if (raw.startswith("/") or not raw.endswith(".md")
                or any(part in {"", ".", ".."} for part in parts)
                or any(char in raw for char in "\\[]|#:\x00")
                or any(ord(char) < 32 for char in raw)):
            raise ValueError("related note path must be a safe vault-relative .md path")
        path = vault
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("related note path cannot traverse a symlink")
        if not path.resolve().is_relative_to(vault):
            raise ValueError("related note path escapes the selected vault")
        if not path.is_file():
            raise ValueError(f"related note does not exist in the vault: {raw}")
        stat = path.stat()
        identity = (stat.st_dev, stat.st_ino)
        if identity in seen_files:
            raise ValueError("duplicate related note path")
        seen_files.add(identity)
        with path.open(encoding="utf-8", errors="replace") as stream:
            heading = re.search(r"(?m)^# (.+)$", stream.read(8192))
        title = _wiki_alias(heading.group(1) if heading else path.stem.replace("-", " "))
        related.append({"path": raw, "reason": reason, "title": title})
    return related


def _source_documents(snapshot: Path) -> tuple[dict, dict]:
    manifest, docs = load_snapshot(snapshot)
    if manifest.get("schema_version") != "research-snapshot-v1":
        raise ValueError("expected a public-paper research snapshot")
    papers = manifest.get("papers")
    if not isinstance(papers, list) or not papers or len(papers) != len(docs):
        raise ValueError("snapshot paper manifest does not match its corpus")
    seen: set[str] = set()
    for item in papers:
        if not isinstance(item, dict):
            raise ValueError("snapshot paper manifest contains an invalid entry")
        doc_id = item.get("doc_id")
        if (not isinstance(doc_id, str) or doc_id in seen or doc_id not in docs
                or not re.fullmatch(r"arxiv_\d{4}_\d{4,5}v\d+", doc_id)):
            raise ValueError("snapshot paper identity is missing or duplicated")
        seen.add(doc_id)
        doc = docs[doc_id]
        meta = doc.get("metadata", {})
        arxiv_id = item.get("arxiv_id")
        if (not isinstance(arxiv_id, str) or not ARXIV_ID.fullmatch(arxiv_id)
                or "v" not in arxiv_id or doc_id != "arxiv_" + arxiv_id.replace(".", "_")
                or doc.get("doc_id") != doc_id or doc.get("title") != item.get("title")
                or meta.get("arxiv_id") != arxiv_id
                or meta.get("source_url") != f"https://arxiv.org/abs/{arxiv_id}"
                or item.get("source_url") != meta["source_url"]
                or item.get("coverage") != meta.get("coverage")):
            raise ValueError(f"snapshot source identity mismatch for {doc_id}")
        corpus_file = snapshot / "corpus" / f"{doc_id}.json"
        if not corpus_file.is_file() or item.get("sha256") != content_hash(corpus_file):
            raise ValueError(f"snapshot source hash mismatch for {doc_id}")
    return manifest, docs


def _selected_papers(selection: dict, manifest: dict, docs: dict,
                     week_start: date, week_end: date, vault: Path) -> list[dict]:
    if selection.get("schema_version") != SELECTION_VERSION:
        raise ValueError("unsupported weekly selection schema")
    if selection.get("corpus_hash") != manifest["corpus_hash"]:
        raise ValueError("selection and snapshot corpus hashes differ")
    if selection.get("review_status") not in {"human_reviewed", "agent_authored_draft"}:
        raise ValueError("review_status must be human_reviewed or agent_authored_draft")
    papers = selection.get("papers")
    if not isinstance(papers, list) or not 1 <= len(papers) <= 10:
        raise ValueError("select between one and ten papers")
    manifest_by_id = {item["doc_id"]: item for item in manifest["papers"]}
    seen_ids: set[str] = set()
    seen_papers: set[str] = set()
    reviewed: list[dict] = []
    in_week_count = 0
    for row in papers:
        if not isinstance(row, dict):
            raise ValueError("selected paper must be an object")
        doc_id = row.get("doc_id")
        if not isinstance(doc_id, str) or doc_id not in docs:
            raise ValueError("selected paper is absent from the snapshot")
        doc = docs[doc_id]
        source = manifest_by_id[doc_id]
        arxiv_id = source["arxiv_id"]
        base_id = arxiv_id.split("v", 1)[0]
        if doc_id in seen_ids or base_id in seen_papers:
            raise ValueError("duplicate selected paper or arXiv revision")
        seen_ids.add(doc_id)
        seen_papers.add(base_id)
        if row.get("arxiv_id") != arxiv_id or row.get("source_url") != source["source_url"]:
            raise ValueError(f"selected source identity mismatch for {doc_id}")
        metadata = doc["metadata"]
        try:
            submitted = date.fromisoformat(metadata["submitted"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"paper {doc_id} has no valid submitted date") from exc
        updated_raw = metadata.get("updated")
        try:
            updated = date.fromisoformat(updated_raw) if updated_raw else None
        except (TypeError, ValueError) as exc:
            raise ValueError(f"paper {doc_id} has an invalid updated date") from exc
        if updated is not None and updated < submitted:
            raise ValueError(f"paper {doc_id} has an updated date before submission")
        # A pinned revision belongs to its revision week, not the older week
        # when the underlying paper was first submitted.
        version_date = updated or submitted
        role = row.get("role", "new_or_revised")
        if role not in {"new_or_revised", "context"}:
            raise ValueError("paper role must be new_or_revised or context")
        submitted_this_week = week_start <= submitted <= week_end
        if role == "new_or_revised":
            if not week_start <= version_date <= week_end:
                raise ValueError(
                    f"paper {doc_id} pinned version is neither new nor revised "
                    "in the selected week"
                )
            if row.get("context_reason") is not None:
                raise ValueError("context_reason is only valid for a context paper")
            timing = "New this week" if submitted_this_week else "Revised this week"
            context_reason = None
            in_week_count += 1
        else:
            if version_date >= week_start:
                raise ValueError(f"context paper {doc_id} must predate the selected week")
            context_reason = _nonempty(row.get("context_reason"), "context_reason",
                                       max_chars=240)
            timing = "Background context; not counted as this week's new research"
        why = _nonempty(row.get("why_it_matters"), "why_it_matters", max_chars=500)
        limitations = row.get("limitations")
        if not isinstance(limitations, list) or not 1 <= len(limitations) <= 8:
            raise ValueError("each paper needs one to eight limitations")
        limitations = [_nonempty(item, "limitation", max_chars=500)
                       for item in limitations]
        related_notes = _related_notes(row.get("related_notes"), vault)
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 8:
            raise ValueError("each paper needs one to eight exact evidence spans")
        spans = []
        seen_spans: set[tuple[int, int]] = set()
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("evidence span must be an object")
            if item.get("doc_id", doc_id) != doc_id:
                raise ValueError(f"evidence source identity mismatch for {doc_id}")
            start, end, quote = item.get("start"), item.get("end"), item.get("quote")
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(doc["text"])
                    or not isinstance(quote, str) or not quote or len(quote) > 2000
                    or doc["text"][start:end] != quote):
                raise ValueError(f"invalid exact evidence span for {doc_id}")
            if (start, end) in seen_spans:
                raise ValueError(f"duplicate evidence span for {doc_id}")
            seen_spans.add((start, end))
            spans.append({"start": start, "end": end, "quote": quote})
        reviewed.append({"doc": doc, "source": source, "why": why,
                         "limitations": limitations, "evidence": spans,
                         "role": role, "timing": timing, "context_reason": context_reason,
                         "related_notes": related_notes,
                         "submitted": submitted.isoformat(),
                         "updated": updated.isoformat() if updated else None,
                         "version_date": version_date.isoformat()})
    if in_week_count == 0:
        raise ValueError("weekly digest needs at least one new or revised paper in the week")
    return reviewed


def _frontmatter(fields: dict) -> str:
    return "---\n" + yaml.safe_dump(fields, sort_keys=False, allow_unicode=True).strip() + "\n---\n"


def _paper_note(item: dict, *, week: str, weekly_link: str,
                corpus_hash: str, review_status: str,
                analysis_note: str | None) -> str:
    doc, source = item["doc"], item["source"]
    human_reviewed = review_status == "human_reviewed"
    lines = [
        _frontmatter({
            "record_id": f"paper-{source['arxiv_id']}", "kind": "paper",
            "arxiv_id": source["arxiv_id"], "source_url": source["source_url"],
            "source_snapshot": corpus_hash,
            "review_status": ("human_reviewed_declared" if human_reviewed
                              else "agent_authored_draft"),
            "evidence_status": ("exact_spans_verified_semantic_support_human_reviewed"
                                if human_reviewed else "exact_spans_verified_semantic_unreviewed"),
            "coverage": source["coverage"],
            "weekly_role": item["role"], "submitted": item["submitted"],
            "updated": item["updated"], "pinned_version_date": item["version_date"],
        }),
        f"# {_inline(doc['title'])}", "",
        f"[Pinned arXiv version]({source['source_url']}) · "
        f"Extract coverage: `{source['coverage']}` · [[{weekly_link}|Week {week} digest]]", "",
        f"**{item['timing']}.** Pinned version: {item['version_date']} · "
        f"Submitted: {item['submitted']} · "
        f"Updated: {item['updated'] or 'not recorded'}", "",
        *(["**Analysis scope:** " + _inline(analysis_note), ""] if analysis_note else []),
        "## Why it matters", "", _inline(item["why"]), "",
        ("## Reviewed evidence" if human_reviewed else "## Evidence to review"), "",
        ("These excerpts were selected in a declared human review. Character offsets and exact "
         "text were verified against the frozen extract; semantic support was not judged by code."
         if human_reviewed else
         "Agent-authored draft awaiting human review. Character offsets and exact text were "
         "verified against the frozen extract; relevance and semantic support are unreviewed."),
        "",
    ]
    for span in item["evidence"]:
        lines.extend([f"**Offsets {span['start']}–{span['end']}**", "",
                      _fenced_quote(span["quote"]), ""])
    if item["context_reason"]:
        lines.extend(["## Why this older paper is here", "",
                      _inline(item["context_reason"]), ""])
    if item["related_notes"]:
        lines.extend(["## Connections to my vault", ""])
        lines.extend(
            f"- [[{note['path']}|{note['title']}]] — {_inline(note['reason'])}"
            for note in item["related_notes"]
        )
        lines.append("")
    lines.extend(["## Limitations", ""])
    lines.extend(f"- {_inline(value)}" for value in item["limitations"])
    lines.extend(["", "The snapshot contains selected extracted text, not a complete review "
                  "of figures, tables, or related literature.", ""])
    return "\n".join(lines)


def _digest_note(*, week: str, topic: str, corpus_hash: str,
                 papers: list[tuple[dict, str]], review_status: str,
                 next_measurement: str | None, analysis_note: str | None) -> str:
    human_reviewed = review_status == "human_reviewed"
    lines = [
        _frontmatter({
            "record_id": f"weekly-ai-research-{week}", "kind": "weekly_digest",
            "iso_week": week, "topic": topic, "source_snapshot": corpus_hash,
            "review_status": ("human_reviewed_declared" if human_reviewed
                              else "agent_authored_draft"),
        }),
        f"# AI research · {week}", "", f"**Topic:** {_inline(topic)}", "",
        f"**Selected papers:** {len(papers)} · **Snapshot:** `{corpus_hash}`", "",
        ("This is a bounded, human-selected update, not an exhaustive literature search. "
         "Exact quote offsets were checked automatically; the selection declares human review "
         "of relevance and meaning."
         if human_reviewed else
         "Agent-authored draft for human review, not an exhaustive literature search. "
         "Exact quote offsets were checked automatically; relevance and semantic support "
         "have not been independently reviewed."), "",
    ]
    if analysis_note:
        lines.extend(["**Analysis scope:** " + _inline(analysis_note), ""])
    lines.extend(["## Papers", ""])
    for item, link in papers:
        source, doc = item["source"], item["doc"]
        lines.extend([
            f"### [[{link}|{_wiki_alias(doc['title'])}]]", "",
            f"**{item['timing']}.** Pinned version: {item['version_date']} · "
            f"Submitted: {item['submitted']} · "
            f"Updated: {item['updated'] or 'not recorded'}", "",
            f"**Why it matters:** {_inline(item['why'])}", "",
            f"**Source:** [arXiv {source['arxiv_id']}]({source['source_url']}) "
            f"· `{source['coverage']}`", "",
        ])
        evidence_heading = "Reviewed evidence" if human_reviewed else "Evidence to review"
        lines.extend([
            f"**Evidence:** [[{link}#{evidence_heading}|"
            f"{len(item['evidence'])} exact source excerpts in the paper note]]", "",
        ])
        lines.extend(["**Limitations:**", ""])
        lines.extend(f"- {_inline(value)}" for value in item["limitations"])
        lines.append("")
        if item["context_reason"]:
            lines.extend(["**Why included as context:** "
                          + _inline(item["context_reason"]), ""])
        if item["related_notes"]:
            lines.extend(["**Related vault notes:**", ""])
            lines.extend(
                f"- [[{note['path']}|{note['title']}]] — {_inline(note['reason'])}"
                for note in item["related_notes"]
            )
            lines.append("")
    if next_measurement:
        lines.extend(["## Next measurement to consider", "",
                      "Proposal, not a finding: " + _inline(next_measurement), ""])
    return "\n".join(lines)


def plan_weekly_digest(snapshot: Path, selection_file: Path, vault: Path,
                       *, output_root: str = "AI Research") -> DigestPlan:
    """Validate the complete batch and render it in memory, without writing."""
    snapshot = snapshot.expanduser().resolve(strict=True)
    vault = vault.expanduser().resolve(strict=True)
    if not vault.is_dir():
        raise ValueError("vault must be a directory")
    root = _relative_root(output_root)
    selection = json.loads(selection_file.read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise ValueError("selection must be a JSON object")
    week, week_start, week_end = _week(selection.get("week"))
    topic = _nonempty(selection.get("topic"), "topic", max_chars=120)
    next_raw = selection.get("next_measurement")
    next_measurement = (_nonempty(next_raw, "next_measurement", max_chars=600)
                        if next_raw is not None else None)
    analysis_raw = selection.get("analysis_note")
    analysis_note = (_nonempty(analysis_raw, "analysis_note", max_chars=500)
                     if analysis_raw is not None else None)
    manifest, docs = _source_documents(snapshot)
    reviewed = _selected_papers(selection, manifest, docs, week_start, week_end, vault)
    corpus_hash = manifest["corpus_hash"]
    review_status = selection["review_status"]
    weekly = _within_vault(vault, vault / root / "Weekly" / f"{week}.md")
    weekly_link = (root / "Weekly" / week).as_posix()
    files = []
    digest_items = []
    for item in reviewed:
        doc, source = item["doc"], item["source"]
        year = date.fromisoformat(item["submitted"]).year
        name = f"{source['arxiv_id']} - {_slug(doc['title'])}"
        relative = root / "Papers" / str(year) / name
        path = _within_vault(vault, vault / f"{relative}.md")
        files.append((path, _paper_note(item, week=week, weekly_link=weekly_link,
                                        corpus_hash=corpus_hash,
                                        review_status=review_status,
                                        analysis_note=analysis_note)))
        # The dotted arXiv ID looks like a file extension to the graph parser;
        # an explicit .md suffix also makes the Obsidian target unambiguous.
        digest_items.append((item, f"{relative.as_posix()}.md"))
    files.append((weekly, _digest_note(week=week, topic=topic, corpus_hash=corpus_hash,
                                       papers=digest_items,
                                       review_status=review_status,
                                       next_measurement=next_measurement,
                                       analysis_note=analysis_note)))
    if len({path for path, _ in files}) != len(files):
        raise ValueError("two selected papers map to the same output path")
    for path, _ in files:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"weekly output already exists: {path}")
    return DigestPlan(vault, week, corpus_hash, review_status, tuple(files))


def write_weekly_digest(plan: DigestPlan, *, confirm: bool = False) -> dict:
    """Create every planned Markdown note exclusively; remove partial files on failure."""
    if not confirm:
        raise ValueError("explicit --confirm is required to write the vault")
    if plan.review_status != "human_reviewed":
        raise ValueError("a declared human_reviewed selection is required to write the vault")
    for path, _ in plan.files:
        _within_vault(plan.vault, path)
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"weekly output already exists: {path}")
    created: list[Path] = []
    try:
        for path, text in plan.files:
            path.parent.mkdir(parents=True, exist_ok=True)
            _within_vault(plan.vault, path)
            with path.open("x", encoding="utf-8") as stream:
                created.append(path)
                stream.write(text)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return plan.summary("written")
