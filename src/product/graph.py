"""Deterministic, read-only links over a frozen Obsidian vault snapshot.

This is a navigation index, not a semantic claim that two notes agree. Missing
and ambiguous targets stay unresolved instead of becoming invented graph edges.
"""

from __future__ import annotations

import posixpath
import re
from collections import defaultdict
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

import yaml

from src.research.agent import load_snapshot

GRAPH_VERSION = "obsidian-links-v1"
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---(?:\n|$)", re.S)
_WIKILINK = re.compile(r"(?<!\\)\[\[([^\]\n]+)\]\]")
_MARKDOWN_LINK = re.compile(r"(?<!\\)(?<!\!)\[[^\]\n]+\]\((<[^>\n]+>|[^)\n]+)\)")
_TYPED_FIELDS = ("supersedes", "supersedes_doc_id", "derived_from")


def _frontmatter(text: str) -> tuple[dict, str]:
    match = _FRONTMATTER.match(text)
    if match is None:
        return {}, text
    try:
        metadata = yaml.safe_load(match[1]) or {}
    except yaml.YAMLError:
        metadata = {}
    return (metadata if isinstance(metadata, dict) else {}), text[match.end():]


def _stable_id(metadata: dict, doc_id: str) -> str:
    for field in ("record_id", "memory_id"):
        value = metadata.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return doc_id


def _prose_lines(body: str) -> str:
    """Remove fenced/inline code so example syntax does not create note edges."""
    rendered = []
    fence_char, fence_size = None, 0
    for line in body.splitlines():
        match = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        if match:
            marker = match[1]
            if fence_char is None:
                fence_char, fence_size = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_size:
                fence_char, fence_size = None, 0
            rendered.append("")
            continue
        if fence_char is not None:
            rendered.append("")
            continue
        rendered.append(re.sub(r"(`+)(.*?)\1", "", line))
    return "\n".join(rendered)


def _candidate_paths(path: str, source_path: str, *, wikilink: bool) -> list[str]:
    """Return safe, possible vault-relative paths without touching the live vault."""
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        return []
    part = PurePosixPath(path)
    if any(component in {"", ".", ".."} for component in part.parts[1:]):
        # Parent traversal is only allowed for ordinary relative Markdown links.
        if wikilink or not path.startswith(("./", "../")):
            return []
    suffixes = [path] if PurePosixPath(path).suffix else [path + ".md", path]
    candidates = []
    for suffix in suffixes:
        if wikilink and not suffix.startswith(("./", "../")):
            candidates.append(posixpath.normpath(suffix))
        relative = posixpath.normpath(posixpath.join(posixpath.dirname(source_path), suffix))
        if relative not in candidates:
            candidates.append(relative)
    return [candidate for candidate in candidates
            if candidate != ".." and not candidate.startswith("../")
            and not candidate.startswith("/")]


def _reference_values(value: object) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [item.strip() for item in value
                if isinstance(item, str) and item.strip()]
    return []


def build_graph(snapshot: Path) -> dict:
    """Index source links and typed frontmatter edges from one immutable snapshot.

    Every resolved edge uses snapshot doc IDs. ``record_id`` (or ``memory_id``)
    gives each node a stable identity across snapshots when the note declares one.
    If a path, title, or record ID cannot be resolved uniquely, it is reported in
    ``unresolved`` and never linked to an arbitrary node.
    """
    manifest, docs = load_snapshot(snapshot)
    nodes: dict[str, dict] = {}
    metadata_by_doc: dict[str, dict] = {}
    prose_by_doc: dict[str, str] = {}
    by_record: dict[str, list[str]] = defaultdict(list)
    by_path: dict[str, str] = {}
    by_stem: dict[str, list[str]] = defaultdict(list)
    for doc_id, doc in sorted(docs.items()):
        source_path = doc.get("metadata", {}).get("source_path")
        if not isinstance(source_path, str) or not source_path:
            continue
        path = PurePosixPath(source_path)
        if path.suffix.lower() != ".md":
            note_metadata, prose = {}, ""
        else:
            note_metadata, prose = _frontmatter(doc["text"])
        record_id = _stable_id(note_metadata, doc_id)
        node = {
            "doc_id": doc_id,
            "record_id": record_id,
            "title": doc["title"],
            "source_path": source_path,
            "source_kind": doc["metadata"].get("source_kind"),
            "kind": note_metadata.get("kind"),
            "effective_date": str(note_metadata["effective_date"])
            if note_metadata.get("effective_date") is not None else None,
        }
        nodes[doc_id] = node
        metadata_by_doc[doc_id] = note_metadata
        prose_by_doc[doc_id] = _prose_lines(prose)
        by_record[record_id].append(doc_id)
        by_path[source_path] = doc_id
        by_stem[path.stem].append(doc_id)

    edges: set[tuple[str, str, str]] = set()
    unresolved: set[tuple[str, str, str, str]] = set()

    def add(source: str, relation: str, raw_target: str,
            matches: list[str], reason: str = "missing") -> None:
        unique = sorted(set(matches))
        if len(unique) == 1:
            edges.add((source, unique[0], relation))
        else:
            unresolved.add((source, relation, raw_target,
                            "ambiguous" if unique else reason))

    for doc_id, node in nodes.items():
        if not node["source_path"].lower().endswith(".md"):
            continue
        metadata = metadata_by_doc[doc_id]
        for field in _TYPED_FIELDS:
            for raw in _reference_values(metadata.get(field)):
                if field == "supersedes_doc_id":
                    matches = [raw] if raw in nodes else []
                    relation = "supersedes"
                else:
                    matches = by_record.get(raw, [])
                    if not matches and raw in nodes:
                        matches = [raw]
                    relation = field
                add(doc_id, relation, raw, matches)

        prose = prose_by_doc[doc_id]
        for match in _WIKILINK.finditer(prose):
            raw = match[1].split("|", 1)[0].split("#", 1)[0].strip()
            if not raw:  # Heading link within the current note.
                continue
            if "://" in raw:
                continue
            paths = _candidate_paths(raw, node["source_path"], wikilink=True)
            if not paths:
                matches = []
            elif "/" not in raw:
                # A bare name is safe only when unique across the snapshot. A
                # same-folder guess could silently follow the wrong note later.
                name = PurePosixPath(raw)
                matches = ([candidate for path, candidate in by_path.items()
                            if PurePosixPath(path).name == name.name]
                           if name.suffix else by_stem.get(name.stem, []))
            else:
                matches = [by_path[path] for path in paths if path in by_path]
                if not matches:
                    matches = [candidate for path, candidate in by_path.items()
                               if path.endswith("/" + raw)
                               or path.endswith("/" + raw + ".md")]
            add(doc_id, "wikilink", raw, matches,
                "unsafe_path" if not paths else "missing")

        for match in _MARKDOWN_LINK.finditer(prose):
            raw = match[1].strip()
            if raw.startswith("<") and raw.endswith(">"):
                raw = raw[1:-1]
            try:
                parsed = urlsplit(raw)
            except ValueError:
                continue
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue  # External URL or a same-document heading anchor.
            path = unquote(parsed.path)
            paths = _candidate_paths(path, node["source_path"], wikilink=False)
            matches = [by_path[candidate] for candidate in paths if candidate in by_path]
            add(doc_id, "markdown_link", raw, matches,
                "unsafe_path" if not paths else "missing")

    return {
        "schema_version": GRAPH_VERSION,
        "corpus_hash": manifest["corpus_hash"],
        "nodes": nodes,
        "edges": [{"source": source, "target": target, "type": relation}
                  for source, target, relation in sorted(edges)],
        "unresolved": [
            {"source": source, "type": relation, "target_ref": target,
             "reason": reason}
            for source, relation, target, reason in sorted(unresolved)
        ],
    }


def one_hop_neighbors(graph: dict, identifier: str) -> list[dict]:
    """Return both outgoing links and backlinks for a doc ID or stable record ID."""
    nodes = graph["nodes"]
    matches = ([identifier] if identifier in nodes else
               [doc_id for doc_id, node in nodes.items()
                if node["record_id"] == identifier])
    if len(matches) != 1:
        raise ValueError("node identifier is missing or ambiguous")
    selected = matches[0]
    adjacent: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for edge in graph["edges"]:
        if edge["source"] == selected:
            adjacent[edge["target"]].add(("outgoing", edge["type"]))
        if edge["target"] == selected:
            adjacent[edge["source"]].add(("incoming", edge["type"]))
    return [{**nodes[doc_id], "connections": [
        {"direction": direction, "type": relation}
        for direction, relation in sorted(adjacent[doc_id])
    ]} for doc_id in sorted(adjacent)]
