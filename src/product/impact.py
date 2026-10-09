"""Conservative, source-triggered candidate review over one frozen vault.

This finds an earlier decision worth inspecting. Lexical overlap and links are
navigation signals, never evidence that the new source changes the decision.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from src.product.graph import build_graph
from src.product.memory import _navigation_only, _record_context
from src.research.agent import load_snapshot

IMPACT_VERSION = "envoy-impact-v1"
_FRONTMATTER = re.compile(r"\A---\n.*?\n---(?:\n|$)", re.S)
_WORDS = re.compile(r"[a-z][a-z0-9]{2,}")
_TITLE = re.compile(r"^# (\S.*)$", re.M)
_STOPWORDS = {
    "about", "after", "again", "also", "and", "are", "because", "before", "been",
    "between", "could", "date", "decision", "does", "each", "evidence", "first",
    "for", "from", "had", "has", "have", "into", "kind", "later", "measured",
    "more", "next", "note", "observed", "only", "our", "outcome", "plan",
    "project", "record", "result", "review", "run", "said", "same", "should",
    "source", "than", "that", "the", "their", "there", "these", "this",
    "through", "was", "were", "will", "with", "would", "your",
}
_DECISION_KINDS = {"decision", "correction", "lesson"}
_REVISION_SOURCE_KINDS = {"result", "correction"}


def _terms(value: str) -> set[str]:
    return set(_WORDS.findall(value.casefold())) - _STOPWORDS


def _body(text: str) -> str:
    frontmatter = _FRONTMATTER.match(text)
    return text[frontmatter.end():] if frontmatter else text


def visible_title(doc: dict) -> str:
    """Use the visible H1 for captured notes with opaque UUID filenames."""
    if doc.get("metadata", {}).get("source_path", "").lower().endswith(".md"):
        heading = _TITLE.search(_body(doc["text"]))
        if heading:
            return heading[1].strip()[:200]
    return doc["title"]


def _paragraphs(text: str) -> list[tuple[int, str]]:
    frontmatter = _FRONTMATTER.match(text)
    body_start = frontmatter.end() if frontmatter else 0
    result = []
    cursor = body_start
    for paragraph in text[body_start:].split("\n\n"):
        start = text.find(paragraph, cursor)
        cursor = start + len(paragraph) + 2
        clean = paragraph.strip()
        if (not clean or clean.startswith("#") and "\n" not in clean
                or _navigation_only(clean) or clean == "## Linked history"):
            continue
        result.append((start, paragraph))
    return result


def _passage(doc: dict, query_terms: set[str]) -> dict | None:
    paragraphs = _paragraphs(doc["text"])
    if not paragraphs:
        return None
    paragraphs.sort(
        key=lambda item: (-len(query_terms & _terms(item[1])), item[0]),
    )
    start, paragraph = paragraphs[0]
    if len(paragraph) > 1_200:
        match = next((match for match in _WORDS.finditer(paragraph)
                      if match.group().casefold() in query_terms), None)
        offset = max(0, (match.start() if match else 0) - 160)
        start += offset
        paragraph = paragraph[offset:offset + 1_200]
    end = start + len(paragraph)
    return {"start": start, "end": end, "quote": doc["text"][start:end]}


def _node(graph: dict, docs: dict, doc_id: str) -> dict:
    node = graph["nodes"][doc_id]
    return {**node, "title": visible_title(docs[doc_id]),
            "record": _record_context(docs[doc_id])}


def _rank_prior(graph: dict, docs: dict, source_id: str) -> tuple[str | None, set[str]]:
    source = docs[source_id]
    source_terms = _terms(visible_title(source) + " " + _body(source["text"]))
    source_title_terms = _terms(visible_title(source))
    source_terms.discard("learning")
    candidates = {
        doc_id: node for doc_id, node in graph["nodes"].items()
        if doc_id != source_id and node["kind"] in _DECISION_KINDS
    }
    if not candidates:
        return None, set()
    linked = {edge["target"] if edge["source"] == source_id else edge["source"]
              for edge in graph["edges"]
              if source_id in (edge["source"], edge["target"])}
    superseded = {edge["target"] for edge in graph["edges"]
                  if edge["type"] == "supersedes" and edge["source"] != source_id}
    document_terms = {
        doc_id: _terms(visible_title(docs[doc_id]) + " " + _body(docs[doc_id]["text"]))
        for doc_id in candidates
    }
    counts = Counter(term for terms in document_terms.values() for term in terms)
    ranked = []
    for doc_id, node in candidates.items():
        shared = source_terms & document_terms[doc_id]
        direct_link = doc_id in linked
        title_bridge = bool(
            source_title_terms & document_terms[doc_id]
            or _terms(visible_title(docs[doc_id])) & source_terms
        )
        if not direct_link and (len(shared) < 2 or not title_bridge):
            continue
        overlap = sum(math.log(1 + len(candidates) / (1 + counts[term]))
                      for term in shared)
        # A linked document is an explicit navigation candidate. It is still
        # not a semantic contradiction or automatic reason to revise it.
        score = (100 if direct_link else 0) + overlap
        if node["kind"] == "decision":
            score += 1
        if doc_id in superseded:
            score -= 10
        if _record_context(docs[doc_id]).get("review_status") == "user_approved":
            score += 0.5
        ranked.append((score, node.get("effective_date") or "", doc_id, shared))
    if not ranked:
        return None, set()
    ranked.sort(reverse=True)
    _, _, doc_id, shared = ranked[0]
    return doc_id, shared


def build_impact_review(snapshot: Path, source_path: str) -> dict:
    """Return exact passages and a cautious draft after a selected source arrives."""
    if not isinstance(source_path, str) or not source_path.strip():
        raise ValueError("source_path must name a selected vault document")
    manifest, docs = load_snapshot(snapshot)
    matches = [doc_id for doc_id, doc in docs.items()
               if doc["metadata"].get("source_path") == source_path]
    if len(matches) != 1:
        raise ValueError("source_path is absent or ambiguous in this snapshot")
    source_id = matches[0]
    graph = build_graph(snapshot)
    if source_id not in graph["nodes"]:
        raise ValueError("source_path is not a graph-indexed vault document")
    prior_id, shared = _rank_prior(graph, docs, source_id)
    source_node = _node(graph, docs, source_id)
    prior_node = _node(graph, docs, prior_id) if prior_id else None
    query_terms = shared or _terms(source_node["title"])
    evidence = []
    for doc_id in (source_id, prior_id):
        if doc_id is None:
            continue
        passage = _passage(docs[doc_id], query_terms)
        if passage is None:
            continue
        doc = docs[doc_id]
        evidence.append({
            "evidence_id": f"E{len(evidence) + 1}", "doc_id": doc_id,
            "title": visible_title(doc), **passage,
            "source_path": doc["metadata"].get("source_path"),
            "source_kind": doc["metadata"].get("source_kind"),
            "record": _record_context(doc),
        })
    source_quote = next((item for item in evidence if item["doc_id"] == source_id), None)
    prior_quote = next((item for item in evidence if item["doc_id"] == prior_id), None)
    if source_quote is None:
        raise ValueError("source has no reviewable body passage")
    status = "possible_impact" if prior_node is not None and prior_quote else "no_candidate"
    source_kind = source_node["kind"]
    proposal = None
    if status == "possible_impact" and source_kind in _REVISION_SOURCE_KINDS:
        proposal = {
            "title": f"Revisit {prior_node['title']}",
            "kind": "correction",
            "text": (
                f"Review the new {source_kind} in '{source_node['title']}' against "
                f"'{prior_node['title']}'. State what was observed, whether the earlier "
                "decision still applies, and what remains uncertain. Replace this "
                "draft with your conclusion before approving."
            ),
            "effective_date": source_node.get("effective_date"),
            "supersedes": prior_id,
            "evidence_ids": [source_quote["evidence_id"], prior_quote["evidence_id"]],
            "requires_edit": True,
            "origin": "local_review_scaffold",
        }
    if status == "no_candidate":
        reason = "No earlier decision with a clear link or enough shared project terms was found."
    elif proposal is None:
        reason = (
            "A related earlier conclusion was found, but this source is not marked as a "
            "measured result or correction. It may be a plan; no revision was drafted."
        )
    else:
        reason = (
            "A related earlier conclusion was found. Review both passages before "
            "deciding whether the new result changes it."
        )
    timeline = []
    for role, node, passage in (("prior", prior_node, prior_quote),
                                ("source", source_node, source_quote)):
        if node is not None and passage is not None:
            timeline.append({
                "role": role, "doc_id": node["doc_id"], "title": node["title"],
                "kind": node["kind"], "effective_date": node["effective_date"],
                "summary": passage["quote"][:280],
            })
    return {
        "schema_version": IMPACT_VERSION,
        "retriever": "linked_and_lexical_candidate_v1",
        "status": status,
        "source": source_node,
        "prior": prior_node if status == "possible_impact" else None,
        "reason": reason,
        "evidence": evidence,
        "proposal": proposal,
        "timeline": timeline,
        "corpus_hash": manifest["corpus_hash"],
        "warning": (
            "Links and word overlap suggest what to review; exact quotations prove "
            "provenance, not semantic support or contradiction."
        ),
    }
