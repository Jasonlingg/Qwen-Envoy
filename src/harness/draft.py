"""Validate a structured paper answer and render an unreviewed Obsidian draft.

This gate checks citation structure, not whether a passage entails a claim.
Only the trusted question and verified bundle can introduce source material.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import yaml

DRAFT_VERSION = "research-question-draft-v1"
_ROOT_FIELDS = {"schema_version", "claims", "open_questions", "proposed_applications"}
_CLAIM_FIELDS = {"text", "passage_ids"}
_MAX_CLAIMS = 10
_MAX_QUESTIONS = 5
_MAX_APPLICATIONS = 5
_MAX_TEXT = 600


def _line(value: object, label: str, *, limit: int = _MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must be nonempty text of at most {limit} characters")
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise ValueError(f"{label} contains a control character")
    collapsed = " ".join(value.split())
    # Keep model prose as plain text, so it cannot add links, headings, or HTML.
    return re.sub(r"([\\`*_\[\]<>#])", r"\\\1", collapsed)


def _fenced_quote(quote: str) -> str:
    longest = max((len(run) for run in re.findall(r"`+", quote)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{quote}\n{fence}"


def _cited_line(
    value: object, label: str, aliases: Mapping[str, str], *, limit: int = _MAX_TEXT
) -> str:
    if not isinstance(value, Mapping) or set(value) != _CLAIM_FIELDS:
        raise ValueError(f"{label} has missing or unexpected fields")
    text = _line(value.get("text"), label, limit=limit)
    ids = value.get("passage_ids")
    if (
        not isinstance(ids, list)
        or not 1 <= len(ids) <= 5
        or any(not isinstance(item, str) for item in ids)
        or len(set(ids)) != len(ids)
        or any(item not in aliases for item in ids)
    ):
        raise ValueError(f"{label} needs one to five known passage IDs")
    references = " ".join(f"[{aliases[item]}](#{aliases[item].lower()})" for item in ids)
    return f"{text} {references}"


def render_question_draft(bundle: Mapping[str, object], draft: Mapping[str, object]) -> str:
    """Render only cited claims and explicitly labeled questions/applications.

    ``bundle`` must come from the host's independent provenance verifier. This
    function does not treat a model's candidate findings as verified claims.
    """
    if bundle.get("schema_version") != "research-evidence-bundle-v1":
        raise ValueError("expected a verified research evidence bundle")
    if (
        bundle.get("provenance_status") != "valid"
        or bundle.get("semantic_support_status") != "not_reviewed"
    ):
        raise ValueError("draft requires valid provenance and unreviewed semantics")
    passages = bundle.get("passages")
    if not isinstance(passages, list) or not passages:
        raise ValueError("draft requires at least one verified passage")
    passage_by_id: dict[str, dict] = {}
    for passage in passages:
        if not isinstance(passage, dict):
            raise ValueError("bundle passage is invalid")
        passage_id = passage.get("passage_id")
        if not isinstance(passage_id, str) or not passage_id or passage_id in passage_by_id:
            raise ValueError("bundle passage IDs must be unique")
        passage_by_id[passage_id] = passage
    if not isinstance(draft, Mapping) or set(draft) != _ROOT_FIELDS:
        raise ValueError("draft has missing or unexpected fields")
    if draft.get("schema_version") != DRAFT_VERSION:
        raise ValueError("unsupported draft schema")
    claims = draft.get("claims")
    if not isinstance(claims, list) or not 1 <= len(claims) <= _MAX_CLAIMS:
        raise ValueError("draft needs one to ten cited claims")
    questions = draft.get("open_questions")
    applications = draft.get("proposed_applications")
    if not isinstance(questions, list) or len(questions) > _MAX_QUESTIONS:
        raise ValueError("open_questions must be a bounded list")
    if not isinstance(applications, list) or len(applications) > _MAX_APPLICATIONS:
        raise ValueError("proposed_applications must be a bounded list")

    aliases = {passage_id: f"E{index}" for index, passage_id in enumerate(passage_by_id, start=1)}
    claim_lines = [f"- {_cited_line(item, 'claim', aliases)}" for item in claims]
    question_lines = [
        f"- Question: {_cited_line(item, 'open question', aliases, limit=300)}"
        for item in questions
    ]
    application_lines = [
        f"- Proposed application (not verified): "
        f"{_cited_line(item, 'proposed application', aliases)}"
        for item in applications
    ]
    run_id = bundle.get("run_id")
    question = bundle.get("question")
    corpus_hash = bundle.get("corpus_hash")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("bundle has no run ID")
    if not isinstance(corpus_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", corpus_hash):
        raise ValueError("bundle has no pinned corpus hash")
    title = _line(question, "trusted question", limit=4000)
    metadata = {
        "kind": "research_question_draft",
        "review_status": "agent_authored_draft",
        "semantic_support_status": "not_reviewed",
        "run_id": run_id,
        "source_snapshot": corpus_hash,
    }
    lines = [
        "---",
        yaml.safe_dump(metadata, sort_keys=False).strip(),
        "---",
        "",
        f"# {title}",
        "",
        "Draft for human review. Exact source spans were checked; claim support was not.",
        "",
        "## Claims to review",
        "",
        *claim_lines,
        "",
    ]
    if question_lines:
        lines.extend(["## Open questions", "", *question_lines, ""])
    if application_lines:
        lines.extend(["## Possible applications", "", *application_lines, ""])
    lines.extend(["## Source passages", ""])
    for passage_id, passage in passage_by_id.items():
        alias = aliases[passage_id]
        doc_id = passage.get("doc_id")
        start, end, quote = passage.get("start"), passage.get("end"), passage.get("quote")
        if (
            not isinstance(doc_id, str)
            or type(start) is not int
            or type(end) is not int
            or not isinstance(quote, str)
        ):
            raise ValueError("bundle passage is incomplete")
        lines.extend(
            [
                f"### {alias}",
                "",
                f"Source `{_line(doc_id, 'document ID', limit=200)}` · offsets {start}–{end} "
                f"· passage `{passage_id}`",
                "",
                _fenced_quote(quote),
                "",
            ]
        )
    return "\n".join(lines)
