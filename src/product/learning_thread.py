"""Dated learning threads in a working index outside the reviewed Obsidian vault.

The index records a user's initial view, proposed changes, and explicitly
accepted revisions. It never writes to the vault or promotes an agent draft.
Reference IDs are pointers for human review, not verified semantic support.
"""

from __future__ import annotations

import copy
import fcntl
import json
import os
import re
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import yaml

SCHEMA_VERSION = "learning-thread-v1"
_THREAD_ID = re.compile(r"thread_[0-9a-f]{32}\Z")
_PROPOSAL_ID = re.compile(r"proposal_[0-9a-f]{32}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RELATIONS = {"supports", "challenges", "context", "unclear"}


def _required(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value.strip()


def _date(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("effective_date must be an ISO date")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ValueError("effective_date must be an ISO date") from exc


def _time(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str):
        raise ValueError("timestamp must have a timezone")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("timestamp must be an ISO datetime with a timezone") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must have a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _refs(values: list[str] | tuple[str, ...], name: str) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list of reference IDs")
    refs = [_required(value, name) for value in values]
    if len(refs) != len(set(refs)):
        raise ValueError(f"{name} contains duplicate reference IDs")
    return refs


def _assessments(values: list[dict] | tuple[dict, ...], refs: list[str]) -> list[dict]:
    """Store user judgments separately from mechanically checked source spans."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("assessments must be a list")
    result = []
    seen = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("assessment must be an object")
        reference = _required(value.get("reference"), "assessment reference")
        relation = value.get("relation")
        if reference not in refs or reference in seen or relation not in _RELATIONS:
            raise ValueError("assessment must name a selected reference and valid relation")
        result.append({"reference": reference, "relation": relation,
                       "basis": "user_assessment_not_verified_support"})
        seen.add(reference)
    return result


def _store_dir(store_dir: Path) -> Path:
    root = Path(store_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("store_dir must be a directory")
    return root


def _path(root: Path, thread_id: str) -> Path:
    if not isinstance(thread_id, str) or not _THREAD_ID.fullmatch(thread_id):
        raise ValueError("invalid thread_id")
    return root / f"{thread_id}.json"


@contextmanager
def _locked(root: Path):
    with (root / ".learning-threads.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _read(path: Path) -> dict:
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError("learning thread does not exist") from exc
    if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported learning thread state")
    return state


def _write(path: Path, state: dict) -> None:
    """Replace a complete JSON document, so readers never see a partial update."""
    temporary = path.with_name(f".{path.stem}-{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def create_thread(
    store_dir: Path, *, question: str, current_view: str, effective_date: str,
    evidence_refs: list[str] | tuple[str, ...] = (),
    observation_refs: list[str] | tuple[str, ...] = (),
    recorded_at: str | None = None,
) -> dict:
    """Record a user-stated starting view, without touching the vault."""
    revision = {
        "revision_id": f"revision_{uuid4().hex}",
        "view": _required(current_view, "current_view"),
        "reason": "Initial view stated by the user.",
        "effective_date": _date(effective_date),
        "recorded_at": _time(recorded_at),
        "accepted_by": "user_initial_view",
        "proposal_id": None,
        "supersedes_revision_id": None,
        "evidence_refs": _refs(evidence_refs, "evidence_refs"),
        "observation_refs": _refs(observation_refs, "observation_refs"),
        "assessments": [],
    }
    root = _store_dir(store_dir)
    state = {
        "schema_version": SCHEMA_VERSION,
        "thread_id": f"thread_{uuid4().hex}",
        "question": _required(question, "question"),
        "revisions": [revision],
        "proposals": [],
        "reference_warning": (
            "Reference IDs are pointers only; source provenance and semantic support "
            "are not verified by the thread store."
        ),
    }
    with _locked(root):
        path = _path(root, state["thread_id"])
        if path.exists():
            raise FileExistsError("learning thread ID already exists")
        _write(path, state)
    return _public(state)


def propose_revision(
    store_dir: Path, thread_id: str, *, revised_view: str, reason: str,
    effective_date: str, evidence_refs: list[str] | tuple[str, ...] = (),
    observation_refs: list[str] | tuple[str, ...] = (),
    assessments: list[dict] | tuple[dict, ...] = (),
    proposed_at: str | None = None,
) -> dict:
    """Save a draft based on the current revision; never change the accepted view."""
    checked_evidence = _refs(evidence_refs, "evidence_refs")
    checked_observations = _refs(observation_refs, "observation_refs")
    proposal = {
        "proposal_id": f"proposal_{uuid4().hex}",
        "revised_view": _required(revised_view, "revised_view"),
        "reason": _required(reason, "reason"),
        "effective_date": _date(effective_date),
        "proposed_at": _time(proposed_at),
        "evidence_refs": checked_evidence,
        "observation_refs": checked_observations,
        "assessments": _assessments(assessments, checked_evidence + checked_observations),
        "status": "pending",
        "accepted_revision_id": None,
        "staged_sha256": None,
    }
    root = _store_dir(store_dir)
    path = _path(root, thread_id)
    with _locked(root):
        state = _read(path)
        proposal["based_on_revision_id"] = state["revisions"][-1]["revision_id"]
        state["proposals"].append(proposal)
        _write(path, state)
    return copy.deepcopy(proposal)


def record_staged_draft(
    store_dir: Path, thread_id: str, proposal_id: str, sha256: str,
) -> dict:
    """Bind a pending proposal to the exact Markdown bytes staged in the inbox."""
    if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
        raise ValueError("invalid proposal_id")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise ValueError("sha256 must be 64 lowercase hex characters")
    root = _store_dir(store_dir)
    with _locked(root):
        path = _path(root, thread_id)
        state = _read(path)
        matches = [item for item in state["proposals"]
                   if item["proposal_id"] == proposal_id]
        if len(matches) != 1:
            raise ValueError("proposal does not exist")
        proposal = matches[0]
        if proposal["status"] != "pending":
            raise ValueError("only a pending proposal can record a staged draft")
        prior_hash = proposal.get("staged_sha256")
        if prior_hash is not None and prior_hash != sha256:
            raise ValueError("proposal is already bound to a different staged draft")
        if prior_hash is None:
            proposal["staged_sha256"] = sha256
            _write(path, state)
    return copy.deepcopy(proposal)


def discard_pending_proposal(
    store_dir: Path, thread_id: str, proposal_id: str,
) -> dict:
    """Remove a failed-to-stage proposal from actionable work, retaining an audit row."""
    if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
        raise ValueError("invalid proposal_id")
    root = _store_dir(store_dir)
    with _locked(root):
        path = _path(root, thread_id)
        state = _read(path)
        matches = [item for item in state["proposals"]
                   if item["proposal_id"] == proposal_id]
        if len(matches) != 1:
            raise ValueError("proposal does not exist")
        proposal = matches[0]
        if proposal["status"] != "pending":
            raise ValueError("only a pending proposal can be discarded")
        proposal["status"] = "discarded"
        _write(path, state)
    return copy.deepcopy(proposal)


def approve_revision(
    store_dir: Path, thread_id: str, proposal_id: str, *,
    approved_by: str, confirmed: bool = False, approved_at: str | None = None,
) -> dict:
    """Record an explicit user acceptance in the working index, not Obsidian promotion.

    A stale proposal cannot overwrite a more recently accepted view. A caller
    must supply the user's affirmative action as ``confirmed=True``.
    """
    if confirmed is not True:
        raise ValueError("explicit user confirmation is required")
    reviewer = _required(approved_by, "approved_by")
    if not isinstance(proposal_id, str) or not _PROPOSAL_ID.fullmatch(proposal_id):
        raise ValueError("invalid proposal_id")
    timestamp = _time(approved_at)
    root = _store_dir(store_dir)
    path = _path(root, thread_id)
    with _locked(root):
        state = _read(path)
        matches = [item for item in state["proposals"]
                   if item["proposal_id"] == proposal_id]
        if len(matches) != 1:
            raise ValueError("proposal does not exist")
        proposal = matches[0]
        if proposal["status"] == "stale":
            raise ValueError("proposal is stale; review the newer accepted view")
        if proposal["status"] != "pending":
            raise ValueError("proposal is not pending")
        prior = state["revisions"][-1]
        if proposal["based_on_revision_id"] != prior["revision_id"]:
            raise ValueError("proposal is stale; review the newer accepted view")
        revision = {
            "revision_id": f"revision_{uuid4().hex}",
            "view": proposal["revised_view"],
            "reason": proposal["reason"],
            "effective_date": proposal["effective_date"],
            "recorded_at": timestamp,
            "accepted_by": reviewer,
            "proposal_id": proposal_id,
            "supersedes_revision_id": prior["revision_id"],
            "evidence_refs": proposal["evidence_refs"],
            "observation_refs": proposal["observation_refs"],
            "assessments": copy.deepcopy(proposal.get("assessments", [])),
        }
        proposal["status"] = "accepted_in_working_index"
        proposal["accepted_revision_id"] = revision["revision_id"]
        for other in state["proposals"]:
            if (other is not proposal and other["status"] == "pending"
                    and other["based_on_revision_id"] == prior["revision_id"]):
                other["status"] = "stale"
                other["superseded_by_revision_id"] = revision["revision_id"]
        state["revisions"].append(revision)
        _write(path, state)
    return copy.deepcopy(revision)


def _public(state: dict) -> dict:
    return {
        "schema_version": state["schema_version"],
        "thread_id": state["thread_id"],
        "question": state["question"],
        "current_approved_view": copy.deepcopy(state["revisions"][-1]),
        "timeline": copy.deepcopy(state["revisions"]),
        "pending_proposals": copy.deepcopy([
            proposal for proposal in state["proposals"] if proposal["status"] == "pending"
        ]),
        "reference_warning": state["reference_warning"],
        "vault_promotion_status": "not_promoted_by_thread_store",
    }


def get_thread(store_dir: Path, thread_id: str) -> dict:
    """Return the accepted timeline and current view plus outstanding proposals."""
    state = _read(_path(_store_dir(store_dir), thread_id))
    return _public(state)


def list_threads(store_dir: Path) -> list[dict]:
    """List brief public summaries without following symlinks or unrelated files."""
    root = _store_dir(store_dir)
    summaries = []
    for path in root.iterdir():
        if path.is_symlink() or not path.is_file() or not path.name.endswith(".json"):
            continue
        thread_id = path.name[:-5]
        if not _THREAD_ID.fullmatch(thread_id):
            continue
        state = _read(path)
        if state.get("thread_id") != thread_id:
            raise ValueError(f"thread ID differs from filename: {path.name}")
        summaries.append({
            "thread_id": thread_id,
            "question": state["question"],
            "current_approved_view": copy.deepcopy(state["revisions"][-1]),
            "revision_count": len(state["revisions"]),
            "pending_proposal_count": sum(
                proposal["status"] == "pending" for proposal in state["proposals"]
            ),
        })
    return sorted(summaries, key=lambda item: (item["question"].casefold(), item["thread_id"]))


def render_proposal_markdown(store_dir: Path, thread_id: str, proposal_id: str) -> str:
    """Render an unapproved draft for the harness to stage in the vault inbox."""
    state = _read(_path(_store_dir(store_dir), thread_id))
    matches = [item for item in state["proposals"]
               if item["proposal_id"] == proposal_id]
    if len(matches) != 1:
        raise ValueError("proposal does not exist")
    proposal = matches[0]
    if proposal["status"] == "stale":
        raise ValueError("proposal is stale; review the newer accepted view")
    if proposal["status"] != "pending":
        raise ValueError("only a pending proposal can be staged as a draft")
    if proposal["based_on_revision_id"] != state["revisions"][-1]["revision_id"]:
        raise ValueError("proposal is stale; review the newer accepted view")
    metadata = {
        "kind": "learning_thread_proposal",
        "review_status": "agent_authored_draft",
        "thread_id": thread_id,
        "proposal_id": proposal_id,
        "based_on_revision_id": proposal["based_on_revision_id"],
        "effective_date": proposal["effective_date"],
    }
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).strip()
    lines = [
        "---", frontmatter, "---", "", f"# {state['question']}", "",
        "## Current view", "", state["revisions"][-1]["view"], "",
        "## Proposed view", "", proposal["revised_view"], "",
        "## Why review this", "", proposal["reason"], "",
        "## Evidence references", "",
        *[f"- {ref}" for ref in proposal["evidence_refs"]], "",
        "## Reviewed vault references", "",
        *[f"- {ref}" for ref in proposal["observation_refs"]], "",
        "## Your assessment of selected passages", "",
        *[f"- {item['relation']}: `{item['reference']}`"
          for item in proposal.get("assessments", [])], "",
        "> These labels are the user's interpretation, not verified source support.", "",
        "> These references are pointers; source provenance and semantic support require review.",
        "", "Staging this draft does not change the accepted view or reviewed Obsidian "
        "library. A later explicit acceptance updates the working thread only.", "",
    ]
    return "\n".join(lines)
