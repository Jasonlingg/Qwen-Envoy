"""Immutable, dated system-map revisions for a learning thread.

Map revisions live in the app's working state, never in the reviewed vault.
An index selects the current revision; each revision is written once. The
source spans stored on edges are pointers, not judgments of semantic support.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

SCHEMA_VERSION = "system-map-v1"
_THREAD_ID = re.compile(r"thread_[0-9a-f]{32}\Z")
_REVISION_ID = re.compile(r"maprev_[0-9a-f]{32}\Z")


def _thread_dir(store_dir: Path, thread_id: str) -> Path:
    if not isinstance(thread_id, str) or _THREAD_ID.fullmatch(thread_id) is None:
        raise ValueError("invalid thread_id")
    return Path(store_dir) / thread_id


@contextmanager
def _locked(store_dir: Path):
    store_dir.mkdir(parents=True, exist_ok=True)
    with (store_dir / ".system-maps.lock").open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.stem}-{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_index(folder: Path, thread_id: str, question: str) -> dict:
    path = folder / "index.json"
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "thread_id": thread_id,
                "question": question, "revisions": []}
    index = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(index, dict) or index.get("schema_version") != SCHEMA_VERSION
            or index.get("thread_id") != thread_id
            or index.get("question") != question
            or not isinstance(index.get("revisions"), list)):
        raise ValueError("system map index is invalid")
    revisions = index["revisions"]
    if (any(not isinstance(item, dict)
            or _REVISION_ID.fullmatch(str(item.get("id", ""))) is None
            or not isinstance(item.get("created_at"), str)
            or not isinstance(item.get("summary"), str)
            for item in revisions)
            or len({item["id"] for item in revisions}) != len(revisions)):
        raise ValueError("system map revision index is invalid")
    return index


def _revision(folder: Path, revision_id: str) -> dict:
    if not isinstance(revision_id, str) or _REVISION_ID.fullmatch(revision_id) is None:
        raise ValueError("invalid map revision ID")
    revision = json.loads((folder / f"{revision_id}.json").read_text(encoding="utf-8"))
    if (not isinstance(revision, dict) or revision.get("id") != revision_id
            or not isinstance(revision.get("nodes"), list)
            or not isinstance(revision.get("edges"), list)):
        raise ValueError("system map revision is invalid")
    return revision


def get_map(store_dir: Path, thread_id: str, question: str,
            revision_id: str | None = None) -> dict:
    """Read the current or a selected immutable revision."""
    folder = _thread_dir(store_dir, thread_id)
    with _locked(Path(store_dir)):
        index = _read_index(folder, thread_id, question)
        history = list(reversed(index["revisions"]))
        if revision_id is None:
            selected = index["revisions"][-1]["id"] if index["revisions"] else None
        else:
            if revision_id not in {row["id"] for row in index["revisions"]}:
                raise ValueError("map revision does not exist")
            selected = revision_id
        revision = (_revision(folder, selected) if selected else {
            "id": None, "created_at": None, "summary": "", "nodes": [], "edges": [],
        })
        return {"thread_id": thread_id, "question": question,
                "revision": revision, "history": history}


def save_map(store_dir: Path, thread_id: str, question: str, *,
             nodes: list[dict], edges: list[dict], summary: str) -> dict:
    """Append one map revision and atomically point the index at it."""
    folder = _thread_dir(store_dir, thread_id)
    with _locked(Path(store_dir)):
        folder.mkdir(parents=True, exist_ok=True)
        index = _read_index(folder, thread_id, question)
        revision = {
            "id": f"maprev_{uuid4().hex}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "summary": summary, "nodes": nodes, "edges": edges,
        }
        path = folder / f"{revision['id']}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(revision, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _sync_dir(folder)
        index["revisions"].append({
            "id": revision["id"], "created_at": revision["created_at"],
            "summary": summary,
        })
        _atomic_write(folder / "index.json", index)
        return {"thread_id": thread_id, "question": question,
                "revision": revision, "history": list(reversed(index["revisions"]))}
