"""Retry-safe submission and revision candidate collection for one topic.

This is a metadata stage, not a relevance judgment or a weekly digest. arXiv
only filters submissions by date, so revisions use a bounded, descending
last-updated scan. An incomplete scan never advances the weekly cursor.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
import re
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

from src.research.papers import discover, fetch

PROFILE_SCHEMA = "weekly-radar-topics-v1"
STATE_SCHEMA = "weekly-radar-candidates-v2"
_TOPIC_ID = re.compile(r"[a-z][a-z0-9-]{0,63}")
_VERSIONED_ID = re.compile(r"(\d{4}\.\d{4,5})v([1-9]\d*)")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_ATOM = {"a": "http://www.w3.org/2005/Atom",
         "os": "http://a9.com/-/spec/opensearch/1.1/"}


def _day(value: str) -> date:
    if not isinstance(value, str) or not _DATE.fullmatch(value):
        raise ValueError("dates must use YYYY-MM-DD")
    return date.fromisoformat(value)


def discover_updates(
    query: str, since: str, until: str, limit: int = 100, scan_limit: int = 1000,
    *, page_size: int = 100, downloader: Callable[[str], bytes] = fetch,
    delay: float = 3.1,
) -> dict:
    """Find currently visible versions updated in a window by bounded paging.

    The API has no update-date *filter*. We sort by ``lastUpdatedDate`` and
    scan until the first entry older than ``since`` or the result set ends.
    Hitting either cap before then raises, so callers cannot claim coverage.
    Historical replay may be impossible within the bound because newer results
    must be scanned first. This retrieves latest versions, not every intermediate
    version that may have existed between weekly runs.
    """
    start, end = _day(since), _day(until)
    if start > end or not 1 <= limit <= 100 or not 1 <= scan_limit <= 2000:
        raise ValueError("require since <= until, limit 1..100, scan_limit 1..2000")
    if not 1 <= page_size <= 2000 or delay < 0:
        raise ValueError("page_size must be 1..2000 and delay must be nonnegative")
    if not isinstance(query, str) or not query.strip():
        raise ValueError("arXiv query must be nonempty")
    scanned = 0
    papers: list[dict] = []
    urls: list[str] = []
    seen_ids: set[str] = set()
    last_updated: datetime | None = None
    total_results: int | None = None
    reached_start = False
    while scanned < scan_limit and not reached_start:
        size = min(page_size, scan_limit - scanned)
        url = "https://export.arxiv.org/api/query?" + urlencode({
            "search_query": query, "sortBy": "lastUpdatedDate",
            "sortOrder": "descending", "start": scanned, "max_results": size,
        })
        if urls:
            time.sleep(delay)
        root = ET.fromstring(downloader(url))
        urls.append(url)
        total_text = root.findtext("os:totalResults", None, _ATOM)
        page_start = root.findtext("os:startIndex", None, _ATOM)
        try:
            page_total = int(total_text) if total_text is not None else -1
            reported_start = int(page_start) if page_start is not None else -1
        except ValueError as exc:
            raise ValueError("arXiv update scan returned invalid paging metadata") from exc
        if (page_total < 0 or reported_start != scanned
                or total_results is not None and page_total != total_results):
            raise ValueError("arXiv update scan returned inconsistent paging metadata")
        total_results = page_total
        entries = root.findall("a:entry", _ATOM)
        if scanned + len(entries) > page_total:
            raise ValueError("arXiv update scan returned too many entries")
        if scanned + len(entries) < page_total and len(entries) < size:
            raise ValueError("arXiv update scan returned a short page")
        if not entries:
            reached_start = True
            break
        for entry in entries:
            identifier = entry.findtext("a:id", "", _ATOM).rsplit("/", 1)[-1]
            if _VERSIONED_ID.fullmatch(identifier) is None:
                raise ValueError("arXiv update scan returned an error or unversioned ID")
            if identifier in seen_ids:
                raise ValueError("arXiv update scan repeated an ID across pages")
            seen_ids.add(identifier)
            updated = entry.findtext("a:updated", "", _ATOM)
            submitted = entry.findtext("a:published", "", _ATOM)
            try:
                updated_at = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                datetime.fromisoformat(submitted.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError("arXiv update scan returned invalid dates") from exc
            if updated_at.tzinfo is None:
                raise ValueError("arXiv update scan returned a timezone-free date")
            if last_updated is not None and updated_at > last_updated:
                raise ValueError("arXiv update scan was not sorted by last update")
            last_updated = updated_at
            updated_day = updated_at.astimezone(timezone.utc).date()
            if updated_day < start:
                reached_start = True
                break
            if updated_day > end:
                continue
            paper = {
                "arxiv_id": identifier,
                "title": " ".join(entry.findtext("a:title", "", _ATOM).split()),
                "abstract": " ".join(entry.findtext("a:summary", "", _ATOM).split()),
                "submitted": submitted, "updated": updated,
                "source_url": f"https://arxiv.org/abs/{identifier}",
            }
            _paper(paper)
            papers.append(paper)
            if len(papers) >= limit:
                raise ValueError("update scan filled the result limit; narrow the window")
        scanned += len(entries)
        if scanned == page_total:
            reached_start = True
    if not reached_start:
        raise ValueError("update scan exhausted scan_limit before reaching since")
    return {
        "status": "complete", "query_urls": urls, "since": since, "until": until,
        "limit": limit, "scan_limit": scan_limit, "scanned_count": scanned,
        "total_results": total_results,
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "papers": papers,
        "coverage_note": "Latest matching arXiv versions in the completed update scan; "
                         "intermediate versions between runs may be unavailable.",
    }


def _profile_hash(profile: dict) -> str:
    encoded = json.dumps(profile, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_topic(path: Path, topic_id: str) -> dict:
    """Load one pinned topic profile from a small JSON profile collection."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError("unsupported topic-profile schema")
    topics = payload.get("topics")
    if not isinstance(topics, list) or not topics:
        raise ValueError("topic profiles must contain a nonempty topics list")
    ids = [item.get("id") for item in topics if isinstance(item, dict)]
    if (len(ids) != len(topics) or any(not isinstance(value, str) for value in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("topic IDs must be unique")
    for topic in topics:
        if (not isinstance(topic.get("id"), str)
                or not _TOPIC_ID.fullmatch(topic["id"])
                or not isinstance(topic.get("query"), str)
                or not topic["query"].strip()):
            raise ValueError("each topic needs a safe ID and nonempty arXiv query")
        if any(not isinstance(topic.get(field), str) or not topic[field].strip()
               for field in ("name", "project_context")):
            raise ValueError("each topic needs a name and project context")
        for field in ("positive_examples", "exclusions", "keywords"):
            values = topic.get(field)
            if (not isinstance(values, list)
                    or any(not isinstance(value, str) or not value.strip()
                           for value in values)):
                raise ValueError(f"topic {topic['id']} needs a {field} list of strings")
    matches = [item for item in topics if item["id"] == topic_id]
    if not matches:
        raise ValueError(f"unknown topic: {topic_id}")
    return matches[0]


def normalize_title(title: str) -> str:
    """Conservative title key for duplicate discovery records."""
    normalized = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(re.findall(r"\w+", normalized.replace("_", " ")))


def _version_key(item: dict) -> tuple[str, int]:
    match = _VERSIONED_ID.fullmatch(item["arxiv_id"])
    assert match is not None
    return match.group(1), int(match.group(2))


def _paper(value: object) -> tuple[str, str, dict]:
    if not isinstance(value, dict):
        raise ValueError("discovery paper must be an object")
    identifier = value.get("arxiv_id")
    title = value.get("title")
    if not isinstance(identifier, str):
        raise ValueError("discovery paper lacks an arXiv ID")
    match = _VERSIONED_ID.fullmatch(identifier)
    if match is None:
        raise ValueError(f"discovery paper needs a versioned arXiv ID: {identifier}")
    if not isinstance(title, str) or not normalize_title(title):
        raise ValueError(f"discovery paper has no usable title: {identifier}")
    for field in ("abstract", "submitted", "updated", "source_url"):
        if not isinstance(value.get(field), str):
            raise ValueError(f"discovery paper lacks {field}: {identifier}")
    if not value["submitted"] or not value["updated"]:
        raise ValueError(f"discovery paper lacks dates: {identifier}")
    try:
        datetime.fromisoformat(value["submitted"].replace("Z", "+00:00"))
        datetime.fromisoformat(value["updated"].replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"discovery paper has invalid dates: {identifier}") from exc
    if value["source_url"] != f"https://arxiv.org/abs/{identifier}":
        raise ValueError(f"discovery paper has a mismatched source URL: {identifier}")
    revision = {field: value[field] for field in (
        "arxiv_id", "title", "abstract", "submitted", "updated", "source_url"
    )}
    return match.group(1), normalize_title(title), revision


def _merge_record(left: dict, right: dict) -> None:
    left["base_arxiv_ids"] = sorted(set(left["base_arxiv_ids"] + right["base_arxiv_ids"]))
    left["normalized_titles"] = sorted(set(
        left["normalized_titles"] + right["normalized_titles"]
    ))
    existing = {item["arxiv_id"] for item in left["versions"]}
    left["versions"].extend(item for item in right["versions"]
                            if item["arxiv_id"] not in existing)
    left["versions"].sort(key=_version_key)
    left["candidate_id"] = f"arxiv:{left['base_arxiv_ids'][0]}"


def _merge_paper(candidates: list[dict], value: object) -> tuple[bool, bool]:
    base_id, title_key, revision = _paper(value)
    matches = [index for index, candidate in enumerate(candidates)
               if (base_id in candidate["base_arxiv_ids"]
                   or title_key in candidate["normalized_titles"])]
    if not matches:
        candidates.append({
            "candidate_id": f"arxiv:{base_id}",
            "base_arxiv_ids": [base_id],
            "normalized_titles": [title_key],
            "versions": [revision],
        })
        return True, True
    target = candidates[matches[0]]
    for index in reversed(matches[1:]):
        _merge_record(target, candidates.pop(index))
    target["base_arxiv_ids"] = sorted(set(target["base_arxiv_ids"] + [base_id]))
    target["candidate_id"] = f"arxiv:{target['base_arxiv_ids'][0]}"
    target["normalized_titles"] = sorted(set(target["normalized_titles"] + [title_key]))
    if any(item["arxiv_id"] == revision["arxiv_id"] for item in target["versions"]):
        return False, False
    target["versions"].append(revision)
    target["versions"].sort(key=_version_key)
    return False, True


def _state(path: Path, topic_id: str, profile_hash: str) -> dict:
    if not path.exists():
        return {"schema_version": STATE_SCHEMA, "topic_id": topic_id,
                "profile_sha256": profile_hash, "last_successful_until": None,
                "candidates": []}
    state = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(state, dict) or state.get("schema_version") != STATE_SCHEMA
            or state.get("topic_id") != topic_id
            or not isinstance(state.get("candidates"), list)):
        raise ValueError("candidate state has an unsupported schema or topic")
    if state.get("profile_sha256") != profile_hash:
        raise ValueError("topic profile changed; use a new state directory or topic ID")
    cursor = state.get("last_successful_until")
    if cursor is not None:
        _day(cursor)
    return state


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@contextmanager
def _topic_lock(state_path: Path):
    """Serialize each topic's read/fetch/commit across processes and threads."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = state_path.with_name(f".{state_path.name}.lock")
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def collect_candidates(
    topic: dict, state_dir: Path, *, until: str, since: str | None = None,
    limit: int = 100, overlap_days: int = 2, update_scan_limit: int = 1000,
    provider: Callable[[str, str, str, int], dict] = discover,
    updates_provider: Callable[[str, str, str, int, int], dict] | None = None,
) -> dict:
    """Fetch both bounded channels; commit cursor and store together.

    The first run needs ``since``. Later runs begin before the last successful
    end date, so retries and overlapping provider responses cannot duplicate a
    version. Provider errors and incomplete scans leave the state untouched.
    """
    topic_id = topic.get("id") if isinstance(topic, dict) else None
    query = topic.get("query") if isinstance(topic, dict) else None
    if (not isinstance(topic_id, str) or not _TOPIC_ID.fullmatch(topic_id)
            or not isinstance(query, str) or not query.strip()):
        raise ValueError("topic needs a safe ID and nonempty arXiv query")
    if not 1 <= limit <= 100 or not 1 <= overlap_days <= 31 or not 1 <= update_scan_limit <= 2000:
        raise ValueError("limit must be 1..100, overlap_days 1..31, update_scan_limit 1..2000")
    if updates_provider is None:
        if provider is not discover:
            raise ValueError("a custom submission provider needs a custom update provider")
        updates_provider = discover_updates
    end = _day(until)
    path = state_dir / f"{topic_id}.json"
    with _topic_lock(path):
        return _collect_locked(topic, path, end=end, until=until, since=since,
                               limit=limit, overlap_days=overlap_days,
                               update_scan_limit=update_scan_limit,
                               provider=provider, updates_provider=updates_provider)


def _collect_locked(
    topic: dict, path: Path, *, end: date, until: str, since: str | None,
    limit: int, overlap_days: int, update_scan_limit: int,
    provider: Callable[[str, str, str, int], dict],
    updates_provider: Callable[[str, str, str, int, int], dict],
) -> dict:
    topic_id = topic["id"]
    query = topic["query"]
    state = _state(path, topic_id, _profile_hash(topic))
    cursor = state["last_successful_until"]
    if cursor is None:
        if since is None:
            raise ValueError("the first run requires since")
        start = _day(since)
    else:
        if since is not None:
            raise ValueError("since is only allowed on the first run")
        previous = _day(cursor)
        if end < previous:
            raise ValueError("until cannot precede the successful cursor")
        start = previous - timedelta(days=overlap_days - 1)
    if start > end:
        raise ValueError("since cannot follow until")
    submitted = provider(query, start.isoformat(), end.isoformat(), limit)
    _check_response(submitted, start.isoformat(), until, limit, "submission")
    # The public API permits only one request every three seconds. The normal
    # submission provider makes one call; the update provider spaces its pages.
    if provider is discover and updates_provider is discover_updates:
        time.sleep(3.1)
    updated = updates_provider(query, start.isoformat(), end.isoformat(), limit,
                               update_scan_limit)
    _check_response(updated, start.isoformat(), until, limit, "update")
    if updated.get("scan_limit") not in (None, update_scan_limit):
        raise ValueError("update response has a mismatched scan_limit")
    candidates = copy.deepcopy(state["candidates"])
    new_candidates = new_versions = 0
    for paper in submitted["papers"] + updated["papers"]:
        added_candidate, added_version = _merge_paper(candidates, paper)
        new_candidates += added_candidate
        new_versions += added_version
    candidates.sort(key=lambda candidate: candidate["candidate_id"])
    changed = cursor != until or candidates != state["candidates"]
    if changed:
        state["candidates"] = candidates
        state["last_successful_until"] = until
        state["last_successful_run"] = {
            "since": start.isoformat(), "until": until,
            "submitted_query_url": submitted.get("query_url"),
            "updated_query_urls": updated.get("query_urls"),
            "submitted_retrieved_at": submitted.get("retrieved_at"),
            "updated_retrieved_at": updated.get("retrieved_at"),
            "submitted_fetched_count": len(submitted["papers"]),
            "updated_fetched_count": len(updated["papers"]),
            "updated_scanned_count": updated.get("scanned_count"),
            "update_scan_limit": update_scan_limit,
        }
        _atomic_json(path, state)
    return {
        "topic_id": topic_id, "state_path": str(path),
        "profile_sha256": state["profile_sha256"],
        "since": start.isoformat(), "until": until,
        "fetched_count": len(submitted["papers"]) + len(updated["papers"]),
        "submitted_fetched_count": len(submitted["papers"]),
        "updated_fetched_count": len(updated["papers"]),
        "updated_scanned_count": updated.get("scanned_count"),
        "new_candidates": new_candidates, "new_versions": new_versions,
        "candidate_count": len(candidates), "cursor": until,
        "state_changed": changed,
        "coverage_note": "Submission-date search plus completed bounded last-updated scan. "
                         "Only latest visible versions are discoverable; later historical "
                         "replay may exceed the scan bound.",
    }


def _check_response(response: object, since: str, until: str, limit: int,
                    channel: str) -> None:
    if (not isinstance(response, dict) or not isinstance(response.get("papers"), list)
            or response.get("failures") or response.get("status") not in (None, "complete")):
        raise ValueError(f"{channel} discovery response is incomplete")
    for field, expected in (("since", since), ("until", until), ("limit", limit)):
        if field in response and response[field] != expected:
            raise ValueError(f"{channel} discovery response has a mismatched {field}")
    if len(response["papers"]) >= limit:
        raise ValueError(f"{channel} discovery filled the result limit; narrow the window")
