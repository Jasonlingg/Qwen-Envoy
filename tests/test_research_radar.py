"""Offline checks for the weekly candidate cursor and revision-aware store."""

import json
import subprocess
import sys
import threading
from itertools import permutations
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from src.research.radar import collect_candidates, discover_updates, load_topic

PROFILES = Path("data/research/topic_profiles_v1.json")


def collect(*args, **kwargs):
    kwargs.setdefault("updates_provider", lambda *_: {"papers": []})
    return collect_candidates(*args, **kwargs)


def paper(identifier: str, title: str = "Tool-Using Search Agents") -> dict:
    return {
        "arxiv_id": identifier, "title": title,
        "abstract": "Synthetic example, not a real paper.",
        "submitted": "2026-09-03T00:00:00Z",
        "updated": "2026-09-03T00:00:00Z",
        "source_url": f"https://arxiv.org/abs/{identifier}",
    }


def atom_feed(*records: tuple[str, str, str], total: int | None = None,
              start: int = 0) -> bytes:
    entries = "".join(
        f"<entry><id>https://arxiv.org/abs/{identifier}</id>"
        f"<title>Tool-Using Search Agents</title><summary>Search tools.</summary>"
        f"<published>{submitted}</published><updated>{updated}</updated></entry>"
        for identifier, submitted, updated in records
    )
    count = len(records) if total is None else total
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:os="http://a9.com/-/spec/opensearch/1.1/">'
        f'<os:totalResults>{count}</os:totalResults>'
        f'<os:startIndex>{start}</os:startIndex>{entries}</feed>'
    ).encode()


def test_update_scan_pages_past_future_records_and_finds_old_paper_revision():
    pages = [
        atom_feed(
            ("2609.09999v1", "2026-09-22T00:00:00Z", "2026-09-22T00:00:00Z"),
            ("2503.01234v2", "2025-03-03T00:00:00Z", "2026-09-18T00:00:00Z"),
            total=4,
        ),
        atom_feed(
            ("2609.01234v1", "2026-09-15T00:00:00Z", "2026-09-15T00:00:00Z"),
            ("2503.09999v1", "2025-03-01T00:00:00Z", "2026-08-30T00:00:00Z"),
            total=4, start=2,
        ),
    ]
    urls = []

    def downloader(url):
        urls.append(url)
        page = int(parse_qs(urlsplit(url).query)["start"][0]) // 2
        return pages[page]

    result = discover_updates(
        'all:"search tools"', "2026-09-14", "2026-09-21", limit=10,
        scan_limit=6, page_size=2, downloader=downloader, delay=0,
    )
    assert result["status"] == "complete" and result["scanned_count"] == 4
    assert [item["arxiv_id"] for item in result["papers"]] == [
        "2503.01234v2", "2609.01234v1",
    ]
    assert len(urls) == 2
    assert all(parse_qs(urlsplit(url).query)["sortBy"] == ["lastUpdatedDate"]
               for url in urls)
    assert [parse_qs(urlsplit(url).query)["start"] for url in urls] == [["0"], ["2"]]


def test_update_scan_refuses_truncated_or_unordered_coverage():
    in_window = atom_feed(
        ("2503.01234v2", "2025-03-03T00:00:00Z", "2026-09-18T00:00:00Z"),
        ("2503.09999v2", "2025-03-04T00:00:00Z", "2026-09-17T00:00:00Z"),
        total=3,
    )
    with pytest.raises(ValueError, match="exhausted scan_limit"):
        discover_updates(
            "all:search", "2026-09-14", "2026-09-21", limit=10,
            scan_limit=2, page_size=2, downloader=lambda *_: in_window, delay=0,
        )
    unordered = atom_feed(
        ("2503.09999v2", "2025-03-04T00:00:00Z", "2026-09-17T00:00:00Z"),
        ("2503.01234v2", "2025-03-03T00:00:00Z", "2026-09-18T00:00:00Z"),
    )
    with pytest.raises(ValueError, match="not sorted"):
        discover_updates(
            "all:search", "2026-09-14", "2026-09-21", limit=10,
            scan_limit=2, page_size=2, downloader=lambda *_: unordered, delay=0,
        )
    short = atom_feed(
        ("2503.01234v2", "2025-03-03T00:00:00Z", "2026-09-18T00:00:00Z"),
        total=3,
    )
    with pytest.raises(ValueError, match="short page"):
        discover_updates(
            "all:search", "2026-09-14", "2026-09-21", limit=10,
            scan_limit=4, page_size=2, downloader=lambda *_: short, delay=0,
        )


def test_duplicate_update_pages_are_not_accepted_as_complete():
    first = atom_feed(
        ("2503.01234v2", "2025-03-03T00:00:00Z", "2026-09-18T00:00:00Z"),
        total=3,
    )
    second = atom_feed(
        ("2503.01234v2", "2025-03-03T00:00:00Z", "2026-09-18T00:00:00Z"),
        total=3, start=1,
    )
    pages = [first, second]
    with pytest.raises(ValueError, match="repeated an ID"):
        discover_updates(
            "all:search", "2026-09-14", "2026-09-21", limit=10,
            scan_limit=3, page_size=1,
            downloader=lambda url: pages[int(parse_qs(urlsplit(url).query)["start"][0])],
            delay=0,
        )


def test_old_paper_revision_merges_and_failed_scan_keeps_cursor(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    old = paper("2503.01234v1")
    old["submitted"] = old["updated"] = "2025-03-03T00:00:00Z"
    collect(
        topic, tmp_path, since="2026-09-01", until="2026-09-07",
        provider=lambda *_: {"papers": [old]},
    )
    state_path = tmp_path / "tool-using-agents.json"
    before = state_path.read_bytes()

    def incomplete(*_):
        raise ValueError("update scan exhausted scan_limit")

    with pytest.raises(ValueError, match="exhausted scan_limit"):
        collect(topic, tmp_path, until="2026-09-21",
                provider=lambda *_: {"papers": []}, updates_provider=incomplete)
    assert state_path.read_bytes() == before

    revision = {**old, "arxiv_id": "2503.01234v2",
                "source_url": "https://arxiv.org/abs/2503.01234v2",
                "updated": "2026-09-18T00:00:00Z"}
    report = collect(
        topic, tmp_path, until="2026-09-21",
        provider=lambda *_: {"papers": []},
        updates_provider=lambda *_: {"status": "complete", "papers": [revision],
                                     "scanned_count": 3, "query_urls": ["offline:updates"]},
    )
    assert report["new_versions"] == 1
    assert report["submitted_fetched_count"] == 0
    assert report["updated_fetched_count"] == 1
    state = json.loads(state_path.read_text())
    assert state["schema_version"] == "weekly-radar-candidates-v2"
    assert state["last_successful_until"] == "2026-09-21"
    assert [item["arxiv_id"] for item in state["candidates"][0]["versions"]] == [
        "2503.01234v1", "2503.01234v2",
    ]


def test_profile_is_explicit_and_hashed(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    assert topic["query"]
    assert topic["positive_examples"] and topic["exclusions"]
    with pytest.raises(ValueError, match="unknown topic"):
        load_topic(PROFILES, "missing")
    response = {"papers": [], "query_url": "offline", "retrieved_at": "fixed"}
    result = collect(
        topic, tmp_path, since="2026-09-01", until="2026-09-07",
        provider=lambda *_: response,
    )
    assert len(result["profile_sha256"]) == 64
    assert result["cursor"] == "2026-09-07"


def test_overlap_retry_and_revisions_are_idempotent(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    calls = []
    responses = [
        {"papers": [paper("2609.01234v1")], "query_url": "week-one"},
        {"papers": [paper("2609.01234v1"), paper("2609.01234v2"),
                    paper("2609.09999v1", "Tool Using Search Agents")],
         "query_url": "week-two"},
    ]

    def provider(query, since, until, limit):
        calls.append((query, since, until, limit))
        return responses[min(len(calls) - 1, 1)]

    first = collect(
        topic, tmp_path, since="2026-09-01", until="2026-09-14", provider=provider,
    )
    second = collect(topic, tmp_path, until="2026-09-21", provider=provider)
    state_path = tmp_path / "tool-using-agents.json"
    before_retry = state_path.read_bytes()
    retry = collect(topic, tmp_path, until="2026-09-21", provider=provider)
    state = json.loads(state_path.read_text())

    assert first["new_candidates"] == 1 and first["new_versions"] == 1
    assert calls[1][1:3] == ("2026-09-13", "2026-09-21")
    assert second["new_candidates"] == 0 and second["new_versions"] == 2
    assert retry["new_candidates"] == retry["new_versions"] == 0
    assert retry["state_changed"] is False
    assert state_path.read_bytes() == before_retry
    assert state["last_successful_until"] == "2026-09-21"
    assert len(state["candidates"]) == 1
    candidate = state["candidates"][0]
    assert candidate["base_arxiv_ids"] == ["2609.01234", "2609.09999"]
    assert [v["arxiv_id"] for v in candidate["versions"]] == [
        "2609.01234v1", "2609.01234v2", "2609.09999v1",
    ]


def test_failed_fetch_or_malformed_batch_never_advances_cursor(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    state_path = tmp_path / "tool-using-agents.json"

    def fail(*_):
        raise OSError("provider unavailable")

    with pytest.raises(OSError, match="provider unavailable"):
        collect(
            topic, tmp_path, since="2026-09-01", until="2026-09-07", provider=fail,
        )
    assert not state_path.exists()

    collect(
        topic, tmp_path, since="2026-09-01", until="2026-09-07",
        provider=lambda *_: {"papers": [paper("2609.01234v1")]},
    )
    initial = state_path.read_bytes()
    with pytest.raises(ValueError, match="versioned arXiv ID"):
        collect(
            topic, tmp_path, until="2026-09-14",
            provider=lambda *_: {"papers": [paper("2609.01234v2"),
                                           paper("2609.09999")]},
        )
    assert state_path.read_bytes() == initial
    with pytest.raises(ValueError, match="incomplete"):
        collect(
            topic, tmp_path, until="2026-09-14",
            provider=lambda *_: {"papers": [], "status": "partial"},
        )
    assert state_path.read_bytes() == initial
    with pytest.raises(ValueError, match="filled the result limit"):
        collect(
            topic, tmp_path, until="2026-09-14", limit=1,
            provider=lambda *_: {"papers": [paper("2609.09999v1")]},
        )
    assert state_path.read_bytes() == initial
    with pytest.raises(ValueError, match="mismatched until"):
        collect(
            topic, tmp_path, until="2026-09-14",
            provider=lambda *_: {"papers": [], "until": "2026-09-13"},
        )
    assert state_path.read_bytes() == initial


def test_changed_profile_does_not_silently_reuse_old_cursor(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    collect(
        topic, tmp_path, since="2026-09-01", until="2026-09-07",
        provider=lambda *_: {"papers": []},
    )
    with pytest.raises(ValueError, match="topic profile changed"):
        collect(
            {**topic, "query": 'all:"new query"'}, tmp_path, until="2026-09-14",
            provider=lambda *_: {"papers": []},
        )


def test_submission_only_v1_cursor_cannot_be_reused(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    path = tmp_path / "tool-using-agents.json"
    path.write_text(json.dumps({"schema_version": "weekly-radar-candidates-v1",
                                "topic_id": topic["id"],
                                "last_successful_until": "2026-09-07",
                                "candidates": []}))
    with pytest.raises(ValueError, match="unsupported schema"):
        collect(topic, tmp_path, until="2026-09-14",
                provider=lambda *_: pytest.fail("should reject before any fetch"))


def test_same_title_candidate_id_is_order_independent(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    records = [
        paper("2609.09999v1", "Tool-Using Search Agents"),
        paper("2609.01234v2", "Tool Using Search Agents"),
        paper("2609.01234v1", "TOOL USING SEARCH AGENTS"),
    ]
    candidates = []
    for index, ordering in enumerate(permutations(records)):
        folder = tmp_path / str(index)
        collect(
            topic, folder, since="2026-09-01", until="2026-09-07",
            provider=lambda *_, rows=ordering: {"papers": list(rows)},
        )
        state = json.loads((folder / "tool-using-agents.json").read_text())
        candidates.append(state["candidates"])
    assert all(value == candidates[0] for value in candidates)
    assert candidates[0][0]["candidate_id"] == "arxiv:2609.01234"


def test_concurrent_collectors_serialize_read_fetch_and_commit(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    collect(
        topic, tmp_path, since="2026-09-01", until="2026-09-07",
        provider=lambda *_: {"papers": [paper("2609.01234v1")]},
    )
    first_fetching = threading.Event()
    release_first = threading.Event()
    second_started = threading.Event()
    second_fetching = threading.Event()
    errors = []

    def first_provider(*_):
        first_fetching.set()
        if not release_first.wait(timeout=3):
            raise TimeoutError("first provider was not released")
        return {"papers": [paper("2609.01234v2")]}

    def second_provider(*_):
        second_fetching.set()
        return {"papers": [paper("2609.01234v3")]}

    def first_run():
        try:
            collect(topic, tmp_path, until="2026-09-14", provider=first_provider)
        except Exception as exc:
            errors.append(exc)

    def second_run():
        second_started.set()
        try:
            collect(topic, tmp_path, until="2026-09-21", provider=second_provider)
        except Exception as exc:
            errors.append(exc)

    first = threading.Thread(target=first_run)
    second = threading.Thread(target=second_run)
    first.start()
    try:
        assert first_fetching.wait(timeout=3)
        second.start()
        assert second_started.wait(timeout=3)
        assert not second_fetching.wait(timeout=0.1)
    finally:
        release_first.set()
        first.join(timeout=3)
        if second.ident is not None:
            second.join(timeout=3)
    assert not first.is_alive() and not second.is_alive()
    assert not errors
    assert second_fetching.is_set()
    state = json.loads((tmp_path / "tool-using-agents.json").read_text())
    assert state["last_successful_until"] == "2026-09-21"
    assert [item["arxiv_id"] for item in state["candidates"][0]["versions"]] == [
        "2609.01234v1", "2609.01234v2", "2609.01234v3",
    ]


def test_concurrent_older_window_cannot_move_cursor_back(tmp_path):
    topic = load_topic(PROFILES, "tool-using-agents")
    collect(
        topic, tmp_path, since="2026-09-01", until="2026-09-07",
        provider=lambda *_: {"papers": []},
    )
    future_fetching = threading.Event()
    release_future = threading.Event()
    stale_started = threading.Event()
    stale_fetched = threading.Event()
    errors = []

    def future_provider(*_):
        future_fetching.set()
        if not release_future.wait(timeout=3):
            raise TimeoutError("future provider was not released")
        return {"papers": [paper("2609.01234v1")]}

    def future_run():
        try:
            collect(topic, tmp_path, until="2026-09-21", provider=future_provider)
        except Exception as exc:
            errors.append(exc)

    def stale_provider(*_):
        stale_fetched.set()
        return {"papers": []}

    def stale_run():
        stale_started.set()
        try:
            collect(
                topic, tmp_path, until="2026-09-14", provider=stale_provider,
            )
        except Exception as exc:
            errors.append(exc)

    future = threading.Thread(target=future_run)
    stale = threading.Thread(target=stale_run)
    future.start()
    try:
        assert future_fetching.wait(timeout=3)
        stale.start()
        assert stale_started.wait(timeout=3)
        assert not stale_fetched.wait(timeout=0.1)
    finally:
        release_future.set()
        future.join(timeout=3)
        if stale.ident is not None:
            stale.join(timeout=3)
    assert not future.is_alive() and not stale.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], ValueError)
    assert "cannot precede" in str(errors[0])
    assert not stale_fetched.is_set()
    state = json.loads((tmp_path / "tool-using-agents.json").read_text())
    assert state["last_successful_until"] == "2026-09-21"


def test_one_command_demo_runs_without_network():
    result = subprocess.run(
        [sys.executable, "scripts/weekly_radar_candidates.py", "--demo"],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["mode"] == "offline_synthetic_demo"
    assert payload["cursor"] == "2026-09-21"
    assert payload["retry"]["state_changed"] is False
    assert len(payload["candidates"]) == 1
    assert len(payload["candidates"][0]["versions"]) == 3
