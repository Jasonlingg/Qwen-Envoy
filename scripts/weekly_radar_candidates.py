"""Collect weekly paper candidates and revisions without ranking or model calls.

Offline smoke: python scripts/weekly_radar_candidates.py --demo
Live example: python scripts/weekly_radar_candidates.py --topic tool-using-agents \
    --since 2026-09-01 --until 2026-09-07
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.research.radar import collect_candidates, load_topic  # noqa: E402

DEFAULT_PROFILES = Path("data/research/topic_profiles_v1.json")


def _demo() -> dict:
    topic = load_topic(DEFAULT_PROFILES, "tool-using-agents")
    first = {
        "papers": [
            {"arxiv_id": "2609.01234v1", "title": "Tool-Using Search Agents",
             "abstract": "An example candidate, not a real paper.",
             "submitted": "2026-09-03T00:00:00Z", "updated": "2026-09-03T00:00:00Z",
             "source_url": "https://arxiv.org/abs/2609.01234v1"},
        ],
        "retrieved_at": "2026-09-14T00:00:00Z", "query_url": "offline-demo:first",
    }
    second = {
        "papers": [
            {**first["papers"][0], "arxiv_id": "2609.09999v1",
             "title": "Tool Using Search Agents", "submitted": "2026-09-19T00:00:00Z",
             "updated": "2026-09-19T00:00:00Z",
             "source_url": "https://arxiv.org/abs/2609.09999v1"},
        ],
        "retrieved_at": "2026-09-21T00:00:00Z", "query_url": "offline-demo:second",
    }
    revision = {**first["papers"][0], "arxiv_id": "2609.01234v2",
                "updated": "2026-09-18T00:00:00Z",
                "source_url": "https://arxiv.org/abs/2609.01234v2"}
    update_calls = 0

    def updates_provider(*_):
        nonlocal update_calls
        update_calls += 1
        return {"status": "complete", "papers": [] if update_calls == 1 else [revision],
                "scanned_count": 2, "query_urls": [f"offline-demo:updates:{update_calls}"]}

    with tempfile.TemporaryDirectory(prefix="envoy-radar-demo-") as directory:
        state_dir = Path(directory)
        first_run = collect_candidates(
            topic, state_dir, since="2026-09-01", until="2026-09-14",
            provider=lambda *_: first, updates_provider=updates_provider,
        )
        second_run = collect_candidates(
            topic, state_dir, until="2026-09-21", provider=lambda *_: second,
            updates_provider=updates_provider,
        )
        retry = collect_candidates(
            topic, state_dir, until="2026-09-21", provider=lambda *_: second,
            updates_provider=updates_provider,
        )
        state = json.loads(Path(second_run["state_path"]).read_text())
    for report in (first_run, second_run, retry):
        report.pop("state_path")  # The demonstration's temporary store has been removed.
    return {
        "mode": "offline_synthetic_demo", "persistence": "ephemeral; no files retained",
        "first": first_run,
        "second": second_run, "retry": retry,
        "cursor": state["last_successful_until"],
        "candidates": state["candidates"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Run a synthetic offline two-week demo")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--topic", help="Topic ID in --profiles")
    parser.add_argument("--state-dir", type=Path,
                        default=Path("out/research/weekly-radar-candidates"))
    parser.add_argument("--since", help="YYYY-MM-DD; required on the first run only")
    parser.add_argument("--until", help="YYYY-MM-DD")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--overlap-days", type=int, default=2)
    parser.add_argument("--update-scan-limit", type=int, default=1000,
                        help="Maximum latest-updated records to scan before refusing coverage")
    parser.add_argument("--fixture", type=Path,
                        help="Offline JSON with separate submitted and updated responses")
    args = parser.parse_args(argv)
    try:
        if args.demo:
            print(json.dumps(_demo(), indent=2))
            return 0
        if not args.topic or not args.until:
            parser.error("--topic and --until are required unless --demo is used")
        topic = load_topic(args.profiles, args.topic)
        if args.fixture:
            fixture = json.loads(args.fixture.read_text(encoding="utf-8"))
            if (not isinstance(fixture, dict)
                    or not isinstance(fixture.get("submitted"), dict)
                    or not isinstance(fixture.get("updated"), dict)):
                raise ValueError("fixture needs submitted and updated response objects")

            def provider(*_):
                return fixture["submitted"]

            def updates_provider(*_):
                return fixture["updated"]

        else:
            provider = None
            updates_provider = None
        kwargs = {"until": args.until, "since": args.since, "limit": args.limit,
                  "overlap_days": args.overlap_days,
                  "update_scan_limit": args.update_scan_limit}
        if provider is not None:
            kwargs["provider"] = provider
            kwargs["updates_provider"] = updates_provider
        result = collect_candidates(topic, args.state_dir, **kwargs)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
