"""Preview or publish a reviewed weekly AI-paper digest into an Obsidian vault."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product.weekly_digest import plan_weekly_digest, write_weekly_digest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, type=Path,
                        help="Frozen public-paper research snapshot")
    parser.add_argument("--selection", required=True, type=Path,
                        help="Explicit human-reviewed selection JSON")
    parser.add_argument("--vault", required=True, type=Path,
                        help="Existing Obsidian vault root")
    parser.add_argument("--output-root", default="AI Research",
                        help="Relative folder inside the vault (default: AI Research)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--confirm", action="store_true",
                      help="Actually create the paper notes and weekly digest")
    mode.add_argument("--show-content", action="store_true",
                      help="Print rendered Markdown to stdout; make no vault changes")
    args = parser.parse_args(argv)
    try:
        plan = plan_weekly_digest(args.snapshot, args.selection, args.vault,
                                  output_root=args.output_root)
        result = (write_weekly_digest(plan, confirm=True) if args.confirm else
                  plan.summary("ready" if plan.review_status == "human_reviewed"
                               else "draft_preview_only"))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.show_content:
            for path, content in plan.files:
                print(f"\n===== {path.relative_to(plan.vault)} =====\n")
                print(content, end="" if content.endswith("\n") else "\n")
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
