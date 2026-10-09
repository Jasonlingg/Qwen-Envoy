"""Local learning-memory demo using real vault files and verified source passages.

`inspect`/`recall` use a lexical baseline. `review-code-exec` only reads an
existing transcript; this CLI never runs Qwen, Nemotron, or model-authored code.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.product.memory import (  # noqa: E402
    CAPTURE_KINDS,
    approve_note,
    capture_note,
    freeze_vault,
    inspect_evidence,
    review_code_exec_transcript,
)


def _text(args: argparse.Namespace) -> str:
    return args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")


def _save_json(payload: dict, output: Path | None) -> None:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        print(output)
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def _add_text(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="The note text")
    group.add_argument("--text-file", type=Path, help="UTF-8 file containing the note text")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="Append a user-supplied note to the vault")
    capture.add_argument("--vault", type=Path, required=True)
    capture.add_argument("--title", required=True)
    capture.add_argument("--kind", default="source",
                         choices=sorted(CAPTURE_KINDS))
    capture.add_argument("--effective-date")
    _add_text(capture)

    snapshot = commands.add_parser("snapshot", help="Freeze an opt-in vault collection")
    snapshot.add_argument("--vault", type=Path, required=True)
    snapshot.add_argument(
        "--collection", default=".", help="Relative folder; default is vault root"
    )
    snapshot.add_argument("--output", type=Path, required=True)

    for name in ("inspect", "recall"):
        command = commands.add_parser(name, help="Review exact passages from lexical search")
        command.add_argument("--snapshot", type=Path, required=True)
        command.add_argument("--query", required=True)
        command.add_argument("--top-k", type=int, default=5)
        command.add_argument("--output", type=Path, help="Save a review JSON for approval")

    transcript = commands.add_parser(
        "review-code-exec", help="Verify spans in a saved code-execution transcript"
    )
    transcript.add_argument("--snapshot", type=Path, required=True)
    transcript.add_argument("--transcript", type=Path, required=True)
    transcript.add_argument("--question-id")
    transcript.add_argument("--output", type=Path, help="Save the verified packet")

    approve = commands.add_parser("approve", help="Append a user-approved dated revision")
    approve.add_argument("--vault", type=Path, required=True)
    approve.add_argument("--snapshot", type=Path, required=True)
    approve.add_argument("--review", type=Path, required=True)
    approve.add_argument("--title", required=True)
    approve.add_argument("--kind", required=True, choices=["lesson", "correction", "decision"])
    approve.add_argument("--evidence", nargs="+", required=True, metavar="EID",
                         help="Evidence IDs reviewed by the user, e.g. E1 E2")
    approve.add_argument("--effective-date")
    approve.add_argument("--supersedes", help="Snapshot document or record ID this note corrects")
    approve.add_argument("--confirm", action="store_true",
                         help="Required: I reviewed the evidence and approve saving this note")
    _add_text(approve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "capture":
        print(capture_note(args.vault, title=args.title, text=_text(args), kind=args.kind,
                           effective_date=args.effective_date))
    elif args.command == "snapshot":
        result = freeze_vault(args.vault, args.output, args.collection)
        print(json.dumps({"snapshot": str(args.output), "status": result["status"],
                          "documents": len(result["papers"]),
                          "corpus_hash": result["corpus_hash"]}, indent=2))
    elif args.command in {"inspect", "recall"}:
        _save_json(inspect_evidence(args.snapshot, args.query, args.top_k), args.output)
    elif args.command == "review-code-exec":
        _save_json(review_code_exec_transcript(args.snapshot, args.transcript,
                                              args.question_id), args.output)
    elif args.command == "approve":
        if not args.confirm:
            raise ValueError("--confirm is required after reviewing the selected evidence")
        review = json.loads(args.review.read_text())
        print(approve_note(args.vault, args.snapshot, review, title=args.title,
                           text=_text(args), kind=args.kind, evidence_ids=args.evidence,
                           effective_date=args.effective_date, supersedes=args.supersedes))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
