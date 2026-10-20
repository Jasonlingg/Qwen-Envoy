"""Package a stopped native collection for human review without assigning grades."""

import argparse
import json
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.native_pilot_packet import package  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.source / "execution-manifest.json").read_text())
    questions = json.loads((ROOT / manifest["questions_file"]).read_text())
    package(args.source, args.destination, manifest, questions, secrets.randbits(64))
    rubric = ROOT / "benchmarks/ml-workflow-pilot-v1/reviewer-rubric.json"
    (args.destination / "review/rubric.json").write_bytes(rubric.read_bytes())
    (args.destination / "review/README.md").write_text(
        "# Human review packet\n\n"
        "Read each question and its two anonymous answer files, then rubric.json. "
        "Record initial scores and source-context checks in worksheet.json before "
        "opening operator files or collection-report.json. Those reveal identities.\n\n"
        "Keep human and model-assisted review separate. All verdicts are currently "
        "unknown. Check material claims against primary sources, not just citation "
        "presence. Record source-check/initial-review time separately. Time any "
        "corrections up to ten minutes per answer; retain unfinished/censored work. "
        "Mark any formatting/source cues that reveal an arm.\n")
    print("Created review inputs and operator receipts; no quality scores assigned.")


if __name__ == "__main__":
    main()
