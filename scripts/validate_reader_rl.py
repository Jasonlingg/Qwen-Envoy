"""Offline reward/trainer report. Exit 2 means not ready; never starts training."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.reader_rl_validation import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-keys", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.reviewed_keys, args.output)
    print(json.dumps(result, indent=2))
    return 2  # Exposed diagnostic success can never authorize a launch.


if __name__ == "__main__":
    raise SystemExit(main())
