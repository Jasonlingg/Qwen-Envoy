"""Prepare/check a frozen two-case smoke locally. This CLI cannot run inference."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION
from src.eval.assigned_smoke_check import check
from src.eval.assigned_smoke_plan import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "check"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-root", type=Path)
    parser.add_argument("--sandbox-image")
    parser.add_argument("--reader-guidance", choices=["baseline", "focused-v1"],
                        default="baseline", help="Opt-in candidate; prepare phase only")
    args = parser.parse_args()
    if args.phase == "prepare" and (args.library_root is None or not args.sandbox_image):
        parser.error("prepare needs --library-root and a pinned --sandbox-image")
    # Preparation never needs hub access, provider credentials or a model process.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION,
                                               local_files_only=True, trust_remote_code=False)
    if args.phase == "prepare":
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        prepare(args.library_root, args.output, sandbox_image=args.sandbox_image,
                tokenizer=tokenizer, implementation_commit=commit,
                reader_guidance=args.reader_guidance)
    print(json.dumps(check(args.output, tokenizer=tokenizer), indent=2))


if __name__ == "__main__":
    main()
