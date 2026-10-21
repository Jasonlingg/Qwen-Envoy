"""Export identity-redacted review inputs or summarize explicitly authored reviews."""
# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION
from src.eval.reader_comparison_plan import check, file_hash
from src.eval.reader_comparison_review import export, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['export', 'report'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--review-dir', type=Path, required=True)
    parser.add_argument('--reviews', type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION,
        local_files_only=True, trust_remote_code=False)
    plan = check(args.prepared, tokenizer=tokenizer)
    if args.phase == 'export':
        export(args.prepared, args.run_dir, args.review_dir)
        return
    if not args.reviews:
        parser.error('report requires a completed --reviews file; blank templates are not grades')
    packet = json.loads((args.review_dir / 'review-packet.json').read_text())
    key = json.loads((args.review_dir / 'model-key.json').read_text())
    if packet['plan_sha256'] != file_hash(args.prepared / 'plan.json'):
        raise ValueError('review belongs to a different preparation')
    for cell, binding in key['key'].items():
        case_id, _ = cell.split(':')
        raw = args.run_dir / f"{case_id}--{binding['arm']}.packet.json"
        if file_hash(raw) != binding['packet_sha256']:
            raise ValueError('raw packet changed after review export')
    report = summarize(plan, json.loads((args.run_dir / 'results.json').read_text()),
                       packet, key, json.loads(args.reviews.read_text()))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
