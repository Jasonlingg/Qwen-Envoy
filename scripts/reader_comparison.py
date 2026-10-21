"""Prepare/check locally, or run one separately authorized bounded comparison."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.runpod_qwen_bootstrap import BASE, BASE_REVISION
from src.env.repl import PersistentREPL
from src.eval.assigned_smoke_lifecycle import supervise
from src.eval.assigned_smoke_plan import write_json
from src.eval.reader_comparison_plan import LIMITS, check, file_hash, prepare
from src.eval.reader_comparison_run import run


def authorize(path, plan_sha):
    record = json.loads(path.read_text())
    if (record.get('approved') is not True or record.get('attempts') != 1
            or record.get('plan_sha256') != plan_sha or record.get('limits') != LIMITS
            or type(record.get('current_gpu_hourly_usd')) not in {float, int}
            or not 0 < record['current_gpu_hourly_usd'] <= 1
            or record.get('provider_dollar_cap') is not False
            or not record.get('user_authorization') or not record.get('rate_checked_at')):
        raise ValueError('new comparison needs explicit plan-bound approval and current rate')
    return {'record': record, 'sha256': file_hash(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'check', 'run', 'worker'])
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--library-root', type=Path)
    parser.add_argument('--sandbox-image')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--authorization-file', type=Path)
    args = parser.parse_args()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE, revision=BASE_REVISION,
        local_files_only=True, trust_remote_code=False)
    if args.phase == 'prepare':
        if not args.library_root or not args.sandbox_image:
            parser.error('prepare needs cached --library-root and pinned --sandbox-image')
        prepare(args.library_root, args.prepared, sandbox_image=args.sandbox_image,
                tokenizer=tokenizer)
    plan = check(args.prepared, tokenizer=tokenizer)
    if args.phase in {'prepare', 'check'}:
        print(json.dumps({'status': plan['status'], 'cells': len(plan['schedule']),
            'model_calls': 0, 'limits': LIMITS}, indent=2))
        return
    if not args.output:
        parser.error('live phases need a new --output directory')
    plan_sha = file_hash(args.prepared / 'plan.json')
    if args.phase == 'run':
        if not args.authorization_file:
            parser.error('run needs a new comparison --authorization-file')
        approval = authorize(args.authorization_file, plan_sha)
        if not PersistentREPL._docker_available(plan['sandbox_image']):
            raise ValueError('pinned Docker unavailable; GPU was not activated')
        args.output.mkdir(parents=True, exist_ok=False)
        with (ROOT / 'out/reader-comparison-LIVE_ATTEMPT').open('x') as stream:
            stream.write(str(args.output.resolve()) + '\n')
        write_json(args.output / 'runtime.json', {'approval': approval,
            'plan_sha256': plan_sha, 'started_at': datetime.now(timezone.utc).isoformat(),
            'implementation_commit': subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()})
        command = [sys.executable, str(Path(__file__).resolve()), 'worker', '--prepared',
                   str(args.prepared.resolve()), '--output', str(args.output.resolve())]
        supervise(plan['endpoint_id'], args.output, command, key=credential(),
                  seconds=LIMITS['watchdog_seconds'])
    else:
        marker = ROOT / 'out/reader-comparison-LIVE_ATTEMPT'
        runtime = json.loads((args.output / 'runtime.json').read_text())
        approval = runtime['approval']
        if (marker.read_text().strip() != str(args.output.resolve())
                or runtime['plan_sha256'] != plan_sha
                or approval['record'].get('plan_sha256') != plan_sha
                or approval['record'].get('approved') is not True
                or approval['record'].get('limits') != LIMITS):
            raise ValueError('worker is not bound to this exclusive approved session')
        with (args.output / 'WORKER_STARTED').open('x'):
            pass
        summary = run(args.prepared, args.output, tokenizer=tokenizer, key=credential())
        if summary['status'] != 'completed_pending_claim_review':
            raise RuntimeError('incomplete comparison retained; no retry or promotion')


def credential():
    key = os.environ.get('RUNPOD_API_KEY')
    if not key:
        key = tomllib.loads((Path.home() / '.runpod/config.toml').read_text()).get('apikey')
    if not key:
        raise ValueError('Runpod credential required')
    return key


if __name__ == '__main__':
    main()
