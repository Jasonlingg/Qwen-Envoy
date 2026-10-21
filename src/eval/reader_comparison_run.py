"""One fixed paired pass through real MCP; only Docker executes model Python."""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

from src.eval.assigned_smoke_mcp import investigate_through_mcp
from src.eval.assigned_smoke_plan import write_json
from src.eval.assigned_smoke_run import audit_summary, track_sandbox
from src.eval.reader_comparison_plan import ARMS, LIMITS, check
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.assigned_reader import build_assigned_reader
from src.product.packet_verifier import verify_investigation_packet
from src.research.agent import load_snapshot


def request_audit_summary(path):
    result = audit_summary(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    completed = [r for r in rows if r.get('event') == 'completed']
    covered = 0
    for row in completed:
        usage = row.get('response', {}).get('usage', {})
        values = [usage.get(k) for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')]
        if (all(type(n) is int and n >= 0 for n in values)
                and values[2] > 0 and values[0] + values[1] == values[2]):
            covered += 1
    result['usage_requests_complete'] = covered
    result['usage_complete'] = (0 < covered == result['completed'] == result['submitted']
        == result['reserved'] and result['failed'] == 0
        and len({r['request'] for r in completed}) == len(completed))
    return result


def identity(plan, arm):
    pins = plan['models']
    if arm == 'base':
        return {'checkpoint': f"{pins['base']}@{pins['base_revision']}",
                'base': pins['base'], 'base_revision': pins['base_revision'],
                'adapter_applied': False, 'served_model': ARMS[arm]}
    checkpoint = f"{pins['adapter']}@{pins['adapter_revision']}/{pins['subfolder']}"
    return {**pins, 'checkpoint': checkpoint,
            'adapter_applied': True, 'served_model': ARMS[arm]}


def action_metrics(packet):
    metrics = {name: [] for name in ('execution_error_steps', 'action_rejection_steps',
        'submission_rejection_steps', 'gap_rejection_steps', 'repeated_call_rejections')}
    for trace in packet.get('trajectory', []):
        observation = trace.get('observation', '')
        for prefix, field in [('Action rejected:', 'action_rejection_steps'),
            ('Submission rejected:', 'submission_rejection_steps'),
            ('Gap report rejected:', 'gap_rejection_steps')]:
            if observation.startswith(prefix):
                metrics[field].append(trace['step'])
        if re.search(r'Traceback|(?:Type|Value|Name|Syntax|Runtime|Key|Index)Error:|ERROR:',
                     observation):
            metrics['execution_error_steps'].append(trace['step'])
        if 'identical' in observation and 'already executed' in observation:
            metrics['repeated_call_rejections'].append(trace['step'])
    return metrics


def run(prepared, output, *, tokenizer, key, transport_factory=RunpodQueueTransport,
        reader_factory=build_assigned_reader, invoke=investigate_through_mcp):
    plan = check(prepared, tokenizer=tokenizer)
    if (output / 'results.json').exists():
        raise FileExistsError('existing comparison results cannot be overwritten')
    with (output / 'CELLS_STARTED').open('x'):
        pass
    cases = {c['case_id']: c for c in plan['cases']}
    frozen = {(p['case_id'], p['arm']): p['payload'] for p in
              json.loads((prepared / 'model_inputs.json').read_text())}
    results = []

    def save():
        summary = {'status': 'completed_pending_claim_review' if len(results) == 24 and
            all(r['mechanical_check'] == 'passed' for r in results) else 'incomplete',
            'cells': results, 'unattempted_cells': plan['schedule'][len(results):],
            'requests_reserved': sum(r.get('audit', {}).get('reserved', 0) for r in results),
            'semantic_support': 'not_reviewed', 'human_review': 'not_performed',
            'actual_spend_usd': None, 'reward': 'not_computed'}
        write_json(output / 'results.json', summary)
        return summary

    save()
    for item in plan['schedule']:
        case, arm = cases[item['case_id']], item['arm']
        name = f"{case['case_id']}--{arm}"
        prefix, audit = output / name, output / f'{name}.requests.jsonl'
        with audit.open('x'):
            pass  # Exclusive cell ledger: never reset after failure or uncertain dispatch.
        transport = None
        started = time.monotonic()
        result = {**item, 'mechanical_check': 'failed', 'semantic_support': 'not_reviewed'}
        try:
            transport = transport_factory(plan['endpoint_id'], key, tokenizer, audit,
                requests_max=LIMITS['requests_per_cell'], served_model=ARMS[arm])
            first = True

            def guarded(url, payload, headers, timeout):
                nonlocal first
                if first and payload != frozen[(case['case_id'], arm)]:
                    raise ValueError('first live request differs from frozen input')
                first = False
                return transport(url, payload, headers, timeout)

            model = identity(plan, arm)
            reader = reader_factory('qwen',
                qwen_endpoint=f"https://api.runpod.ai/v2/{plan['endpoint_id']}",
                qwen_model=ARMS[arm], qwen_checkpoint=model['checkpoint'],
                max_steps=LIMITS['action_steps_per_cell'], sandbox_image=plan['sandbox_image'],
                policy_transport=guarded, reader_guidance=plan['guidance'])
            if hashlib.sha256(reader.system_prompt.encode()).hexdigest() != plan['prompt_sha256']:
                raise ValueError('live prompt differs from preparation')
            reader.model_identity.update(model)
            if reader.repl_factory is None:
                track_sandbox(reader, output)
            snapshot = Path(plan['papers'][case['paper_version']]['snapshot'])
            packet = invoke(reader, case['question'], snapshot, prefix)
            write_json(prefix.with_suffix('.packet.json'), packet)
            manifest, docs = load_snapshot(snapshot)
            verified = verify_investigation_packet(packet, manifest, docs,
                allowed_prompt_sha256s=frozenset({plan['prompt_sha256']}),
                expected_reader_mode='assigned', expected_reader_guidance=plan['guidance'])
            write_json(prefix.with_suffix('.verified.json'), verified)
            result.update(status=packet['status'], **action_metrics(packet))
            if packet['status'] in {'error', 'unavailable'} or first:
                raise ValueError('reader failed or made no model request')
            result['mechanical_check'] = 'passed'
        except Exception as exc:
            result.update(error_type=type(exc).__name__, mechanical_check='failed')
        finally:
            if transport is not None:
                transport.audit.close()
            result.update(seconds=time.monotonic() - started, audit=request_audit_summary(audit))
            if result['audit']['failed']:
                result['mechanical_check'] = 'failed'
            results.append(result)
            summary = save()  # Preserve earlier packets/results before any fail-fast exit.
        if result['mechanical_check'] != 'passed':
            break
    return summary
