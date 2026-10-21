"""Identity-redacted claim review; exact quotes never supply semantic grades."""
from __future__ import annotations

import json
import random
import statistics

from src.eval.assigned_smoke_plan import digest, write_json
from src.eval.reader_comparison_plan import DATA, LIMITS, file_hash


def export(prepared, run_dir, output):
    if output.exists():
        raise FileExistsError('review exports cannot be overwritten')
    plan = json.loads((prepared / 'plan.json').read_text())
    results = json.loads((run_dir / 'results.json').read_text())
    runtime = json.loads((run_dir / 'runtime.json').read_text())
    references = json.loads((prepared / 'references.json').read_text())
    if (file_hash(prepared / 'plan.json') != (prepared / 'plan.sha256').read_text().strip()
            or digest(references) != plan['references_sha256']
            or file_hash(DATA / 'references.json') != plan['data_sha256']['references.json']):
        raise ValueError('frozen review references or plan changed')
    if runtime.get('plan_sha256') != file_hash(prepared / 'plan.json'):
        raise ValueError('run belongs to a different prepared plan')
    if (results['status'] != 'completed_pending_claim_review'
            or [{'case_id': c['case_id'], 'arm': c['arm']} for c in results['cells']]
            != plan['schedule']
            or any(c['mechanical_check'] != 'passed' for c in results['cells'])):
        raise ValueError('incomplete runs remain diagnostic; no complete paired review')
    refs = {r['case_id']: r for r in references['cases']}
    rng, questions, rows, key = random.Random(42), [], [], {}
    for case in plan['cases']:
        arms = ['base', 'v5']
        rng.shuffle(arms)
        samples = []
        for slot, arm in zip(['A', 'B'], arms):
            name = f"{case['case_id']}--{arm}"
            packet_path = run_dir / f'{name}.packet.json'
            packet = json.loads(packet_path.read_text())
            key[f"{case['case_id']}:{slot}"] = {'arm': arm,
                'packet_sha256': file_hash(packet_path)}
            samples.append({'slot': slot, 'status': packet['status'],
                'candidate_claims': packet.get('candidate_claims', []),
                'evidence': packet.get('evidence', []),
                'research_gaps': packet.get('research_gaps', [])})
            rows.append({'case_id': case['case_id'], 'slot': slot, 'verdict': None,
                         'condition_error': None, 'unsupported_numeric': None, 'notes': None})
        questions.append({**case, **refs[case['case_id']], 'samples': samples})
    packet = {'plan_sha256': file_hash(prepared / 'plan.json'),
              'plan_content_sha256': digest(plan), 'questions': questions,
              'instructions': 'Judge every factual clause against its own cited evidence and '
              'the frozen requirements. A real quote is not automatically relevant/supporting. '
              'Pass requires every requirement; partial is not a pass. Record condition swaps '
              'and unsupported numeric claims separately, including in qualified partials. '
              'Use complete extracted source when needed; do not turn absence in excerpts '
              'into absence in the full paper. Reference context alone cannot repair an '
              'unsupported candidate citation. Keep model-key.json and raw run logs unseen.'}
    output.mkdir(parents=True)
    write_json(output / 'review-packet.json', packet)
    write_json(output / 'model-key.json', {'key': key, 'results_sha256':
        digest(results), 'review_packet_sha256': digest(packet)})
    write_json(output / 'review-template.json', {'reviewer_kind': None,
        'reviewer_identity': None, 'reviewed_at': None, 'rows': rows})


def summarize(plan, results, review_packet, model_key, review):
    if (review_packet['plan_content_sha256'] != digest(plan)
            or model_key['results_sha256'] != digest(results)
            or model_key['review_packet_sha256'] != digest(review_packet)):
        raise ValueError('review binding differs from recorded run')
    if (len(results['cells']) != LIMITS['questions'] * LIMITS['arms']
            or [{'case_id': c['case_id'], 'arm': c['arm']} for c in results['cells']]
            != plan['schedule']
            or results['unattempted_cells'] or results['status'] != 'completed_pending_claim_review'
            or any(r['mechanical_check'] != 'passed' for r in results['cells'])):
        raise ValueError('incomplete comparison is inconclusive')
    key, rows = model_key['key'], review['rows']
    if (review.get('reviewer_kind') not in {'human', 'model_assisted'}
            or not review.get('reviewer_identity') or not review.get('reviewed_at')
            or len(rows) != 24 or len({(r['case_id'], r['slot']) for r in rows}) != 24
            or {f"{r['case_id']}:{r['slot']}" for r in rows} != set(key)):
        raise ValueError('complete attributed source-visible review required')
    kinds = {c['case_id']: c['kind'] for c in plan['cases']}
    counts = {arm: {'supported_passes': 0, 'gap_passes': 0, 'condition_errors': 0,
                    'unsupported_numeric': 0, 'execution_failure_episodes': 0}
              for arm in ('base', 'v5')}
    for row in rows:
        if (row['verdict'] not in {'pass', 'partial', 'fail'} or not row.get('notes')
                or type(row['condition_error']) is not bool
                or type(row['unsupported_numeric']) is not bool
                or (row['verdict'] == 'pass' and
                    (row['condition_error'] or row['unsupported_numeric']))):
            raise ValueError('review requires explicit claim judgments and notes')
        arm = key[f"{row['case_id']}:{row['slot']}"]['arm']
        count = counts[arm]
        field = 'gap_passes' if kinds[row['case_id']] == 'target_gap' else 'supported_passes'
        count[field] += row['verdict'] == 'pass'
        count['condition_errors'] += row['condition_error']
        count['unsupported_numeric'] += row['unsupported_numeric']
    telemetry_complete = True
    for arm, count in counts.items():
        cells = [c for c in results['cells'] if c['arm'] == arm]
        count['execution_failure_episodes'] = sum(bool(c['execution_error_steps']) for c in cells)
        usages = [c['audit'].get('reported_usage', {}).get('total_tokens') for c in cells]
        complete = (all(type(u) is int and u > 0 for u in usages)
                    and all(c['audit'].get('usage_complete') is True for c in cells))
        telemetry_complete &= complete
        count['reported_total_tokens'] = sum(usages) if complete else None
        count['median_seconds'] = statistics.median(c['seconds'] for c in cells)
    base, v5 = counts['base'], counts['v5']
    timings = {(c['case_id'], c['arm']): c['seconds'] for c in results['cells']}
    if len(timings) != 24 or any(type(s) not in {int, float} or s <= 0
                               for s in timings.values()):
        raise ValueError('complete positive paired timings required')
    latency_ratio = statistics.median(timings[(c['case_id'], 'v5')] /
        timings[(c['case_id'], 'base')] for c in plan['cases'])
    favorable = (telemetry_complete and v5['supported_passes'] >= base['supported_passes'] + 2
        and all(v5[k] <= base[k] for k in ('condition_errors', 'unsupported_numeric',
                                         'execution_failure_episodes'))
        and v5['reported_total_tokens'] <= 1.25 * base['reported_total_tokens']
        and latency_ratio <= 1.25)
    return {'arms': counts, 'decision': 'favorable_development_signal' if favorable else
        ('inconclusive_missing_telemetry' if not telemetry_complete else 'threshold_not_met'),
        'median_paired_latency_ratio_v5_over_base': latency_ratio,
        'reviewer_kind': review['reviewer_kind'], 'human_review':
        'performed' if review['reviewer_kind'] == 'human' else 'not_performed',
        'limitation': 'One small development comparison. No held-out or statistically validated '
        'improvement claim, no reward, no automatic training authorization.',
        'actual_spend_usd': results.get('actual_spend_usd')}
