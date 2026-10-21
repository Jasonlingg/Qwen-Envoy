"""Offline, source-pinned paired development comparison; no inference here."""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from src.eval.assigned_smoke_plan import MODEL_IDENTITY, ROOT, digest, initial_payload, write_json
from src.product.arxiv_library import ArxivLibrary
from src.product.paper_assignment import paper_assignment
from src.product.qwen_investigator import effective_system_prompt
from src.research.agent import load_snapshot

DATA = ROOT / 'examples/base-v5-reader-comparison'
ARMS = {'base': 'envoy-base-only', 'v5': 'envoy'}
LIMITS = {'questions': 12, 'arms': 2, 'attempts_per_cell': 1, 'action_steps_per_cell': 6,
          'requests_per_cell': 6, 'requests_total': 144, 'input_tokens_per_request': 7168,
          'output_tokens_per_request': 1024, 'input_tokens_reserved_total': 1032192,
          'output_tokens_reserved_total': 147456, 'watchdog_seconds': 900,
          'inference_resubmissions': 0, 'max_active_gpu_workers': 1}


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = list((ROOT / 'src').rglob('*.py')) + [
        ROOT / p for p in ('scripts/arxiv_research_mcp.py', 'scripts/runpod_qwen_bootstrap.py',
                          'scripts/reader_comparison.py',
                          'scripts/review_reader_comparison.py')]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)}


def schedule(cases):
    order = list(cases)
    random.Random(42).shuffle(order)
    return [{'case_id': case['case_id'], 'arm': arm}
            for i, case in enumerate(order)
            for arm in (['base', 'v5'] if i % 2 == 0 else ['v5', 'base'])]


def prompt():
    return effective_system_prompt('public_papers', 'default', 'v2',
                                   reader_mode='assigned', reader_guidance='focused-v1')


def inputs(cases, papers, tokenizer):
    items = []
    for case in cases:
        assignment = papers[case['paper_version']]['assignment']
        for arm, alias in ARMS.items():
            payload = {**initial_payload(prompt(), assignment, case['question']), 'model': alias}
            tokens = tokenizer.apply_chat_template(payload['messages'], tokenize=True,
                add_generation_prompt=True, enable_thinking=False, return_dict=False)
            if (not isinstance(tokens, list) or any(type(t) is not int for t in tokens)
                    or not 0 < len(tokens) <= LIMITS['input_tokens_per_request']):
                raise ValueError('initial prompt must be a bounded flat token list')
            items.append({'case_id': case['case_id'], 'arm': arm, 'payload': payload,
                          'initial_prompt_tokens': len(tokens)})
    return items


def checked_data(papers):
    cases = json.loads((DATA / 'cases.json').read_text())
    references = json.loads((DATA / 'references.json').read_text())
    if (len(cases) != LIMITS['questions'] or len({c['case_id'] for c in cases}) != len(cases)
            or any(sum(c['kind'] == k for c in cases) != 4
                   for k in ('supported', 'condition_binding', 'target_gap'))
            or references['human_review'] != 'not_performed'
            or references['sent_to_model'] is not False or references['held_out'] is not False
            or [r['case_id'] for r in references['cases']] != [c['case_id'] for c in cases]):
        raise ValueError('unknown question/reference contract')
    for case, ref in zip(cases, references['cases']):
        if not re.fullmatch(r'[a-z][a-z0-9-]+', case['case_id']):
            raise ValueError('unsafe case identifier')
        manifest, docs = load_snapshot(Path(papers[case['paper_version']]['snapshot']))
        doc = next(iter(docs.values()))
        if (manifest['corpus_hash'] != papers[case['paper_version']]['corpus_hash']
                or paper_assignment(manifest, docs) != papers[case['paper_version']]['assignment']
                or digest(doc) != papers[case['paper_version']]['document_sha256']):
            raise ValueError('frozen paper changed')
        for context in ref['source_contexts']:
            start, end = context['start'], context['end']
            if (type(start) is not int or type(end) is not int
                    or not 0 <= start < end <= len(doc['text'])
                    or context['text'] != doc['text'][start:end]):
                raise ValueError('reference is not exact source text')
    return cases, references


def prepare(library_root, output, *, sandbox_image, tokenizer):
    if output.exists():
        raise FileExistsError('frozen preparation cannot be overwritten')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}', sandbox_image):
        raise ValueError('Docker image must be pinned')
    library = ArxivLibrary(Path(library_root).resolve(strict=True))
    cases = json.loads((DATA / 'cases.json').read_text())
    papers = {}
    for version in sorted({c['paper_version'] for c in cases}):
        snapshot = library.view([version])  # Cached only, never open/fetch here.
        manifest, docs = load_snapshot(snapshot)
        doc = next(iter(docs.values()))
        if doc['metadata']['coverage'] == 'abstract_only':
            raise ValueError('comparison requires full extracted text')
        papers[version] = {'snapshot': str(snapshot), 'corpus_hash': manifest['corpus_hash'],
            'assignment': paper_assignment(manifest, docs), 'document_sha256': digest(doc)}
    cases, references = checked_data(papers)
    model_inputs = inputs(cases, papers, tokenizer)
    plan = {'schema': 'reader-comparison-v1', 'status': 'prepared_not_run',
        'model_calls': 0, 'activation_approved': False, 'limits': LIMITS, 'arms': ARMS,
        'models': MODEL_IDENTITY, 'endpoint_id': 'q05pjrvguakieg', 'cases': cases,
        'schedule': schedule(cases), 'papers': papers, 'sandbox_image': sandbox_image,
        'guidance': 'focused-v1', 'prompt_sha256': hashlib.sha256(prompt().encode()).hexdigest(),
        'hypothesis': 'The existing v5 adapter improves supported answering over base Qwen3-8B '
            'under the same assigned-paper MCP harness, beyond merely increasing refusals.',
        'expected_signal': 'More of the eight answerable cases meet every source-visible claim '
            'requirement, without more condition swaps, fabricated numbers or execution '
            'failures.',
        'decision_rule': 'Exploratory only: at least two additional supported-answer passes out '
            'of eight, no additional condition-swap/fabricated-number cases or '
            'execution-failure '
            'episodes, and no more than 25% higher total reported tokens or median paired latency. '
            'Require all 24 cells and source-visible review; missing telemetry/reviews means '
            'inconclusive. Report four target-gap cases separately. A favorable signal only '
            'justifies an untouched evaluation; it does not prove improvement or authorize '
            'training.',
        'split': 'New assistant-authored development questions; no established training/'
            'pretraining '
            'disjointness. Questions and references frozen before inference.',
        'reward': 'not_computed', 'human_review': 'not_performed',
        'budget': 'One exclusive session, 24 fixed cells, six durable reserved requests each. '
            'All format retries count; no resubmission after uncertain dispatch. One A5000; '
            '900-second watchdog includes cold start, then job/container cleanup and shutdown '
            'readback. Not a hard dollar ceiling; current rate and separate activation approval '
            'must be recorded before running. Stop on infrastructure failure; retain prior '
            'evidence.',
        'provider_dollar_cap': False, 'actual_spend_usd': None,
        'implementation_sha256': sources(), 'data_sha256': {
            name: file_hash(DATA / name) for name in ('cases.json', 'references.json')},
        'inputs_sha256': digest(model_inputs), 'references_sha256': digest(references)}
    output.mkdir(parents=True)
    write_json(output / 'plan.json', plan)
    write_json(output / 'model_inputs.json', model_inputs)
    write_json(output / 'references.json', references)
    (output / 'system-prompt.txt').write_text(prompt())
    (output / 'plan.sha256').write_text(file_hash(output / 'plan.json') + '\n')
    return plan


def check(output, *, tokenizer):
    if file_hash(output / 'plan.json') != (output / 'plan.sha256').read_text().strip():
        raise ValueError('plan changed')
    plan = json.loads((output / 'plan.json').read_text())
    for key, value in {'schema': 'reader-comparison-v1', 'limits': LIMITS, 'arms': ARMS,
                       'models': MODEL_IDENTITY, 'status': 'prepared_not_run', 'model_calls': 0,
                       'activation_approved': False, 'guidance': 'focused-v1',
                       'endpoint_id': 'q05pjrvguakieg'}.items():
        if plan.get(key) != value:
            raise ValueError(f'comparison contract changed: {key}')
    if plan['implementation_sha256'] != sources():
        raise ValueError('comparison implementation changed')
    if plan['data_sha256'] != {n: file_hash(DATA / n) for n in ('cases.json', 'references.json')}:
        raise ValueError('question/reference files changed')
    cases, references = checked_data(plan['papers'])
    if json.loads((output / 'references.json').read_text()) != references:
        raise ValueError('frozen references changed')
    if (cases != plan['cases'] or schedule(cases) != plan['schedule']
            or digest(references) != plan['references_sha256']
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', plan['sandbox_image'])
            or hashlib.sha256(prompt().encode()).hexdigest() != plan['prompt_sha256']
            or (output / 'system-prompt.txt').read_text() != prompt()):
        raise ValueError('prepared prompt, sources or schedule changed')
    expected = inputs(cases, plan['papers'], tokenizer)
    if (json.loads((output / 'model_inputs.json').read_text()) != expected
            or digest(expected) != plan['inputs_sha256']):
        raise ValueError('prepared model inputs changed')
    return plan
