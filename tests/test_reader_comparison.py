"""No paid calls: source isolation, matched routing, budgets and failure retention."""
import json
import os

import pytest

from scripts.reader_comparison import authorize
from src.eval import reader_comparison_plan as planning
from src.eval.reader_comparison_run import run
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.assigned_reader import build_assigned_reader
from tests.arxiv_fakes import article_page, make_library
from tests.test_assigned_reader import SourceREPL
from tests.test_assigned_smoke_plan import IMAGE, Tokenizer


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    library, network = make_library(tmp_path / 'library',
                                    {'2309.06180v1': {'html': article_page()}})
    library.open('2309.06180v1')
    doc = library.document('2309.06180v1')
    data = tmp_path / 'data'
    data.mkdir()
    cases = [{'case_id': f'case-{i}', 'paper_version': '2309.06180v1',
              'kind': ['target_gap', 'supported', 'condition_binding'][i // 4],
              'question': f'Question {i}?'} for i in range(12)]
    refs = {'human_review': 'not_performed', 'sent_to_model': False, 'held_out': False,
            'cases': [{'case_id': c['case_id'], 'requirements': ['PRIVATE REFERENCE'],
                       'source_contexts': [{'start': 0, 'end': 10, 'text': doc['text'][:10]}]}
                      for c in cases]}
    for name, value in [('cases.json', cases), ('references.json', refs)]:
        (data / name).write_text(json.dumps(value))
    monkeypatch.setattr(planning, 'DATA', data)
    prepared = tmp_path / 'prepared'
    before = len(network.calls)
    image = os.environ.get('ENVOY_ASSIGNED_TEST_DOCKER_IMAGE', IMAGE)
    plan = planning.prepare(library.root, prepared, sandbox_image=image,
                            tokenizer=Tokenizer())
    assert len(network.calls) == before
    return plan, prepared, doc, data


def test_paired_inputs_only_differ_by_model_and_do_not_leak_rubric(frozen):
    plan, prepared, _, _ = frozen
    assert planning.check(prepared, tokenizer=Tokenizer()) == plan
    inputs = json.loads((prepared / 'model_inputs.json').read_text())
    assert 'PRIVATE REFERENCE' not in json.dumps(inputs)
    assert 'source_contexts' not in json.dumps(inputs)
    assert 'target_gap' not in json.dumps(inputs)
    for a, b in zip(inputs[::2], inputs[1::2]):
        assert a['payload']['model'] == 'envoy-base-only'
        assert b['payload']['model'] == 'envoy'
        assert {k: v for k, v in a['payload'].items() if k != 'model'} == {
            k: v for k, v in b['payload'].items() if k != 'model'}
    assert sum(x['arm'] == 'base' for x in plan['schedule'][::2]) == 6
    assert planning.LIMITS['requests_total'] == 24 * 6
    assert planning.LIMITS['input_tokens_reserved_total'] == 144 * 7168
    assert planning.LIMITS['output_tokens_reserved_total'] == 144 * 1024


@pytest.mark.parametrize('target', ['model_inputs.json', 'system-prompt.txt', 'plan.json'])
def test_frozen_artifact_change_rejected(frozen, target):
    _, prepared, _, _ = frozen
    (prepared / target).write_text('{}')
    with pytest.raises(ValueError):
        planning.check(prepared, tokenizer=Tokenizer())


def test_review_source_and_runtime_changes_fail_closed(frozen, monkeypatch):
    plan, prepared, _, data = frozen
    (data / 'references.json').write_text('{}')
    with pytest.raises(ValueError, match='reference files changed'):
        planning.check(prepared, tokenizer=Tokenizer())
    monkeypatch.setattr(planning, 'sources', lambda: {})
    with pytest.raises(ValueError, match='implementation changed'):
        planning.check(prepared, tokenizer=Tokenizer())
    with pytest.raises(FileExistsError):
        planning.prepare(prepared, prepared, sandbox_image=IMAGE, tokenizer=Tokenizer())


def scripted_run(frozen, output, *, fail_at=None, docker=False, missing_usage_at=None):
    _, prepared, doc, _ = frozen
    posts = []

    def factory(endpoint, key, tokenizer, audit, **kwargs):
        assert kwargs['requests_max'] == 6
        model = kwargs['served_model']
        actions = iter([f'print(read("{doc["doc_id"]}"))',
            f'submit(answer="Scripted fact", citations=["{doc["doc_id"]}"], '
            f'evidence=[{{"doc_id":"{doc["doc_id"]}","start":0,"end":10}}])'])

        def http(method, path, payload=None):
            posts.append(payload['input']['openai_input'])
            if len(posts) == fail_at:
                raise RuntimeError('uncertain dispatch')
            return {'id': f'job-{len(posts)}', 'status': 'COMPLETED', 'output': {
                'model': model, 'choices': [{'message': {'content': next(actions)}}],
                'usage': ({} if len(posts) == missing_usage_at else
                          {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110})}}
        return RunpodQueueTransport(endpoint, key, tokenizer, audit, http=http, **kwargs)

    def reader(*args, **kwargs):
        value = build_assigned_reader(*args, **kwargs)
        if not docker:
            value.repl_factory = lambda _: SourceREPL(doc['text'])
        return value

    output.mkdir()
    (output / 'runtime.json').write_text(json.dumps({
        'plan_sha256': planning.file_hash(prepared / 'plan.json')}))
    result = run(prepared, output, tokenizer=Tokenizer(), key='fake',
                 transport_factory=factory, reader_factory=reader)
    return result, posts


def test_all_24_real_mcp_cells_match_frozen_inputs(frozen, tmp_path):
    output = tmp_path / 'run'
    summary, posts = scripted_run(frozen, output)
    assert summary['status'] == 'completed_pending_claim_review'
    assert len(summary['cells']) == 24 and summary['requests_reserved'] == len(posts) == 48
    assert not summary['unattempted_cells'] and summary['semantic_support'] == 'not_reviewed'
    assert len(list(output.glob('*.mcp.json'))) == 24
    assert all(c['mechanical_check'] == 'passed' for c in summary['cells'])
    assert all(c['execution_error_steps'] == [] for c in summary['cells'])
    packets = [json.loads(p.read_text()) for p in output.glob('*.packet.json')]
    assert {p['model_identity']['adapter_applied'] for p in packets} == {False, True}
    with pytest.raises(FileExistsError):
        run(frozen[1], output, tokenizer=Tokenizer(), key='fake')
    assert json.loads((output / 'results.json').read_text()) == summary


def test_uncertain_call_stops_without_retry_and_keeps_previous_pair(frozen, tmp_path):
    output = tmp_path / 'partial'
    summary, posts = scripted_run(frozen, output, fail_at=5)
    assert len(posts) == 5 and summary['requests_reserved'] == 5
    assert summary['status'] == 'incomplete' and len(summary['unattempted_cells']) == 21
    assert [c['mechanical_check'] for c in summary['cells']] == ['passed', 'passed', 'failed']
    assert len(list(output.glob('*.verified.json'))) >= 2
    assert json.loads((output / 'results.json').read_text()) == summary


def test_new_approval_cannot_reuse_old_smoke_allowance(frozen, tmp_path):
    plan, prepared, _, _ = frozen
    path = tmp_path / 'approval.json'
    record = {'approved': True, 'attempts': 1,
              'plan_sha256': planning.file_hash(prepared / 'plan.json'),
              'limits': plan['limits'], 'current_gpu_hourly_usd': 0.69,
              'provider_dollar_cap': False, 'user_authorization': 'test fixture',
              'rate_checked_at': '2026-10-20T00:00:00Z'}
    path.write_text(json.dumps(record))
    assert authorize(path, record['plan_sha256'])['record'] == record
    record['limits'] = {'model_requests_total': 12}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='new comparison'):
        authorize(path, planning.file_hash(prepared / 'plan.json'))


def test_missing_one_request_usage_is_not_hidden_by_positive_cell_sum(frozen, tmp_path):
    summary, _ = scripted_run(frozen, tmp_path / 'missing-usage', missing_usage_at=2)
    assert summary['status'] == 'completed_pending_claim_review'
    audit = summary['cells'][0]['audit']
    assert audit['reported_usage']['total_tokens'] == 110
    assert audit['completed'] == 2 and audit['usage_requests_complete'] == 1
    assert audit['usage_complete'] is False


@pytest.mark.skipif(not os.environ.get('ENVOY_ASSIGNED_TEST_DOCKER_IMAGE'),
                    reason='set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE for real Docker')
def test_both_arms_execute_scripted_model_python_in_real_docker(frozen, tmp_path):
    output = tmp_path / 'docker'
    summary, posts = scripted_run(frozen, output, fail_at=5, docker=True)
    assert len(posts) == 5
    assert [c['mechanical_check'] for c in summary['cells'][:2]] == ['passed', 'passed']
    assert {c['arm'] for c in summary['cells'][:2]} == {'base', 'v5'}
    for cell in summary['cells'][:2]:
        packet = json.loads((output / f"{cell['case_id']}--{cell['arm']}.packet.json").read_text())
        assert packet['execution'] == 'docker_only'
