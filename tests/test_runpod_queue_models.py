"""A comparison must opt in to base routing and cannot cross model ledgers."""
import json

import pytest

from src.policies.runpod_queue import RunpodQueueTransport
from tests.test_runpod_queue import Tokenizer, payload


def test_base_alias_is_explicit_and_response_bound(tmp_path):
    calls = []

    def http(method, path, body=None):
        calls.append((method, path, body))
        return {'id': 'base-1', 'status': 'COMPLETED', 'output': {
            'model': 'envoy-base-only', 'choices': [{'message': {'content': 'print(1)'}}]}}

    audit = tmp_path / 'base.requests.jsonl'
    transport = RunpodQueueTransport('abc', 'secret', Tokenizer(), audit,
        requests_max=1, served_model='envoy-base-only', http=http)
    with pytest.raises(ValueError, match='unsupported reader'):
        transport('', payload(), {}, 120)
    assert transport.remaining == 1
    transport('', payload(model='envoy-base-only'), {}, 120)
    assert calls[0][2]['input']['openai_input']['model'] == 'envoy-base-only'
    assert json.loads(audit.read_text().splitlines()[0])['served_model'] == 'envoy-base-only'
    with pytest.raises(ValueError, match='refusing reset'):
        RunpodQueueTransport('abc', 'secret', Tokenizer(), audit, requests_max=1)
    assert RunpodQueueTransport('abc', 'secret', Tokenizer(), audit, requests_max=1,
        served_model='envoy-base-only').remaining == 0


def test_base_request_rejects_adapter_response_and_consumes_budget(tmp_path):
    calls = []

    def http(method, path, body=None):
        calls.append(path)
        return {'id': 'wrong-1', 'status': 'COMPLETED', 'output': {
            'model': 'envoy', 'choices': [{'message': {'content': 'bad'}}]}}

    transport = RunpodQueueTransport('abc', 'secret', Tokenizer(), tmp_path / 'audit',
        requests_max=1, served_model='envoy-base-only', http=http)
    with pytest.raises(ValueError, match='invalid adapter'):
        transport('', payload(model='envoy-base-only'), {}, 120)
    assert transport.remaining == 0
    assert calls == ['/run', '/cancel/wrong-1']


def test_adapter_ledger_cannot_be_reused_for_base(tmp_path):
    path = tmp_path / 'audit'
    RunpodQueueTransport('abc', 'secret', Tokenizer(), path, requests_max=1)
    with pytest.raises(ValueError, match='refusing reset'):
        RunpodQueueTransport('abc', 'secret', Tokenizer(), path,
            requests_max=1, served_model='envoy-base-only')
    with pytest.raises(ValueError, match='unknown served model'):
        RunpodQueueTransport('abc', 'secret', Tokenizer(), tmp_path / 'other',
            served_model='arbitrary')
