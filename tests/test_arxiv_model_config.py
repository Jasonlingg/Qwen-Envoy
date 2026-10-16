"""V2 policy, runner and verified packet use one effective prompt identity."""
import hashlib
import json

import pytest

from src.product.arxiv_mcp_config import build_reader
from src.product.packet_verifier import verify_investigation_packet
from src.research.agent import load_snapshot
from tests.arxiv_fakes import article_page, make_library
from tests.test_qwen_investigator import FakeREPL


@pytest.mark.parametrize('kind', ['qwen', 'nemotron'])
def test_configured_v2_policy_prompt_matches_packet(tmp_path, monkeypatch, kind):
    monkeypatch.setenv('NEBIUS_API_KEY', 'test')
    library, _ = make_library(tmp_path, {'2609.12345v2': {'html': article_page()}})
    library.open('2609.12345v2')
    snapshot = library.view(['2609.12345v2'])
    doc_id = library.document('2609.12345v2')['doc_id']
    reader = build_reader(kind, qwen_endpoint='http://test', qwen_model='fake',
                          qwen_checkpoint='pinned-fake', harness_version='v2')
    policy = reader.policy_factory()
    action = (f'submit(answer="A SUBMIT: marker.", citations=["{doc_id}"], '
              f'evidence=[{{"doc_id":"{doc_id}","start":0,"end":10}}])')
    replies = iter(['print(search("cache"))', f'print(read("{doc_id}"))', action])
    sent = []
    def transport(url, payload, headers, timeout):
        sent.append(json.loads(json.dumps(payload)))
        return {'choices': [{'message': {'content': next(replies)}}]}
    policy._transport = transport
    reader.policy_factory = lambda: policy
    reader.repl_factory = lambda _: FakeREPL()
    reader.max_steps = 3
    packet = reader.investigate('What marker?', snapshot)
    prompt_hash = hashlib.sha256(policy.system_prompt.encode()).hexdigest()
    assert packet['status'] == 'evidence_found'
    assert packet['system_prompt_sha256'] == prompt_hash
    assert sent[0]['messages'][0]['content'] == reader.system_prompt == policy.system_prompt
    assert 'Never call submit()' not in str(sent)
    manifest, docs = load_snapshot(snapshot)
    verified = verify_investigation_packet(packet, manifest, docs,
                                           allowed_prompt_sha256s=frozenset({prompt_hash}))
    assert verified['prompt_hash'] == prompt_hash
