import pytest

from src.env.tools_v2 import HARNESS_V2_SHA256, V2_PROMPT_ADDENDUM
from src.product.qwen_investigator import FORMAT_RETRY_MESSAGE, QwenInvestigator
from tests.test_qwen_investigator import FakePolicy, FakeREPL, _investigator, _snapshot


def _v2(actions, repl, *, max_steps=5):
    policy = FakePolicy(actions)
    worker = QwenInvestigator(
        policy_factory=lambda: policy,
        model_identity={"checkpoint": "nemotron", "seed": 42, "temperature": 0},
        max_steps=max_steps, repl_factory=lambda corpus: repl, harness_version="v2",
    )
    return worker, policy


def test_v1_packets_record_harness_version_and_prompt_is_unchanged(tmp_path):
    snapshot, _, _, _ = _snapshot(tmp_path)
    worker, _ = _investigator(
        ['print(search("coverage"))', 'print(search("failed"))'], FakeREPL(), max_steps=2)
    assert V2_PROMPT_ADDENDUM not in worker.system_prompt
    packet = worker.investigate("What failed?", snapshot)
    assert packet["harness_version"] == "v1" and "harness_shim_sha256" not in packet


def test_v2_submit_call_is_converted_and_accepted(tmp_path):
    snapshot, _, doc_id, document = _snapshot(tmp_path)
    quote = "source coverage was incomplete"
    start = document["text"].index(quote)
    actions = [
        'print(search("coverage"))',
        f'print(read("{doc_id}"))',
        f'submit(answer="Coverage was incomplete.", citations=["{doc_id}"], '
        f'evidence=[{{"doc_id": "{doc_id}", "start": {start}, "end": {start + len(quote)}}}])',
    ]
    worker, _ = _v2(actions, FakeREPL())
    assert worker.system_prompt.endswith(V2_PROMPT_ADDENDUM)
    packet = worker.investigate("Why did the test fail?", snapshot)
    assert packet["status"] == "evidence_found"
    assert packet["harness_version"] == "v2"
    assert packet["harness_shim_sha256"] == HARNESS_V2_SHA256
    assert packet["trajectory"][-1]["submit_call_converted"] is True


def test_v2_variable_submit_is_rejected_without_execution(tmp_path):
    snapshot, _, _, _ = _snapshot(tmp_path)
    repl = FakeREPL()
    worker, policy = _v2(["submit(answer=ans)", 'print(search("coverage"))'], repl, max_steps=2)
    packet = worker.investigate("What failed?", snapshot)
    assert "literal values" in packet["trajectory"][0]["observation"]
    assert all("submit" not in code for code, _ in repl.executed)


def test_v2_prose_gets_two_free_retries_then_uses_a_step(tmp_path):
    snapshot, _, _, _ = _snapshot(tmp_path)
    prose = "The paper says the answer is blue."
    worker, policy = _v2([prose, prose, prose, 'print(search("coverage"))'],
                         FakeREPL(), max_steps=2)
    packet = worker.investigate("What failed?", snapshot)
    first = packet["trajectory"][0]
    assert [r["rejected_action"] for r in first["format_retries"]] == [prose, prose]
    assert first["action"] == prose            # third prose reply consumed step 1
    assert policy.observations[1] == FORMAT_RETRY_MESSAGE
    assert packet["trajectory"][1]["action"] == 'print(search("coverage"))'
    assert packet["model_requests_attempted"] == 4


def test_v2_valid_python_is_not_retried(tmp_path):
    snapshot, _, _, _ = _snapshot(tmp_path)
    worker, _ = _v2(
        ['print(search("coverage"))', 'print(search("failed"))'], FakeREPL(), max_steps=2)
    packet = worker.investigate("What failed?", snapshot)
    assert "format_retries" not in packet["trajectory"][0]
    assert packet["model_requests_attempted"] == 2


def test_unknown_harness_version_is_rejected():
    with pytest.raises(ValueError, match="harness_version"):
        QwenInvestigator(policy_factory=lambda: None, model_identity={"m": 1},
                         harness_version="v3")


@pytest.mark.parametrize('template', [
    'search("{doc}", "coverage")', 'search("coverage", "{doc}")',
    'search("coverage", doc_id="{doc}")', 'search("{doc}", query="coverage")',
    'search(query="coverage", doc_id="{doc}")',
    'search(*["{doc}", "coverage"])',
    'search(**dict(doc_id="{doc}", query="coverage"))',
    'd = "{doc}"; search(d, "coverage")',
])
def test_scoped_search_requires_discovery_and_counts_as_inspection(tmp_path, template):
    snapshot, _, doc_id, document = _snapshot(tmp_path)
    action = template.format(doc=doc_id)
    start = document['text'].index('source coverage')
    submit = (f'submit(answer="Coverage incomplete", citations=["{doc_id}"], '
              f'evidence=[{{"doc_id":"{doc_id}","start":{start},"end":{start+15}}}])')
    repl = FakeREPL()
    worker, _ = _v2([action, 'print(search("coverage"))', action, submit], repl)
    packet = worker.investigate('Why?', snapshot)
    assert 'first call search' in packet['trajectory'][0]['observation']
    assert packet['inspection_steps'] == 1
    assert packet['status'] == 'evidence_found'
    assert len(repl.executed) == 2


def test_effective_prompt_keeps_v1_and_removes_v2_nemotron_conflict(tmp_path):
    from src.product.qwen_investigator import (
        NEMOTRON_PAPER_SYSTEM_PROMPT,
        effective_system_prompt,
    )
    assert (effective_system_prompt('public_papers', 'nemotron', 'v1')
            == NEMOTRON_PAPER_SYSTEM_PROMPT)
    prompt = effective_system_prompt('public_papers', 'nemotron', 'v2')
    assert prompt.endswith(V2_PROMPT_ADDENDUM)
    assert 'There is no\nsubmit()' not in prompt
    snapshot, _, _, _ = _snapshot(tmp_path)
    policy = FakePolicy(['print(search("coverage"))', 'print(search("coverage again"))'])
    worker = QwenInvestigator(lambda: policy, {'m': 'test'}, source_domain='public_papers',
                              protocol_variant='nemotron', harness_version='v2',
                              repl_factory=lambda _: FakeREPL(), max_steps=2)
    result = worker.investigate('Why?', snapshot)
    assert worker.system_prompt == prompt
    assert 'Never call submit()' not in str(result['trajectory'])
    import hashlib
    assert result['system_prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
