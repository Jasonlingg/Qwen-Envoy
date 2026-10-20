import json
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import pytest

from src.policies.runpod_queue import RunpodQueueTransport


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, enable_thinking=False,
                              return_dict=False)
        return [1] * len(messages[0]["content"])


def payload(**updates):
    return {"model": "envoy", "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 1024, "chat_template_kwargs": {"enable_thinking": False}, **updates}


def test_success_uses_one_submission_and_enforces_shared_budget(tmp_path):
    calls = []
    def http(method, path, body=None):
        calls.append((method, path, body))
        return {"id": "job-1", "status": "COMPLETED", "output": [{
            "model": "envoy", "choices": [{"message": {"content": "print(1)"}}]}]}
    audit = tmp_path / "audit.jsonl"
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), audit,
                                      requests_max=1, http=http)
    assert transport("unused", payload(), {}, 120)["model"] == "envoy"
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport("unused", payload(), {}, 120)
    assert len(calls) == 1
    assert calls[0][2]["policy"]["executionTimeout"] == 120000
    assert "secret" not in audit.read_text()
    restarted = RunpodQueueTransport("abc", "secret", Tokenizer(), audit, requests_max=1)
    assert restarted.remaining == 0
    with pytest.raises(ValueError, match="refusing reset"):
        RunpodQueueTransport("abc", "secret", Tokenizer(), audit)


def test_uncertain_submission_is_not_retried_or_refunded(tmp_path):
    calls = []
    def http(*args):
        calls.append(args)
        raise RuntimeError("lost connection")
    audit = tmp_path / "audit.jsonl"
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), audit,
                                      requests_max=1, http=http)
    with pytest.raises(RuntimeError, match="lost connection"):
        transport("unused", payload(), {}, 120)
    with pytest.raises(RuntimeError, match="budget exhausted"):
        transport("unused", payload(), {}, 120)
    assert len(calls) == 1
    assert json.loads(audit.read_text().splitlines()[-1])["reservation_retained"] is True
    restarted = RunpodQueueTransport("abc", "secret", Tokenizer(), audit,
                                      requests_max=1, http=http)
    assert restarted.remaining == 0


def _process_request(path):
    def http(*args):
        return {"id": "job-1", "status": "COMPLETED", "output": [{
            "model": "envoy", "choices": [{"message": {"content": "print(1)"}}]}]}
    from pathlib import Path
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), Path(path),
                                      requests_max=3, http=http)
    try:
        transport("unused", payload(), {}, 120)
    except RuntimeError as exc:
        assert "budget exhausted" in str(exc)
        return False
    return True


def test_multiple_processes_share_one_hard_budget(tmp_path):
    path = tmp_path / "audit.jsonl"
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        results = list(pool.map(_process_request, [str(path)] * 8))
    assert sum(results) == 3
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len([r for r in rows if r["event"] == "budget"]) == 1
    assert sorted(r["request"] for r in rows if r["event"] == "reserved") == [1, 2, 3]


def test_corrupt_session_ledger_cannot_restore_budget(tmp_path):
    path = tmp_path / "audit.jsonl"
    RunpodQueueTransport("abc", "secret", Tokenizer(), path)
    with path.open("a") as stream:
        stream.write('{"event": "reserved"')
    with pytest.raises(ValueError, match="corrupt"):
        RunpodQueueTransport("abc", "secret", Tokenizer(), path)


@pytest.mark.parametrize("changes", [{"model": "envoy-base-only"}, {"stream": True},
                                     {"max_tokens": 2048}, {"n": 2},
                                     {"best_of": 4}, {"max_completion_tokens": 9999},
                                     {"chat_template_kwargs": {"enable_thinking": True}}])
def test_request_cannot_bypass_worker_limits(tmp_path, changes):
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), tmp_path / "audit")
    with pytest.raises(ValueError):
        transport("unused", payload(**changes), {}, 120)
    assert transport.remaining == 12


def test_poll_existing_job_then_reject_base_model_output(tmp_path, monkeypatch):
    monkeypatch.setattr("src.policies.runpod_queue.time.sleep", lambda _: None)
    calls = []
    def http(method, path, body=None):
        calls.append((method, path))
        if path == "/run":
            return {"id": "job-1", "status": "IN_QUEUE"}
        if path == "/status/job-1":
            return {"status": "COMPLETED", "output": [{
                "model": "envoy-base-only", "choices": [{"message": {"content": "oops"}}]}]}
        return {}
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), tmp_path / "audit", http=http)
    with pytest.raises(ValueError, match="invalid adapter"):
        transport("unused", payload(), {}, 120)
    assert calls == [("POST", "/run"), ("GET", "/status/job-1"), ("POST", "/cancel/job-1")]
    assert transport.remaining == 11


def test_http_redirect_and_error_do_not_leak_credentials(tmp_path, monkeypatch):
    from urllib.error import HTTPError
    from src.policies import runpod_queue
    seen = []
    class Opener:
        def open(self, req, **kwargs):
            seen.append(req.full_url)
            raise HTTPError(req.full_url, 302, "secret", {}, None)
    monkeypatch.setattr(runpod_queue.request, "build_opener", lambda *args: Opener())
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), tmp_path / "audit")
    with pytest.raises(RuntimeError, match="^Runpod HTTP 302; no retry$"):
        transport("https://evil.example/", payload(), {"Authorization": "untrusted"}, 120)
    assert seen == ["https://api.runpod.ai/v2/abc/run"]
    assert runpod_queue._NoRedirect().redirect_request(None, None, 302, "", {}, "https://evil.example") is None


def test_bounded_http_response(tmp_path, monkeypatch):
    from src.policies import runpod_queue
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, size):
            assert size == 1_000_001
            return b"x" * size
    class Opener:
        def open(self, *args, **kwargs): return Response()
    monkeypatch.setattr(runpod_queue.request, "build_opener", lambda *args: Opener())
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), tmp_path / "audit")
    with pytest.raises(ValueError, match="response too large"):
        transport("unused", payload(), {}, 120)


def test_context_limit_fails_before_submission(tmp_path):
    transport = RunpodQueueTransport("abc", "secret", Tokenizer(), tmp_path / "audit")
    with pytest.raises(ValueError, match="input limit"):
        transport("unused", payload(messages=[{"content": "x" * 7169}]), {}, 120)
    assert transport.remaining == 12


def test_batch_encoding_cannot_be_miscounted_as_two_tokens(tmp_path):
    class WrongTokenizer:
        def apply_chat_template(self, *args, **kwargs):
            return {"input_ids": [1] * 9000, "attention_mask": [1] * 9000}
    transport = RunpodQueueTransport("abc", "secret", WrongTokenizer(), tmp_path / "audit")
    with pytest.raises(ValueError, match="flat token-id list"):
        transport("unused", payload(), {}, 120)
    assert transport.remaining == 12
