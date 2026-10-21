"""One-attempt smoke wiring: real queue adapter, MCP and Docker with fake provider I/O."""
import json
import os

import pytest

from src.eval.experiment_smoke import checked_plan, plan, run
from src.policies.runpod_queue import RunpodQueueTransport
from tests.arxiv_fakes import article_page, make_library
from tests.experiment_fakes import FINISH
from tests.test_arxiv_research_mcp import FakeDiscovery
from tests.test_experiment_budget import Tokenizer


def test_frozen_smoke_rejects_changed_question(tmp_path):
    saved = plan()
    saved["question"] = "different"
    (tmp_path / "plan.json").write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="frozen plan"):
        checked_plan(tmp_path)


@pytest.mark.skipif(not os.environ.get("ENVOY_ASSIGNED_TEST_DOCKER_IMAGE"),
                    reason="set ENVOY_ASSIGNED_TEST_DOCKER_IMAGE")
def test_smoke_queue_mcp_docker_progress_and_exclusive_attempt(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    library.open("2609.12345")
    quote = "Eviction saves 40 percent of memory"
    start = library.document("2609.12345")["text"].index(quote)
    actions = iter(['search_papers("memory")', 'read_paper("2609.12345", "Memory?")',
        'print(search_within("arxiv_2609_12345v2", "memory"))',
        f'print(passage("arxiv_2609_12345v2", {start}, {len(quote)}))',
        'submit(answer="Memory savings reported", citations=["arxiv_2609_12345v2"], '
        f'evidence=[{{"doc_id":"arxiv_2609_12345v2","start":{start},'
        f'"end":{start+len(quote)}}}])', FINISH])
    calls = []
    def factory(endpoint, key, tokenizer, audit, **kwargs):
        def http(method, path, payload=None):
            calls.append((method, path))
            return {"id": f"job-{len(calls)}", "status": "COMPLETED", "output": {
                "model": "envoy", "choices": [{"message": {"content": next(actions)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10}}}
        return RunpodQueueTransport(endpoint, key, tokenizer, audit, http=http, **kwargs)
    (tmp_path / "plan.json").write_text(json.dumps(plan()))
    result = run(tmp_path, tokenizer=Tokenizer(), key="fixture", transport_factory=factory,
                 discovery=FakeDiscovery())
    assert result["status"] == "brief_ready" and len(calls) == 6
    assert result["budget"]["requests_max"] == 12
    assert json.loads((tmp_path / "progress.json").read_text())["evidence"]
    assert json.loads((tmp_path / "mcp.json").read_text())["result"] == result
    ledger = [json.loads(line) for line in
              (tmp_path / "research.requests.jsonl").read_text().splitlines()]
    assert len([row for row in ledger if row["event"] == "reserved"]) == 6
    with pytest.raises(FileExistsError):
        run(tmp_path, tokenizer=Tokenizer(), key="fixture", transport_factory=factory)
    assert len(calls) == 6


def test_uncertain_transport_retains_result_but_fails_worker(tmp_path):
    calls = []
    def factory(endpoint, key, tokenizer, audit, **kwargs):
        def http(*args):
            calls.append(args)
            raise TimeoutError("uncertain submission")
        return RunpodQueueTransport(endpoint, key, tokenizer, audit, http=http, **kwargs)
    (tmp_path / "plan.json").write_text(json.dumps(plan()))
    with pytest.raises(RuntimeError, match="partial result retained"):
        run(tmp_path, tokenizer=Tokenizer(), key="fixture", transport_factory=factory,
            discovery=FakeDiscovery())
    result = json.loads((tmp_path / "mcp.json").read_text())["result"]
    assert result["status"] == "inference_failed" and len(calls) == 1
    assert (tmp_path / "progress.json").exists()
