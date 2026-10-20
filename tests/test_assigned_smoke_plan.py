"""Preparation cannot run inference; references are frozen outside model inputs."""
import hashlib
import json
from pathlib import Path

import pytest

from src.eval import assigned_smoke_plan as plan_module
from src.eval.assigned_smoke_check import check
from src.eval.assigned_smoke_plan import CASES, LIMITS, prepare, source_digest
from src.policies.runpod_queue import RunpodQueueTransport
from src.product.assigned_reader import build_assigned_reader
from src.research.agent import load_snapshot
from tests.arxiv_fakes import article_page, make_library
from tests.test_assigned_reader import SourceREPL

IMAGE = "sha256:" + "a" * 64


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": True, "add_generation_prompt": True,
                          "enable_thinking": False, "return_dict": False}
        return list(range(100))


@pytest.fixture
def cached(tmp_path, monkeypatch):
    html = article_page().replace(b"</article>", (
        "<p>We measure walltime improvements with a batch size of 1 on a single TPU-v4 "
        "for both sampling settings. The coefficient α depends on hardware.</p></article>"
    ).encode())
    library, network = make_library(tmp_path, {"2211.17192v2": {"html": html}})
    library.open("2211.17192v2")
    monkeypatch.setattr(plan_module, "EXPECTED_SOURCE_SHA256",
                        source_digest(library.document("2211.17192v2")))
    return library, network, tmp_path / "prepared"


def make(cached):
    library, _, output = cached
    return prepare(library.root, output, sandbox_image=IMAGE, tokenizer=Tokenizer())


def test_prepare_is_cached_only_and_freezes_private_references(cached, monkeypatch):
    library, network, output = cached
    fetches = len(network.calls)
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not invoke models or Docker")
    monkeypatch.setattr("src.product.qwen_investigator.QwenInvestigator.investigate", forbidden)
    monkeypatch.setattr("src.env.repl.PersistentREPL.start_session", forbidden)
    monkeypatch.setattr("urllib.request.urlopen", forbidden)
    plan = make(cached)
    assert len(network.calls) == fetches
    assert plan["model_calls"] == 0 and plan["activation"]["approved"] is False
    inputs = (output / "model_inputs.json").read_text()
    assert "TPU-v4" not in inputs and "batch size of 1" not in inputs
    assert "reference_answer" not in inputs and "source_context" not in inputs
    refs = json.loads((output / "references.json").read_text())
    assert refs["sent_to_model"] is False and refs["human_review"] == "not_performed"
    assert "zero/negative speedup" in refs["cases"][1]["reference_constraint"]
    assert check(output, tokenizer=Tokenizer())["initial_prompt_tokens"] == [100, 100]
    template = json.loads((output / "claim-review-template.json").read_text())
    assert all(row["verdict"] is None and row["reviewer_kind"] is None
               for row in template["cases"])


def test_prepared_payload_is_actual_assigned_policy_initial_payload(cached):
    plan = make(cached)
    _, _, output = cached
    inputs = json.loads((output / "model_inputs.json").read_text())
    sent = []
    def transport(url, payload, headers, timeout):
        sent.append(payload)
        return {"choices": [{"message": {"content": 'submit(answer="Unanswerable")'}}]}
    reader = build_assigned_reader("qwen", qwen_endpoint="http://scripted", qwen_model="envoy",
        qwen_checkpoint="test", policy_transport=transport, max_steps=2)
    _, docs = load_snapshot(Path(plan["snapshot"]))
    reader.repl_factory = lambda _: SourceREPL(next(iter(docs.values()))["text"])
    reader.investigate(CASES[0]["question"], plan["snapshot"])
    assert inputs[0]["payload"] == sent[0]


def test_preparation_is_never_overwritten(cached):
    make(cached)
    output = cached[2]
    before = (output / "plan.json").read_bytes()
    with pytest.raises(FileExistsError):
        make(cached)
    assert before == (output / "plan.json").read_bytes()


@pytest.mark.parametrize("filename", ["plan.json", "model_inputs.json", "references.json",
                                     "claim-review-template.json", "system-prompt.txt"])
def test_changed_frozen_artifact_is_rejected(cached, filename):
    make(cached)
    (cached[2] / filename).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        check(cached[2], tokenizer=Tokenizer())


@pytest.mark.parametrize("field", ["limits", "model"])
def test_rehashed_plan_cannot_silently_change_runtime_contract(cached, field):
    make(cached)
    output = cached[2]
    plan = json.loads((output / "plan.json").read_text())
    plan[field] = {}
    raw = json.dumps(plan).encode()
    (output / "plan.json").write_bytes(raw)
    (output / "plan.sha256").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="contract|pin"):
        check(output, tokenizer=Tokenizer())


def test_changed_implementation_hash_is_rejected(cached):
    make(cached)
    output = cached[2]
    plan = json.loads((output / "plan.json").read_text())
    plan["implementation_sha256"]["src/env/repl.py"] = "changed"
    raw = json.dumps(plan).encode()
    (output / "plan.json").write_bytes(raw)
    (output / "plan.sha256").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="implementation changed"):
        check(output, tokenizer=Tokenizer())


def test_changed_cached_source_is_rejected_before_creating_session(cached, monkeypatch):
    monkeypatch.setattr(plan_module, "EXPECTED_SOURCE_SHA256", "bad pin")
    with pytest.raises(ValueError, match="pinned source"):
        make(cached)
    assert not cached[2].exists()


def test_oversized_initial_prompt_fails_before_preparation(cached):
    class Huge(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return [1] * 7169
    with pytest.raises(ValueError, match="input limit"):
        prepare(cached[0].root, cached[2], sandbox_image=IMAGE, tokenizer=Huge())
    assert not cached[2].exists()


def test_six_request_case_ledgers_bound_both_cases_to_twelve(tmp_path):
    posts = []
    def fake_http(method, path, payload=None):
        posts.append((method, path))
        return {"id": f"job-{len(posts)}", "status": "COMPLETED", "output": {
            "model": "envoy", "choices": [{"message": {"content": "scripted"}}]}}
    payload = {"model": "envoy", "max_tokens": 1024, "messages": [],
               "chat_template_kwargs": {"enable_thinking": False}}
    for case in CASES:
        path = tmp_path / f'{case["case_id"]}.jsonl'
        transport = RunpodQueueTransport("test", "scripted-key", Tokenizer(), path,
            requests_max=LIMITS["model_requests_per_case"], http=fake_http)
        for _ in range(6):
            transport("ignored", payload, {}, 1)
        with pytest.raises(RuntimeError, match="budget exhausted"):
            transport("ignored", payload, {}, 1)
        resumed = RunpodQueueTransport("test", "scripted-key", Tokenizer(), path,
            requests_max=6, http=fake_http)
        assert resumed.remaining == 0
    assert len(posts) == LIMITS["model_requests_total"] == 12
    assert 12 * 7168 == LIMITS["input_tokens_reserved_total"]
    assert 12 * 1024 == LIMITS["output_tokens_reserved_total"]
