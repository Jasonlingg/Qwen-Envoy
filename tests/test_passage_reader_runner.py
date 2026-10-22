"""Base routing must stay explicit through the real MCP and note records."""
import asyncio
import json
import shutil

import pytest

from scripts import try_passage_reader as runner
from tests.test_passage_reader import ScopedREPL, fixture


class Tokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return [1] * 100

    def encode(self, value):
        return [1] * len(value)


def test_base_alias_survives_mcp_policy_queue_and_saved_note(tmp_path, monkeypatch):
    library, _, _, _, _, windows, _, _, _ = fixture(tmp_path / "source")
    paper = "2609.12345v2"
    text = library.document(paper)["text"]
    output = tmp_path / "attempt"
    output.mkdir()
    shutil.copytree(library.root, output / "library")
    case = {"case_id": "awq-protocol", "question": "What saves memory?",
            "arxiv_id": paper, "windows": windows}
    runner.save(output / "cases.json", [case])
    runner.save(output / "review-input.json", {"cases": [{
        "case_id": case["case_id"], "requirements": ["Eviction saves 40 percent of memory."]}]})
    model = runner.identity({"models": runner.MODEL_IDENTITY}, "base")
    monkeypatch.setattr(runner, "check", lambda *_: {"model_identity": model})
    original = runner.build_assigned_reader

    def build(*args, **kwargs):
        assert kwargs["qwen_model"] == "envoy-base-only"
        assert kwargs["qwen_checkpoint"] == model["checkpoint"]
        result = original(*args, **kwargs)
        result.repl_factory = lambda _: ScopedREPL(text)
        return result

    monkeypatch.setattr(runner, "build_assigned_reader", build)
    monkeypatch.setattr(runner, "track_sandbox", lambda *_: None)
    tokenizer = Tokenizer()
    asyncio.run(runner.run_cases(output, tokenizer, scripted=True))
    scripted = json.loads((output / "scripted/awq-protocol.json").read_text())
    actions = iter(step["action"] for step in scripted["packet"]["trajectory"])
    dispatched = []

    class Queue:
        def __init__(self, endpoint, key, tokenizer, audit, **kwargs):
            assert kwargs["served_model"] == "envoy-base-only"
            self.remaining = 12
            self.audit = audit.open("a")

        def __call__(self, url, payload, headers, timeout):
            assert payload["model"] == "envoy-base-only"
            dispatched.append(payload)
            self.remaining -= 1
            return {"model": payload["model"], "choices": [{"message": {
                "content": next(actions)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20}}

    monkeypatch.setattr(runner, "RunpodQueueTransport", Queue)
    result = asyncio.run(runner.run_cases(output, tokenizer, "test-key"))
    report = json.loads((output / "live/awq-protocol.json").read_text())
    assert len(dispatched) == 2  # Includes the identical-note retry, with no new call.
    assert result["cases"][0]["same_note_retry_without_dispatch"]
    assert report["status"] == "evidence_found" and report["note_attached"]
    assert report["packet"]["model_identity"]["adapter_applied"] is False
    assert report["packet"]["model_identity"]["served_model"] == "envoy-base-only"
    assert report["verified"]["model_identity"]["checkpoint"] == model["checkpoint"]
    assert report["semantic_support"] == "not_reviewed"


def valid_plan(output):
    runner.save(output / "cases.json", [])
    runner.save(output / "review-input.json", {})
    plan = {"model_arm": "base", "model_identity": runner.identity(
        {"models": runner.MODEL_IDENTITY}, "base"), "limits": runner.LIMITS,
        "endpoint": runner.ENDPOINT, "sandbox_image": runner.IMAGE,
        "implementation_sha256": runner.hashes(), "source_sha256": {},
        "cases_sha256": runner.sha(output / "cases.json"),
        "review_input_sha256": runner.sha(output / "review-input.json")}
    runner.save(output / "plan.json", plan)
    return plan


@pytest.mark.parametrize("change", ["requested_arm", "model_identity", "implementation"])
def test_model_arm_or_pinned_implementation_mismatch_is_rejected(tmp_path, change):
    plan = valid_plan(tmp_path)
    assert runner.check(tmp_path, model_arm="base") == plan
    if change == "requested_arm":
        requested = "v5"
    else:
        requested = "base"
        if change == "model_identity":
            plan["model_identity"]["served_model"] = "envoy"
        else:
            plan["implementation_sha256"]["scripts/try_passage_reader.py"] = "changed"
        runner.save(tmp_path / "plan.json", plan)
    with pytest.raises(ValueError, match="prepared implementation"):
        runner.check(tmp_path, model_arm=requested)
