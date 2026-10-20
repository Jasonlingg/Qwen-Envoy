"""Candidate preparation pins guidance/MCP scope without activation or inference."""
# ruff: noqa: F811
import hashlib
import json

import pytest

from src.eval.assigned_smoke_candidate import candidate_contract
from src.eval.assigned_smoke_check import check
from src.eval.assigned_smoke_plan import LIMITS, prepare
from src.product.assigned_guidance import FOCUSED
from src.product.qwen_investigator import effective_system_prompt
from tests.test_assigned_smoke_plan import IMAGE, Tokenizer, cached  # noqa: F401


def make(cached):
    return prepare(cached[0].root, cached[2], sandbox_image=IMAGE, tokenizer=Tokenizer(),
                   reader_guidance=FOCUSED)


def test_candidate_preparation_freezes_prompt_limits_and_actual_mcp_path(cached, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not activate, start Docker or call a model")
    monkeypatch.setattr("src.env.repl.PersistentREPL.start_session", forbidden)
    monkeypatch.setattr("src.eval.assigned_smoke_control.cli", forbidden)
    monkeypatch.setattr("src.product.qwen_investigator.QwenInvestigator.investigate", forbidden)
    before = len(cached[1].calls)
    plan = make(cached)
    assert len(cached[1].calls) == before
    assert plan["limits"] == LIMITS and plan["model_calls"] == 0
    assert plan["activation"]["approved"] is False
    assert all(plan[k] == v for k, v in candidate_contract().items())
    prompt = (cached[2]/"system-prompt.txt").read_text()
    assert prompt == effective_system_prompt("public_papers", "default", "v2",
        reader_mode="assigned", reader_guidance=FOCUSED)
    inputs = (cached[2]/"model_inputs.json").read_text()
    assert "reference_answer" not in inputs and "source_context" not in inputs
    assert "batch size of 1" not in inputs and "TPU-v4" not in inputs
    assert "src/product/assigned_guidance.py" in plan["implementation_sha256"]
    assert "src/eval/assigned_smoke_mcp.py" in plan["implementation_sha256"]
    assert "scripts/run_assigned_reader_smoke.py" in plan["implementation_sha256"]
    assert check(cached[2], tokenizer=Tokenizer())["execution_path"] == "mcp_in_process"


@pytest.mark.parametrize("field", ["reader_guidance", "execution_path", "e2e_scope",
                                   "decision_rule"])
def test_candidate_scope_cannot_be_rehashed_to_bypass_its_contract(cached, field):
    plan = make(cached)
    plan[field] = "changed"
    raw = json.dumps(plan).encode()
    (cached[2]/"plan.json").write_bytes(raw)
    (cached[2]/"plan.sha256").write_text(hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError, match="reader_guidance|contract"):
        check(cached[2], tokenizer=Tokenizer())
