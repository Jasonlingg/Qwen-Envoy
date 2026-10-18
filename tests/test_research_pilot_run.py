import json
from pathlib import Path

import pytest

from scripts.run_research_workflow_pilot import main
from src.eval.research_pilot_budget import RequestBudget
from src.eval.research_pilot_bundle import prepare
from src.eval.research_pilot_host import SYSTEM_PROMPT
from src.eval.research_pilot_review import build_review, score_review
from src.eval.research_pilot_run import run_pilot
from src.policies.openai_compatible import OpenAICompatiblePolicy
from tests.arxiv_fakes import FakeInvestigator
from tests.test_research_pilot_bundle_tools import sample

ROOT = Path(__file__).resolve().parents[1]
QUOTE = "Eviction saves 40 percent of memory"


@pytest.mark.parametrize("final_usage_missing", [False, True])
def test_injected_clients_complete_both_arms_and_never_become_live_evidence(
        tmp_path, final_usage_missing):
    library, question, path = sample(tmp_path)
    bundle = tmp_path / "bundle"
    prepare(path, library.root, bundle, ROOT)
    version = question["arxiv_ids"][0]
    start = library.document(version)["text"].index(QUOTE)
    evidence = {"arxiv_id": version, "start": start, "end": start + len(QUOTE)}
    finish = {"tool": "finish", "arguments": {"answer": "The authors report memory savings.",
        "claims": [{"text": "The authors report memory savings.", "evidence": [evidence]}],
        "limitations": ["Scripted test, not a model result."]}}
    budget = RequestBudget()
    class Reader(FakeInvestigator):
        def investigate(self, question, snapshot):
            budget.transport("qwen", lambda messages: 100, 8192, lambda *args: {
                "usage": {"prompt_tokens": 100, "completion_tokens": 1}})("unused", {
                    "model": "fake", "messages": [{"role": "user", "content": question}],
                    "max_tokens": 1024}, {}, 1)
            return super().investigate(question, snapshot)
    def factory(question_id, arm):
        first = ({"tool": "get_passage", "arguments": {"arxiv_id": version,
                 "start": start, "length": len(QUOTE)}} if arm == "direct" else
                 {"tool": "investigate", "arguments": {"question": question["question"],
                                                         "arxiv_ids": [version]}})
        actions = iter([first, finish])
        def upstream(*args):
            action = next(actions)
            usage = (None if final_usage_missing and arm == "delegated"
                     and action["tool"] == "finish" else
                     {"prompt_tokens": 100, "completion_tokens": 1})
            return {"choices": [{"message": {"content": json.dumps(action)}}], "usage": usage}
        return OpenAICompatiblePolicy(endpoint="http://127.0.0.1:1/v1", model="fake",
            system_prompt=SYSTEM_PROMPT, action_cleaner=lambda raw: raw,
            transport=budget.transport(f"host:{question_id}:{arm}", lambda m: 100, 8192,
                                       upstream))
    result = run_pilot(bundle, tmp_path / "run", host_factory=factory,
        investigator=Reader({version: QUOTE}), budget=budget, config={}, repo=ROOT,
        execution_kind="fake_test")
    assert result["status"] == ("incomplete" if final_usage_missing else "complete")
    assert result["actual_model_requests"] == 5 and result["actual_qwen_requests"] == 1
    assert all(row["status"] == "submitted" for row in result["results"])
    assert result["results"][0]["provenance_checks"]["claims"][0]["evidence"][0]["quote"] == QUOTE
    assert (tmp_path / "run/reader-packets/001.json").is_file()
    packet, key = build_review(result)
    assert all(row["verdict"] is None for row in packet["rows"])
    assert score_review(result, packet, key)["decision"] == "inconclusive"
    with pytest.raises(ValueError, match="already exists"):
        run_pilot(bundle, tmp_path / "run", host_factory=factory,
                  investigator=Reader({}), budget=budget, config={}, repo=ROOT)


def test_run_requires_explicit_execute_before_reading_config_or_dispatch(tmp_path):
    with pytest.raises(ValueError, match="explicit --execute"):
        main(["run", "--bundle", str(tmp_path / "missing"), "--config", str(tmp_path / "missing"),
              "--output", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_initial_question_failure_stops_before_spending_on_later_arms(tmp_path):
    library, _, path = sample(tmp_path)
    prepare(path, library.root, tmp_path / "bundle", ROOT)
    class InvalidHost:
        system_prompt = SYSTEM_PROMPT
        history = []
        def act(self, observation):
            return "invalid JSON"
    result = run_pilot(tmp_path / "bundle", tmp_path / "run",
        host_factory=lambda *args: InvalidHost(), investigator=FakeInvestigator({}),
        budget=RequestBudget(), config={}, repo=ROOT, execution_kind="fake_test")
    assert result["status"] == "incomplete" and len(result["results"]) == 1
    assert result["actual_qwen_requests"] == 0
    assert result["stop_reason"].startswith("Initial-question smoke failed")
