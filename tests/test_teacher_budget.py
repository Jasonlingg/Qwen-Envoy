from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from src.policies.teacher_budget import BudgetedMessages, BudgetStop, TeacherBudget


def response(inputs=100, outputs=10):
    return SimpleNamespace(model="claude-sonnet-5", usage=SimpleNamespace(
        input_tokens=inputs, output_tokens=outputs))


def test_concurrent_reservations_cannot_spend_same_money(tmp_path):
    budget = TeacherBudget(tmp_path / "budget.json", .01)
    def reserve(_):
        try:
            return budget.reserve(100, 100, "question")
        except BudgetStop:
            return None
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            tokens = [t for t in pool.map(reserve, range(20)) if t]
        assert len(tokens) == 3
        assert budget.summary()["accounted_usd"] <= .01
    finally:
        budget.close()


def test_settled_spend_survives_resume_and_cap_cannot_change(tmp_path):
    path = tmp_path / "budget.json"
    budget = TeacherBudget(path, .01)
    token = budget.reserve(100, 100, "q1")
    budget.settle(token, response())
    budget.close()
    with pytest.raises(ValueError, match="change the cap"):
        TeacherBudget(path, 10)
    budget = TeacherBudget(path, .01)
    try:
        assert budget.summary()["accounted_usd"] == .0003
        with pytest.raises(BudgetStop, match="Another process"):
            TeacherBudget(path, .01)
    finally:
        budget.close()


def test_unknown_usage_is_not_refunded_on_resume(tmp_path):
    path = tmp_path / "budget.json"
    budget = TeacherBudget(path, .1)
    token = budget.reserve(100, 100, "q1")
    with pytest.raises(BudgetStop, match="Missing usage"):
        budget.settle(token, SimpleNamespace())
    assert budget.summary()["unsettled_requests"] == 1
    budget.close()
    with pytest.raises(BudgetStop, match="Unsettled"):
        TeacherBudget(path, .1)


def test_budget_checks_before_paid_request_and_accounts_failed_episodes(tmp_path):
    calls = []
    class Messages:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(input_tokens=100)
        def create(self, **kwargs):
            calls.append(kwargs)
            return response()
    budget = TeacherBudget(tmp_path / "budget.json", .001)
    wrapper = BudgetedMessages(Messages(), budget, "q1")
    try:
        with pytest.raises(BudgetStop, match="cannot cover"):
            wrapper.create(model="claude-sonnet-5", system="system", messages=[], max_tokens=100)
        assert not calls
    finally:
        budget.close()


def test_network_failure_retains_full_reservation(tmp_path):
    class Messages:
        def count_tokens(self, **kwargs):
            return SimpleNamespace(input_tokens=100)
        def create(self, **kwargs):
            raise ConnectionError("Disconnected after sending")
    budget = TeacherBudget(tmp_path / "budget.json", .1)
    try:
        with pytest.raises(BudgetStop, match="unknown charge"):
            BudgetedMessages(Messages(), budget, "q1").create(
                model="claude-sonnet-5", system="system", messages=[], max_tokens=100)
        assert budget.summary()["accounted_usd"] == .003288
        assert budget.summary()["unsettled_requests"] == 1
    finally:
        budget.close()


def test_teacher_export_uses_exact_inference_prompt_and_observations():
    from scripts.export_qasper_sft_data import _to_conversation
    from src.policies.code_execution import QASPER_SYSTEM_PROMPT
    from scripts.generate_qasper_teacher_batch import SYSTEM_PROMPT
    row = {"student_protocol": "qasper-span-v1", "question": "question",
           "initial_observation": "Question: question\n", "trajectory": [
               {"action": "print(1)", "raw_action": "print(1)\n", "observation": "1\n"},
               {"action": "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []"}]}
    messages = _to_conversation(row, 32000)["messages"]
    assert SYSTEM_PROMPT == QASPER_SYSTEM_PROMPT == messages[0]["content"]
    assert messages[1]["content"] == "Question: question\n"
    assert messages[2]["content"] == "print(1)\n"
    assert messages[3]["content"] == "1\n"


def test_full_teacher_episode_executes_and_exports_without_privileged_hint(tmp_path, monkeypatch):
    import anthropic
    import json
    from scripts.generate_qasper_teacher_batch import _run_one, _stop
    from scripts.export_qasper_sft_data import _to_conversation
    from src.env.corpus import Corpus
    corpus_path = tmp_path / "corpus"
    corpus_path.mkdir()
    (corpus_path / "paper.json").write_text(json.dumps(
        {"doc_id": "paper", "title": "Paper", "text": "The classifier is Random Forest."}))
    corpus = Corpus(str(corpus_path))
    corpus.load(build_index=False)
    actions = ['print(passage("paper", start=0, length=32))',
               'SUBMIT: Random Forest CITATIONS: ["paper"] EVIDENCE: [{"doc_id":"paper","start":0,"end":32}]']
    class Messages:
        def create(self, **kwargs):
            result = response()
            result.content = [SimpleNamespace(text=actions.pop(0))]
            return result
    monkeypatch.setattr(anthropic, "Anthropic", lambda **kwargs:
                        SimpleNamespace(messages=Messages(), close=lambda: None))
    q = {"id": "q1", "question": 'Use the known paper (doc_id: "paper"): Which classifier?',
         "answer": "Random Forest", "expected_citations": ["paper"],
         "expected_answerability": "sufficient", "grader_notes": ["PRIVATE_SENTINEL"],
         "answer_annotations": [{"answer_type": "extractive", "answer_text": "Random Forest",
                                 "unanswerable": False, "evidence": [
                                     {"doc_id": "paper", "start": 0, "end": 32}]}]}
    _stop.clear()
    row = _run_one(q, corpus, str(corpus_path), 10, "claude-sonnet-5", True)
    assert row["status"] == "completed"
    assert row["qasper_score"]["reward"] == 1.0
    assert len(row["teacher_usage"]) == 2
    exported = _to_conversation(row, 32000)
    assert "PRIVATE_SENTINEL" not in json.dumps(exported)
    assert "TEACHER-ONLY" not in json.dumps(exported)
    assert "Random Forest" in exported["messages"][3]["content"]
