import json

import pytest

from src.eval.known_paper_diagnostic import run_trial, training_prompt, TRAINING_PROMPT_SHA256
from tests.arxiv_fakes import article_page, make_library


class Policy:
    def __init__(self, actions):
        self.actions, self.observations = iter(actions), []
    def reset(self): pass
    def act(self, observation):
        self.observations.append(observation)
        return next(self.actions)


class Repl:
    def __init__(self, output="source text"):
        self.output, self.closed, self.actions = output, False, []
    def start_session(self): pass
    def execute(self, action, **kwargs):
        self.actions.append(action)
        return self.output
    def kill_session(self): self.closed = True


def setup(tmp_path):
    library, _ = make_library(tmp_path, {"2609.12345v2": {"html": article_page()}})
    library.open("2609.12345v2")
    return library.view(["2609.12345v2"]), library.document("2609.12345v2")["doc_id"]


def test_known_paper_can_be_inspected_without_product_discovery_gate(tmp_path):
    snapshot, doc = setup(tmp_path)
    repl = Repl()
    policy = Policy([f'print(search_within("{doc}", "batch size"))',
                     f'SUBMIT: 1 CITATIONS: ["{doc}"]'])
    result = run_trial(policy, snapshot, "known paper question", doc, image="pinned",
                       repl_factory=lambda _: repl)
    assert result["status"] == "submitted" and result["answer"] == "1"
    assert result["source_inspection_actions"] == 1 and result["evidence"] == []
    assert result["semantic_support"] == "not_reviewed" and result["reward"] == "not_computed"
    assert len(repl.actions) == 1 and repl.closed
    assert "batch size" in repl.actions[0]
    assert policy.observations[0] == "Question: known paper question\n"


def test_submission_without_inspection_is_not_falsely_counted(tmp_path):
    snapshot, doc = setup(tmp_path)
    repl = Repl()
    result = run_trial(Policy([f'SUBMIT: 1 CITATIONS: ["{doc}"]']), snapshot,
                       "question", doc, image="pinned", repl_factory=lambda _: repl)
    assert result["status"] == "submitted" and result["source_inspection_actions"] == 0
    assert result["semantic_support"] == "not_reviewed"


def test_execution_errors_are_recorded_and_do_not_reset_step_budget(tmp_path):
    snapshot, doc = setup(tmp_path)
    repl = Repl("STDERR: Traceback: TypeError: missing query")
    result = run_trial(Policy(['search_within("bad")'] * 2), snapshot,
                       "question", doc, image="pinned", max_steps=2, repl_factory=lambda _: repl)
    assert result["status"] == "incomplete" and result["model_requests_attempted"] == 2
    assert all(t["execution_error"] for t in result["trajectory"])
    assert result["source_inspection_actions"] == 0 and repl.closed


def test_docker_failure_never_uses_local_python(tmp_path, monkeypatch):
    snapshot, doc = setup(tmp_path)
    monkeypatch.setattr("src.eval.known_paper_diagnostic.PersistentREPL._docker_available",
                        lambda _: False)
    result = run_trial(Policy([]), snapshot, "question", doc, image="missing")
    assert result["status"] == "error" and result["model_requests_attempted"] == 0
    assert result["execution"] == "docker_only"


def test_training_prompt_is_pinned():
    from pathlib import Path
    import hashlib
    prompt = training_prompt(Path('data/sft/qasper-v5/train.jsonl'))
    assert hashlib.sha256(prompt.encode()).hexdigest() == TRAINING_PROMPT_SHA256
