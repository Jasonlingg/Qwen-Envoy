"""Replay must reproduce V1 exactly and never execute saved Python on the host."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import replay_harness_v2 as replay
from src.eval import harness_replay


@pytest.mark.parametrize(("text", "expected"), [
    ("ok [1, 2]", None),
    ("Traceback (most recent call last):\nTypeError: unexpected max_results", "TypeError"),
    ("  File x\nSyntaxError: invalid syntax", "SyntaxError"),
    ("TypeError: search(): bad. Usage: search(query", "usage_message"),
    ("{'error': 'Unknown document'}", "tool_error"),
])
def test_classify_observation(text, expected):
    assert replay.classify_observation(text) == expected


def test_v2_does_not_start_without_exact_v1(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("V2 started despite a mismatched V1 baseline")
    monkeypatch.setattr(replay, "DockerREPL", forbidden)
    with pytest.raises(ValueError, match="exact V1 match"):
        replay.compare_v2([{"v1_exact_match": False}], Path("corpus"), "image")
    with pytest.raises(ValueError, match="exact V1 match"):
        replay.compare_v2([], Path("corpus"), "image")


def test_changed_historical_source_is_rejected_before_import(tmp_path, monkeypatch):
    monkeypatch.setattr(harness_replay.subprocess, "check_output", lambda *a, **k: b"bad source")
    with pytest.raises(ValueError, match="Historical source hash mismatch"):
        harness_replay._historical_module(tmp_path, "env.py", "0" * 64)


def test_unlabelled_docker_image_is_rejected(monkeypatch):
    monkeypatch.setattr(harness_replay.subprocess, "check_output",
                        lambda *a, **k: b'[{"Id":"sha256:123","Config":{}}]')
    with pytest.raises(ValueError, match="sandbox=v1 label"):
        harness_replay.inspect_image("image")


def test_compact_report_preserves_every_action_and_binds_observations():
    report = {"questions": [{"question_id": "q1", "initial_observation": "question",
                             "actions": [{"step": 1, "action": "search('x')",
                                          "v1_observation": "saved output",
                                          "v1_exact_match": True}]}]}
    compact = harness_replay.compact_report(report)
    item = compact["questions"][0]["actions"][0]
    assert item["action"] == "search('x')" and item["v1_exact_match"] is True
    assert item["v1_observation_sha256"] == harness_replay.sha256("saved output")
    assert item["v1_observation_chars"] == len("saved output")
    assert report["questions"][0]["actions"][0]["v1_observation"] == "saved output"


def test_v2_reports_per_action_without_running_submission_or_prose(tmp_path, monkeypatch):
    executed = []

    class FakeDocker:
        def __init__(self, **kwargs):
            assert kwargs["image"] == "recorded-image"
            assert kwargs["extra_preamble"]

        def start_session(self):
            pass

        def execute(self, code, timeout=30):
            executed.append(code)
            return "TypeError: bad. Usage: search(query)"

        def kill_session(self):
            executed.append("CLOSED")

    monkeypatch.setattr(replay, "DockerREPL", FakeDocker)
    cases = [
        ("SUBMIT: answer CITATIONS: []", False, "recorded_submission"),
        ('print(search("duplicate"))', False, "recorded_verifier_skip"),
        ('submit(answer="Unanswerable", citations=[], evidence=[])', True, "submit_call_converted"),
        ("submit(answer=variable)", True, "submit_call_rejected"),
        ("This is explanatory prose.", True, "format_retry"),
        ('print(search("new"))', True, "executed"),
    ]
    question = {"question_id": "q1", "v1_exact_match": True, "actions": [
        {"action": action, "v1_executed": run, "recorded_observation": "saved",
         "v1_observation": "saved"} for action, run, _ in cases]}
    result = replay.compare_v2([question], tmp_path, "recorded-image")
    assert [item["v2_outcome"] for item in question["actions"]] == [case[2] for case in cases]
    assert executed == ['print(search("new"))', "CLOSED"]
    assert result["counts"]["v2"]["usage_message"] == 1
    assert question["counts"]["v2"]["format_retry"] == 1


def test_baseline_compares_every_field_and_passes_python_only_to_docker(tmp_path, monkeypatch):
    host_marker = tmp_path / "must-not-exist"
    action = f"__import__('pathlib').Path({str(host_marker)!r}).write_text('unsafe')"
    seen = []

    class FakeDocker:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def execute(self, code, timeout=30):
            seen.append(code)
            return "actual observation"

    monkeypatch.setattr(harness_replay, "DockerREPL", FakeDocker)
    loaded = []
    fake_corpus = SimpleNamespace(load=lambda **kwargs: loaded.append(kwargs))
    monkeypatch.setattr(harness_replay, "Corpus", lambda **kwargs: fake_corpus)
    module = SimpleNamespace()

    class FakeEnvironment:
        def __init__(self, **kwargs):
            assert kwargs["use_docker"] is True
            assert kwargs["corpus"] is fake_corpus and loaded == [{"build_index": False}]
            self.kwargs = kwargs

        def reset(self):
            self.repl = module.PersistentREPL(use_docker=True, corpus_path="frozen")
            return "original initial observation"

        def step(self, code):
            observation = self.repl.execute(code)
            self.record = SimpleNamespace(model_dump=lambda: {
                "step": 1, "action": code, "observation": observation,
                "reward": 0.0, "done": False,
            })

        def get_trajectory(self):
            return [self.record]

        def close(self):
            pass

    module.DocumentExplorationEnv = FakeEnvironment
    row = {"question_id": "q1", "question": "Question?", "trajectory": [{
        "step": 1, "action": action, "observation": "recorded observation",
        "reward": 0.0, "done": True,
    }]}
    manifest = {"max_steps": 15, "require_evidence": True, "evidence_verifier": True,
                "verifier_feedback_budget": 1, "escalate_after_verifier_failure": False}
    result = harness_replay.replay_baseline(module, row, {}, tmp_path, manifest, "sha256:123")
    assert seen == [{"image": "sha256:123", "corpus_path": "frozen"}, action]
    assert not host_marker.exists()
    assert not result["v1_exact_match"]
    assert result["actions"][0]["v1_executed"] is True
    assert result["actions"][0]["v1_done"] is False
    assert result["actions"][0]["recorded_done"] is True
    assert result["actions"][0]["v1_tool_observation"] == "actual observation"
    assert result["initial_observation"] == "original initial observation"
    with pytest.raises(ValueError, match="host execution is forbidden"):
        module.PersistentREPL(use_docker=False, corpus_path="frozen")
