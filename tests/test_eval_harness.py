"""Attempt timing and completed work survive endpoint failures."""

from types import SimpleNamespace

import pytest

from src.env.document_env import StepRecord
from src.eval import harness
from src.policies.openai_compatible import OpenAICompatiblePolicy


class FakeEnvironment:
    def __init__(self, *, questions, **kwargs):
        self.questions = questions
        self.trajectory = []
        self.closed = False

    def reset(self, **kwargs):
        return "question"

    def step(self, action):
        self.trajectory.append(StepRecord(
            step=len(self.trajectory) + 1, action=action, observation="found source",
            reward=0, done=action == "finish",
        ))
        return "found source", 0.0, action == "finish", {}

    def get_trajectory(self):
        return self.trajectory

    def get_episode_info(self):
        return None

    def close(self):
        self.closed = True


def _setup(monkeypatch, ticks):
    questions = [{"id": "q", "question": "Question", "answer": "a"}]
    env = FakeEnvironment(questions=questions)
    monkeypatch.setattr(harness, "DocumentExplorationEnv", lambda **kwargs: env)
    clock = iter(ticks)
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(harness.time, "time", lambda: pytest.fail("must use monotonic timing"))
    return questions, env


def test_failed_attempt_retains_elapsed_time_and_completed_steps(monkeypatch):
    questions, env = _setup(monkeypatch, [100.0, 100.5, 105.0])
    actions = iter(["read('paper')"])

    def act(observation):
        action = next(actions, None)
        if action is None:
            raise RuntimeError("model endpoint returned HTTP 400: context length exceeded")
        return action

    result = harness._run_one_question(
        None, questions, 0, "base", lambda: SimpleNamespace(act=act), 15, False, "unused",
    )
    assert result.status == "error"
    assert result.steps == len(result.trajectory) == 1
    assert result.trajectory[0].action == "read('paper')"
    assert result.duration_seconds == 5.0
    assert "context length exceeded" in result.error
    assert env.closed


def test_success_and_failure_measure_same_attempt_boundary(monkeypatch):
    questions, env = _setup(monkeypatch, [100.0, 100.5, 104.0, 105.0])
    result = harness._run_one_question(
        None, questions, 0, "base", lambda: SimpleNamespace(act=lambda _: "finish"),
        15, False, "unused",
    )
    assert result.status == "completed"
    assert result.steps == 1
    assert result.duration_seconds == 5.0
    assert env.closed


def test_policy_initialization_failure_has_measured_duration(monkeypatch):
    questions, env = _setup(monkeypatch, [100.0, 102.0])

    def unavailable_policy():
        raise RuntimeError("load failed")

    result = harness._run_one_question(
        None, questions, 0, "base", unavailable_policy, 15, False, "unused",
    )
    assert result.status == "error"
    assert result.steps == 0
    assert result.duration_seconds == 2.0
    assert env.closed


def test_failed_harness_attempt_retains_provider_usage(monkeypatch):
    questions, _ = _setup(monkeypatch, [100.0, 100.5, 105.0])
    calls = 0

    def transport(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("model endpoint returned HTTP 400: context length exceeded")
        return {
            "choices": [{"message": {"content": "print('source')"}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }

    result = harness._run_one_question(
        None, questions, 0, "base", lambda: OpenAICompatiblePolicy(
            endpoint="http://localhost/v1", model="local", transport=transport,
        ), 15, False, "unused",
    )
    assert result.status == "error"
    assert result.steps == 1
    usage = result.policy_metadata["token_usage"]
    assert usage["request_count"] == 2
    assert usage["failed_request_count"] == 1
    assert usage["observed_prompt_tokens"] == 20
    assert usage["observed_completion_tokens"] == 5
    assert usage["prompt_tokens"] is usage["completion_tokens"] is None


def test_broken_telemetry_does_not_replace_original_harness_failure(monkeypatch):
    questions, _ = _setup(monkeypatch, [100.0, 100.5, 105.0])

    def act(observation):
        raise RuntimeError("original failure")

    def metadata():
        raise RuntimeError("telemetry failure detail")

    result = harness._run_one_question(
        None, questions, 0, "base",
        lambda: SimpleNamespace(act=act, eval_metadata=metadata), 15, False, "unused",
    )
    assert result.error == "RuntimeError: original failure"
    assert result.policy_metadata == {"metadata_collection_failed": True}


def test_generated_reasoning_is_saved_per_step_but_not_executed(monkeypatch):
    questions, env = _setup(monkeypatch, [100.0, 100.5, 104.0, 105.0])
    responses = iter([
        {"choices": [{
            "message": {
                "reasoning": "print('not executable')",
                "content": "finish",
            },
            "logprobs": {"content": [{"token": "finish", "logprob": -0.25}]},
        }]},
    ])
    result = harness._run_one_question(
        None, questions, 0, "qwen", lambda: OpenAICompatiblePolicy(
            endpoint="http://localhost/v1", model="qwen",
            transport=lambda *args: next(responses),
        ), 15, False, "unused",
    )
    assert result.status == "completed"
    assert result.trajectory[0].action == "finish"
    assert result.trajectory[0].reasoning == "print('not executable')"
    assert result.trajectory[0].logprob_diagnostics["prefix"] == [
        {"token": "finish", "logprob": -0.25}
    ]
    assert env.closed


def test_unfinished_thinking_never_reaches_environment(monkeypatch):
    questions, env = _setup(monkeypatch, [100.0, 100.5, 105.0])
    result = harness._run_one_question(
        None, questions, 0, "qwen", lambda: OpenAICompatiblePolicy(
            endpoint="http://localhost/v1", model="qwen",
            transport=lambda *args: {"choices": [{"message": {
                "content": "<think>print('do not run')",
            }}]},
        ), 15, False, "unused",
    )
    assert result.status == "error"
    assert result.steps == 0
    assert env.trajectory == []
    assert "unfinished thinking" in result.error
