"""Tests for the Gym-compatible document exploration environment."""

import subprocess
from types import SimpleNamespace

import pytest

from src.env.corpus import Corpus
from src.env.document_env import DocumentExplorationEnv, parse_submission
from src.eval.harness import run_single


@pytest.fixture(scope="module")
def corpus() -> Corpus:
    subprocess.run(["python", "scripts/setup_corpus.py"], check=True, capture_output=True)
    c = Corpus(corpus_path="data/corpus")
    c.load(build_index=False)
    return c


@pytest.fixture(scope="module")
def questions() -> list[dict]:
    return [{
        "id": "test_q",
        "question": "What was Apex Corp's net income in 2024?",
        "answer": "Apex Corp's net income was $28.2M in 2024.",
        "expected_citations": ["apex_corp_2024_financial"],
    }]


@pytest.fixture
def env(corpus: Corpus, questions: list[dict]) -> DocumentExplorationEnv:
    e = DocumentExplorationEnv(
        corpus=corpus, questions=questions, max_steps=5, use_docker=False,
    )
    yield e
    e.close()


def test_reset_returns_observation(env: DocumentExplorationEnv) -> None:
    obs = env.reset(question_idx=0)
    assert "Question:" in obs
    assert "Apex Corp" in obs


def test_question_only_observation_avoids_duplicate_policy_instructions(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        use_docker=False,
        include_preamble=False,
    )
    try:
        observation = env.reset(question_idx=0)
        assert observation == f"Question: {questions[0]['question']}\n"
        assert "SUBMIT: <answer>" not in observation
        assert "Python REPL" not in observation
    finally:
        env.close()


def test_step_with_code(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    obs, reward, done, info = env.step('print("hello")')
    assert "hello" in obs
    assert reward == 0.0
    assert done is False


def test_step_with_submit(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    obs, reward, done, info = env.step(
        'SUBMIT: Net income was $28.2M CITATIONS: ["apex_corp_2024_financial"] '
        'EVIDENCE: [{"doc_id":"apex_corp_2024_financial","start":0,"end":20}]'
    )
    assert done is True
    assert reward > 0.0
    assert "reward_breakdown" in info
    assert info["predicted_evidence"] == [{
        "doc_id": "apex_corp_2024_financial", "start": 0, "end": 20,
    }]


def test_max_steps_triggers_done(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    for _ in range(5):
        obs, reward, done, info = env.step('print("step")')
        if done:
            break
    assert done is True


def test_trajectory_recorded(env: DocumentExplorationEnv) -> None:
    env.reset(question_idx=0)
    env.step('print("a")')
    env.step('SUBMIT: answer CITATIONS: []')
    traj = env.get_trajectory()
    assert len(traj) == 2


def test_legacy_verifier_allows_invalid_submission_after_feedback(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=5,
        use_docker=False,
        require_evidence=True,
        evidence_verifier=True,
        verifier_feedback_budget=1,
        escalate_after_verifier_failure=False,
    )
    try:
        env.reset(question_idx=0)
        observation, reward, done, info = env.step(
            "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []"
        )
        assert done is False
        assert reward == 0.0
        assert "The refusal is premature" in observation
        assert info["verifier_intervention"].kind == "submission"

        # Explicit legacy-v1 mode reproduces the archived fail-open behavior.
        _, _, done, info = env.step("SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []")
        assert done is True
        assert "reward_breakdown" in info
        episode = env.get_episode_info()
        assert episode is not None
        assert len(episode.verifier_events) == 1
        assert episode.escalation is None
    finally:
        env.close()


def test_verifier_rejects_invalid_submission_after_feedback(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus, questions=questions, max_steps=5, use_docker=False,
        require_evidence=True, evidence_verifier=True, verifier_feedback_budget=1,
    )
    try:
        env.reset(question_idx=0)
        invalid = "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []"
        _, first_reward, first_done, _ = env.step(invalid)
        assert first_reward == 0.0 and first_done is False

        handoff, reward, done, info = env.step(invalid)
        assert done is True and reward == 0.0
        assert info["escalated"] is True
        assert "reward_breakdown" not in info
        assert "[ESCALATE_TO_HOST]" in handoff
        episode = env.get_episode_info()
        assert episode is not None
        assert episode.final_reward is None
        assert episode.escalation is not None
        assert episode.escalation.candidate_action == invalid
    finally:
        env.close()


def test_failed_verifier_recovery_has_no_scored_prediction(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus, questions=questions, max_steps=5, use_docker=False,
        require_evidence=True, evidence_verifier=True, verifier_feedback_budget=1,
    )
    invalid = "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []"
    try:
        result = run_single(env, SimpleNamespace(act=lambda _observation: invalid), 0)
        assert result.status == "escalated"
        assert result.predicted_answer == ""
        assert result.predicted_citations == []
        assert result.predicted_evidence == []
        assert result.reward == result.outcome_reward == 0.0
        assert result.escalation is not None
    finally:
        env.close()


def test_verifier_rejects_invalid_submission_on_final_step(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus, questions=questions, max_steps=1, use_docker=False,
        require_evidence=True, evidence_verifier=True, verifier_feedback_budget=1,
    )
    try:
        env.reset(question_idx=0)
        _, reward, done, info = env.step("SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []")
        assert done is True and reward == 0.0
        assert info["escalated"] is True
        assert "reward_breakdown" not in info
        episode = env.get_episode_info()
        assert episode is not None
        assert episode.final_reward is None
        assert episode.escalation is not None
    finally:
        env.close()


def test_verifier_does_not_execute_duplicate_after_feedback_exhaustion(
    corpus: Corpus, questions: list[dict], monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus, questions=questions, max_steps=5, use_docker=False,
        evidence_verifier=True, verifier_feedback_budget=1,
    )
    search = 'print(search_within("apex_corp_2024_financial", "net income 2024"))'
    try:
        env.reset(question_idx=0)
        _, _, done, _ = env.step(search)
        assert done is False
        _, _, done, info = env.step(search)
        assert done is False
        assert info["verifier_intervention"].kind == "duplicate_action"
        monkeypatch.setattr(env.repl, "execute", lambda *_args, **_kwargs: pytest.fail(
            "rejected duplicate reached the Python executor"
        ))
        _, reward, done, info = env.step(search)
        assert done is True and reward == 0.0
        assert info["escalated"] is True
        assert info["escalation"].kind == "duplicate_action"
    finally:
        env.close()


def test_evidence_verifier_escalates_after_two_failed_recoveries(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=7,
        use_docker=False,
        evidence_verifier=True,
        verifier_feedback_budget=2,
        escalate_after_verifier_failure=True,
    )
    search = (
        'print(search_within("apex_corp_2024_financial", "net income 2024"))'
    )
    try:
        env.reset(question_idx=0)
        _, _, done, _ = env.step(search)
        assert done is False

        feedback, _, done, _ = env.step("SUBMIT: Unanswerable CITATIONS: []")
        assert done is False
        assert "recovery 1/2" in feedback
        assert "DIFFERENT Python investigation action" in feedback

        feedback, _, done, info = env.step(search)
        assert done is False
        assert "recovery 2/2" in feedback
        assert info["verifier_intervention"].kind == "duplicate_action"

        handoff, reward, done, info = env.step(search)
        assert done is True
        assert reward == 0.0
        assert info["escalated"] is True
        assert "[ESCALATE_TO_HOST]" in handoff
        episode = env.get_episode_info()
        assert episode is not None
        assert episode.escalation is not None
        assert episode.escalation.kind == "duplicate_action"
        assert len(episode.verifier_events) == 2
    finally:
        env.close()


def test_evidence_verifier_accepts_successful_recovery(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=7,
        use_docker=False,
        evidence_verifier=True,
        verifier_feedback_budget=1,
    )
    try:
        env.reset(question_idx=0)
        env.step(
            'print(search_within("apex_corp_2024_financial", "net income 2024"))'
        )
        _, _, done, _ = env.step("SUBMIT: Unanswerable CITATIONS: []")
        assert done is False

        _, _, done, _ = env.step(
            'print(search_within("apex_corp_2024_financial", "profit after tax"))'
        )
        assert done is False
        _, reward, done, info = env.step(
            'SUBMIT: Net income was $28.2M CITATIONS: ["apex_corp_2024_financial"]'
        )
        assert done is True
        assert reward > 0.0
        assert "reward_breakdown" in info
        episode = env.get_episode_info()
        assert episode is not None
        assert episode.escalation is None
    finally:
        env.close()


def test_persistent_abstention_escalates_after_two_recovery_searches(
    corpus: Corpus, questions: list[dict]
) -> None:
    env = DocumentExplorationEnv(
        corpus=corpus,
        questions=questions,
        max_steps=8,
        use_docker=False,
        evidence_verifier=True,
        verifier_feedback_budget=2,
        escalate_after_verifier_failure=True,
    )
    doc_id = "apex_corp_2024_financial"
    try:
        env.reset(question_idx=0)
        env.step(f'print(search_within("{doc_id}", "net income 2024"))')

        first, _, done, _ = env.step("SUBMIT: Unanswerable CITATIONS: []")
        assert done is False
        assert "recovery 1/2" in first

        env.step(f'print(search_within("{doc_id}", "profit after tax"))')
        second, _, done, _ = env.step("SUBMIT: Unanswerable CITATIONS: []")
        assert done is False
        assert "recovery 2/2" in second
        assert "still abstains" in second

        env.step(f'print(scan("{doc_id}", r"net income|profit", max_hits=6))')
        handoff, reward, done, info = env.step("SUBMIT: Unanswerable CITATIONS: []")
        assert done is True
        assert reward == 0.0
        assert info["escalated"] is True
        assert "[ESCALATE_TO_HOST]" in handoff
        episode = env.get_episode_info()
        assert episode is not None
        assert episode.escalation is not None
        assert episode.escalation.kind == "submission"
        assert episode.escalation.state["successful_tool_actions"] == 3
    finally:
        env.close()


class TestParseSubmission:
    def test_valid_submission(self) -> None:
        result = parse_submission('SUBMIT: answer here CITATIONS: ["doc1", "doc2"]')
        assert result is not None
        answer, citations = result
        assert answer == "answer here"
        assert citations == ["doc1", "doc2"]

    def test_not_a_submission(self) -> None:
        assert parse_submission('print("hello")') is None

    def test_submit_no_citations(self) -> None:
        result = parse_submission("SUBMIT: just the answer")
        assert result is not None
        assert result[0] == "just the answer"
        assert result[1] == []

    def test_case_insensitive(self) -> None:
        result = parse_submission('submit: answer CITATIONS: ["doc1"]')
        assert result is not None
