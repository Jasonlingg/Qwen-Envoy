import json

import pytest

from src.eval.research_pilot_host import MAX_ACTION_CHARS, SYSTEM_PROMPT, run_host

QUESTION = {"id": "memory", "question": "What memory reduction is reported?",
            "arxiv_ids": ["2510.03215v2"]}
SPAN = {"arxiv_id": "2510.03215v2", "start": 0, "end": 10}


def action(tool, **arguments):
    return json.dumps({"tool": tool, "arguments": arguments})


def finish(claims=None):
    return action("finish", answer="Evidence is insufficient.",
                  claims=[] if claims is None else claims, limitations=["Selected papers only."])


class Policy:
    system_prompt = SYSTEM_PROMPT

    def __init__(self, actions):
        self.actions = iter(actions)
        self.observations = []

    def act(self, observation):
        self.observations.append(observation)
        value = next(self.actions)
        if isinstance(value, Exception):
            raise value
        return value


def test_direct_cannot_delegate_even_when_callback_is_supplied():
    called = []
    policy = Policy([action("investigate", question="q", arxiv_ids=QUESTION["arxiv_ids"]),
                     finish()])
    result = run_host(QUESTION, "direct", policy, {"investigate": lambda **kw: called.append(kw)})
    assert result["status"] == "submitted"
    assert not called and result["investigate_calls"] == 0
    assert "unavailable" in policy.observations[1]
    assert "investigate" not in json.loads(policy.observations[0])["available_tools"]


def test_delegated_cannot_finish_without_attempting_reader():
    result = run_host(QUESTION, "delegated", Policy([finish()]), {}, max_turns=1)
    assert result["status"] == "incomplete" and result["submission"] is None
    assert "must call investigate" in result["trajectory"][0]["output"]


def test_failed_paper_and_fallback_are_preserved_without_hidden_retries():
    called = []
    reader_result = {"status": "evidence_found", "papers": [
        {"arxiv_id": "2510.03215v2", "status": "error", "error": "timed out"},
        {"arxiv_id": "2402.01030v4", "status": "evidence_found"}], "evidence": [SPAN]}
    def investigate(**arguments):
        called.append(arguments)
        return reader_result
    policy = Policy([action("investigate", question="q", arxiv_ids=QUESTION["arxiv_ids"]),
                     action("get_passage", arxiv_id="2510.03215v2", start=0, length=10), finish()])
    result = run_host(QUESTION, "delegated", policy, {
        "investigate": investigate, "get_passage": lambda **kw: {"text": "some quote"}})
    assert result["status"] == "submitted" and len(called) == 1
    assert result["delegation_failures"] == reader_result["papers"][:1]
    assert result["fallback_tool_calls"] == 1
    assert result["trajectory"][0]["output"]["evidence"] == [SPAN]


def test_tool_exception_becomes_feedback_and_reader_failure_can_be_reported():
    def unavailable(**kw):
        raise TimeoutError("source not reachable")
    policy = Policy([action("investigate", question="q", arxiv_ids=QUESTION["arxiv_ids"]),
                     finish()])
    result = run_host(QUESTION, "delegated", policy, {"investigate": unavailable})
    assert result["status"] == "submitted" and result["investigate_calls"] == 1
    assert "TimeoutError" in policy.observations[1]
    assert result["delegation_failures"][0]["status"] == "tool_error"


def test_factual_finish_requires_external_frozen_corpus_validator():
    claims = [{"text": "Authors report a reduction.", "evidence": [SPAN]}]
    result = run_host(QUESTION, "direct", Policy([finish(claims)]), {}, max_turns=1)
    assert result["status"] == "incomplete"
    assert "validator" in result["trajectory"][0]["output"]
    checked = []
    def validate(submission):
        checked.append(submission)
        return {"source_spans_valid": True, "semantic_support": "not_reviewed"}
    result = run_host(QUESTION, "direct", Policy([finish(claims)]), {}, validate_finish=validate)
    assert result["status"] == "submitted" and len(checked) == 1
    assert result["checks"]["source_spans_valid"] is True
    assert result["semantic_support"] == "not_reviewed"


def test_frozen_corpus_validation_failure_does_not_accept_finish():
    def invalid(submission):
        raise ValueError("span exceeds frozen paper length")
    claims = [{"text": "A claim.", "evidence": [SPAN]}]
    result = run_host(QUESTION, "direct", Policy([finish(claims), finish()]), {},
                      validate_finish=invalid, max_turns=2)
    assert result["status"] == "incomplete"
    assert result["submission"] is None
    assert "frozen paper" in result["trajectory"][0]["output"]


@pytest.mark.parametrize("raw", [
    "```json\n{}\n```", "print('never execute')", "[]", "null",
    json.dumps({"tool": "finish", "arguments": {}, "extra": True}),
    '{"tool":"finish","tool":"get_passage","arguments":{}}',
    '{"tool":"get_passage","arguments":{"start":NaN}}',
    "x" * (MAX_ACTION_CHARS + 1),
])
def test_raw_json_only_and_bounded_actions(raw):
    result = run_host(QUESTION, "direct", Policy([raw]), {}, max_turns=1)
    assert result["status"] == "incomplete" and result["submission"] is None
    assert len(result["trajectory"][0]["raw_action"]) <= MAX_ACTION_CHARS


@pytest.mark.parametrize("span", [
    {**SPAN, "start": True}, {**SPAN, "end": -1}, {**SPAN, "end": 0},
    {**SPAN, "quote": "extra"}, {"arxiv_id": "2510.03215v2", "start": 0},
])
def test_strict_span_shape_before_external_validation(span):
    checked = []
    claims = [{"text": "Claim.", "evidence": [span]}]
    result = run_host(QUESTION, "direct", Policy([finish(claims)]), {}, max_turns=1,
                      validate_finish=lambda value: checked.append(value))
    assert result["status"] == "incomplete" and not checked


def test_model_failure_is_honest_and_timed():
    result = run_host(QUESTION, "direct", Policy([RuntimeError("budget exhausted")]), {})
    assert result["status"] == "error" and result["submission"] is None
    assert result["model_requests_attempted"] == 1
    assert result["elapsed_seconds"] >= result["trajectory"][0]["model_seconds"] >= 0


def test_unexpected_validator_failure_is_honest_error():
    def unavailable(submission):
        raise OSError("frozen corpus is unavailable")
    result = run_host(QUESTION, "direct", Policy([finish()]), {}, validate_finish=unavailable)
    assert result["status"] == "error" and result["submission"] is None
    assert "OSError" in result["error"]


def test_invalid_tool_argument_failure_is_returned_without_retry():
    calls = []
    def passage(arxiv_id, start, length):
        calls.append(arxiv_id)
        if arxiv_id not in QUESTION["arxiv_ids"]:
            raise ValueError("paper is outside this question's frozen set")
    policy = Policy([action("get_passage", arxiv_id="2601.00001v1", start=0, length=10), finish()])
    result = run_host(QUESTION, "direct", policy, {"get_passage": passage})
    assert result["status"] == "submitted" and calls == ["2601.00001v1"]
    assert "outside" in policy.observations[1]


def test_initial_observation_omits_reference_answers():
    question = {**QUESTION, "gold_answer": "Do not leak this."}
    policy = Policy([finish()])
    run_host(question, "direct", policy, {})
    assert "Do not leak this" not in policy.observations[0]


@pytest.mark.parametrize("kwargs", [
    {"arm": "unknown"}, {"max_turns": True}, {"max_turns": 0}, {"question": {}},
    {"tools": {"find_evidence": None}},
])
def test_configuration_validation(kwargs):
    arguments = {"question": QUESTION, "arm": "direct", "policy": Policy([]), "tools": {}}
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        run_host(**arguments)
