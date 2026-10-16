"""Interrupted replays may reuse only complete, bound, exact V1 questions."""

import copy
import json

import pytest

from src.eval.harness_replay_checkpoint import _hash, checkpointed_baselines


def _source(qid):
    return {"question_id": qid, "question": "Question?", "trajectory": [{
        "step": 1, "action": "print('safe')", "observation": "safe\n",
        "reward": 0.0, "done": False,
    }]}


def _result(row):
    saved = row["trajectory"][0]
    return {"question_id": row["question_id"], "question": row["question"],
            "initial_observation": "Question: Question?\n", "v1_exact_match": True,
            "actions": [{"step": 1, "action": saved["action"],
                         "recorded_observation": saved["observation"],
                         "v1_observation": saved["observation"], "v1_exact_match": True,
                         "v1_reward": 0.0, "recorded_reward": 0.0,
                         "v1_done": False, "recorded_done": False,
                         "normalization": "none", "v1_executed": True,
                         "v1_tool_observation": "safe\n"}]}


def test_interruption_preserves_complete_rows_and_resume_runs_only_pending(tmp_path):
    rows, checkpoint = [_source("q1"), _source("q2")], tmp_path / "checkpoint.json"

    def interrupted(pair):
        if pair[0]["question_id"] == "q2":
            raise RuntimeError("simulated interruption")
        return _result(pair[0])

    kwargs = {"checkpoint": checkpoint, "identity": {"run": "r", "image": "sha"},
              "workers": 1}
    with pytest.raises(RuntimeError, match="interruption"):
        checkpointed_baselines(rows, [None, None], interrupted, resume=False, **kwargs)
    saved = json.loads(checkpoint.read_text())
    assert list(saved["questions"]) == ["q1"]
    called = []

    def resumed(pair):
        called.append(pair[0]["question_id"])
        return _result(pair[0])

    result, reused = checkpointed_baselines(rows, [None, None], resumed, resume=True, **kwargs)
    assert reused == 1 and called == ["q2"]
    assert [row["question_id"] for row in result] == ["q1", "q2"]


@pytest.mark.parametrize("change", [
    "identity", "action", "incomplete", "observation", "false_flag"])
def test_stale_or_changed_checkpoint_is_rejected(tmp_path, change):
    row, checkpoint = _source("q1"), tmp_path / "checkpoint.json"
    kwargs = {"checkpoint": checkpoint, "identity": {"run": "r", "corpus": "c"}, "workers": 1}
    checkpointed_baselines([row], [None], lambda pair: _result(pair[0]), resume=False, **kwargs)
    changed = copy.deepcopy(row)
    if change == "identity":
        kwargs["identity"] = {"run": "r", "corpus": "different"}
    elif change == "action":
        changed["trajectory"][0]["action"] = "print('changed')"
    else:
        saved = json.loads(checkpoint.read_text())
        result = saved["questions"]["q1"]["result"]
        if change == "incomplete":
            result["actions"] = []
        elif change == "observation":
            result["actions"][0]["v1_observation"] = "changed"
        else:
            result["actions"][0]["v1_exact_match"] = "true"
        saved["questions"]["q1"]["result_hash"] = _hash(result)
        checkpoint.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="Checkpoint"):
        checkpointed_baselines([changed], [None], lambda pair: pytest.fail("must not run"),
                               resume=True, **kwargs)


def test_existing_checkpoint_requires_explicit_resume(tmp_path):
    row, checkpoint = _source("q1"), tmp_path / "checkpoint.json"
    kwargs = {"checkpoint": checkpoint, "identity": {"run": "r"}, "workers": 1}
    checkpointed_baselines([row], [None], lambda pair: _result(pair[0]), resume=False, **kwargs)
    with pytest.raises(ValueError, match="use --resume"):
        checkpointed_baselines([row], [None], lambda pair: _result(pair[0]), resume=False, **kwargs)
