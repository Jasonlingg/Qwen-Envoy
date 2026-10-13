"""Small deterministic guards for migrating reviewed QASPER action traces."""

import json

import pytest

from scripts.replay_qasper_envoybench_sft import (
    ReplayRejectedError,
    _append_journal,
    _conversation,
    _observation_payload,
    _read_journal,
)
from src.eval.sft_quality import trajectory_hash
from src.policies.code_execution import QASPER_SYSTEM_PROMPT


def test_only_the_step_footer_may_change() -> None:
    source = "{'text': 'same bytes'}\n\n\n[Step 2/10/10]"
    runtime = "{'text': 'same bytes'}\n\n\n[Step 2/15/15]"
    changed = "{'text': 'different bytes'}\n\n\n[Step 2/15/15]"
    assert _observation_payload(source, step=2, budget=10) == _observation_payload(
        runtime, step=2, budget=15,
    )
    assert _observation_payload(source, step=2, budget=10) != _observation_payload(
        changed, step=2, budget=15,
    )


def test_source_step_five_guidance_can_map_to_runtime_ordinary_footer() -> None:
    payload = "{'text': 'real passage'}\n"
    source = (
        payload + "\n\n[Step 5/10 — 5 steps left. "
        "Run sufficiency check: can you answer now? If yes → SUBMIT immediately.]"
    )
    runtime = payload + "\n\n[Step 5/15/15]"
    assert _observation_payload(source, step=5, budget=10) == payload
    assert _observation_payload(runtime, step=5, budget=15) == payload


@pytest.mark.parametrize(
    "observation,step,budget",
    [
        ("data\n\n\n[Step 2/10/10]", 1, 10),
        ("data\n\n\n[Step 2/10/10]", 2, 15),
        ("data\n\n\n[Step 2/10/9]", 2, 10),
        ("data\n[Step 2/10/10]", 2, 10),
        ("data\n\n[Step 5/10 — 4 steps left. Run sufficiency check: can you answer now? "
         "If yes → SUBMIT immediately.]", 5, 10),
    ],
)
def test_unexpected_or_malformed_footer_is_rejected(
    observation: str, step: int, budget: int,
) -> None:
    with pytest.raises(ReplayRejectedError, match="unexpected_step_footer"):
        _observation_payload(observation, step=step, budget=budget)


def test_conversation_preserves_the_runtime_initial_turn_and_observation() -> None:
    row = {
        "initial_observation": "runtime preamble\n\nQuestion: Q?\n",
        "trajectory": [
            {"action": "print(read('doc'))", "observation": "real bytes\n\n\n[Step 1/15/15]",
             "done": False},
            {"action": 'SUBMIT: A CITATIONS: ["doc"] EVIDENCE: []',
             "observation": "Submitted. Reward: 1.000", "done": True},
        ],
    }
    messages = _conversation(row)["messages"]
    assert messages == [
        {"role": "system", "content": QASPER_SYSTEM_PROMPT},
        {"role": "user", "content": "runtime preamble\n\nQuestion: Q?\n"},
        {"role": "assistant", "content": "print(read('doc'))"},
        {"role": "user", "content": "real bytes\n\n\n[Step 1/15/15]"},
        {"role": "assistant", "content": 'SUBMIT: A CITATIONS: ["doc"] EVIDENCE: []'},
    ]


def test_resume_rejects_changed_context_and_tampered_outcome(tmp_path) -> None:
    source = {
        "question_id": "q", "question": "Q?",
        "trajectory": [{"action": "print('real')", "observation": "real", "done": False}],
    }
    candidate = {**source, "source_trajectory_sha256": trajectory_hash(source),
                 "runtime_protocol": "envoybench-v1"}
    candidate["review"] = {"trajectory_sha256": trajectory_hash(candidate)}
    journal = tmp_path / "partial.jsonl"
    journal.touch()
    context = {"type": "header", "source_sha256": "pinned", "runtime": "envoybench-v1"}
    head = _append_journal(journal, context, None)
    _append_journal(journal, {
        "type": "accepted", "source_index": 0, "question_id": "q",
        "source_trajectory_sha256": trajectory_hash(source), "candidate": candidate,
    }, head)
    accepted, rejected, _ = _read_journal(journal, context, [source])
    assert [row["question_id"] for row in accepted] == ["q"]
    assert rejected == []
    with pytest.raises(ValueError, match="source, runtime, or holdout hash changed"):
        _read_journal(journal, {**context, "source_sha256": "changed"}, [source])

    records = [json.loads(line) for line in journal.read_text().splitlines()]
    records[1]["candidate"]["trajectory"][0]["observation"] = "fabricated"
    journal.write_text("".join(json.dumps(item) + "\n" for item in records))
    with pytest.raises(ValueError, match="Tampered replay journal entry"):
        _read_journal(journal, context, [source])
