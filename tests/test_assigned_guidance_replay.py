"""Replay must match the baseline and cannot invent a new model continuation."""
import copy

import pytest

from scripts.replay_assigned_guidance import (
    RecordedActionsExhaustedError,
    RecordedPolicy,
    require_baseline,
)


def test_recorded_policy_stops_instead_of_fabricating_a_response_to_new_feedback():
    policy = RecordedPolicy(['print(read("id"))'])
    assert policy.act("initial prompt") == 'print(read("id"))'
    with pytest.raises(RecordedActionsExhaustedError, match="no recorded response"):
        policy.act("new feedback")
    assert policy.observations == ["initial prompt", "new feedback"]


@pytest.mark.parametrize("field", ["trajectory", "evidence", "status", "inspection_steps"])
def test_baseline_mismatch_cannot_be_compared_as_a_candidate_improvement(field):
    original = {"trajectory": [{"action": "read", "observation": "source"}],
                "evidence": [{"start": 0, "end": 6}], "status": "evidence_found",
                "inspection_steps": 1}
    packet = copy.deepcopy(original)
    policy = RecordedPolicy([])
    policy.observations = ["the exact prior observation"]
    requests = [{"payload": {"messages": [{"content": policy.observations[0]}]}}]
    require_baseline(packet, policy, original, requests)
    packet[field] = None
    with pytest.raises(ValueError, match=f"recorded run: {field}"):
        require_baseline(packet, policy, original, requests)


def test_matching_outputs_with_different_model_inputs_are_not_a_reproduced_baseline():
    policy = RecordedPolicy([])
    policy.observations = ["different source text"]
    with pytest.raises(ValueError, match="model-input observations"):
        require_baseline({}, policy, {}, [{"payload": {"messages": [{"content": "old"}]}}])
