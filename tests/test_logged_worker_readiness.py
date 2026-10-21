"""Readiness evidence cannot be borrowed from an old worker or activation."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from src.eval.assigned_smoke_lifecycle import logged_readiness, supervise


@pytest.mark.parametrize("worker,age,fitness,expected", [
    ("current", 1, True, True), ("old", 1, True, False),
    ("current", -1, True, False), ("current", 1, False, False)])
def test_both_current_readiness_events_required(worker, age, fitness, expected):
    since = datetime.now(timezone.utc)
    stamp = (since + timedelta(seconds=age)).isoformat()
    lines = ["vLLM is healthy"] + (["All fitness checks passed."] if fitness else [])
    logs = "\n".join(json.dumps({"workerId": worker, "ts": stamp, "line": line})
                     for line in lines)
    assert logged_readiness(logs, {"current"}, since) is expected


def test_elapsed_absolute_deadline_never_activates(tmp_path):
    calls = []
    def control(command, endpoint):
        calls.append(command)
        if command == "get":
            return {"workersMin": 0, "workersMax": 0,
                    "gpuIds": ["NVIDIA RTX A5000"], "gpuCount": 1}
        return {"jobs": {"inQueue": 0, "inProgress": 0}, "workers": {"ready": 0}}
    with pytest.raises(TimeoutError, match="original session deadline"):
        supervise("test", tmp_path, [], key="fake", control=control,
                  worker_reader=lambda: [], deadline_unix=0)
    assert calls == ["get", "health"]
