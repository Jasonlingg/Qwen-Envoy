import json

import pytest

from scripts.run_assigned_reader_smoke import authorization
from src.eval.assigned_smoke_plan import LIMITS


def test_replacement_record_pins_limits_and_original_plan(tmp_path):
    path = tmp_path / "approval.json"
    record = {"approved": True, "replacement_attempts": 1,
              "prepared_sha256": "original", "limits": LIMITS}
    path.write_text(json.dumps(record))
    assert authorization(path, "original")["record"] == record
    with pytest.raises(ValueError, match="approval"):
        authorization(path, "changed")
    record["limits"] = {**LIMITS, "model_requests_total": 24}
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="approval"):
        authorization(path, "original")
