"""Watchdog arithmetic and scoped teardown with fake CLI; no cloud mutations."""
import copy
import json

import pytest

from scripts.watch_reader_gpu import cleanup, validate, watch


def record():
    return {"started": 100, "deadline": 7300, "budget_usd": 5,
            "hourly_total_upper_bound": 1.9,
            "pod": {"id": "owned_pod", "createdAt": "2026-10-24T19:58:31.04Z"},
            "volume": {"id": "owned_volume"}}


@pytest.mark.parametrize("field,value", [("deadline", 7301), ("budget_usd", 5.01),
    ("hourly_total_upper_bound", 3), ("started", float("nan"))])
def test_watchdog_rejects_unbounded_or_unaffordable_lifetime(field, value):
    value_record = record()
    value_record[field] = value
    with pytest.raises(ValueError):
        validate(value_record)


def test_watchdog_is_independent_of_training_and_deletes_only_owned_resources(tmp_path):
    value = record()
    validate(value)
    (tmp_path / "owned.json").write_text(json.dumps(value))
    calls, deleted = [], set()

    def cli(kind, action, ident):
        assert ident in {"owned_pod", "owned_volume"}
        calls.append((kind, action, ident))
        if action == "delete":
            deleted.add(ident)
            return {}
        return {"code": "not_found"} if ident in deleted else {
            "id": ident, "createdAt": "2026-10-24T19:58:31.04Z"}

    proof = watch(tmp_path, cli, clock=lambda: 7300)
    assert all(p["absent"] for p in proof.values())
    assert [c[0] for c in calls if c[1] == "delete"] == ["pod", "network-volume"]
    assert (tmp_path / "watchdog-cleanup.json").exists()


def test_watchdog_refuses_resource_identity_mismatch():
    value = copy.deepcopy(record())
    with pytest.raises(ValueError, match="create receipt"):
        cleanup(value, lambda *args: {"id": "someone_else"})


def test_mcp_and_cli_creation_timestamp_formats_identify_same_resource():
    from scripts.watch_reader_gpu import creation_time

    assert creation_time("2026-10-24T19:58:31.04Z") == creation_time(
        "2026-10-24 19:58:31.04 +0000 UTC")
    assert creation_time("2026-10-24T19:58:31.05Z") != creation_time(
        "2026-10-24 19:58:31.04 +0000 UTC")


def test_cleanup_accepts_cli_timestamp_format_and_checks_absence():
    value = record()
    value["pod"]["createdAt"] = "2026-10-24T19:58:31.04Z"
    deleted = set()

    def call(kind, action, ident):
        if action == "delete":
            deleted.add(ident)
            return {}
        return {"code": "not_found"} if ident in deleted else {
            "id": ident, "createdAt": "2026-10-24 19:58:31.04 +0000 UTC"}

    assert all(p["absent"] for p in cleanup(value, call).values())


def pilot_record():
    from pathlib import Path

    from src.eval.reader_improvement import file_hash

    path = Path("examples/reader-rl-pilot-v2/protocol.json")
    value = record()
    value.update(deadline=13600, budget_usd=8, pilot_protocol=str(path),
                 pilot_protocol_sha256=file_hash(path))
    value["paid_authorization_receipt"] = {
        "authorized": True, "authorization": "synthetic fixture; no real permission or launch",
        "protocol_sha256": file_hash(path), "budget_usd": 8,
        "maximum_lifetime_seconds": 13500,
    }
    return value


def test_larger_pilot_guard_requires_its_own_authorization_and_keeps_old_default():
    value = pilot_record()
    validate(value)
    value.pop("pilot_protocol")
    with pytest.raises(ValueError,match="allowance"):
        validate(value)


@pytest.mark.parametrize(
    "mutation", ["pending", "missing", "budget", "protocol", "over_time", "rate"])
def test_pilot_guard_rejects_unapproved_or_unaffordable_changes(mutation):
    value = pilot_record()
    if mutation == "pending":
        value["paid_authorization_receipt"]["authorized"] = False
    elif mutation == "missing":
        value.pop("paid_authorization_receipt")
    elif mutation == "budget":
        value["paid_authorization_receipt"]["budget_usd"] = 9
    elif mutation == "protocol":
        value["pilot_protocol_sha256"] = "0" * 64
    elif mutation == "over_time":
        value["deadline"] += 1
    else:
        value["hourly_total_upper_bound"] = 1.93
    with pytest.raises(ValueError):
        validate(value)
