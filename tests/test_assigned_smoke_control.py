"""Only canonical v2 records can identify the worker whose logs unlock inference."""
import copy
import json
from datetime import datetime, timezone

import pytest

from src.eval import assigned_smoke_lifecycle as lifecycle
from src.eval.assigned_smoke_control import WorkerCatalog
from src.eval.assigned_smoke_plan import MODEL_IDENTITY


def response():
    return {"summary": {"total": 1}, "workers": [{"id": "v2-current", "status": "IDLE",
        "env": {"private": "must-not-persist"}, "isStale": False}]}


def test_catalog_uses_fixed_control_host_bounded_get_and_drops_environment(monkeypatch):
    calls = []
    def request(client, method, path, payload=None):
        calls.append((client.base, method, path, payload))
        return response()
    monkeypatch.setattr("src.policies.runpod_queue.RunpodQueueTransport._request", request)
    result = WorkerCatalog("endpoint", "fake-key").list()
    assert calls == [("https://api.runpod.io/v2/serverless/endpoint", "GET", "/workers", None)]
    assert result == [{"id": "v2-current", "status": "IDLE", "isStale": False}]
    assert "private" not in json.dumps(result)


@pytest.mark.parametrize("bad", [None, [], {}, {"workers": []},
    {"summary": {"total": 0}, "workers": [{"id": "x", "status": "IDLE"}]},
    {"summary": {"total": True}, "workers": [{"id": "x", "status": "IDLE"}]}])
def test_malformed_catalog_fails_closed(bad):
    with pytest.raises(ValueError, match="v2 worker"):
        WorkerCatalog("endpoint", "fake", fetch=lambda: bad).list()


@pytest.mark.parametrize("change", [{"id": "../other"}, {"status": "unknown"},
                                    {"isStale": "false"}])
def test_invalid_worker_record_fails_closed(change):
    raw = response()
    raw["workers"][0].update(change)
    with pytest.raises(ValueError, match="v2 worker"):
        WorkerCatalog("endpoint", "fake", fetch=lambda: raw).list()


def test_duplicate_ids_are_rejected():
    raw = response()
    raw["summary"]["total"] = 2
    raw["workers"].append(copy.deepcopy(raw["workers"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        WorkerCatalog("endpoint", "fake", fetch=lambda: raw).list()


def test_worker_identity_rejects_unpinned_adapter():
    identity = {"base_repo": MODEL_IDENTITY["base"],
        "base_revision": MODEL_IDENTITY["base_revision"],
        "adapter_repo": MODEL_IDENTITY["adapter"],
        "adapter_revision": MODEL_IDENTITY["adapter_revision"],
        "adapter_sha256": MODEL_IDENTITY["adapter_sha256"],
        "adapter_subfolder": MODEL_IDENTITY["subfolder"],
        "served_adapter": "envoy", "gpu": "NVIDIA RTX A5000"}
    def logs():
        return json.dumps({"line": "ENVOY_PINNED_IDENTITY " + json.dumps(identity),
                           "workerId": "current", "ts": "2026-10-20T12:00:00Z"})
    kwargs = {"active_workers": {"current"}, "since": datetime(2026, 10, 20, tzinfo=timezone.utc)}
    observed = lifecycle.worker_identity(logs(), **kwargs)
    assert observed["adapter_sha256"] == identity["adapter_sha256"]
    assert lifecycle.worker_identity(logs(), **{**kwargs, "active_workers": {"other"}}) is None
    later = datetime(2026, 10, 21, tzinfo=timezone.utc)
    assert lifecycle.worker_identity(logs(), **{**kwargs, "since": later}) is None
    identity["adapter_revision"] = "HEAD"
    with pytest.raises(ValueError, match="identity"):
        lifecycle.worker_identity(logs(), **kwargs)
