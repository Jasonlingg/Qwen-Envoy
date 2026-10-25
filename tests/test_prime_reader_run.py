"""Budget/process failure tests; no models, optimizer, cloud or paid requests."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from src.eval.reader_improvement import file_hash
from src.training import prime_budget, prime_run
from src.training.prime_budget import Budget, proxy
from src.training.prime_run import (
    Processes,
    execute,
    prepare,
    validate_plan,
    verify_base_assets,
    write,
)


def payload():
    return {"model": "envoy", "token_ids": [1, 2, 3], "sampling_params": {
        "temperature": 1., "top_p": 1., "top_k": -1, "min_p": 0., "max_tokens": 1024}}


def test_atomic_reservations_enforce_global_limits_across_policy_cycles(tmp_path):
    limits = {"requests": 3, "reserved_tokens": 8192}
    first = Budget(tmp_path, limits, time.time() + 60, {"version": 0})
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: reserve_or_none(first), range(12)))
    assert sum(x is not None for x in results) == 3
    second = Budget(tmp_path, limits, time.time() + 60, {"version": 1})
    with pytest.raises(RuntimeError, match="budget"):
        second.reserve(payload())
    assert second.state["reserved_tokens"] == 3 * 1027
    assert all(r["policy"] == {"version": 0} for r in second.state["requests"])


def reserve_or_none(budget):
    try:
        return budget.reserve(payload())
    except RuntimeError:
        return None


@pytest.mark.parametrize("expired", [False, True])
def test_token_and_time_limits_reject_before_dispatch(tmp_path, expired):
    b = Budget(tmp_path, {"requests": 50, "reserved_tokens": 8192},
               time.time() + (-1 if expired else 60), {})
    if not expired:
        p = payload()
        p["token_ids"] = [1] * 7168
        b.reserve(p)
    before = copy.deepcopy(b.state)
    with pytest.raises(RuntimeError):
        b.reserve(payload())
    assert b.state == before


def test_uncertain_call_is_never_refunded_and_closes_future_cycles(tmp_path):
    limits = {"requests": 50, "reserved_tokens": 8192}
    b = Budget(tmp_path, limits, time.time() + 60, {})
    ordinal = b.reserve(payload())
    b.finish(ordinal, False)
    resumed = Budget(tmp_path, limits, time.time() + 60, {})
    assert resumed.state["reserved_tokens"] == 1027
    with pytest.raises(RuntimeError):
        resumed.reserve(payload())


def test_process_death_after_reservation_cannot_allow_more_requests(tmp_path):
    limits = {"requests": 50, "reserved_tokens": 8192}
    Budget(tmp_path, limits, time.time() + 60, {}).reserve(payload())
    with pytest.raises(RuntimeError):
        Budget(tmp_path, limits, time.time() + 60, {}).reserve(payload())


@pytest.mark.parametrize("field,value", [("max_tokens", 1025), ("temperature", 0),
    ("top_p", .9), ("top_k", 20), ("n", 2)])
def test_changed_sampling_is_rejected_before_reservation(tmp_path, field, value):
    b = Budget(tmp_path, {"requests": 2, "reserved_tokens": 8192}, time.time() + 60, {})
    p = payload()
    p["sampling_params"][field] = value
    with pytest.raises(ValueError):
        b.reserve(p)
    assert not b.state["requests"]


@pytest.mark.parametrize("mode", ["ok", "redirect", "oversize", "bad_json", "failure"])
def test_real_loopback_proxy_bounds_errors_and_never_follows_redirects(tmp_path, monkeypatch, mode):
    calls = []

    class Engine(BaseHTTPRequestHandler):
        def do_POST(self):
            calls.append(self.path)
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b'{"choices":[]}'
            if mode == "oversize":
                body = b"x" * (2 * 1024 * 1024 + 1)
            elif mode == "bad_json":
                body = b"not JSON"
            self.send_response(302 if mode == "redirect" else 500 if mode == "failure" else 200)
            self.send_header("Location", "/escape")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    engine = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    thread = threading.Thread(target=engine.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(prime_budget, "ENGINE", f"http://127.0.0.1:{engine.server_port}")
    write(tmp_path / "started.json", {"deadline": time.time() + 30})
    plan = {"limits": {"requests": 3, "reserved_tokens": 8192}}
    try:
        with proxy(tmp_path, plan, {"version": 0}) as url:
            with httpx.Client(trust_env=False) as client:
                r = client.post(url.removesuffix("/v1") + "/inference/v1/generate", json=payload())
                assert r.status_code == (200 if mode == "ok" else 502)
                if mode != "ok":
                    r = client.post(url.removesuffix("/v1") + "/inference/v1/generate",
                                    json=payload())
                    assert r.status_code == 502
        ledger = json.loads((tmp_path / "dispatches.json").read_text())
        assert len(ledger["requests"]) == 1 and ledger["reserved_tokens"] == 1027
        assert ledger["closed"] is (mode != "ok")
        assert calls == ["/inference/v1/generate"]
    finally:
        engine.shutdown()
        engine.server_close()
        thread.join(timeout=5)


def test_prepared_run_uses_only_train_role_and_refuses_drift_or_restart(tmp_path):
    output = tmp_path / "run"
    plan = prepare(output, tmp_path / "assets")
    assert validate_plan(output) == plan
    assert len(plan["train_case_ids"]) == 81
    assert plan["quality_improvement_measured"] is False
    changed = copy.deepcopy(plan)
    changed["train_case_ids"][0] = "locked-evaluation-question"
    write(output / "plan.json", changed)
    with pytest.raises(ValueError, match="drift"):
        validate_plan(output)
    held = "qasper_train_2f4acd34eb2d09db9b5ad9b1eb82cb4a88c13f5b--rl-short-fact"
    changed["train_case_ids"][0] = held
    write(output / "plan.json", changed)
    with pytest.raises(ValueError, match="drift"):
        validate_plan(output)
    write(output / "plan.json", plan)
    write(output / "started.json", {"deadline": 0,
                                   "plan_sha256": file_hash(output / "plan.json")})
    with pytest.raises(FileExistsError):
        execute(output)
    assert not (output / "00-preflight.log").exists()


def test_wall_deadline_stops_owned_process_without_touching_other_processes(tmp_path):
    processes = Processes(time.time() + .3)
    owned = processes.start([sys.executable, "-c", "import time; time.sleep(60)"],
                            tmp_path / "owned.log")
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            processes.wait(owned)
    finally:
        processes.close()
    assert owned.poll() is not None
    assert unrelated.poll() is None
    unrelated.terminate()
    unrelated.wait(timeout=5)


def test_expired_run_cannot_start_a_process(tmp_path):
    with pytest.raises(TimeoutError):
        Processes(time.time() - 1).start([sys.executable, "-c", "raise Exception()"],
                                       tmp_path / "never.log")
    assert not (tmp_path / "never.log").exists()


def test_second_cycle_failure_stops_updates_and_still_writes_cleanup_receipt(tmp_path, monkeypatch):
    """Controller-only fixture, not a native trainer/GPU simulation result."""
    output = tmp_path / "run"
    prepare(output, tmp_path / "assets")
    calls = []

    class ControlledProcesses:
        def __init__(self, deadline):
            self.closed = False

        def start(self, command, log, **kwargs):
            stage = (command[-2], int(command[-1]))
            calls.append(stage)
            return stage

        def wait(self, stage):
            if stage == ("prepare", 2):
                raise RuntimeError("fixture scoring failure")

        def stop(self, child):
            pass

        def close(self):
            calls.append(("cleanup", 0))

    monkeypatch.setattr(prime_run, "Processes", ControlledProcesses)
    with pytest.raises(RuntimeError, match="scoring"):
        execute(output)
    assert ("train", 1) in calls and ("verify", 1) in calls
    assert ("train", 2) not in calls
    assert calls[-1] == ("cleanup", 0)
    result = json.loads((output / "result.json").read_text())
    assert result["verified_updates"] == 1
    assert result["attempted_update_steps"] == [1]
    assert result["all_recorded_containers_absent"] is True


def test_base_asset_identity_uses_content_not_directory_name(tmp_path, monkeypatch):
    import hashlib

    catalog_path = tmp_path / "examples/verifiers-port/prime-base-files.json"
    catalog_path.parent.mkdir(parents=True)
    base = tmp_path / "assets"
    base.mkdir()
    blob, weights = b'{"fixture":true}\n', b"authored tiny weight fixture"
    (base / "config.json").write_bytes(blob)
    (base / "fixture.safetensors").write_bytes(weights)
    write(catalog_path, {"revision": prime_run.BASE_REVISION, "files": {
        "config.json": {"lfs_sha256": None, "git_blob_id": hashlib.sha1(
            f"blob {len(blob)}\0".encode() + blob).hexdigest()},
        "fixture.safetensors": {"lfs_sha256": hashlib.sha256(weights).hexdigest()}}})
    monkeypatch.setattr(prime_run, "ROOT", tmp_path)
    assert len(verify_base_assets(base)) == 2
    (base / "fixture.safetensors").write_bytes(weights + b"x")
    with pytest.raises(ValueError, match="differs"):
        verify_base_assets(base)


def test_optimizer_identity_survives_reload_and_detects_reset_or_missing_state():
    """Tiny CPU AdamW state fixture; no model rollout or native GPU checkpoint."""
    import torch

    from src.training.prime_native import optimizer_fingerprint

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    before = optimizer_fingerprint(optimizer, 1)
    saved = copy.deepcopy(optimizer.state_dict())
    restored = torch.optim.AdamW(model.parameters())
    restored.load_state_dict(saved)
    # Insertion ordering is not a policy change.
    restored.state = dict(reversed(list(restored.state.items())))
    assert optimizer_fingerprint(restored, 1) == before
    with pytest.raises(ValueError, match="reset"):
        optimizer_fingerprint(restored, 2)
    del next(iter(restored.state.values()))["exp_avg"]
    with pytest.raises(ValueError, match="incomplete"):
        optimizer_fingerprint(restored, 1)


@pytest.mark.parametrize("mutation", ["seed", "order"])
def test_pilot_cannot_change_predeclared_seed_or_question_order(tmp_path, mutation):
    output = tmp_path / "run"
    plan = prepare(output, tmp_path / "assets")
    if mutation == "seed":
        plan["seed"] = 43
    else:
        plan["train_case_ids"].reverse()
    write(output / "plan.json", plan)
    with pytest.raises(ValueError, match="drift"):
        validate_plan(output)
