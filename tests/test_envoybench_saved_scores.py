"""Supplementary runs and official F1 never alter the frozen paired review."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from benchmarks.envoybench.demo import build_demo_payload, create_app
from benchmarks.envoybench.export import snapshot_html
from benchmarks.envoybench.run import _file_sha256
from src.eval.artifacts import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/envoybench-v0.1"
SCORE = ROOT / "release/qasper-agent-study/qwen-official-score.json"


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def supplementary(tmp_path: Path) -> tuple[Path, Path]:
    manifest = json.loads((RELEASE / "run/manifest.json").read_text())
    original = json.loads((RELEASE / "run/results.json").read_text())
    key = manifest["models"][0]["key"]
    manifest.update(run_id="separate-run", comparison_id="separate-comparison")
    manifest["models"] = [manifest["models"][0]]
    rows = [
        {**row, "run_id": manifest["run_id"], "comparison_id": manifest["comparison_id"]}
        for row in original
        if row["model_key"] == key
    ]
    rows[0]["trajectory"][0]["observation"] = "supplementary " + "x" * 2000 + " full tail"
    run_dir = tmp_path / "supplementary"
    _write(run_dir / "manifest.json", manifest)
    _write(run_dir / "results.json", rows)
    report = json.loads(SCORE.read_text())
    report.update(
        run_id=manifest["run_id"],
        results_hash=configuration_hash({"results": rows}),
        results_sha256=_file_sha256(run_dir / "results.json"),
    )
    report["models"] = {key: report["models"][key]}
    report["rows"] = [row for row in report["rows"] if row["model_key"] == key]
    score_path = tmp_path / "supplementary-score.json"
    _write(score_path, report)
    return run_dir, score_path


def _options() -> dict:
    return dict(
        run_dir=RELEASE / "run",
        review_dir=RELEASE / "review-prepared",
        judged_review_path=RELEASE / "review-model-assisted/review.json",
        provisional=True,
    )


def test_supplementary_score_and_trace_are_separate_from_frozen_pair(supplementary) -> None:
    run_dir, score_path = supplementary
    baseline = build_demo_payload(**_options())
    store = {}
    payload = build_demo_payload(
        **_options(),
        supplementary_runs=[run_dir],
        qasper_scores=[score_path],
        verified_trace_store=store,
    )
    assert payload["active"] == baseline["active"]
    extra = payload["supplementary_runs"][0]
    assert len(extra["models"]) == 1
    model = extra["models"][0]
    assert model["metrics"]["pass"] is None
    assert model["metrics"]["qasper_answer_f1"] == pytest.approx(0.1991537448471281)
    assert extra["provenance"]["review_kind"] is None
    assert not any("qasper_answer_f1" in m["metrics"] for m in payload["active"]["models"])
    question = extra["cases"][0]["id"]
    key = model["key"]
    assert store[("separate-run", question, key)]["trajectory"][0]["observation"].endswith(
        " full tail"
    )
    assert store[(question, key)] != store[("separate-run", question, key)]
    rendered = snapshot_html(payload, verified_trace_store=store)
    embedded = rendered.split('<script id="envoybench-data" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    exported = json.loads(embedded)["supplementary_runs"][0]
    assert exported["trace_scope"] == "full_saved_trajectories"
    assert exported["cases"][0]["systems"][key]["trajectory"][0]["observation"].endswith(
        " full tail"
    )
    assert "QASPER Agent Studio" in rendered
    assert "Official QASPER Answer F1 (0–100)" in rendered

    client = TestClient(
        create_app(**_options(), supplementary_runs=[run_dir], qasper_scores=[score_path])
    )
    params = dict(question_id=question, model=key, run_id="separate-run")
    response = client.get("/api/demo/trace", params=params)
    assert response.status_code == 200
    assert response.json()["trajectory"][0]["observation"].endswith(" full tail")
    assert client.get("/api/demo/trace", params={**params, "run_id": "missing"}).status_code == 404
    original = client.get("/api/demo/trace", params=dict(question_id=question, model=key))
    assert original.json() != response.json()


def test_official_score_attaches_only_when_explicitly_loaded() -> None:
    before = build_demo_payload(**_options())
    after = build_demo_payload(
        **_options(), qasper_scores=[SCORE], token_diagnostic_run=RELEASE / "token-diagnostic-smoke"
    )
    assert [model["metrics"]["pass"] for model in after["active"]["models"]] == [
        model["metrics"]["pass"] for model in before["active"]["models"]
    ]
    assert [
        model["metrics"]["qasper_answer_f1"] for model in after["active"]["models"]
    ] == pytest.approx([0.1991537448471281, 0.30687301402774547])
    assert all("metrics" not in model for model in after["token_diagnostic"]["models"])


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda report: report.update(results_hash="wrong"), "results_hash"),
        (lambda report: report["rows"].pop(), "missing question/model"),
        (lambda report: report["rows"][0].update(answer_f1=float("nan")), "invalid or duplicate"),
        (lambda report: report["rows"][0].update(predicted_answer="different"), "saved answer"),
    ],
)
def test_score_rejects_mismatched_artifacts(supplementary, change, match) -> None:
    run_dir, score_path = supplementary
    report = copy.deepcopy(json.loads(score_path.read_text()))
    change(report)
    _write(score_path, report)
    with pytest.raises(ValueError, match=match):
        build_demo_payload(**_options(), supplementary_runs=[run_dir], qasper_scores=[score_path])


def test_supplementary_run_does_not_relax_paired_or_blind_review(supplementary, tmp_path) -> None:
    run_dir, _ = supplementary
    with pytest.raises(ValueError, match="at least two models"):
        build_demo_payload(run_dir=run_dir, provisional=True)
    with pytest.raises(ValueError, match="blind human review"):
        create_app(supplementary_runs=[run_dir], blind_review_dir=tmp_path / "blind")


def test_saved_nebius_budget_stop_is_visible_without_a_full_score() -> None:
    run_dir = ROOT / "release/qasper-agent-study/nebius-run"
    store = {}
    payload = build_demo_payload(
        **_options(), supplementary_runs=[run_dir], qasper_scores=[SCORE],
        verified_trace_store=store,
    )
    run = payload["supplementary_runs"][0]
    key = "nemotron_ultra_nebius"
    assert run["run_status"] == "incomplete"
    assert run["completion"] == {
        "planned": 40, "recorded": 39, "finished": 38,
        "interrupted": 1, "not_attempted": 1,
    }
    assert run["usage_budget"]["estimated_usd"] == 1.947862
    assert run["models"][0]["metrics"]["pass"] is None
    assert run["models"][0]["metrics"].get("qasper_answer_f1") is None
    assert run["cases"][-2]["systems"][key]["status"] == "error"
    assert run["cases"][-1]["systems"][key]["status"] == "not_attempted"
    assert run["cases"][-1]["systems"][key]["trajectory"] == []
    assert (run["provenance"]["run_id"], run["cases"][-1]["id"], key) not in store
    html = snapshot_html(payload, verified_trace_store=store)
    assert "not_attempted" in html and "budget stop" in html
