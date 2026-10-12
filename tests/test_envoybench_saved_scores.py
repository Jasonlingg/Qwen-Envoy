"""Supplementary runs and official F1 never alter the frozen paired review."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from benchmarks.envoybench import saved_payload
from benchmarks.envoybench.demo import build_demo_payload, create_app
from benchmarks.envoybench.export import snapshot_html
from benchmarks.envoybench.run import _file_sha256
from src.eval.artifacts import configuration_hash

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release/envoybench-v0.1"
SCORE = ROOT / "release/qasper-agent-study/qwen-official-score.json"
NEMOTRON_SCORE = ROOT / "release/qasper-agent-study/nemotron-official-score.json"
AMENDED_RUN = ROOT / "release/qasper-agent-study/nebius-amended-run"
RETRIEVAL_RUN = ROOT / "release/qasper-agent-study/retrieval-baseline"


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


def test_completed_bm25_baseline_is_visible_and_traceable_on_provisional_first_page() -> None:
    store = {}
    payload = build_demo_payload(**_options(), verified_trace_store=store)
    card = payload["active"]["retrieval_baseline"]
    assert card["status"] == "complete"
    assert card["question_count"] == 40
    assert card["model_request_count"] == 80
    assert [model["qasper_answer_f1"] for model in card["models"]] == pytest.approx(
        [0.4151989695911006, 0.5886030262883601]
    )
    assert [model["answerable_answer_f1"] for model in card["models"]] == pytest.approx(
        [0.330397939182201, 0.17720605257672012]
    )
    assert [(model["correct_refusals"], model["false_refusals"])
            for model in card["models"]] == [(10, 3), (20, 9)]
    assert "support is unreviewed" in card["caveat"]
    run = next(run for run in payload["supplementary_runs"]
               if run["provenance"]["run_id"] == card["run_id"])
    assert run["kind"] == "retrieval_baseline_run"
    assert run["architecture"] == "envoybench-known-paper-bm25-v1"
    assert len(run["cases"]) == 40
    assert run["models"][1]["metrics"]["qasper_answer_f1"] == pytest.approx(
        card["models"][1]["qasper_answer_f1"]
    )
    case = run["cases"][0]
    system = case["systems"]["qwen_v5_retrieval"]
    assert system["retrieval_passages"][0]["passage_id"].startswith("qasper_")
    assert system["prompt_tokens_local"] > 0
    assert system["trajectory"][0]["observation"] == (
        "Single answer request; no tool execution."
    )
    assert (card["run_id"], case["id"], "qwen_v5_retrieval") in store

    client = TestClient(create_app(**_options()))
    response = client.get("/api/demo/trace", params={
        "question_id": case["id"], "model": "qwen_v5_retrieval", "run_id": card["run_id"],
    })
    assert response.status_code == 200
    assert response.json()["trajectory"][0]["observation"] == (
        "Single answer request; no tool execution."
    )


def test_completed_bm25_requires_matching_provisional_source_and_keeps_prepared_fallback(
    monkeypatch, tmp_path: Path,
) -> None:
    fixture = build_demo_payload()
    assert fixture["supplementary_runs"] == []
    assert fixture["active"].get("retrieval_baseline") is None

    monkeypatch.setattr(saved_payload, "RELEASE_RETRIEVAL_BASELINE", tmp_path / "absent")
    payload = build_demo_payload(**_options())
    assert payload["active"]["retrieval_baseline"]["status"] == "prepared_not_run"
    assert payload["supplementary_runs"] == []

    manifest = json.loads((RELEASE / "run/manifest.json").read_text())
    manifest["run_id"] = "different-source-run"
    assert saved_payload._completed_retrieval_payload(
        manifest, saved_payload.DEFAULT_DATASET, RETRIEVAL_RUN, None,
    ) is None


def test_completed_bm25_rejects_changed_frozen_packets(tmp_path: Path) -> None:
    for name in ("plan", "manifest", "packets", "results", "official-score"):
        shutil.copyfile(RETRIEVAL_RUN / f"{name}.json", tmp_path / f"{name}.json")
    packets_path = tmp_path / "packets.json"
    packets = json.loads(packets_path.read_text())
    packets["packets"][0]["passages"][0]["text"] += " changed"
    _write(packets_path, packets)
    source = json.loads((RELEASE / "run/manifest.json").read_text())
    with pytest.raises(ValueError, match="manifest or packets differ"):
        saved_payload._completed_retrieval_payload(
            source, saved_payload.DEFAULT_DATASET, tmp_path, None,
        )


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


def test_amended_nebius_run_shows_lineage_cost_and_full_official_f1() -> None:
    payload = build_demo_payload(
        **_options(), supplementary_runs=[AMENDED_RUN],
        qasper_scores=[SCORE, NEMOTRON_SCORE],
    )
    run = payload["supplementary_runs"][0]
    amendment = run["amended_continuation"]
    score = json.loads(NEMOTRON_SCORE.read_text())
    key = "nemotron_ultra_nebius"
    assert run["run_status"] == "complete"
    assert len(run["cases"]) == 40
    assert amendment["original_completed_count"] == 38
    assert amendment["original_budget_cap_usd"] == 2.0
    assert amendment["amended_budget_cap_usd"] == 25.0
    assert amendment["restarted_question_id"] == run["cases"][38]["id"]
    assert amendment["first_attempt_question_id"] == run["cases"][39]["id"]
    assert run["cases"][0]["episode_origin"] == "Original completed episode"
    assert "restarted" in run["cases"][38]["episode_origin"]
    assert "first attempt" in run["cases"][39]["episode_origin"]
    assert run["usage_budget"]["estimated_usd"] == pytest.approx(2.035422)
    assert run["models"][0]["metrics"]["qasper_answer_f1"] == score["models"][key]["answer_f1"]
    assert run["models"][0]["metrics"]["pass"] is None
    assert run["provenance"]["review_kind"] is None
    assert "Amended continuation" in snapshot_html(payload)
