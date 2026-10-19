import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from scripts.export_ml_experiment_case_study import SELECTED, build_export
from src.product.ml_experiment_evidence import EvidenceResolver
from src.product.ml_experiment_models import ExperimentBrief
from src.product.ml_experiment_types import LocalArtifact

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/ml-experiment-qwen-v5"
EXPORT = EXAMPLE / "exports/qwen-v5-task5-development.json"


def test_development_export_matches_frozen_parents_and_preserves_unknowns():
    if not (ROOT / "benchmarks/envoybench/data/test_candidate/corpus").is_dir():
        pytest.skip("Rebuilding requires the pinned local QASPER corpus; never download in tests")
    rebuilt = build_export()
    assert rebuilt == json.loads(EXPORT.read_text())
    assert rebuilt["summary"]["models"]["qwen_base"]["answerable_passes"] == 3
    assert rebuilt["summary"]["models"]["qwen_v5"]["answerable_passes"] == 3
    assert rebuilt["summary"]["timing"]["qwen_base"]["positive_duration_count"] == 29
    assert rebuilt["measurements"]["qwen_base"]["latency"]["value"] is None
    assert rebuilt["measurements"]["qwen_v5"]["cost"]["value"] is None
    assert all(case["existing_review"]["kind"] == "model_assisted"
               for case in rebuilt["cases"].values())
    assert all(len(json.dumps(case, ensure_ascii=False)) <= 4_000
               for case in rebuilt["cases"].values())


def test_example_is_a_specific_unrun_proposal_not_a_model_improvement_claim():
    brief = ExperimentBrief.model_validate_json((EXAMPLE / "proposal.json").read_text())
    assert brief.evaluation.purpose == "development"
    assert set(brief.evaluation.sample_ids) == set(SELECTED.values())
    assert brief.candidate_outcomes_already_seen == "no"
    assert not brief.missing_fields()
    prompts = json.loads((EXAMPLE / "review-prompts.json").read_text())
    for label in ("baseline", "candidate"):
        expected = hashlib.sha256(prompts[label].encode()).hexdigest()
        assert getattr(brief, label).identity["prompt_sha256"].value == expected
    assert brief.baseline.identity["model_revision"] == brief.candidate.identity["model_revision"]
    resolver = EvidenceResolver(library_root=ROOT / "out/unused", artifact_root=EXAMPLE / "exports")
    receipts = resolver([ref for ref in brief.references if isinstance(ref, LocalArtifact)])
    assert len(receipts) == 7
    assert all(receipt.semantic_support == "not_reviewed" for receipt in receipts)
    claims = json.loads((EXAMPLE / "research-claims.json").read_text())
    assert len(claims) == 3 and all(c["independent_human_review"] == "pending" for c in claims)


def test_real_example_roundtrip_leaves_proposal_draft_and_history_inconclusive(tmp_path):
    pytest.importorskip("mcp")
    from scripts.demo_ml_experiment_workflow import run_demo
    from src.product.ml_experiment_store import ExperimentStore

    library = ROOT / "out/arxiv-mcp-live-library"
    versions = [r["arxiv_id"] for r in json.loads((EXAMPLE / "research-claims.json").read_text())]
    if any(not (library / "papers" / v / "document.json").is_file() for v in versions):
        pytest.skip("Demonstration needs the three pinned paper caches; never fetch in tests")
    report = asyncio.run(run_demo(store_root=tmp_path / "records",
        artifact_root=tmp_path / "exports", library_root=library))
    assert report["restart_verified"] and report["retry_verified"]
    assert report["proposal"]["status"] == "draft" and not report["proposal"]["results"]
    assert report["historical"]["assessment"]["overall"] == "inconclusive"
    store = ExperimentStore(tmp_path / "records")
    result = store.get(report["historical"]["experiment_id"])["selected_result"]
    proxy = next(m for m in result["result"]["baseline"]["measurements"]
                 if m["metric_id"] == "provisional_pass_rate")
    assert proxy["review"]["kind"] == "model_assisted"
    assert report["model_requests"] == report["gpu_launches"] == 0
