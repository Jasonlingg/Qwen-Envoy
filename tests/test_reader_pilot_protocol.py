"""Pre-run artifact consistency; no locked reference decoding or model calls."""
from pathlib import Path

from src.eval.passage_pilot_replay import PROMPT_SHA
from src.eval.reader_improvement import file_hash, load
from src.training.reader_data import PILOT_REVIEW, load_bundle, reviewed_training_ids


def test_protocol_pins_reviewed_train_pool_and_original_disjoint_evaluation_ids():
    bundle = Path("data/rl/reader-facts-v1")
    _, roles, _, cards = load_bundle(bundle)
    protocol = load(Path("examples/reader-rl-pilot-v1/protocol.json"))
    assert protocol["identities"]["bundle_manifest_sha256"] == file_hash(bundle / "manifest.json")
    assert protocol["identities"]["training_review_sha256"] == file_hash(PILOT_REVIEW)
    assert protocol["identities"]["prompt_sha256"] == PROMPT_SHA
    assert protocol["training_case_ids"] == reviewed_training_ids(bundle, roles, cards)
    assert protocol["evaluation_question_ids"] == [
        c["question_id"] for c in roles["model_evaluation_locked"]]
    assert protocol["evaluation_paper_ids"] == [
        c["doc_id"] for c in roles["model_evaluation_locked"]]
    held = set(protocol["held_training_case_ids"])
    assert held.isdisjoint(protocol["training_case_ids"])
    assert held.isdisjoint(c["case_id"] for c in roles["model_evaluation_locked"])
    assert protocol["locked_evaluation_references_opened"] is False
    assert protocol["budget_proposal"]["paid_authorization"] is False
    assert protocol["review"]["independent_human_review"] is None


def test_proposed_phase_caps_fit_total_workload_and_do_not_claim_quality_from_reward():
    protocol = load(Path("examples/reader-rl-pilot-v1/protocol.json"))
    phases = protocol["phases"]
    assert sum(p["max_work_seconds"] for p in phases.values()) <= (
        protocol["budget_proposal"]["maximum_total_owned_GPU_lifetime_seconds"])
    assert phases["training"]["initialization"] == "original_pinned_v5_not_the_integration_adapter"
    assert protocol["success_rule"]["training_reward_or_answer_f1_alone_sufficient"] is False
    assert protocol["success_rule"]["required_complete_paired_episodes"] == (
        len(protocol["evaluation_question_ids"]) * 2)


def test_current_manifest_binds_local_repair_without_held_out_model_results():
    from src.training.reader_data import ROOT

    record = load(ROOT / "examples/reader-rl-pilot-v1/SHA256SUMS.json")
    assert record["model_outputs_observed"] is True
    assert record["held_out_model_outputs_observed"] is False
    assert record["stage"].startswith("local_repair_after_failed_training_integration")
    assert record["locked_reference_contents_reviewed"] is False
    # v1 is a historical source receipt. The explicit v2 successor binds current
    # implementations without rewriting the source hashes of earlier runs.
    current = load(ROOT / "examples/reader-rl-pilot-v2/SHA256SUMS.json")
    assert current["parent_pilot_receipt_sha256"] == file_hash(
        ROOT / "examples/reader-rl-pilot-v1/SHA256SUMS.json")
    for name, sha in record["artifacts"].items():
        expected = current["artifacts"][name] if name in current["superseded_source_paths"] else sha
        assert file_hash(ROOT / name) == expected, name
    assert record["artifacts"]["examples/reader-rl-pilot-v1/protocol.json"] == (
        "1d0038b7c5d2f84327ac1bdcf55d939bcbdbfcbf1d5a991d07ef81ffb05aa8d9")


def test_v2_amendment_keeps_evaluation_fixed_and_fits_its_declared_phase_budget():
    from src.training.reader_data import TRAINING_REVISION, load_training_bundle

    old = load(Path("examples/reader-rl-pilot-v1/protocol.json"))
    current = load(Path("examples/reader-rl-pilot-v2/protocol.json"))
    *_, selected, version = load_training_bundle(Path("data/rl/reader-facts-v1"), TRAINING_REVISION)
    assert current["training_case_ids"] == selected
    assert current["identities"]["reward_version"] == version
    assert current["identities"]["reward_revision_sha256"] == file_hash(TRAINING_REVISION)
    for key in ("evaluation_question_ids", "evaluation_paper_ids", "success_rule", "review"):
        assert current[key] == old[key]
    assert current["phases"]["paired_evaluation"] == old["phases"]["paired_evaluation"]
    budget = current["budget_proposal"]
    assert sum(p["max_work_seconds"] for p in current["phases"].values()) + budget[
        "maximum_setup_export_cleanup_seconds"] == budget[
            "maximum_total_owned_GPU_lifetime_seconds"]
    assert budget["paid_authorization"] is False
    assert budget["maximum_total_owned_GPU_lifetime_seconds"] * budget[
        "maximum_combined_hourly_rate_usd"] / 3600 <= budget[
            "estimated_cost_ceiling_usd"] * budget["watchdog_spend_fraction"]
