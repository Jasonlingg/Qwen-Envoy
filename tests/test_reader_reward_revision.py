"""Training-only reward revision: source bindings, strict support and opt-in transport."""

import copy
from pathlib import Path

import pytest

from src.eval.reader_improvement import digest, file_hash, load, write
from src.eval.reader_rl_audit import submit
from src.eval.reader_rl_validation import assess
from src.training.prime_run import prepare, validate_plan
from src.training.reader_data import (
    REVISED_REWARD_VERSION,
    REWARD_VERSION,
    TRAINING_REVISION,
    load_bundle,
    load_training_bundle,
    reviewed_training_ids,
)

BUNDLE = Path("data/rl/reader-facts-v1")
PROTOCOL = Path("examples/reader-rl-pilot-v2/protocol.json")
HELD = "qasper_train_65ad17f614b7345f0077424c04c94971c831585b--rl-short-fact"


def test_revision_changes_only_reviewed_training_variants_and_preserves_all_sources():
    original = load_bundle(BUNDLE)
    manifest, roles, contexts, cards, selected, version = load_training_bundle(
        BUNDLE, TRAINING_REVISION
    )
    assert (manifest, roles, contexts) == original[:3]
    assert version == REVISED_REWARD_VERSION and len(selected) == 80
    assert HELD not in selected
    assert len(set(reviewed_training_ids(BUNDLE, original[1], original[3])) - set(selected)) == 1
    revised = {cid for cid in cards if cards[cid] != original[3][cid]}
    assert len(revised) == 4
    added = 0
    for cid in cards:
        before, after = original[3][cid], cards[cid]
        assert {k: v for k, v in before.items() if k != "accepted_answers"} == {
            k: v for k, v in after.items() if k != "accepted_answers"
        }
        assert (
            after["accepted_answers"][: len(before["accepted_answers"])]
            == before["accepted_answers"]
        )
        added += len(after["accepted_answers"]) - len(before["accepted_answers"])
        if before["role"] != "train":
            assert before == after
    assert added == 6
    assert load_training_bundle(BUNDLE)[-1] == REWARD_VERSION
    assert load_bundle(BUNDLE) == original  # No mutation of historical keys.


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "validation",
        "locked",
        "unknown_alias",
        "duplicate",
        "anonymous",
        "hold_alias",
        "missing_hold",
    ],
)
def test_revision_rejects_reference_drift_or_unreviewed_changes(tmp_path, mutation):
    revision = copy.deepcopy(load(TRAINING_REVISION))
    _, roles, _, cards = load_bundle(BUNDLE)
    if mutation == "stale":
        revision["parent_bundle_sha256"] = "0" * 64
    elif mutation in {"validation", "locked"}:
        role = "reward_validation" if mutation == "validation" else "model_evaluation_locked"
        revision["changes"][0]["case_id"] = roles[role][0]["case_id"]
    elif mutation == "unknown_alias":
        revision["changes"][0]["aliases"][0]["variant"]["answer"] += " unsupported addition"
    elif mutation == "duplicate":
        revision["changes"].append(copy.deepcopy(revision["changes"][0]))
    elif mutation == "anonymous":
        revision["reviewer_identity"] = ""
    elif mutation == "hold_alias":
        cid = revision["changes"][0]["case_id"]
        revision["holds"] = [
            {"case_id": cid, "original_card_sha256": digest(cards[cid]), "reason": "fixture"}
        ]
    else:
        revision["holds"] = []
    path = tmp_path / "revision.json"
    write(path, revision)
    with pytest.raises(ValueError):
        load_training_bundle(BUNDLE, path)


def test_new_wording_still_requires_actual_inspected_support_and_rejects_wrong_answers():
    _, _, contexts, cards, _, _ = load_training_bundle(BUNDLE, TRAINING_REVISION)
    _, _, _, old = load_bundle(BUNDLE)
    cid = next(
        cid
        for cid, card in cards.items()
        if any(v["answer"] == "300 hours" for v in card["accepted_answers"])
    )
    context, card = contexts[cid], cards[cid]
    support = card["claims"][0]["support"][0]
    span = {"doc_id": context["doc_id"], "start": support["start"], "end": support["end"]}
    record = {
        "context_id": cid,
        "action": submit("300 hours", context["doc_id"], [span]),
        "inspected_windows": context["windows"],
    }
    assert assess(record, context, old[cid])["reason"] == "answer_not_in_reviewed_variants"
    assert assess(record, context, card)["reward"] == 1
    assert assess({**record, "inspected_windows": []}, context, card)["reward"] == 0
    wrong_span = {**span, "end": span["start"] + 5}
    assert (
        assess(
            {**record, "action": submit("300 hours", context["doc_id"], [wrong_span])},
            context,
            card,
        )["reward"]
        == 0
    )
    for wrong in ["300 days", "30 hours", "300 hours plus an unsupported assertion"]:
        assert (
            assess({**record, "action": submit(wrong, context["doc_id"], [span])}, context, card)[
                "reward"
            ]
            == 0
        )


def test_prepared_pilot_binds_opt_in_revision_and_rejects_caps_or_selection_drift(tmp_path):
    protocol = load(PROTOCOL)
    run = tmp_path / "run"
    phase = protocol["phases"]["training"]
    plan = prepare(
        run,
        tmp_path / "assets",
        updates=10,
        requests=1024,
        tokens=8388608,
        seconds=phase["max_work_seconds"],
        pilot_protocol=PROTOCOL,
    )
    assert validate_plan(run) == plan
    assert plan["reward_version"] == REVISED_REWARD_VERSION
    assert plan["reward_revision_sha256"] == file_hash(TRAINING_REVISION)
    assert len(plan["train_case_ids"]) == 80 and HELD not in plan["train_case_ids"]
    for key, value in [
        ("reward_revision_sha256", "0" * 64),
        ("train_case_ids", plan["train_case_ids"] + [HELD]),
    ]:
        changed = {**plan, key: value}
        write(run / "plan.json", changed)
        with pytest.raises(ValueError):
            validate_plan(run)
    with pytest.raises(ValueError, match="prospective"):
        prepare(
            tmp_path / "wrong-caps",
            tmp_path / "assets",
            updates=10,
            seconds=3600,
            pilot_protocol=PROTOCOL,
        )
    assert not (tmp_path / "wrong-caps").exists()


def test_revised_taskset_excludes_holds_and_carries_source_bound_reward_to_verifiers():
    adapter = pytest.importorskip(
        "src.training.verifiers_adapter",
        reason="requires isolated Verifiers 0.4.0 environment",
        exc_type=ImportError,
    )

    config = adapter.ReaderTasksetConfig(reward_revision=TRAINING_REVISION)
    tasks = list(adapter.ReaderTaskset(config))
    assert len(tasks) == 80 and HELD not in {t.data.case_id for t in tasks}
    chosen = next(t.data for t in tasks if t.data.case_id.startswith("qasper_train_7efbe48"))
    _, _, _, card = adapter.task_case(chosen)
    assert "300 hours" in {v["answer"] for v in card["accepted_answers"]}
    with pytest.raises(ValueError, match="held"):
        adapter.task_case(chosen.model_copy(update={"case_id": HELD}))
    with pytest.raises(ValueError, match="revision changed"):
        adapter.task_case(chosen.model_copy(update={"reward_revision_sha256": "0" * 64}))
