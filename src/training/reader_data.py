"""Load the frozen reader pool and score its source-reviewed short-fact contract."""

import copy
from pathlib import Path

from src.eval.reader_improvement import digest, file_hash, inside, load
from src.eval.reader_rl_audit import submit
from src.eval.reader_rl_validation import VERSION, assess, check_keys
from src.product.passage_scope import PASSAGES_V2, make_scope
from src.research.agent import load_snapshot
from src.research.qasper import QASPER_REVISION

ROOT = Path(__file__).resolve().parents[2]
REWARD_VERSION = "reader-short-fact-reference-v1"
PILOT_REVIEW = ROOT / "examples/reader-rl-pilot-v1/training-review.json"
TRAINING_REVISION = ROOT / "examples/reader-rl-pilot-v2/reward-revision.json"
REVISED_REWARD_VERSION = "reader-short-fact-reference-v2"


def reviewed_training_ids(bundle, roles, cards, review_path=PILOT_REVIEW):
    """Apply attributed pilot holds without modifying the frozen source/reward keys.

    Validates the review's bindings, not the correctness of its semantic judgments.
    """
    review = load(review_path)
    if (review.get("schema_version") != "reader-pilot-source-excerpt-review-v1"
            or review.get("bundle_manifest_sha256") != file_hash(bundle / "manifest.json")
            or review.get("reviewer_kind") != "model_assisted"
            or not review.get("reviewer_identity") or not review.get("scope")
            or review.get("human_review") is not None
            or review.get("locked_evaluation_references_opened") is not False):
        raise ValueError("source-bound, attributed training review required")
    expected = {c["case_id"] for c in roles["train"]}
    rows = review.get("rows", [])
    if (len(rows) != len(expected) or {r.get("case_id") for r in rows} != expected):
        raise ValueError("review must cover each training question exactly once")
    selected = []
    for row in rows:
        card = cards[row["case_id"]]
        if (row.get("card_sha256") != digest(card)
                or row.get("context_sha256") != card["context_sha256"]
                or not row.get("reason")
                or row.get("decision") not in {"retain_for_pilot", "hold"}):
            raise ValueError("training review has stale bindings or an invalid decision")
        if row["decision"] == "retain_for_pilot":
            selected.append(row["case_id"])
    if not selected:
        raise ValueError("no reviewed training questions remain")
    return selected


def check_references(inputs, keys):
    """Reuse the unchanged diagnostic's structural/source-binding checks only."""
    if keys.get("reward_version") != REWARD_VERSION or keys.get("usage") != "rl_reference_keys":
        raise ValueError("wrong short-fact reward reference version")
    return check_keys(
        {**keys, "schema_version": VERSION, "usage": "exposed_diagnostic_only"}, inputs
    )


def score_episode(episode, context, card):
    if episode.get("infrastructure_ok") is not True:
        raise ValueError("discard entire group on infrastructure failure; never zero-reward it")
    packet = episode["packet"]
    if (
        episode["question_id"] != context["question_id"]
        or packet["question"] != context["question"]
    ):
        raise ValueError("episode/question mismatch is an infrastructure error, not zero reward")
    if packet["status"] != "evidence_found" or episode.get("verified") is None:
        return {"eligible": True, "reward": 0.0, "reason": "no_verified_supported_terminal"}
    answer = episode["verified"]["candidate_answer"]
    spans = [{k: s[k] for k in ("doc_id", "start", "end")} for s in packet["evidence"]]
    return assess(
        {
            "context_id": card["context_id"],
            "action": submit(answer, context["doc_id"], spans),
            "inspected_windows": packet["inspected_passage_windows"],
        },
        context,
        card,
    )


def load_bundle(bundle):
    manifest = load(bundle / "manifest.json")
    if (
        manifest["schema_version"] != "reader-rl-fact-pool-v1"
        or manifest["source_revision"] != QASPER_REVISION
        or manifest["reward_version"] != REWARD_VERSION
    ):
        raise ValueError("unknown or unpinned RL fact pool")
    for name, expected in manifest["artifact_sha256"].items():
        if file_hash(inside(bundle, name)) != expected:
            raise ValueError("frozen RL source/reference file changed: " + name)
    inputs, cases = load(bundle / "inputs.json"), load(bundle / "cases.json")
    cards = check_references(inputs, load(bundle / "keys.json"))
    roles = {
        role: [c for c in cases if c["role"] == role]
        for role in ("train", "reward_validation", "model_evaluation_locked")
    }
    papers = {role: {c["doc_id"] for c in rows} for role, rows in roles.items()}
    if any(
        papers[a] & papers[b]
        for a, b in (
            ("train", "reward_validation"),
            ("train", "model_evaluation_locked"),
            ("reward_validation", "model_evaluation_locked"),
        )
    ):
        raise ValueError("RL roles overlap at paper level")
    history = load(bundle / "history.json")
    if (
        "question_to_paper" not in history
        or history.get("raw_source_question_ids_resolved") is not True
    ):
        raise ValueError("question-only prior history must be closed to whole papers")
    if load(bundle / "historical-evaluation-reservations/questions.json") != history_reservations(
        history
    ):
        raise ValueError("future inventory reservations differ from prior question/paper closure")
    for role, rows in roles.items():
        count_key = {
            "train": "train_questions",
            "reward_validation": "reward_validation_questions",
            "model_evaluation_locked": "model_evaluation_questions",
        }[role]
        if len(rows) != manifest[count_key]:
            raise ValueError("RL role count differs from frozen manifest")
        if len(papers[role]) != len(rows) or len({c["question_id"] for c in rows}) != len(rows):
            raise ValueError("expected one unique question per paper")
        excluded = (
            history["excluded_training_paper_ids"]
            if role == "train"
            else history["all_prior_paper_ids"]
        )
        if papers[role] & set(excluded):
            raise ValueError("prior evaluation/validation paper entered a forbidden role")
        excluded_qids = (
            history["excluded_training_question_ids"]
            if role == "train"
            else history["all_prior_question_ids"]
        )
        if {c["question_id"] for c in rows} & set(excluded_qids):
            raise ValueError("prior question-only history entered a forbidden role")
        if any(c["source_split"] != ("train" if role == "train" else "validation") for c in rows):
            raise ValueError("official question split differs from RL role")
        for case in rows:
            split = "train" if role == "train" else "validation"
            if not case["question_id"].startswith("qasper_" + split + "_"):
                raise ValueError("source question identity differs from official split")
            _, docs = load_snapshot(inside(bundle, case["snapshot"]))
            if (
                set(docs) != {case["doc_id"]}
                or docs[case["doc_id"]]["metadata"]["source_split"] != split
                or docs[case["doc_id"]]["metadata"]["source_revision"] != QASPER_REVISION
                or make_scope(docs, case["scope"]["windows"], version=PASSAGES_V2) != case["scope"]
            ):
                raise ValueError("source snapshot/scope identity differs from case")
            if role != "model_evaluation_locked":
                context = inputs["contexts"][case["case_id"]]
                if (
                    context["docs"] != docs
                    or context["question"] != case["question"]
                    or context["windows"] != case["scope"]["windows"]
                    or cards[case["case_id"]]["role"] != role
                ):
                    raise ValueError("reward source/question/role differs from runtime case")
    if {c["case_id"] for c in roles["train"] + roles["reward_validation"]} != set(cards):
        raise ValueError("missing or extra RL reward reference cards")
    lock = load(bundle / "model-evaluation-lock.json")
    if lock["question_ids"] != [c["question_id"] for c in roles["model_evaluation_locked"]] or lock[
        "paper_ids"
    ] != [c["doc_id"] for c in roles["model_evaluation_locked"]]:
        raise ValueError("model-evaluation lock differs from exported cases")
    return manifest, roles, inputs["contexts"], cards


def load_training_bundle(bundle, revision=None):
    """Opt-in, source-bound training aliases/holds; the historical bundle stays intact.

    This validates review provenance and structure, not semantic correctness.
    Validation/evaluation cards, support spans and the scoring algorithm cannot change.
    """
    manifest, roles, contexts, cards = load_bundle(bundle)
    selected = reviewed_training_ids(bundle, roles, cards)
    if revision is None:
        return manifest, roles, contexts, cards, selected, REWARD_VERSION
    review = load(revision)
    audit_path = inside(ROOT, review["audit_manifest"])
    audit = load(audit_path)
    if (review.get("schema_version") != "reader-training-alias-revision-v1"
            or review.get("parent_bundle_sha256") != file_hash(bundle / "manifest.json")
            or review.get("parent_training_review_sha256") != file_hash(PILOT_REVIEW)
            or review.get("parent_reward_version") != REWARD_VERSION
            or review.get("reward_version") != REVISED_REWARD_VERSION
            or review.get("audit_manifest_sha256") != file_hash(audit_path)
            or review.get("reviewer_kind") != "model_assisted"
            or not review.get("reviewer_identity") or not review.get("reviewed_at")
            or review.get("human_review") is not None
            or review.get("locked_evaluation_references_opened") is not False):
        raise ValueError("attributed, source-bound training revision required")
    for name, expected in audit["artifact_sha256"].items():
        if file_hash(inside(audit_path.parent, name)) != expected:
            raise ValueError("training wording audit artifact changed")
    judgments = {r["row_id"]: r for r in load(audit_path.parent / "answer-review.json")}
    proposals = load(audit_path.parent / "proposed-aliases.json")
    train = {c["case_id"] for c in roles["train"]}
    revised, seen = copy.deepcopy(cards), set()
    for change in review.get("changes", []):
        cid = change["case_id"]
        if (cid not in train or cid in seen
                or change.get("original_card_sha256") != digest(cards[cid])
                or not change.get("aliases")):
            raise ValueError("only unique source-bound training cards may gain aliases")
        seen.add(cid)
        for alias in change["aliases"]:
            row = judgments.get(alias.get("source_review_row_id"), {})
            proposal = {"case_id": cid, **alias}
            if (proposal not in proposals or row.get("case_id") != cid
                    or row.get("classification") != "complete_source_supported_answer"
                    or row.get("candidate_answer") != alias["variant"]["answer"]):
                raise ValueError("alias lacks its complete-answer source review")
            revised[cid]["accepted_answers"].append(copy.deepcopy(alias["variant"]))
    held = set()
    for hold in review.get("holds", []):
        cid = hold["case_id"]
        if (cid not in selected or cid in held or cid in seen
                or hold.get("original_card_sha256") != digest(cards[cid])
                or not hold.get("reason")):
            raise ValueError("training hold is stale, duplicate or changes a revised card")
        held.add(cid)
    if not seen or not held:
        raise ValueError("revision must declare its aliases and training holds")
    keys = load(bundle / "keys.json")
    keys["cards"] = [revised[c["context_id"]] for c in keys["cards"]]
    # Keep the original structural validator, including finite complete variants.
    check_references(load(bundle / "inputs.json"), keys)
    return (manifest, roles, contexts, revised,
            [cid for cid in selected if cid not in held], REVISED_REWARD_VERSION)


def history_reservations(history):
    excluded = set(history["excluded_training_question_ids"])
    return {
        "schema_version": "conservative-prior-question-paper-reservations-v1",
        "questions": [
            {"id": qid, "target_doc_ids": [doc]}
            for qid, doc in sorted(history["question_to_paper"].items())
            if qid in excluded
        ],
        "scope": "Prior evaluation/validation or conservatively excluded unknown-purpose "
        "metadata, not a claim every question was evaluated or trained. Register these "
        "canonical question/paper pairs for future identifier-only inventory.",
    }
