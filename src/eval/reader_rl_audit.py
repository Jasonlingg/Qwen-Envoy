"""Offline reward counterexamples, never a deployed reward or training launcher."""
from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

from src.eval.passage_pilot_replay import PROMPT_SHA, verify_case
from src.eval.qasper_reward import normalize, parse_strict, score_submission
from src.eval.reader_improvement import digest, file_hash, inside, load, write
from src.product.packet_verifier import MAX_SPAN_CHARS
from src.product.passage_scope import contains
from src.product.reader_gaps import parse_gap_submission

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "examples/passage-reader-pilot-data"
REGRESSION = ROOT / "examples/reader-improvement-loop"
HYBRID = ROOT / "examples/qasper-hybrid-v5"
SCORERS = ("legacy_overlap", "scoped_overlap", "scoped_exact")


def checked_bundle(bundle):
    validation = load(bundle / "validation.json")
    for name, expected in validation["artifact_sha256"].items():
        if file_hash(inside(bundle, name)) != expected:
            raise ValueError("saved audit input changed: " + name)


def submit(answer, doc_id, spans):
    return ("SUBMIT: " + answer + " CITATIONS: " + json.dumps([doc_id]) +
            " EVIDENCE: " + json.dumps([{k: s[k] for k in ("doc_id", "start", "end")}
                                        for s in spans]))


def gate(action, doc_id, text, windows, inspected):
    """Terminal eligibility only; source semantics cannot be decided here."""
    if (not isinstance(action, str) or len(action) > 8000 or len(inspected) != 2
            or len({(w["start"], w["end"]) for w in inspected}) != 2
            or any(not contains(windows, w["start"], w["end"]) for w in inspected)):
        return None
    try:
        gap = parse_gap_submission(action)
    except (ValueError, TypeError, RecursionError):
        return None
    if gap is not None:
        return "gap"
    parsed = parse_strict(action)
    if parsed is None:
        return None
    _, citations, evidence = parsed
    if citations != [doc_id] or not 1 <= len(evidence) <= 5:
        return None
    for span in evidence:
        if (set(span) != {"doc_id", "start", "end"} or span["doc_id"] != doc_id
                or not contains(inspected, span["start"], span["end"])
                or not 0 <= span["start"] < span["end"] <= len(text)):
            return None
    return "answer"


def score(record, context):
    action = record["action"]
    annotations = context["annotations"]
    legacy = score_submission(action, {"answer_annotations": annotations}, context["docs"],
                              investigated=bool(record["inspected_windows"]))
    label = context["scope_answerability"]
    kind = gate(action, context["doc_id"], context["docs"][context["doc_id"]]["text"],
                context["windows"], record["inspected_windows"])
    overlap = exact = 0.0
    if label == "missing" and kind == "gap":
        overlap = exact = 1.0  # Deliberately tests the weakness of shape-only gap scoring.
    elif label == "present" and kind == "answer":
        answer, _, evidence = parse_strict(action)
        # The frozen runtime validates the full submitted range, then clips each
        # packet quote. Score what the host would actually receive.
        clipped = [{**s, "end": min(s["end"], s["start"] + MAX_SPAN_CHARS)} for s in evidence]
        overlap = score_submission(submit(answer, context["doc_id"], clipped),
            {"answer_annotations": annotations}, context["docs"], investigated=True)["reward"]
        exact = float(any(normalize(answer) == normalize(a["answer_text"])
                          for a in annotations if not a["unanswerable"]))
    return {"legacy_overlap": legacy["reward"], "scoped_overlap": overlap,
            "scoped_exact": exact, "legacy_details": legacy,
            "terminal_eligibility": kind, "source_support": "not_mechanically_proven"}


def pilot_inputs(bundle=PILOT):
    checked_bundle(bundle)
    prepared = bundle / "prepared"
    review = load(prepared / "implementing-assistant-review.json")
    rows = {r["case_id"]: r for r in review["rows"]}
    contexts, records = {}, []
    for case in load(prepared / "cases.json"):
        if case["role"] != "train":
            continue  # No development cases, references or source inspection in this audit.
        cid = case["case_id"]
        row = rows[cid]
        base = bundle / "replay/cases" / cid
        packet, payloads = load(base / "packet.json"), load(base / "payloads.json")
        if verify_case(prepared, case, row, packet, payloads) != load(base / "verified.json"):
            raise ValueError("saved training demonstration no longer verifies")
        doc = load(inside(prepared, case["snapshot"] + "/corpus/" + case["doc_id"] + ".json"))
        contexts[cid] = {"question_id": case["question_id"], "question": case["question"],
            "doc_id": case["doc_id"], "scope_answerability": case["scope_answerability"],
            "windows": case["scope"]["windows"], "docs": {case["doc_id"]: doc},
            "annotations": [case["annotation_for_review_only"]],
            "review_sha256": digest(row), "saved_packet_sha256": file_hash(base / "packet.json"),
            "usage": "reward_diagnostic_only_no_training"}
        inspected = packet["inspected_passage_windows"]
        def add(variant, action, expected, rationale, observed=None, saved=False):
            records.append({"id": cid + ":" + variant, "context_id": cid, "variant": variant,
                "origin": "saved_reviewed_demonstration" if saved else
                    "authored_counterfactual_terminal_not_executed",
                "action": action, "inspected_windows": inspected if observed is None else observed,
                "expected": expected, "rationale": rationale,
                "judgment_origin": "implementing_assistant_source_review_or_counterexample_design",
                "human_review": None})
        add("reviewed", row["final_action"], "acceptable",
            row["notes"], saved=True)
        if case["scope_answerability"] == "missing":
            add("generic_gap", 'submit_gap(missing="Information unavailable", '
                'follow_up_query="research")', "unacceptable",
                "Generic gap and unusably broad query.")
            add("false_impossibility_gap", 'submit_gap(missing="These passages prove the method '
                'can never work", follow_up_query="potato recipes")', "unacceptable",
                "Claims impossibility from missing evidence; query is unrelated.")
            add("legacy_abstention", "SUBMIT: Unanswerable CITATIONS: [] EVIDENCE: []",
                "unacceptable", "Does not implement the current terminal gap contract.")
            continue
        spans = row["supporting_spans"]
        reference = case["annotation_for_review_only"]["answer_text"]
        add("reference_string", submit(reference, case["doc_id"], spans), "reference_control",
            "Original annotated answer string; proxy control, not new semantic grading.")
        add("unrelated_answer", submit("The answer is definitely 17.", case["doc_id"], spans),
            "unacceptable", "Unrelated answer with the accepted genuine source citation.")
        add("negated_reference", submit("It is false that the answer is " + reference,
            case["doc_id"], spans), "unacceptable", "Denies the supplied reference fact.")
        add("unsupported_addition", submit(reference +
            ". The paper proves this eliminates all hallucinations.", case["doc_id"], spans),
            "unacceptable", "Adds an unsupported universal claim to the reference answer.")
        # The other window excludes the annotated paragraph by construction and review.
        distractor = next(w for w in inspected if not any(
            contains([w], s["start"], s["end"]) for s in spans))
        unrelated = [{"doc_id": case["doc_id"], "start": distractor["start"],
                      "end": distractor["start"] + 100}]
        add("irrelevant_real_quote", submit(reference, case["doc_id"], unrelated),
            "unacceptable", "Real inspected quote from the distractor, not the reviewed support.")
        add("uninspected_gold_quote", submit(reference, case["doc_id"], spans),
            "unacceptable", "Only the distractor read is credited in this counterfactual.",
            observed=[distractor])
        add("false_gap", 'submit_gap(missing="The requested fact is missing", '
            'follow_up_query="find the requested fact")', "unacceptable",
            "False refusal despite the complete fact in the approved present scope.")
        if case["question_id"].endswith("ad7b13579823cbc7825421c84d16f23ed863f6ee"):
            bad = ("They use WMT2014 English-to-German for video captioning, VATEX for "
                   "translation, and VQA-v2 for visual question answering.")
            add("swapped_dataset_tasks", submit(bad, case["doc_id"], spans), "unacceptable",
                "Swaps VATEX/video and WMT/translation bindings; identical word multiset to "
                "the reviewed answer. The same citation cannot validate the swapped roles.")
    return contexts, records


def saved_regressions():
    from src.eval.reader_improvement import build_queue, check_review
    queue = build_queue(HYBRID)
    if queue != load(REGRESSION / "queue.json"):
        raise ValueError("saved regression queue changed")
    review = load(REGRESSION / "implementing-assistant-review.json")
    corrections = check_review(queue, review)["correction_drafts"]
    correction_map = {r["case_id"]: r for r in corrections}
    references = {r["id"]: r for r in load(HYBRID / "frozen/review-input.json")["cases"]}
    contexts, records = {}, []
    reviews = {r["case_id"]: r for r in review["rows"]}
    for r in queue["records"]:
        cid = r["case_id"]
        packet = r["original_packet"]
        contexts[cid] = {"question_id": cid, "question": r["question"], "doc_id": r["doc_id"],
            "scope_answerability": "present" if references[cid]["expected_answerability"] ==
                "sufficient" else "missing", "windows": packet["passage_scope"]["windows"],
            "docs": {r["doc_id"]: r["source"]}, "annotations": references[cid]["reference_answers"],
            "saved_packet_sha256": digest(packet), "review_sha256": digest(reviews[cid]),
            "usage": "exposed_regression_only_never_training"}
        for name, action, verdict, rationale in [("actual_v5", packet["trajectory"][-1]["action"],
                "acceptable" if reviews[cid]["verdict"] == "pass" else "unacceptable",
                reviews[cid]["notes"])] + ([("reviewed_correction", correction_map[cid][
                    "proposed_final_action"], "acceptable", correction_map[cid]["rationale"])]
                    if cid in correction_map else []):
            records.append({"id": cid + ":" + name, "context_id": cid, "variant": name,
                "action": action, "inspected_windows": packet["inspected_passage_windows"],
                "origin": "saved_actual_v5" if name == "actual_v5" else "saved_replayed_correction",
                "expected": verdict, "rationale": rationale, "human_review": None,
                "judgment_origin": "existing_implementing_assistant_source_review"})
    return contexts, records


def summarize(results):
    summaries = {}
    good = {r["context_id"]: r for r in results if r["variant"] in {
        "reviewed", "reviewed_correction"}}
    for scorer in SCORERS:
        bad_full = [r["id"] for r in results if r["expected"] == "unacceptable"
                    and r["scores"][scorer] >= 1.0 - 1e-12]
        not_lower = [r["id"] for r in results if r["expected"] == "unacceptable"
                     and r["context_id"] in good and good[r["context_id"]]["scores"][scorer] > 0
                     and r["scores"][scorer] >= good[r["context_id"]]["scores"][scorer] - 1e-12]
        summaries[scorer] = {"unacceptable_full_reward": bad_full,
            "unacceptable_positive_reward": sum(r["expected"] == "unacceptable" and
                r["scores"][scorer] > 0 for r in results),
            "acceptable_zero_reward": [r["id"] for r in results if r["expected"] == "acceptable"
                                       and r["scores"][scorer] == 0],
            "unacceptable_not_below_accepted_target": not_lower,
            "adversarial_gate_passed": not bad_full and not not_lower}
    return summaries


def audit(output):
    if output.exists():
        raise FileExistsError("refusing to overwrite reward audit")
    contexts, records = pilot_inputs()
    regression_contexts, regression_records = saved_regressions()
    contexts.update(regression_contexts)
    records.extend(regression_records)
    results = [{**r, "scores": score(r, contexts[r["context_id"]])} for r in records]
    output.mkdir(parents=True)
    write(output / "inputs.json", {"contexts": contexts, "records": records,
        "human_review": None, "training_eligible": False})
    write(output / "results.json", results)
    summary = {"schema_version": "reader-rl-reward-audit-v1", "records": len(records),
        "origins": dict(Counter(r["origin"] for r in records)), "scorers": summarize(results),
        "training_reward_ready": False, "human_review": None, "new_model_calls": 0,
        "gpu_activations": 0, "optimizer_updates": 0, "effective_runtime_prompt_sha256": PROMPT_SHA,
        "limitation": "Authored adversarial development counterexamples and existing assistant "
        "reviews are not independent semantic labels, new rollouts, or reward calibration. "
        "Passing syntax/provenance cannot establish factual support. No reward is deployed."}
    write(output / "summary.json", summary)
    write(output / "trainer-inspection.json", trainer_observations())
    write(output / "manifest.json", {"artifact_sha256": {p.name: file_hash(p)
        for p in sorted(output.iterdir())}, "implementation_sha256": {p: file_hash(ROOT / p)
        for p in ("src/eval/reader_rl_audit.py", "scripts/audit_reader_rl.py",
                  "src/eval/qasper_reward.py", "src/product/reader_gaps.py")},
        "source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in (
            PILOT / "validation.json", REGRESSION / "queue.json",
            REGRESSION / "implementing-assistant-review.json",
            HYBRID / "frozen/review-input.json")}})
    return summary


def trainer_observations():
    """Inspect source as AST data; never import the CUDA training entry point."""
    path = ROOT / "scripts/train_qasper_grpo.py"
    tree = ast.parse(path.read_text())
    constructors = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == "DocumentExplorationEnv"]
    return {"source_sha256": file_hash(path), "environment_constructors": [{
        "line": n.lineno, "use_docker": next(ast.literal_eval(k.value) for k in n.keywords
                                               if k.arg == "use_docker")} for n in constructors],
        "current_prompt_sha256": PROMPT_SHA, "uses_current_investigator": "QwenInvestigator" in
            path.read_text(), "uses_submit_gap": "submit_gap" in path.read_text(),
        "training_launcher_executed": False}
