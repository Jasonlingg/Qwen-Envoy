"""Offline heuristic probe, not an enabled training reward or semantic judge."""

import argparse
import copy
import json
import re
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.eval.qasper_reward import parse_strict  # noqa: E402
from src.eval.reader_improvement import digest, file_hash, load  # noqa: E402
from src.eval.reader_rl_audit import submit  # noqa: E402
from src.training.reader_data import load_bundle  # noqa: E402
from src.training.reader_f1_reward import assess_f1  # noqa: E402

MAX_WORDS = 32
NUMBER = re.compile(r"[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?[kKmMbB]?")
NEGATION = re.compile(r"\b(?:no|not|never|without|neither|nor|cannot)\b|n['’]t\b", re.I)
UNITS = re.compile(r"(?<![\w'’])(?:ms|Ms|ns|us|µs|μs|s|MB|Mb|GB|Gb|Hz|kHz|MHz|GHz|%|"
                   r"(?i:sentences?|threads?|tweets?))(?!\w)")
COUNT_UNITS = {word: singular for singular in ("sentence", "thread", "tweet")
               for word in (singular, singular + "s")}


def critical(text, *, bind=False):
    text = re.sub(r"\b(?:BIBREF|FIGREF|TABREF)\d+\b", "", text)
    numbers = []
    matches = list(NUMBER.finditer(text))
    for i, match in enumerate(matches):
        token = match.group().replace(",", "")
        scale = {"k": 1000, "m": 1000000, "b": 1000000000}.get(token[-1:].lower(), 1)
        value = Decimal(token[:-1] if scale != 1 else token) * scale
        if bind:
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            tail = " ".join(text[match.end():end].split()[:5])
            unit = next(iter(UNITS.findall(tail)), "")
            numbers.append((value, COUNT_UNITS.get(unit.lower(), unit)))
        else:
            numbers.append(value)
    units = [COUNT_UNITS.get(unit.lower(), unit) for unit in UNITS.findall(text)]
    return sorted(numbers), sorted(units), bool(NEGATION.search(text))


def audit():
    source = Path("examples/reader-f1-reward-v1/results.json")
    previous = load(source)
    if any(file_hash(Path(p)) != sha for p, sha in previous["implementation_sha256"].items()):
        raise ValueError("previous reward implementation changed")
    _, _, contexts, cards = load_bundle(Path("data/rl/reader-facts-v1"))
    rows = copy.deepcopy(previous["rows"])
    for row in list(rows):
        if row["kind"] != "reference":
            continue
        answer, citations, spans = parse_strict(row["record"]["action"])
        variants = [("repeated_reference", (answer + " ") * 40)]
        if "1,873" in answer and "14k" in answer:
            variants.append(("swapped_numeric_bindings", answer.replace(
                "threads", "PLACEHOLDER").replace("tweets", "threads").replace(
                    "PLACEHOLDER", "tweets")))
        for kind, text in variants:
            record = {**row["record"], "action": submit(text, citations[0], spans)}
            rows.append({"role": row["role"], "kind": kind, "record": record,
                         "origin": "authored_fixture_assumed_inspection_not_a_model_rollout"})
    output, groups = [], defaultdict(list)
    for row in rows:
        record = row["record"]
        cid = record["context_id"]
        if cid not in contexts or cards[cid]["role"] != row["role"]:
            raise ValueError("reward context/role unavailable")
        answer = parse_strict(record["action"])[0]
        references = [a["answer_text"] for a in contexts[cid]["annotations"]
                      if not a["unanswerable"] and not a["requires_nontext_evidence"]]
        references += [a["answer"] for a in cards[cid]["accepted_answers"]]
        base = assess_f1(record, contexts[cid], cards[cid])["reward"]
        short = len(answer.split()) <= MAX_WORDS
        compatible = any(critical(answer) == critical(ref) for ref in references)
        bound = any(critical(answer, bind=True) == critical(ref, bind=True) for ref in references)
        result = {k: row[k] for k in ("role", "kind", "origin")}
        result.update(record=record, record_sha256=digest(record), base_reward=base,
                      length_only_reward=base if short else 0.0,
                      guarded_reward=base if short and compatible else 0.0,
                      bound_guarded_reward=base if short and bound else 0.0)
        output.append(result)
        groups[row["kind"]].append(result)
    keys = ("base_reward", "length_only_reward", "guarded_reward", "bound_guarded_reward")
    means = {kind: {"count": len(rs), **{k: sum(r[k] for r in rs) / len(rs) for k in keys}}
             for kind, rs in sorted(groups.items())}
    checks = {
        "reference_credit_preserved": all(r["bound_guarded_reward"] == r["base_reward"] == 1
                                          for r in groups["reference"]),
        "proposed_paraphrases_keep_credit": all(r["bound_guarded_reward"] > 0
                                               for r in groups["paraphrase"]),
        "numeric_and_negation_exploits_zero": all(r["bound_guarded_reward"] == 0 for kind in (
            "decimal_normalization_collision", "negated_reference") for r in groups[kind]),
        "all_prior_evidence_negatives_zero": all(r["bound_guarded_reward"] == 0 for kind in (
            "real_but_insufficient_quote", "uninspected", "extra_unrelated_quote",
            "all_assigned_windows") for r in groups[kind]),
        "repetition_zero": all(r["bound_guarded_reward"] == 0
                               for r in groups["repeated_reference"]),
        "swapped_numeric_bindings_zero": bool(groups["swapped_numeric_bindings"]) and all(
            r["bound_guarded_reward"] == 0 for r in groups["swapped_numeric_bindings"]),
    }
    return {"version": "guarded-f1-heuristic-probe-v1", "source_audit_sha256": file_hash(source),
            "source_manifest_sha256": previous["source_manifest_sha256"],
            "source_revision": previous["source_revision"], "max_final_answer_words": MAX_WORDS,
            "script_sha256": file_hash(Path(__file__)), "rows": output, "by_kind": means,
            "checks": checks, "development_gate_passed": all(checks.values()),
            "training_ready": False, "active_reward_changed": False,
            "independent_human_review": None, "new_model_calls": 0, "gpu_activations": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        out.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("by_kind", "checks")}, indent=2))
