"""Build a reproducible Obsidian thread from the public Qwen evaluation report.

These are report-derived notes, not a copy of a private vault or a claim that an
experiment was preregistered. The builder writes only into an empty directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports" / "qasper-scale-sft-2026-09-25.json"
DEFAULT_OUTPUT = ROOT / "data" / "product_memory" / "qwen_experiment_vault"


def _note(record_id: str, kind: str, report_date: str, status: str,
          source_path: str, source_sha256: str, title: str,
          paragraphs: list[str]) -> str:
    header = (
        "---\n"
        f"record_id: {record_id}\n"
        f"kind: {kind}\n"
        f"effective_date: '{report_date}'\n"
        f"status: {status}\n"
        "review_status: source_derived_unreviewed\n"
        "origin: public_eval_report\n"
        f"source_repo_path: {source_path}\n"
        f"source_sha256: {source_sha256}\n"
        "---\n\n"
    )
    return header + f"# {title}\n\n" + "\n\n".join(paragraphs) + "\n"


def _validated_report(report_path: Path) -> tuple[dict, str, str]:
    source = report_path.resolve(strict=True)
    try:
        source_path = source.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("report must be a file inside this repository") from exc
    raw = source.read_bytes()
    report = json.loads(raw)
    if report.get("schema_version") != "envoy-public-eval-report-v1":
        raise ValueError("unsupported public evaluation report schema")
    date.fromisoformat(report["date"])
    question_ids = report["protocol"]["question_ids"]
    if len(question_ids) != 40 or len(set(question_ids)) != 40:
        raise ValueError("expected 40 unique reported question IDs")
    results = report["results"]
    breakdown = report["semantic_passes_by_answerability"]
    for arm in ("starting_sft", "epoch_1", "epoch_2"):
        semantic = results[arm]["semantic"]
        if sum(semantic[key] for key in ("pass", "partial", "fail")) != 40:
            raise ValueError(f"{arm} semantic counts do not total 40")
        if sum(breakdown[arm][key] for key in ("sufficient", "insufficient")) != semantic["pass"]:
            raise ValueError(f"{arm} answerability breakdown disagrees with pass count")
    return report, source_path, hashlib.sha256(raw).hexdigest()


def render_notes(report_path: Path = DEFAULT_REPORT) -> dict[str, str]:
    """Render deterministic Markdown; every numeric claim comes from the report."""
    report, source_path, source_sha256 = _validated_report(report_path)
    report_date = date.fromisoformat(report["date"]).isoformat()
    protocol = report["protocol"]
    results = report["results"]
    breakdown = report["semantic_passes_by_answerability"]
    decision_rule = report["decision_rule"]
    decisions = report["decision"]
    review = report["review"]
    rule_lines = "\n".join(f"- {condition}" for condition in decision_rule["promote_only_if"])
    key_rule_prefixes = (
        "semantic mean exceeds ", "at least two more paired wins ",
        "at least 28 of 40 semantic passes",
    )
    key_rules = [condition for condition in decision_rule["promote_only_if"]
                 if condition.startswith(key_rule_prefixes)]
    if len(key_rules) != len(key_rule_prefixes):
        raise ValueError("public report is missing a key promotion gate")
    key_rule_summary = "; ".join(key_rules)
    prior_name = f"01-{report_date}-qwen-continuation-strategy.md"
    result_name = f"02-{report_date}-qwen-measured-result.md"
    decision_name = f"03-{report_date}-qwen-reported-decision.md"
    prior_link = f"[[{Path(prior_name).stem}]]"
    result_link = f"[[{Path(result_name).stem}]]"

    prior = _note(
        f"qwen-qasper-continuation-strategy-report-{report_date}", "decision", report_date,
        "retrospective_reconstruction", source_path, source_sha256,
        "Qwen QASPER continuation strategy (retrospective reconstruction)", [
            "This note reconstructs the next-step strategy after the run and separately "
            "records the promotion rule stated in the public evaluation report. It is "
            "**not** a contemporaneous "
            "preregistration, a user-approved memory revision, or proof that thresholds were "
            "fixed before results. The effective date here is the report date; the source does "
            "not supply an independent timestamp for the rule.",
            f"**Inferred next-step strategy and experiment question:** Test one or two further "
            f"SFT epochs on {report['task']} and ask whether those epochs improve "
            f"the starting SFT checkpoint. The report compares "
            f"`epoch_1` and `epoch_2` to `{decision_rule['compare_to']}` on the same "
            f"{len(protocol['question_ids'])} `{protocol['source_split']}` questions. "
            f"The report-recorded gates include: {key_rule_summary}. "
            "Only the strategy and question are retrospective inferences, not a quoted "
            "original plan. The listed gates are report fields, not inferred from the "
            "reported rejection decision.",
            "**Promotion rule recorded in the report:**\n" + rule_lines,
            "The report sets `no_posthoc_threshold_changes: "
            f"{str(decision_rule['no_posthoc_threshold_changes']).lower()}`. "
            "That is a statement in the report, not independent verification of "
            "when the rule was established. A later revision could supersede the "
            "inferred next-step strategy; it would not invalidate this rule or "
            "rewrite the historical run.",
            f"**Source:** `{source_path}`; SHA-256 `{source_sha256}`. "
            "The report is the authority for these reconstructed fields.",
        ],
    )
    totals = {arm: results[arm]["semantic"]["pass"]
              for arm in ("starting_sft", "epoch_1", "epoch_2")}
    sufficient = {arm: breakdown[arm]["sufficient"] for arm in totals}
    insufficient = {arm: breakdown[arm]["insufficient"] for arm in totals}
    result = _note(
        f"qwen-qasper-measured-result-report-{report_date}", "result", report_date,
        "reported_measurement", source_path, source_sha256,
        "Qwen QASPER continuation measured result", [
            f"The public report's identity-blind, model-assisted semantic review found "
            f"**{totals['starting_sft']}/40** passes for the starting SFT checkpoint, "
            f"**{totals['epoch_1']}/40** after epoch 1, and **{totals['epoch_2']}/40** "
            f"after epoch 2 on the same reported question IDs. This result is linked to the "
            f"reconstructed continuation strategy {prior_link}; the link identifies what "
            "to inspect, "
            "not a proven causal relationship.",
            "**Answerability tradeoff (reported semantic pass counts):** on "
            f"sufficient-evidence items, starting SFT / epoch 1 / epoch 2 passed "
            f"{sufficient['starting_sft']} / {sufficient['epoch_1']} / "
            f"{sufficient['epoch_2']}; on insufficient-evidence items they passed "
            f"{insufficient['starting_sft']} / {insufficient['epoch_1']} / "
            f"{insufficient['epoch_2']}. The public report supplies pass counts by subgroup "
            "but not subgroup denominators. More answerable passes at epoch 1 coincided "
            "with fewer insufficient-evidence passes; this does not by itself establish "
            "the cause of the tradeoff.",
            f"**Review:** {review['method']}; reviewer model `{review['model']}`. "
            f"The review covered {review['judged_question_count']} questions and "
            f"{review['judged_candidate_count']} candidate responses.",
            f"**Scope:** {report['scope_limit']} The starting SFT checkpoint is not base "
            "Qwen, and this QASPER result is not a personal Obsidian-vault evaluation. "
            f"Base model: `{protocol['base_model']}` at revision "
            f"`{protocol['base_revision']}`; starting adapter SHA-256 "
            f"`{protocol['adapter_sha256']['starting_sft']}`.",
            f"**Source:** `{source_path}`; SHA-256 `{source_sha256}`. "
            f"Benchmark SHA-256 `{protocol['benchmark_sha256']}` and corpus-content "
            f"SHA-256 `{protocol['corpus_content_sha256']}` identify the reported evaluation.",
        ],
    )
    reported_decision = _note(
        f"qwen-qasper-rejection-report-{report_date}", "source", report_date,
        "reported_decision_not_user_approved", source_path, source_sha256,
        "Qwen QASPER continuation: decision reported in source", [
            f"The public report records `epoch_1: {decisions['epoch_1']}` and "
            f"`epoch_2: {decisions['epoch_2']}`. Its stated reason is: "
            f"{decisions['reason']}. Neither continuation checkpoint was promoted "
            f"according to the report, which records that the promotion gates were applied "
            f"to this comparison. See the measured result {result_link} and the "
            f"retrospectively reconstructed continuation strategy {prior_link}; that note "
            "separately reproduces the rule recorded in the report.",
            "This is a **source-derived report of a decision**, not a new user-approved "
            "Obsidian correction or a claim that the personal-vault Qwen worker improved. "
            "A later user conclusion should be written as a separate reviewed note, "
            "with links to the relevant evidence.",
            f"**Source:** `{source_path}`; SHA-256 `{source_sha256}`. "
            f"Review method: {review['method']}.",
        ],
    )
    return {prior_name: prior, result_name: result, decision_name: reported_decision}


def build_vault(output: Path, report_path: Path = DEFAULT_REPORT) -> dict[str, Path]:
    """Write the thread to an absent or empty directory; never replace files."""
    notes = render_notes(report_path)
    output = output.expanduser()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise FileExistsError(f"output is not an empty directory: {output}")
    else:
        output.mkdir(parents=True)
    written = {}
    for role, (name, content) in zip(("prior", "result", "decision"), notes.items(),
                                     strict=True):
        path = output / name
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
        written[role] = path
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args(argv)
    for role, path in build_vault(args.output, args.report).items():
        print(f"{role}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
