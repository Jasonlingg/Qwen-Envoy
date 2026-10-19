"""Readable experiment briefs rendered from saved data; no generated recommendations."""

from __future__ import annotations

import html
import json
import re

MAX_DISPLAY_CHARS = 20_000


def _text(value) -> str:
    return re.sub(r"([\\`*_{}\[\]#!|])", r"\\\1", html.escape(str(value)))


def _details(value, indent=0) -> list[str]:
    prefix = " " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            label = _text(key.replace("_", " "))
            if isinstance(item, dict) and set(item) == {"value", "unknown_reason"}:
                item = item["value"] or item["unknown_reason"]
            if isinstance(item, (dict, list)) and item:
                lines.extend([f"{prefix}- {label}:", *_details(item, indent + 2)])
            else:
                lines.append(f"{prefix}- {label}: {_text(item if item is not None else 'Unknown')}")
        return lines
    if isinstance(value, list):
        lines = []
        for index, item in enumerate(value, 1):
            if isinstance(item, dict):
                lines.extend([f"{prefix}- Item {index}:", *_details(item, indent + 2)])
            else:
                lines.append(f"{prefix}- {_text(item)}")
        return lines
    return [prefix + _text(value if value is not None else "Not established yet")]


def render_brief(view: dict) -> tuple[str, bool]:
    brief = view["brief"]
    lines = ["# Experiment brief", "", _text(brief["question"]), "",
             f"Status: {view['status']} · revision {view['revision']}", "",
             "## Current setup", ""]
    for name, item in brief["current_setup"].items():
        if name == "software":
            for package, known in item.items():
                value = known["value"] or known["unknown_reason"]
                lines.append(f"- {_text(package)}: {_text(value)}")
        else:
            lines.append(f"- {name}: {_text(item['value'] or item['unknown_reason'])}")
    for title, name in [("Goal", "goal"), ("Proposed change", "proposed_change"),
                        ("Hypothesis", "hypothesis"), ("Expected signal", "expected_signal"),
                        ("Applicability", "applicability")]:
        lines.extend(["", f"## {title}", "", _text(brief[name] or "Not established yet")])
    for title, name in [("Constraints", "constraints"), ("Alternatives considered", "alternatives"),
                        ("Differences from source settings", "differences"),
                        ("Procedure", "procedure"), ("Unknowns", "unknowns"),
                        ("Limitations", "limitations")]:
        lines.extend(["", f"## {title}", ""])
        lines.extend(f"- {_text(item)}" for item in brief[name])
        if not brief[name]:
            lines.append("Not established yet")
    lines.extend(["", "## Observations", ""])
    for item in brief["observations"]:
        lines.append(f"- {_text(item['text'])} ({item['basis']}; "
                     f"references: {_text(', '.join(item['reference_ids']) or 'none')})")
    for title, name in [("Baseline", "baseline"), ("Candidate", "candidate"),
                        ("Evaluation sample", "evaluation"),
                        ("Controlled conditions", "conditions"),
                        ("Metrics", "metrics"), ("Decision criteria", "criteria"),
                        ("Resource ceilings (recorded, not enforced)", "budget")]:
        lines.extend(["", f"## {title}", ""])
        value = brief[name]
        lines.extend(_details(value) if value else ["Not established yet"])
    lines.extend(["", "## Evidence", "",
                  "Source checks establish provenance; semantic support remains unreviewed."])
    receipts = {item["reference_id"]: item for item in view["evidence_receipts"]}
    for ref in brief["references"]:
        receipt = receipts.get(ref["reference_id"])
        lines.extend(["", f"### {_text(ref['reference_id'])}", ""])
        if receipt:
            lines.append(f"Verification: {receipt['verification']}")
            source_hash = receipt["source_sha256"] or "unverified external reference"
            lines.append(f"Source SHA-256: {source_hash}")
            lines.append("Reference: " + _text(json.dumps(receipt["metadata"], ensure_ascii=False)))
            if receipt["verification"] != "reference_only":
                value = receipt["value"]
                rendered = (value if isinstance(value, str)
                            else json.dumps(value, ensure_ascii=False))
                lines.extend("> " + _text(line) for line in rendered.splitlines())
        else:
            lines.append("Declared reference; source verification has not been performed.")
    missing = (", ".join(name.replace("_", " ").replace(".", " ")
                         for name in view["missing_fields"]) or "No structural fields missing")
    lines.extend(["", "## Planning status", "", "Missing: " + _text(missing),
                  "Candidate outcomes already seen: " + brief["candidate_outcomes_already_seen"],
                  "Author: " + brief["author_kind"],
                  "Freezing records intent and grants no permission to execute or spend."])
    if view["plan_sha256"]:
        lines.append("Frozen plan SHA-256: " + view["plan_sha256"])
    rendered = "\n".join(lines) + "\n"
    truncated = len(rendered) > MAX_DISPLAY_CHARS
    if truncated:
        rendered = (rendered[:MAX_DISPLAY_CHARS]
                    + "\n\nDisplay truncated; see the structured brief.\n")
    return rendered, truncated
