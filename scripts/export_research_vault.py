"""Export Envoy's saved research history as a linked, read-only Obsidian vault."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
from pathlib import Path

from src.product.decision_threads import DecisionThreads
from src.product.experiment_atlas import ExperimentAtlas
from src.product.ml_experiment_store import ExperimentStore
from src.product.research_questions import ResearchQuestions

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ("Atlas", "Experiments", "Questions", "Decisions", "Sources", "Attachments")
MARKER = ".envoy-research-vault"


def safe(value: object) -> str:
    """Render stored text as text, never as Markdown links or instructions."""
    return re.sub(r"([\\`*_{}\[\]#!|<>])", r"\\\1", html.escape(str(value)))


def note(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def link(folder: str, record_id: str, label: str | None = None) -> str:
    # Wikilink aliases are plain text; Markdown backslashes appear literally there.
    alias = re.sub(r"[\[\]|\n\r]", " ", str(label or record_id))
    return f"[[{folder}/{record_id}|{alias}]]"


def section(lines: list[str], title: str, value: object) -> None:
    if value is None or value == [] or value == "":
        return
    lines.extend(["", f"## {title}", ""])
    if isinstance(value, list):
        lines.extend(f"- {safe(item)}" for item in value)
    elif isinstance(value, dict):
        lines.extend(f"- **{safe(key)}:** {safe(item)}" for key, item in value.items())
    else:
        lines.append(safe(value))


def export_atlas(vault: Path) -> list[str]:
    atlas = ExperimentAtlas(ROOT)
    ids = [row["atlas_id"] for row in atlas.find(limit=20)["matches"]]
    source_notes: dict[str, str] = {}
    for atlas_id in ids:
        entry = atlas.get(atlas_id)
        lines = [f"# {safe(atlas_id)}", "", "Curated historical experiment summary. "
                 "Read the primary artifacts before using its claims.", "",
                 f"Atlas SHA-256: `{entry['atlas_sha256']}`"]
        section(lines, "Question and actual attempt", entry["attempt"])
        section(lines, "Saved observation", entry["observation"])
        section(lines, "Decision and next question", entry["decision"])
        lines.extend(["", "## Primary evidence", ""])
        for source in entry["sources"]:
            path = source["path"]
            if path not in source_notes:
                source_id = hashlib.sha256(path.encode()).hexdigest()[:16]
                source_notes[path] = source_id
                original = ROOT / path
                raw = original.read_bytes()
                attachment = vault / "Attachments" / path
                attachment.parent.mkdir(parents=True, exist_ok=True)
                attachment.write_bytes(raw)
                source_lines = [f"# {safe(source['label'])}", "",
                                "Saved primary artifact copied byte-for-byte "
                                "from the repository.", "",
                                f"Repository path: `{safe(path)}`  ",
                                f"SHA-256: `{hashlib.sha256(raw).hexdigest()}`", "",
                                f"[Open saved copy](../Attachments/{path})"]
                if original.suffix == ".json":
                    excerpt = raw[:3000].decode("utf-8", errors="replace")
                    source_lines.extend(["", "## Opening excerpt", "", "```json", excerpt,
                                         "```", "", "For the full record, open the saved copy."])
                note(vault / "Sources" / f"{source_id}.md", source_lines)
            lines.append(f"- {link('Sources', source_notes[path], source['label'])} "
                         f"(`{safe(path)}`)")
        note(vault / "Atlas" / f"{atlas_id}.md", lines)
    return ids


def export_store(vault: Path, root: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    if not root.is_dir():
        return {}, {}, {}
    store = ExperimentStore(root)
    questions = ResearchQuestions(root, store)
    decisions = DecisionThreads(root, store)
    experiment_ids = [p.stem for p in root.glob("experiment_*.json")]
    question_ids = [p.stem for p in root.glob("question_*.json")]
    decision_ids = [p.stem for p in root.glob("decision_*.json")]
    experiment_titles: dict[str, str] = {}
    question_titles: dict[str, str] = {}
    decision_titles: dict[str, str] = {}
    for experiment_id in experiment_ids:
        view = store.get(experiment_id)
        brief = view["brief"]
        experiment_titles[experiment_id] = brief["question"]
        lines = [f"# {safe(brief['question'])}", "", f"Record ID: `{experiment_id}`  ",
                 f"Status: {safe(view['status'])}  ",
                 f"Revision: {view['revision']}", "",
                 "Stored claims and source receipts need semantic review. "
                 "This note is an export, not a new evaluation."]
        for title, key in (("Goal", "goal"), ("Proposed change", "proposed_change"),
                           ("Hypothesis", "hypothesis"), ("Expected signal", "expected_signal"),
                           ("Constraints", "constraints"), ("Unknowns", "unknowns"),
                           ("Decision criteria", "criteria")):
            section(lines, title, brief.get(key))
        if view["plan_sha256"]:
            lines.extend(["", f"Frozen plan SHA-256: `{view['plan_sha256']}`"])
        lines.extend(["", "## Research notes", ""])
        for item in view["research_notes"]:
            note_id = item["note_id"]
            selected = store.get(experiment_id, research_note_id=note_id)["selected_research_note"]
            lines.extend([f"- **{safe(note_id)}** (revision {selected['brief_revision']}; "
                          f"SHA-256 `{selected['sha256']}`): {safe(selected['note']['question'])}"])
            for finding in selected["note"]["findings"]:
                lines.append(f"  - {safe(finding)}")
        lines.extend(["", "## Recorded results", ""])
        for item in view["results"]:
            selected = store.get(experiment_id, run_id=item["run_id"])["selected_result"]
            lines.append(f"- **{safe(item['run_id'])}**: {safe(item['overall'])}; "
                         f"data kind `{safe(item['data_kind'])}`; SHA-256 `{item['sha256']}`")
            section(lines, "Run interpretation", selected["result"].get("interpretation"))
            section(lines, "Run limitations", selected["result"].get("limitations"))
        note(vault / "Experiments" / f"{experiment_id}.md", lines)
    for question_id in question_ids:
        view = questions.get(question_id)
        question_titles[question_id] = view["question"]
        lines = [f"# {safe(view['question'])}", "", f"Record ID: `{question_id}`  ",
                 f"Revision: {view['revision']}", "",
                 "Paper-to-result links record a research path, not causation."]
        for item in view["considered_papers"]:
            paper = item["paper"]
            basis = paper["source"]
            lines.extend(["", f"## Paper: {safe(paper['paper_id'])}", "",
                          f"Source experiment: {link('Experiments', basis['experiment_id'])}",
                          f"- Relevance: {safe(paper['relevance'])}",
                          f"- Limitation: {safe(paper['limitation'])}",
                          f"- Author: `{paper['author_kind']}`"])
        for item in view["paper_to_result_links"]:
            row = item["link"]
            lines.extend(["", f"## Attempt: {safe(row['link_id'])}", "",
                          "Result experiment: "
                          + link("Experiments", row["result"]["experiment_id"]),
                          f"- Run: `{safe(row['result']['run_id'])}`",
                          f"- Tried: {safe(row['method_tried'])}",
                          f"- Observed: {safe(row['observed_outcome'])}",
                          f"- Interpretation: {safe(row['interpretation'])}",
                          f"- Author: `{row['author_kind']}`"])
        for item in view["view_history"]:
            row = item["view"]
            lines.extend(["", f"## View: {safe(row['view_id'])}", "",
                          f"{safe(row['statement'])}", "",
                          f"Reason: {safe(row['reason'])}  ",
                          f"Next question: {safe(row['next_question'])}  ",
                          f"Author: `{row['author_kind']}`"])
        note(vault / "Questions" / f"{question_id}.md", lines)
    for decision_id in decision_ids:
        view = decisions.get(decision_id)
        decision_titles[decision_id] = view["question"]
        lines = [f"# {safe(view['question'])}", "", f"Record ID: `{decision_id}`", "",
                 "Author labels are caller-declared; they do not establish human approval.",
                 "", "## Linked experiments", ""]
        lines.extend(f"- {link('Experiments', item)}" for item in view["experiment_ids"])
        for item in view["decisions"]:
            choice = item["choice"]
            lines.extend(["", f"## {safe(choice['outcome'])}: {safe(choice['decision_id'])}", "",
                          f"{safe(choice['rationale'])}", "",
                          f"Author: `{choice['author_kind']}`  ",
                          f"Recorded: {safe(item['recorded_at'])}"])
            if choice["revisit_when"]:
                lines.append(f"Revisit when: {safe(choice['revisit_when'])}")
            for basis in choice["basis"]:
                details = [f"- Basis: {link('Experiments', basis['experiment_id'])}"]
                if basis["run_id"]:
                    details.append(f"run `{safe(basis['run_id'])}`")
                if basis["research_note_id"]:
                    details.append(f"research note `{safe(basis['research_note_id'])}`")
                lines.append("; ".join(details))
        note(vault / "Decisions" / f"{decision_id}.md", lines)
    return experiment_titles, question_titles, decision_titles


def export(vault: Path, stores: list[Path], refresh: bool = False) -> dict[str, int]:
    vault = vault.expanduser().absolute()
    if vault == ROOT or ROOT in vault.parents and vault.parent == ROOT:
        raise ValueError("Choose a dedicated vault directory")
    if vault.exists() and any(vault.iterdir()):
        if not refresh or not (vault / MARKER).is_file():
            raise ValueError("Existing nonempty directory requires --refresh "
                             "and an Envoy vault marker")
        for name in GENERATED:
            target = vault / name
            if target.is_symlink():
                raise ValueError("Generated vault folder may not be a symlink")
            if target.exists():
                shutil.rmtree(target)
        (vault / "Home.md").unlink(missing_ok=True)
    vault.mkdir(parents=True, exist_ok=True)
    (vault / MARKER).write_text("Generated by scripts/export_research_vault.py\n", encoding="utf-8")
    atlas_ids = export_atlas(vault)
    overview = (ROOT / "README.md").read_bytes()
    (vault / "Attachments").mkdir(exist_ok=True)
    (vault / "Attachments" / "README.md").write_bytes(overview)
    experiments: dict[str, str] = {}
    questions: dict[str, str] = {}
    decisions: dict[str, str] = {}
    for root in stores:
        e, q, d = export_store(vault, root.expanduser())
        experiments.update(e)
        questions.update(q)
        decisions.update(d)
    home = ["# Envoy research vault", "",
            "Open this folder as a vault in Obsidian. Search, Graph view and backlinks "
            "connect the notes below. The generated folders are rebuilt by the exporter; "
            "put your own notes in a separate folder.", "",
            "This is a reading copy. Envoy's record store and repository artifacts remain "
            "the sources of truth. A checked quote or hash proves provenance, not semantic "
            "support; paper-to-run links do not prove causation.", "",
            "## Project overview", "",
            "- [[Attachments/README|Qwen Envoy README]] — byte-for-byte copy of "
            "the repository overview; refresh this vault after README changes.",
            "  Repository-relative links inside the copy may need the original checkout.",
            f"- README SHA-256: `{hashlib.sha256(overview).hexdigest()}`", "",
            "## Previous experiments", ""]
    home.extend(f"- {link('Atlas', item)}" for item in atlas_ids)
    for title, folder, titles in (("Saved experiment records", "Experiments", experiments),
                                  ("Research questions", "Questions", questions),
                                  ("Decision threads", "Decisions", decisions)):
        home.extend(["", f"## {title}", ""])
        home.extend(f"- {link(folder, item, titles[item])} (`{item}`)"
                    for item in sorted(titles))
    note(vault / "Home.md", home)
    return {"atlas": len(atlas_ids), "experiments": len(experiments),
            "questions": len(questions), "decisions": len(decisions)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "out" / "envoy-research-vault")
    parser.add_argument("--store", type=Path, action="append", dest="stores",
                        help="Local Envoy record root; repeat for multiple stores")
    parser.add_argument("--refresh", action="store_true",
                        help="Rebuild generated notes in this vault")
    args = parser.parse_args()
    stores = args.stores or [Path("~/.envoy/ml-experiments"),
                             Path("~/.envoy/research-second-brain/records")]
    counts = export(args.output, stores, args.refresh)
    print(json.dumps({"vault": str(args.output.expanduser().absolute()), **counts}, indent=2))


if __name__ == "__main__":
    main()
