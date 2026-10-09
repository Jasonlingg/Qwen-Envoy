"""Frozen-vault link resolution for the chat memory investigator."""

from src.product.graph import build_graph, one_hop_neighbors
from src.product.memory import freeze_vault


def _note(vault, name, text):
    path = vault / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _by_record(graph):
    return {node["record_id"]: doc_id for doc_id, node in graph["nodes"].items()}


def test_graph_indexes_typed_revision_links_and_backlinks(tmp_path):
    vault = tmp_path / "vault"
    _note(vault, "Learning/Old.md", "---\nrecord_id: old\n---\n# Old\n")
    _note(vault, "Sources/Seed.md", "---\nrecord_id: seed\n---\n# Seed\n")
    _note(vault, "Learning/New.md", """---
record_id: new
kind: correction
effective_date: 2026-09-20
supersedes: old
derived_from:
  - seed
---
# New

I revised [[Old|my earlier view]] after [the seed](../Sources/Seed.md#Result).
An unknown [[Missing]] remains unresolved.
`[[Inline example]]` is code, not a link.
```markdown
[[Fenced example]]
```
[Web](https://example.com) is external.
""")
    snapshot = tmp_path / "snapshot"
    manifest = freeze_vault(vault, snapshot)

    graph = build_graph(snapshot)
    ids = _by_record(graph)
    edges = {(edge["source"], edge["target"], edge["type"])
             for edge in graph["edges"]}
    assert graph["corpus_hash"] == manifest["corpus_hash"]
    assert graph["nodes"][ids["new"]]["effective_date"] == "2026-09-20"
    assert (ids["new"], ids["old"], "supersedes") in edges
    assert (ids["new"], ids["old"], "wikilink") in edges
    assert (ids["new"], ids["seed"], "derived_from") in edges
    assert (ids["new"], ids["seed"], "markdown_link") in edges
    assert {item["target_ref"] for item in graph["unresolved"]} == {"Missing"}

    neighbors = one_hop_neighbors(graph, "old")
    assert len(neighbors) == 1
    assert neighbors[0]["record_id"] == "new"
    assert {tuple(connection.values()) for connection in neighbors[0]["connections"]} == {
        ("incoming", "supersedes"), ("incoming", "wikilink")}


def test_ambiguous_names_and_record_ids_are_not_guessed(tmp_path):
    vault = tmp_path / "vault"
    _note(vault, "A/Topic.md", "---\nrecord_id: duplicate\n---\n# First\n")
    _note(vault, "B/Topic.md", "---\nrecord_id: duplicate\n---\n# Second\n")
    _note(vault, "C/Question.md", """---
record_id: question
derived_from: duplicate
---
# Question

What did [[Topic]] say?
""")
    _note(vault, "A/Question.md", "# Local question\n[[Topic]] and [[Topic.png]]\n")
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)

    graph = build_graph(snapshot)
    assert graph["edges"] == []
    assert {(item["type"], item["target_ref"], item["reason"])
            for item in graph["unresolved"]} == {
                ("derived_from", "duplicate", "ambiguous"),
                ("wikilink", "Topic", "ambiguous"),
                ("wikilink", "Topic.png", "missing"),
            }


def test_relative_links_resolve_inside_snapshot_but_cannot_escape(tmp_path):
    vault = tmp_path / "vault"
    _note(vault, "Old Note.md", "# Old note\n")
    _note(vault, "Nested/Current.md", """# Current

[Old](../Old%20Note.md) and [[../Outside]] and [outside](../../Outside.md).
""")
    snapshot = tmp_path / "snapshot"
    freeze_vault(vault, snapshot)

    graph = build_graph(snapshot)
    current = next(node["doc_id"] for node in graph["nodes"].values()
                   if node["source_path"] == "Nested/Current.md")
    old = next(node["doc_id"] for node in graph["nodes"].values()
               if node["source_path"] == "Old Note.md")
    assert graph["edges"] == [{"source": current, "target": old,
                               "type": "markdown_link"}]
    assert {item["target_ref"]: item["reason"] for item in graph["unresolved"]} == {
        "../Outside": "missing", "../../Outside.md": "unsafe_path"}


def test_approved_note_doc_id_relation_survives_a_new_snapshot(tmp_path):
    vault = tmp_path / "vault"
    _note(vault, "Old.md", "# Old\n")
    first = tmp_path / "first"
    freeze_vault(vault, first)
    old_id = next(iter(build_graph(first)["nodes"]))
    _note(vault, "New.md", f"---\nsupersedes_doc_id: {old_id}\n---\n# New\n")
    second = tmp_path / "second"
    freeze_vault(vault, second)

    graph = build_graph(second)
    new_id = next(node["doc_id"] for node in graph["nodes"].values()
                  if node["source_path"] == "New.md")
    assert {tuple(edge.values()) for edge in graph["edges"]} == {
        (new_id, old_id, "supersedes")}
    assert graph["nodes"][new_id]["record_id"] == new_id
