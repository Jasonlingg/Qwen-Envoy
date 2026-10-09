"""System-map prototype API: durable revisions and verified source pointers."""

import json
from pathlib import Path
from shutil import copytree

from fastapi.testclient import TestClient

from scripts.personal_memory_web import create_app

ROOT = Path(__file__).resolve().parents[1]


def _app(tmp_path):
    vault = copytree(ROOT / "data/product_memory/sample_vault", tmp_path / "vault")
    state = tmp_path / "state"
    return create_app(vault, state), vault, state


def _thread(client):
    response = client.post("/api/lab/threads", json={
        "question": "What makes research answers reliable?",
        "current_view": "Checking source spans seems important.",
        "effective_date": "2026-09-29",
    })
    assert response.status_code == 200, response.text
    return response.json()["thread_id"]


def _nodes():
    return [
        {"id": "retrieval", "type": "process", "label": "Retrieval",
         "description": "Find source passages", "x": 40, "y": 80},
        {"id": "answer", "type": "outcome", "label": "Supported answer",
         "description": "An answer with relevant citations", "x": 320, "y": 80},
    ]


def _seed_map_proposal(state, thread_id, *, edges, nodes=None, **changes):
    path = state / "learning_lab" / "map_proposals" / f"{thread_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    proposal = {
        "thread_id": thread_id,
        "authorship": "curated_demo",
        "title": "Review a possible link",
        "rationale": "A frozen source offers a passage worth checking.",
        "summary": "Connect the source to the supported answer.",
        "base_revision_id": None,
        "proposed_map": {"nodes": _nodes() if nodes is None else nodes, "edges": edges},
        **changes,
    }
    path.write_text(json.dumps(proposal), encoding="utf-8")
    return path


def test_map_empty_save_history_roundtrip_and_no_model_calls(tmp_path):
    app, vault, state = _app(tmp_path)
    with TestClient(app) as client:
        thread_id = _thread(client)
        url = f"/api/lab/threads/{thread_id}/map"
        before = client.get("/api/status").json()["model_calls"]
        empty = client.get(url)
        assert empty.status_code == 200, empty.text
        assert empty.json()["question"] == "What makes research answers reliable?"
        assert empty.json()["revision"] == {
            "id": None, "created_at": None, "summary": "", "nodes": [], "edges": [],
        }
        assert empty.json()["history"] == []

        first = client.put(url, json={
            "summary": "Sketch the suspected relationship.", "nodes": _nodes(),
            "edges": [{"id": "could_help", "source": "retrieval", "target": "answer",
                       "type": "may_affect", "label": "may improve",
                       "explanation": "This is still untested.", "basis": "hypothesis",
                       "evidence": []}],
        })
        assert first.status_code == 200, first.text
        first_id = first.json()["revision"]["id"]
        assert first_id.startswith("maprev_")
        assert first.json()["revision"]["edges"][0]["receipts"] == []
        assert first.json()["semantic_support"] == "not_verified"

        attachment = client.post(
            f"/api/lab/threads/{thread_id}/files?filename=pilot.txt",
            content=b"Our pilot found source checking reduced unsupported answers.",
            headers={"content-type": "text/plain"},
        )
        assert attachment.status_code == 200, attachment.text
        source = attachment.json()
        passage = source["passages"][0]
        second_nodes = first.json()["revision"]["nodes"]
        second_edges = first.json()["revision"]["edges"]
        second_edges[0]["basis"] = "measured"
        second_edges[0]["evidence"] = [{
            "source_id": source["source_id"], "start": passage["start"],
            "end": passage["end"], "relation": "supports",
        }]
        second = client.put(url, json={
            "summary": "Attach the pilot's exact passage.",
            "nodes": second_nodes, "edges": second_edges,
        })
        assert second.status_code == 200, second.text
        receipt = second.json()["revision"]["edges"][0]["receipts"][0]
        assert receipt["status"] == "exact_span_verified"
        assert "reduced unsupported answers" in receipt["quote"]
        assert receipt["relation"] == "supports"
        assert len(second.json()["history"]) == 2
        assert second.json()["history"][0]["id"] == second.json()["revision"]["id"]

        old = client.get(f"{url}/revisions/{first_id}")
        assert old.status_code == 200, old.text
        assert old.json()["revision"]["edges"][0]["basis"] == "hypothesis"
        assert old.json()["revision"]["edges"][0]["evidence"] == []
        assert client.get(url).json()["revision"]["edges"][0]["basis"] == "measured"

        # A hydrated map may be sent back unchanged as the next reviewed revision.
        hydrated = second.json()["revision"]
        third = client.put(url, json={
            "summary": "Keep the current map after review.",
            "nodes": hydrated["nodes"], "edges": hydrated["edges"],
        })
        assert third.status_code == 200, third.text
        assert third.json()["revision"]["edges"][0]["receipts"][0]["quote"] == receipt["quote"]
        assert client.get("/api/status").json()["model_calls"] == before
        assert len(list((state / "learning_lab" / "maps" / thread_id).glob("maprev_*.json"))) == 3
        assert not list(vault.rglob("*system-map*"))
    with TestClient(create_app(vault, state)) as restarted:
        restored = restarted.get(url)
        assert restored.status_code == 200, restored.text
        assert restored.json()["revision"]["id"] == third.json()["revision"]["id"]
        assert len(restored.json()["history"]) == 3


def test_map_rejects_invalid_structure_and_forged_source_spans(tmp_path):
    app, _, _ = _app(tmp_path)
    with TestClient(app) as client:
        thread_id = _thread(client)
        url = f"/api/lab/threads/{thread_id}/map"
        edge = {"id": "link", "source": "retrieval", "target": "answer",
                "type": "flows_to", "basis": "documented", "label": "uses",
                "evidence": []}
        assert client.put(url, json={"summary": "No receipt", "nodes": _nodes(),
                                     "edges": [edge]}).status_code == 400
        assert client.put(url, json={"summary": "Duplicate nodes", "nodes": _nodes() * 2,
                                     "edges": []}).status_code == 400
        assert client.put(url, json={"summary": "Wrong endpoint", "nodes": _nodes(),
                                     "edges": [{**edge, "target": "missing",
                                                "basis": "hypothesis"}]}).status_code == 400
        assert client.put(url, json={"summary": "Self-loop", "nodes": _nodes(),
                                     "edges": [{**edge, "target": "retrieval",
                                                "basis": "hypothesis"}]}).status_code == 400
        assert client.put(url, json={"summary": "Bad coordinates",
                                     "nodes": [{**_nodes()[0], "x": 1e100}],
                                     "edges": []}).status_code == 422

        attachment = client.post(
            f"/api/lab/threads/{thread_id}/files?filename=paper.txt",
            content=b"An inspectable passage.", headers={"content-type": "text/plain"},
        ).json()
        source_id = attachment["source_id"]
        bad = {"source_id": source_id, "start": 0, "end": 99_999,
               "relation": "supports"}
        assert client.put(url, json={"summary": "Forged span", "nodes": _nodes(),
                                     "edges": [{**edge, "evidence": [bad]}]}).status_code == 400
        passage = attachment["passages"][0]
        valid = {"source_id": source_id, "start": passage["start"],
                 "end": passage["end"], "relation": "supports"}
        duplicate = client.put(url, json={
            "summary": "Duplicate span", "nodes": _nodes(),
            "edges": [{**edge, "evidence": [valid, valid]}],
        })
        assert duplicate.status_code == 400
        assert client.get(url).json()["history"] == []


def test_map_receipt_degrades_if_frozen_file_changes(tmp_path):
    app, _, state = _app(tmp_path)
    with TestClient(app) as client:
        thread_id = _thread(client)
        url = f"/api/lab/threads/{thread_id}/map"
        attachment = client.post(
            f"/api/lab/threads/{thread_id}/files?filename=study.txt",
            content=b"One measured result needs review.",
            headers={"content-type": "text/plain"},
        ).json()
        passage = attachment["passages"][0]
        saved = client.put(url, json={
            "summary": "Connect study to outcome.", "nodes": _nodes(),
            "edges": [{"id": "link", "source": "retrieval", "target": "answer",
                       "type": "may_affect", "basis": "measured", "label": "may help",
                       "evidence": [{"source_id": attachment["source_id"],
                                     "start": passage["start"], "end": passage["end"],
                                     "relation": "context"}]}],
        })
        assert saved.status_code == 200, saved.text
        revision_id = saved.json()["revision"]["id"]
        raw = next((state / "learning_lab" / "sources" / thread_id
                    / attachment["source_id"] / "raw").iterdir())
        raw.write_bytes(b"Changed source bytes")
        current = client.get(url)
        assert current.status_code == 200, current.text
        assert current.json()["revision"]["edges"][0]["receipts"][0] == {
            "reference": saved.json()["revision"]["edges"][0]["evidence"][0]["reference"],
            "status": "source_unavailable_or_changed", "relation": "context",
        }
        assert (client.get(f"{url}/revisions/{revision_id}").json()["revision"]
                ["edges"][0]["receipts"][0]["status"] == "source_unavailable_or_changed")


def test_curated_map_proposal_is_read_only_reviewable_and_stales_after_save(tmp_path):
    app, _, state = _app(tmp_path)
    with TestClient(app) as client:
        thread_id = _thread(client)
        map_url = f"/api/lab/threads/{thread_id}/map"
        proposal_url = f"{map_url}/proposal"
        assert client.get(proposal_url).status_code == 404

        attachment = client.post(
            f"/api/lab/threads/{thread_id}/files?filename=pilot.txt",
            content=b"Source checks reduced unsupported answers in this pilot.",
            headers={"content-type": "text/plain"},
        ).json()
        passage = attachment["passages"][0]
        evidence = {"source_id": attachment["source_id"],
                    "start": passage["start"], "end": passage["end"],
                    "relation": "supports"}
        edge = {"id": "link", "source": "retrieval", "target": "answer",
                "type": "may_affect", "basis": "documented", "label": "may improve",
                "evidence": [evidence]}
        _seed_map_proposal(state, thread_id, edges=[edge])
        before_calls = client.get("/api/status").json()["model_calls"]

        response = client.get(proposal_url)
        assert response.status_code == 200, response.text
        proposal = response.json()
        assert set(proposal) == {"authorship", "title", "rationale", "summary",
                                 "base_revision_id", "proposed_map"}
        assert proposal["authorship"] == "curated_demo"
        assert proposal["base_revision_id"] is None
        receipt = proposal["proposed_map"]["edges"][0]["receipts"][0]
        assert receipt["status"] == "exact_span_verified"
        assert receipt["quote"] == "Source checks reduced unsupported answers in this pilot."
        assert receipt["relation"] == "supports"
        assert client.get(map_url).json()["history"] == []
        assert client.get("/api/status").json()["model_calls"] == before_calls

        accepted = client.put(map_url, json={
            "summary": proposal["summary"], **proposal["proposed_map"],
        })
        assert accepted.status_code == 200, accepted.text
        assert len(accepted.json()["history"]) == 1
        assert client.get(proposal_url).status_code == 404


def test_curated_map_proposal_rejects_wrong_thread_schema_and_forged_span(tmp_path):
    app, _, state = _app(tmp_path)
    with TestClient(app) as client:
        thread_id = _thread(client)
        proposal_url = f"/api/lab/threads/{thread_id}/map/proposal"
        edge = {"id": "link", "source": "retrieval", "target": "answer",
                "type": "may_affect", "basis": "hypothesis", "evidence": []}
        path = _seed_map_proposal(state, thread_id, edges=[edge])
        wrong_thread = json.loads(path.read_text(encoding="utf-8"))
        wrong_thread["thread_id"] = "thread_" + "0" * 32
        path.write_text(json.dumps(wrong_thread), encoding="utf-8")
        assert client.get(proposal_url).status_code == 404

        _seed_map_proposal(state, thread_id, edges=[edge], authorship="model_generated")
        assert client.get(proposal_url).status_code == 404

        attachment = client.post(
            f"/api/lab/threads/{thread_id}/files?filename=paper.txt",
            content=b"A brief source passage.",
            headers={"content-type": "text/plain"},
        ).json()
        forged = {**edge, "basis": "documented", "evidence": [{
            "source_id": attachment["source_id"], "start": 0, "end": 999,
            "relation": "supports",
        }]}
        _seed_map_proposal(state, thread_id, edges=[forged])
        assert client.get(proposal_url).status_code == 404
        assert client.get(f"/api/lab/threads/{thread_id}/map").json()["history"] == []
