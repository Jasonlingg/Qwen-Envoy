"""The research-map demo is grounded in frozen reports, not generated output."""

from fastapi.testclient import TestClient

from scripts.launch_research_map_demo import prepare_demo


def test_curated_research_case_has_reviewable_supported_and_challenged_update(tmp_path):
    app, thread_id = prepare_demo(tmp_path)
    with TestClient(app) as client:
        map_url = f"/api/lab/threads/{thread_id}/map"
        current = client.get(map_url).json()
        proposal_response = client.get(f"{map_url}/proposal")
        assert proposal_response.status_code == 200, proposal_response.text
        proposal = proposal_response.json()

        assert proposal["authorship"] == "curated_demo"
        assert proposal["base_revision_id"] == current["revision"]["id"]
        assert len(client.get(f"/api/lab/threads/{thread_id}").json()["sources"]) == 3
        initial = current["revision"]["edges"]
        assert len(initial) == 3
        assert initial[2]["receipts"][0]["status"] == "exact_span_verified"
        assert initial[2]["receipts"][0]["relation"] == "supports"

        proposed_edges = proposal["proposed_map"]["edges"]
        tension = next(edge for edge in proposed_edges if edge["id"] == "sft_belief")
        assert [receipt["relation"] for receipt in tension["receipts"]] == [
            "supports", "challenges", "challenges", "context",
        ]
        assert all(receipt["status"] == "exact_span_verified"
                   for receipt in tension["receipts"])
        assert any(edge["type"] == "tests" for edge in proposed_edges)
        assert len(current["history"]) == 1  # Reading the proposal never accepts it.
        assert client.get("/api/status").json()["model_calls"] == 0

        accepted = client.put(map_url, json={
            "nodes": proposal["proposed_map"]["nodes"],
            "edges": proposed_edges,
            "summary": proposal["summary"],
        })
        assert accepted.status_code == 200, accepted.text
        assert len(accepted.json()["history"]) == 2
        assert client.get(f"{map_url}/proposal").status_code == 404
        assert client.get("/api/status").json()["model_calls"] == 0
