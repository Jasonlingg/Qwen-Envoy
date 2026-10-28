"""The browser workbench uses the same durable stores as experiment-mode MCP."""

from fastapi.testclient import TestClient

from scripts.launch_experiment_workbench import create_app
from src.product.ml_experiment_store import ExperimentStore


def test_workbench_reads_sources_and_recalls_decision_after_restart(tmp_path):
    atlas_root = tmp_path / "atlas"
    (atlas_root / "docs").mkdir(parents=True)
    (atlas_root / "evidence.json").write_text('{"result":"inconclusive"}')
    (atlas_root / "docs" / "EXPERIMENT_ATLAS.md").write_text(
        "| ID | Question and actual attempt | Saved observation | "
        "Decision and next question | Primary evidence |\n"
        "| --- | --- | --- | --- | --- |\n"
        "| reader-test | Did Qwen reader concurrency help? | Inconclusive | "
        "Measure paired tasks | [Evidence](../evidence.json) |\n")
    records = tmp_path / "records"
    experiment_id = ExperimentStore(records).save_brief(
        {"question": "Did Qwen reader concurrency help?"})["experiment_id"]
    app = create_app(records, atlas_root=atlas_root)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.get("/").status_code == 200
        assert client.get("/experiment_workbench.js").status_code == 200
        found = client.get("/api/atlas?q=reader").json()
        assert found["matches"][0]["atlas_id"] == "reader-test"
        source = client.get("/api/atlas/reader-test/source/0").json()
        assert source["text"] == '{"result":"inconclusive"}'
        assert len(source["sha256"]) == 64
        assert client.get("/api/experiments?q=reader").json()["matches"][0][
            "experiment_id"] == experiment_id
        assert client.post("/api/decisions", json={"experiment_id": experiment_id,
            "question": "Should we claim a Qwen reader speedup?"}).status_code == 403
        headers = {"Origin": "http://127.0.0.1"}
        thread = client.post("/api/decisions", headers=headers, json={
            "experiment_id": experiment_id,
            "question": "Should we claim a Qwen reader speedup?"}).json()
        saved = client.post("/api/decisions/record", headers=headers, json={
            "thread_id": thread["thread_id"], "expected_revision": 1,
            "experiment_id": experiment_id, "outcome": "defer",
            "rationale": "The recorded test did not establish a speedup.",
            "revisit_when": "After matched reader tasks are measured."}).json()
        assert saved["latest_decision"]["choice"]["author_kind"] == "agent_proposed"
        assert len(saved["latest_decision"]["basis_receipts"][0]["brief_sha256"]) == 64
        assert client.get("/api/atlas", headers={"Host": "evil.example"}).status_code == 403

    with TestClient(create_app(records, atlas_root=atlas_root),
                    base_url="http://127.0.0.1") as restarted:
        found = restarted.get("/api/decisions?q=reader").json()
        assert found["matches"][0]["thread_id"] == thread["thread_id"]
        resumed = restarted.get(f"/api/decisions/{thread['thread_id']}").json()
        assert resumed == saved
