"""Launch a disposable evidence-map demo using real, frozen project reports.

The starting map and proposed update are manually curated for a product test.
No Qwen, Nemotron, GPU, or external API is called. The uploaded reports and map
revisions live in an isolated temporary workspace and disappear on exit.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import tempfile
from pathlib import Path

import uvicorn
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.personal_memory_web import create_app  # noqa: E402

QUESTION = "Does more SFT improve Qwen's source-supported research answers?"
REPORTS = {
    "qasper": ROOT / "docs" / "DEMO_AND_EVIDENCE.md",
    "memory": ROOT / "docs" / "PERSONAL_MEMORY_READINESS_EVAL.md",
    "sweep": ROOT / "docs" / "LEARNING_LOOP_MODEL_SWEEP.md",
}


def _passage(path: Path, first: str, last: str) -> tuple[int, int]:
    text = path.read_text(encoding="utf-8")
    start = text.index(first)
    end = text.index(last, start) + len(last)
    if end - start > 1_200:
        raise ValueError(f"demo passage is too long: {path.name}")
    return start, end


def _evidence(source_id: str, path: Path, first: str, last: str,
              relation: str) -> dict:
    start, end = _passage(path, first, last)
    return {"source_id": source_id, "start": start, "end": end,
            "relation": relation}


def _request(client: TestClient, method: str, url: str, **kwargs) -> dict:
    response = client.request(method, url, **kwargs)
    if not response.is_success:
        raise RuntimeError(f"{method} {url} failed: {response.status_code} {response.text}")
    return response.json()


def seed_demo(client: TestClient, state_dir: Path) -> str:
    """Create one dated belief and an evidence-backed, unaccepted update."""
    thread = _request(client, "POST", "/api/lab/threads", json={
        "question": QUESTION,
        "current_view": (
            "Working hypothesis: more SFT examples should improve supported answers. "
            "This is a curated starting belief for the demo, not a model finding."
        ),
        "effective_date": "2026-09-29",
    })
    thread_id = thread["thread_id"]
    source_ids = {}
    for key, path in REPORTS.items():
        result = _request(
            client, "POST",
            f"/api/lab/threads/{thread_id}/files?filename={path.name}",
            content=path.read_bytes(),
            headers={"content-type": "text/markdown"},
        )
        source_ids[key] = result["source_id"]

    support = _evidence(
        source_ids["qasper"], REPORTS["qasper"],
        "The earlier targeted SFT checkpoint improved over base Qwen",
        "(9/40 to 19/40 semantic passes).", "supports",
    )
    continuation = _evidence(
        source_ids["qasper"], REPORTS["qasper"],
        "On the later set, starting SFT passed",
        "checkpoint passed the preregistered promotion rule.", "challenges",
    )
    abstention = _evidence(
        source_ids["qasper"], REPORTS["qasper"],
        "On 20 answerable",
        "not a proven causal mechanism.", "context",
    )
    memory = _evidence(
        source_ids["memory"], REPORTS["memory"],
        "Two independent AI-assisted reviewers scored",
        "not independent human review;", "challenges",
    )
    sweep = _evidence(
        source_ids["sweep"], REPORTS["sweep"],
        "Within Qwen3, v5 is the leading",
        "pass**.", "context",
    )

    initial_nodes = [
        {"id": "training", "type": "component", "label": "SFT training data",
         "description": "The examples and task mix used to adapt Qwen.",
         "x": 75, "y": 80},
        {"id": "worker", "type": "process", "label": "Qwen research worker",
         "description": "Searches sources and returns evidence-backed answers.",
         "x": 365, "y": 80},
        {"id": "quality", "type": "outcome", "label": "Supported answers",
         "description": "Complete claims with relevant evidence and honest uncertainty.",
         "x": 685, "y": 80},
        {"id": "belief", "type": "assumption", "label": "Belief: more SFT helps",
         "description": "Curated starting hypothesis; the QASPER result supports a narrow version.",
         "x": 365, "y": 350},
    ]
    initial_edges = [
        {"id": "data_to_worker", "source": "training", "target": "worker",
         "type": "flows_to", "label": "training changes policy",
         "explanation": (
             "SFT changes the worker's behavior; the effect on supported "
             "answers is the question to test."
         ),
         "basis": "hypothesis", "evidence": []},
        {"id": "worker_to_quality", "source": "worker", "target": "quality",
         "type": "may_affect", "label": "quality depends on behavior",
         "explanation": (
             "The worker's search, interpretation, and abstention behavior "
             "can affect answer support."
         ),
         "basis": "hypothesis", "evidence": []},
        {"id": "sft_belief", "source": "belief", "target": "quality",
         "type": "may_affect", "label": "more SFT helps?",
         "explanation": (
             "Targeted SFT improved one locked QASPER comparison. This does not yet "
             "establish that additional training improves this research worker across tasks."
         ),
         "basis": "hypothesis", "evidence": [support]},
    ]
    map_url = f"/api/lab/threads/{thread_id}/map"
    saved = _request(client, "PUT", map_url, json={
        "summary": "Curated starting belief, supported by a narrow QASPER result.",
        "nodes": initial_nodes, "edges": initial_edges,
    })
    base_revision_id = saved["revision"]["id"]

    proposed_nodes = [dict(node) for node in initial_nodes] + [
        {"id": "abstention", "type": "outcome", "label": "Honest abstention",
         "description": "Avoid claiming a result that the sources do not establish.",
         "x": 75, "y": 350},
        {"id": "vault_transfer", "type": "outcome", "label": "Vault-task transfer",
         "description": "Source-supported answers on dated personal notes remain unproven.",
         "x": 685, "y": 350},
        {"id": "next_test", "type": "process", "label": "Next test: held-out vault",
         "description": (
             "Same questions, sources, tools, and decoding for base and trained Qwen; "
             "blind human support review plus failures, latency, and cost."
         ), "x": 365, "y": 590},
    ]
    proposed_nodes[3] = {
        **proposed_nodes[3],
        "label": "Belief: gains are task-specific",
        "description": (
            "Targeted SFT helped one paper task. Extra epochs regressed on another "
            "paired set; the current v5 has not shown vault-task improvement."
        ),
    }
    proposed_edges = [dict(edge) for edge in initial_edges]
    proposed_edges[2] = {
        **proposed_edges[2],
        "label": "mixed evidence",
        "explanation": (
            "A targeted QASPER checkpoint beat base on its locked set. Extra "
            "continuation epochs regressed on a different paired set; the v5 "
            "synthetic-vault screen also failed. These results have different "
            "tasks and review limits, so their raw pass counts are not pooled."
        ),
        "evidence": [support, continuation, memory, sweep],
    }
    proposed_edges.extend([
        {"id": "data_to_abstention", "source": "training",
         "target": "abstention", "type": "may_affect",
         "label": "possible tradeoff", "explanation": (
             "Answerable passes rose in one continuation epoch while "
             "insufficient-evidence passes fell. The report suggests a data-mix "
             "explanation but does not establish its cause."
         ), "basis": "hypothesis", "evidence": [abstention]},
        {"id": "test_to_vault", "source": "next_test", "target": "vault_transfer",
         "type": "tests", "label": "test transfer",
         "explanation": (
             "Compare base and selected trained Qwen on held-out real-vault "
             "questions with blind human support review, execution failures, "
             "latency, and cost."
         ), "basis": "hypothesis", "evidence": []},
    ])
    proposal = {
        "thread_id": thread_id,
        "authorship": "curated_demo",
        "title": "New results narrow the SFT belief",
        "rationale": (
            "The early QASPER gain supports task-specific training. A paired "
            "continuation regressed, and a provisional synthetic-vault screen "
            "found v5 worse than base. Review the exact passages before "
            "accepting this narrower belief."
        ),
        "summary": (
            "Narrow the SFT belief, show conflicting evidence, "
            "and name the next vault test."
        ),
        "base_revision_id": base_revision_id,
        "proposed_map": {"nodes": proposed_nodes, "edges": proposed_edges},
    }
    proposal_dir = state_dir / "learning_lab" / "map_proposals"
    proposal_dir.mkdir(parents=True, exist_ok=True)
    (proposal_dir / f"{thread_id}.json").write_text(
        json.dumps(proposal, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return thread_id


def prepare_demo(workspace: Path):
    """Build an isolated app and return its deep link and thread ID."""
    vault = workspace / "vault"
    state = workspace / "state"
    vault.mkdir(parents=True)
    state.mkdir(parents=True)
    app = create_app(vault, state)
    with TestClient(app) as client:
        thread_id = seed_demo(client, state)
    return app, thread_id


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    def stop_on_term(_signum, _frame):
        raise KeyboardInterrupt

    previous_term = signal.signal(signal.SIGTERM, stop_on_term)
    try:
        with tempfile.TemporaryDirectory(prefix="envoy-research-map-demo-") as temp:
            app, thread_id = prepare_demo(Path(temp))
            print(json.dumps({
                "url": f"http://127.0.0.1:{args.port}/lab/map?thread={thread_id}",
                "question": QUESTION,
                "mode": "curated_source_linked_demo",
                "model_calls": 0,
                "temporary": True,
            }), flush=True)
            uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous_term)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
