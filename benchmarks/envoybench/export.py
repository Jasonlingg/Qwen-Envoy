"""Export a read-only, standalone Studio snapshot; no model or network calls."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from benchmarks.envoybench.demo import HERE, build_demo_payload
from benchmarks.envoybench.run import DEFAULT_DATASET


def snapshot_html(
    payload: dict, *, verified_trace_store: dict[tuple[str, ...], dict] | None = None,
) -> str:
    """Embed verified full turns when supplied, without mutating the live payload."""
    if verified_trace_store is not None:
        payload = copy.deepcopy(payload)
        fields = {"step", "action", "observation", "done", "reasoning", "logprob_diagnostics"}
        for active in [payload["active"], *payload.get("supplementary_runs", [])]:
            for case in active["cases"]:
                for model_key, system in case["systems"].items():
                    if system.get("status") == "not_attempted":
                        if system.get("trajectory"):
                            raise ValueError("unattempted question cannot have a trajectory")
                        continue
                    trace_key = ((active["provenance"]["run_id"], case["id"], model_key)
                                 if active.get("supplementary") else (case["id"], model_key))
                    trace = verified_trace_store.get(trace_key)
                    if (not isinstance(trace, dict) or trace.get("question_id") != case["id"]
                            or trace.get("model_key") != model_key
                            or not isinstance(trace.get("trajectory"), list)
                            or not all(isinstance(step, dict) for step in trace["trajectory"])):
                        raise ValueError(
                            "Full export requires every case's verified saved trajectory"
                        )
                    system["trajectory"] = [
                        {key: value for key, value in step.items() if key in fields}
                        for step in trace["trajectory"]
                    ]
            active["trace_scope"] = "full_saved_trajectories"
    # Keep script terminators inert even when they occur after the old excerpt boundary.
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c")
    embedded = f'<script id="envoybench-data" type="application/json">{encoded}</script>\n'
    html = (HERE / "demo.html").read_text(encoding="utf-8")
    marker = "  <script>"
    if html.count(marker) != 1:
        raise ValueError("Studio HTML needs one main script for snapshot injection")
    return html.replace(marker, embedded + marker, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--review-dir", type=Path)
    parser.add_argument("--judged-review", type=Path)
    parser.add_argument("--reference-review", type=Path)
    parser.add_argument("--provisional", action="store_true")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--token-diagnostic-run", type=Path,
                        help="complete paired dev smoke, displayed separately from scored results")
    parser.add_argument("--supplementary-run", type=Path, action="append", default=[],
                        help="additional saved run, including bounded incomplete attempts")
    parser.add_argument("--qasper-score", type=Path, action="append", default=[],
                        help="official QASPER metric artifact for a loaded run; repeat as needed")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verified_trace_store: dict[tuple[str, ...], dict] = {}
    payload = build_demo_payload(
        run_dir=args.run_dir, review_dir=args.review_dir, judged_review_path=args.judged_review,
        reference_review_path=args.reference_review, provisional=args.provisional,
        dataset=args.dataset,
        token_diagnostic_run=args.token_diagnostic_run,
        supplementary_runs=args.supplementary_run, qasper_scores=args.qasper_score,
        verified_trace_store=verified_trace_store,
    )
    html = snapshot_html(payload, verified_trace_store=verified_trace_store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(html)
    print(f"Read-only snapshot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
