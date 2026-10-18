"""Prepare, explicitly execute, or review a frozen direct-vs-Qwen development pilot."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.research_pilot_budget import RequestBudget  # noqa: E402
from src.eval.research_pilot_bundle import (  # noqa: E402
    load_bundle,
    prepare,
    write_json,
)
from src.eval.research_pilot_run import run_pilot  # noqa: E402
from src.eval.research_pilot_runtime import build_models  # noqa: E402


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Freeze cached papers; zero model calls")
    prep.add_argument("--questions", type=Path,
                      default=ROOT / "data/eval/research-workflow-pilot-v1.json")
    prep.add_argument("--library-root", type=Path, default=ROOT / "out/arxiv-mcp-live-library")
    prep.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="Verify frozen inputs; zero model calls")
    check.add_argument("--bundle", type=Path, required=True)
    check.add_argument("--docker", action="store_true", help="Scripted sandbox check; no model")
    execute = commands.add_parser("run", help="Explicit live inference; no GPU provisioning")
    execute.add_argument("--bundle", type=Path, required=True)
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--execute", action="store_true", help="Permit configured model calls")
    review = commands.add_parser("review", help="Write blank claim-level human review")
    review.add_argument("--run", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("report", help="Score completed review; never generate judgments")
    for option in ("run", "review", "key", "output"):
        score.add_argument("--" + option, type=Path, required=True)
    score.add_argument("--costs", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.questions, args.library_root, args.output, ROOT)
        print(json.dumps({"status": result["status"], "questions": len(result["plan"]) // 2,
                          "bundle_sha256": result["bundle_sha256"], "model_requests": 0}))
    elif args.command == "check":
        manifest, data, library = load_bundle(args.bundle)
        for question in data["questions"]:
            from src.eval.research_pilot_tools import FrozenPilotTools
            tools = FrozenPilotTools(library, question)
            found = tools.find_evidence(question["question"], question["arxiv_ids"], top_k=3)
            for span in found["evidence"]:
                passage = tools.get_passage(span["arxiv_id"], span["start"],
                                             min(1_200, span["end"] - span["start"]))
                if passage["text"] != span["quote"][:len(passage["text"])]:
                    raise ValueError("Source-only round trip mismatch")
        result = {"status": "frozen_inputs_verified", "papers": len(manifest["papers"]),
                  "questions": len(data["questions"]), "model_requests": 0}
        if args.docker:
            from src.eval.research_pilot_probe import probe_sandbox
            digest = subprocess.check_output(["docker", "image", "inspect", "rlm-sandbox",
                "--format", "{{.Id}}"], text=True).strip()
            result["sandbox_probe"] = probe_sandbox(library, data["papers"][0]["arxiv_id"], digest)
            if result["sandbox_probe"]["status"] != "evidence_found":
                raise ValueError("Scripted Docker probe failed")
            result["sandbox_probe"].pop("packet")
        print(json.dumps(result))
    elif args.command == "run":
        if not args.execute:
            raise ValueError("Live inference requires explicit --execute; no calls made")
        if args.output.exists():
            raise ValueError("Run destination already exists")
        config = _read(args.config)
        digest = subprocess.check_output(["docker", "image", "inspect", "rlm-sandbox",
            "--format", "{{.Id}}"], text=True).strip()
        # Persist reservations before each dispatch and after every completion.
        budget = RequestBudget(observer=lambda snapshot: write_json(
            args.output / "requests.json", snapshot))
        host, reader, tokenizers = build_models(config, budget)
        reader.image = digest  # Execute these inspected bytes, not a mutable tag.
        record = {**config, "tokenizers": tokenizers, "docker_image_digest": digest,
                  "reader_model_identity": reader.model_identity,
                  "identity_attestation": "declared endpoints; remote weights not attested",
                  "max_reader_steps": 4, "max_active_readers": 2,
                  "output_limit": 1024, "temperature": 0, "top_p": 1, "seed": 42,
                  "thinking": False, "reasoner": "off", "automatic_retries": 0}
        result = run_pilot(args.bundle, args.output, host_factory=host, investigator=reader,
                          budget=budget, config=record, repo=ROOT)
        print(json.dumps({k: result[k] for k in
                          ("status", "actual_model_requests", "actual_qwen_requests")}))
    elif args.command == "review":
        from src.eval.research_pilot_review import build_review
        if args.output.exists():
            raise ValueError("Review destination already exists")
        packet, key = build_review(_read(args.run))
        write_json(args.output / "review.json", packet)
        write_json(args.output / "identity-key.json", key)
        print(json.dumps({"status": "human_review_not_performed", "answers": len(packet["rows"])}))
    else:
        from src.eval.research_pilot_review import score_review
        result = score_review(_read(args.run), _read(args.review), _read(args.key),
                              costs=_read(args.costs) if args.costs else None)
        write_json(args.output, result)
        print(json.dumps(result))


if __name__ == "__main__":
    main()
