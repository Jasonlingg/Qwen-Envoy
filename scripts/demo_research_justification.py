"""No-model cross-paper justification demo using three public arXiv papers."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.research_justification_mcp import create_server  # noqa: E402
from src.product.arxiv_library import ArxivLibrary  # noqa: E402
from src.product.experiment_discovery import ExperimentDiscovery  # noqa: E402
from src.product.ml_experiment_evidence import EvidenceResolver  # noqa: E402
from src.product.ml_experiment_store import ExperimentStore  # noqa: E402
from src.product.paper_discovery import PaperDiscovery  # noqa: E402

QUESTION = "Should domain mixing be added to our Qwen post-training data curation?"
SOURCES = (
    ("2502.10341", "weborganizer", "Notably, the average accuracy of FineWeb-Edu increases",
     "51.6% baseline accuracy.",
     "supports", "Domain mixing improved FineWeb-Edu in the reported pretraining setup.",
     "Domain mixing added to FineWeb-Edu filtering", "FineWeb-Edu filtering alone"),
    ("2406.11794", "dclm", "model-based filtering is key to assembling",
     "high-quality training set.",
     "qualifies", "Model-based filtering was a key data-curation intervention.",
     "Model-based quality filtering", "Not specified by the selected passage"),
    ("2406.17557", "fineweb", "LLMs pretrained on FineWeb-Edu exhibit",
     "like MMLU and ARC.",
     "qualifies", "FineWeb-Edu improves knowledge and reasoning benchmarks.",
     "Educational-quality filtering", "Not specified by the selected passage"),
)


async def run(output: Path, library_root: Path) -> dict:
    output = output.expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    library = ArxivLibrary(library_root.expanduser())
    store = ExperimentStore(output / "records", evidence_resolver=EvidenceResolver(
        library_root=library.root))
    discovery = ExperimentDiscovery(PaperDiscovery(library.root / "discovery-cache"))
    calls = []

    async def call(client, name, args):
        response = await client.call_tool(name, args)
        calls.append({"tool": name, "is_error": bool(response.is_error)})
        if response.is_error:
            raise ValueError(f"{name} failed: {str(response.content)[:300]}")
        return response.structured_content

    async with Client(create_server(store, library, discovery)) as client:
        refs, comparisons = [], []
        for (arxiv_id, reference_id, anchor, end_anchor, relation, finding,
             intervention, comparator) in SOURCES:
            opened = await call(client, "open_paper", {"arxiv_id": arxiv_id})
            if opened["status"] != "ok" or opened["coverage"] == "abstract_only":
                raise ValueError(f"Full text unavailable for {arxiv_id}")
            version = opened["arxiv_id"]
            text = library.document(version)["text"]
            start = text.index(anchor)
            length = text.index(end_anchor, start) + len(end_anchor) - start
            if length > 1_200:
                raise ValueError(f"Selected passage too long for {version}")
            passage = await call(client, "get_passage", {
                "arxiv_id": version, "start": start, "length": length})
            refs.append({"kind": "paper_span", "reference_id": reference_id,
                         "arxiv_id": version, "start": start, "end": start + length,
                         "quote": passage["text"]})
            comparisons.append({"reference_id": reference_id, "relation": relation,
                "reported_finding": finding, "intervention": intervention,
                "comparator": comparator,
                "setting": "Web-text pretraining; model, data and task differ from Qwen SFT.",
                "transfer_limit": "No direct evidence of a gain on our Qwen post-training task.",
                "independence_note": "Related corpora and evaluations; independence needs review."})

        saved = await call(client, "save_experiment_brief", {"brief": {
            "question": QUESTION, "alternatives": ["Quality filtering alone"],
            "unknowns": ["Whether pretraining findings transfer to our Qwen SFT data."]}})
        experiment_id = saved["experiment_id"]
        note = {"note_id": "weborganizer_quality_comparison", "question": QUESTION,
            "references": refs, "uncertainties": [
                "Quoted passages need claim-level semantic review.",
                "These papers do not establish an effect on our Qwen workload."],
            "justification": {"candidate_method": "Domain mixing after quality filtering",
                "simpler_baseline": "Quality filtering alone",
                "compared_findings": comparisons,
                "transfer_gaps": ["The cited results concern pretraining, not our SFT task.",
                                  "The model scale, corpus and evaluation differ."],
                "cheapest_matched_test": "On a fixed Qwen training pool, compare quality-only "
                    "against quality plus mixing with equal tokens, steps and evaluation.",
                "decision_rule": "Before training, set a minimum worthwhile supported-answer "
                    "gain and error/cost ceilings; adopt mixing only if the matched result "
                    "meets them after claim-level review.",
                "user_reported_context": "The user reports that an author privately emphasized "
                    "data quality; this is not a published source finding."}}
        await call(client, "record_experiment_research", {"experiment_id": experiment_id,
            "expected_revision": 1, "note": note})
        packet = await call(client, "get_research_justification", {
            "experiment_id": experiment_id, "note_id": note["note_id"]})
    report = {"packet": packet, "mcp_calls": calls, "model_requests": 0,
              "claim_level_human_review": "not_done"}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library-root", type=Path, default=Path("~/.envoy/arxiv-library"))
    args = parser.parse_args()
    result = asyncio.run(run(args.output, args.library_root))
    print(json.dumps({"report": str(args.output / "report.json"),
                      "mcp_calls": len(result["mcp_calls"]),
                      "model_requests": result["model_requests"]}))


if __name__ == "__main__":
    main()
