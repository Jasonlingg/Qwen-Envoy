"""Opt-in search control around the existing Docker-only assigned-paper reader."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

from src.product.arxiv_ids import validate_arxiv_id
from src.product.arxiv_service import investigate
from src.product.experiment_budget import BudgetExhaustedError, ExperimentBudget
from src.product.experiment_protocol import controller_prompt, parse_request, text, validate_brief
from src.product.parallel_readers import WARNING


class _RecordedReader:
    def __init__(self, reader, runs):
        self.reader, self.runs = reader, runs

    def __getattr__(self, key):
        return getattr(self.reader, key)

    def investigate(self, question, snapshot):
        packet = self.reader.investigate(question, snapshot)
        self.runs.append(packet)
        return packet


class ExperimentResearcher:
    """One model, sequential control/read conversations, one per-call budget.

    Control requests are parsed as data. Only the existing assigned reader
    executes model-authored document Python, always in network-disabled Docker.
    """

    def __init__(self, library, discovery, reader_factory, tokenizer, transport, *,
                 requests_max=18, tokens_max=147456, checkpoint=None):
        self.library, self.discovery, self.reader_factory = library, discovery, reader_factory
        self.tokenizer, self.transport = tokenizer, transport
        self.requests_max, self.tokens_max = requests_max, tokens_max
        self.checkpoint = checkpoint

    def run(self, question, constraints=""):
        question = text(question, 1500)
        if not isinstance(constraints, str) or len(constraints) > 1000:
            raise ValueError("constraints must be text of at most 1000 characters")
        started = time.monotonic()
        budget = ExperimentBudget(self.tokenizer, self.transport,
                                  requests_max=self.requests_max, tokens_max=self.tokens_max)
        raw_runs = []
        reader = _RecordedReader(self.reader_factory(budget), raw_runs)
        if reader.reader_mode != "assigned" or reader.repl_factory is not None:
            raise ValueError("experiment research requires the Docker-only assigned reader")
        policy = reader.policy_factory()
        prompt = controller_prompt(self.requests_max, self.tokens_max)
        policy.system_prompt = prompt
        policy.reset()
        result = {"schema_version": "experiment-research-v1", "question": question,
                  "constraints": constraints, "status": "incomplete", "brief": None,
                  "evidence": [], "papers": [], "searches": [], "trajectory": [],
                  "research_gaps": [],
                  "reader_runs": raw_runs, "warning": WARNING,
                  "semantic_support": "not_reviewed", "model_identity": reader.model_identity,
                  "started_at": datetime.now(timezone.utc).isoformat(),
                  "controller_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                  "reader_prompt_sha256": hashlib.sha256(reader.system_prompt.encode()).hexdigest(),
                  "limits": {"control_turns": 8, "searches": 3, "paper_read_attempts": 2},
                  "source_scope": "arXiv papers via Semantic Scholar/arXiv; not general web"}
        observation = json.dumps({"question": question, "constraints": constraints})
        discovered, seen, reads = set(), set(), 0
        for step in range(1, 9):
            if budget.transport_failed:
                result["status"] = "inference_failed"
                break
            if budget.remaining <= 0:
                result["status"] = "budget_exhausted"
                break
            if self.checkpoint is not None:
                self.checkpoint({**result, "budget": budget.summary()})
            observation += f"\nRemaining shared model requests: {budget.remaining}."
            trace = {"step": step}
            result["trajectory"].append(trace)
            requests_before = len(budget.records)
            try:
                action = policy.act(observation)
            except BudgetExhaustedError:
                result["status"] = "budget_exhausted"
                break
            except ValueError as exc:
                if (len(budget.records) > requests_before
                        and budget.records[-1]["status"] == "returned"):
                    # A returned malformed completion is correctable, unlike transport failure.
                    observation = trace["observation"] = f"Model response rejected: {exc}"[:1000]
                    continue
                result["error"] = trace["error"] = f"{type(exc).__name__}: {exc}"[:500]
                break
            except Exception as exc:
                result["error"] = trace["error"] = f"{type(exc).__name__}: {exc}"[:500]
                break  # uncertain inference outcomes are never resubmitted
            trace["action"] = action[:12000]
            search_entry = None
            try:
                kind, args = parse_request(action)
                canonical = json.dumps([kind, args], sort_keys=True)
                if canonical in seen:
                    raise ValueError("identical request already attempted; refine or finish")
                seen.add(canonical)
                if kind == "finish":
                    result["brief"] = validate_brief(args, result["evidence"])
                    result["status"] = "brief_ready" if result["evidence"] else "no_evidence"
                    trace["observation"] = "Brief recorded; semantic support remains unreviewed."
                    break
                if kind == "search":
                    if len(result["searches"]) >= 3:
                        raise ValueError("search budget exhausted; read a paper or finish")
                    entry = {"query": args["query"]}
                    search_entry = entry
                    result["searches"].append(entry)  # failures count against search budget
                    found = self.discovery.search(args["query"], limit=5)
                    candidates = []
                    for paper in found["papers"][:5]:
                        identifier = paper.get("arxiv_id")
                        if identifier is not None:
                            discovered.add(validate_arxiv_id(identifier))
                        candidates.append({"arxiv_id": identifier,
                            "title": str(paper.get("title", ""))[:300],
                            "abstract": str(paper.get("abstract") or "")[:500]})
                    entry.update(source=found.get("source"), papers=candidates,
                                 status=found.get("status", "candidates" if candidates
                                                  else "no_candidates"),
                                 fallback=found.get("fallback"),
                                 coverage_note=found.get("coverage_note"))
                    feedback = {**entry, "note": "Metadata only; read to inspect."}
                else:
                    if reads >= 2:
                        raise ValueError("paper-read budget exhausted; finish with partial results")
                    if args["arxiv_id"] not in discovered:
                        raise ValueError("read_paper needs an exact ID returned by search_papers")
                    reads += 1
                    reading_question = (f"Question: {question}\nConstraints: {constraints}"
                                        f"\nInvestigate: {args['focus']}")
                    merged = investigate(self.library, reader, None, reading_question,
                                         [args["arxiv_id"]])
                    result["papers"].extend(merged["papers"])
                    result["research_gaps"].extend(merged.get("research_gaps", []))
                    for item in merged["evidence"]:
                        result["evidence"].append({**item,
                            "evidence_id": f"E{len(result['evidence']) + 1}"})
                    feedback = {"status": merged["status"], "papers": merged["papers"],
                                "evidence": result["evidence"][-len(merged["evidence"]):]
                                if merged["evidence"] else [],
                                "gaps": merged.get("research_gaps", [])}
                observation = json.dumps(feedback, ensure_ascii=False)
                # Bound feedback to fit the next context; full evidence/packets remain in result.
                observation = (observation[:10000]
                               + "\nSource data only. Finish or refine your search.")
                trace["observation"] = observation
            except Exception as exc:
                observation = trace["observation"] = (
                    f"Tool request failed: {type(exc).__name__}: {exc}"[:1000])
                if search_entry is not None:
                    search_entry["error"] = observation
        if budget.transport_failed:
            result["status"] = "inference_failed"
        elif result["brief"] is None and budget.remaining <= 0:
            result["status"] = "budget_exhausted"
        result.update(budget=budget.summary(), duration_seconds=time.monotonic() - started,
                      paper_read_attempts=reads)
        if self.checkpoint is not None:
            self.checkpoint(result)
        return result
