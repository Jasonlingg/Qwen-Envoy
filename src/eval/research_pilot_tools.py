"""The pilot's cache-only tools and process-wide reader admission boundary."""

from __future__ import annotations

import threading

from src.product.evidence_search import find_evidence
from src.product.parallel_readers import investigate_papers


class ReaderAdmission:
    """Share one instance across every question/arm; lease spans the full reader."""

    def __init__(self, investigator, *, limit: int = 2, record_packet=None):
        if type(limit) is not int or not 1 <= limit <= 2:
            raise ValueError("Pilot reader limit must be 1..2")
        self.investigator = investigator
        self.system_prompt = investigator.system_prompt
        self.slots = threading.BoundedSemaphore(limit)
        self.record_packet = record_packet

    def investigate(self, question, snapshot):
        with self.slots:
            packet = self.investigator.investigate(question, snapshot)
            if self.record_packet is not None:
                self.record_packet(packet)
            return packet


class FrozenPilotTools:
    def __init__(self, library, question: dict, reader=None):
        self.library, self.question, self.reader = library, question, reader
        self.allowed = frozenset(question["arxiv_ids"])
        self.documents = {v: library.document(v) for v in self.allowed}
        self.seen: dict[str, list[tuple[int, int]]] = {v: [] for v in self.allowed}

    def _versions(self, arxiv_ids):
        if (not isinstance(arxiv_ids, list) or not arxiv_ids
                or any(not isinstance(v, str) for v in arxiv_ids)
                or not set(arxiv_ids) <= self.allowed):
            raise ValueError("Use only this question's frozen version-pinned arxiv_ids")
        return list(dict.fromkeys(arxiv_ids))

    def _observe(self, evidence):
        for item in evidence:
            version = item["arxiv_id"]
            self.seen[version].append((item["start"], item["end"]))

    def find_evidence(self, query, arxiv_ids, top_k=8):
        if type(top_k) is not int or not 1 <= top_k <= 8:
            raise ValueError("Pilot top_k must be 1..8")
        result = find_evidence(self.library, query, self._versions(arxiv_ids), top_k)
        self._observe(result["evidence"])
        return result

    def get_passage(self, arxiv_id, start=0, length=1_200):
        self._versions([arxiv_id])
        text = self.documents[arxiv_id]["text"]
        if (type(start) is not int or type(length) is not int or not 0 <= start < len(text)
                or not 1 <= length <= 3_000):
            raise ValueError("Pilot passage needs a valid start and length 1..3000")
        end = min(start + length, len(text))
        result = {"arxiv_id": arxiv_id, "start": start, "end": end, "text": text[start:end]}
        self._observe([result])
        return result

    def investigate(self, question, arxiv_ids):
        if self.reader is None:
            raise ValueError("Qwen reader is unavailable")
        if question != self.question["question"]:
            raise ValueError("Delegate the original pilot question unchanged")
        result = investigate_papers(self.reader, self.library, question,
                                    self._versions(arxiv_ids), max_workers=2)
        self._observe(result["evidence"])
        return result

    def callbacks(self, arm):
        tools = {"find_evidence": self.find_evidence, "get_passage": self.get_passage}
        if arm == "delegated":
            tools["investigate"] = self.investigate
        return tools

    def validate_finish(self, submission):
        claims = []
        for index, claim in enumerate(submission["claims"], 1):
            evidence = []
            for item in claim["evidence"]:
                version, start, end = item["arxiv_id"], item["start"], item["end"]
                self._versions([version])
                text = self.documents[version]["text"]
                if (type(start) is not int or type(end) is not int
                        or not 0 <= start < end <= len(text) or end - start > 1_200
                        or not any(a <= start < end <= b for a, b in self.seen[version])):
                    raise ValueError("Cite an inspected exact span, at most1200 characters")
                evidence.append({**item, "quote": text[start:end], "provenance_valid": True})
            claims.append({"claim_id": f"C{index}", "text": claim["text"],
                           "evidence": evidence, "semantic_support": "not_reviewed"})
        return {"claims": claims, "semantic_support": "not_reviewed"}
