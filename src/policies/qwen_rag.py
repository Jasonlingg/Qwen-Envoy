"""One-pass sparse RAG baseline using the same Qwen checkpoint as Envoy."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from rank_bm25 import BM25Okapi

from src.env.corpus import Corpus
from src.policies.qwen_sft_policy import QwenSFTPolicy

RAG_SYSTEM_PROMPT = """You answer research questions from retrieved paper passages.

Use only the supplied passages. Do not write Python and do not request more information.
Distinguish findings reported by papers from advice or inference about the user's system.
If the passages cannot support the requested claim, say so instead of guessing.

Return exactly one submission in this format:
SUBMIT: <concise answer> CITATIONS: ["doc_id"] EVIDENCE: [{"doc_id":"doc_id","start":0,"end":100}]

Citations identify the papers used. Evidence spans must use the exact start and end offsets shown
beside supplied passages. Include every paper needed for a comparison. For a fully unsupported
question, use CITATIONS: [] EVIDENCE: [].
"""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _extract_question(observation: str) -> str:
    marker = "Question:"
    if marker in observation:
        return observation.rsplit(marker, 1)[-1].strip()
    return observation.strip()


class QwenRAGPolicy(QwenSFTPolicy):
    """Retrieve a fixed context once, then ask the SFT checkpoint for one answer."""

    def __init__(
        self,
        corpus: Corpus,
        checkpoint_path: str | None = None,
        base_model: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_docs: int = 4,
        chunks_per_doc: int = 3,
        max_context_chars: int = 12_000,
    ) -> None:
        if top_docs < 1 or chunks_per_doc < 1 or max_context_chars < 500:
            raise ValueError("RAG retrieval limits must be positive")
        super().__init__(
            checkpoint_path=checkpoint_path,
            base_model=base_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=RAG_SYSTEM_PROMPT,
        )
        self.corpus = corpus
        self.top_docs = top_docs
        self.chunks_per_doc = chunks_per_doc
        self.max_context_chars = max_context_chars
        self._answered = False
        self._retrieval_metadata: dict[str, Any] = {}
        self._passages = self._build_passages()
        self._bm25 = BM25Okapi([
            _tokenize(f"{item['title']} {item['title']} {item['text']}")
            for item in self._passages
        ])
        self.config = {
            "retrieval": "BM25 passage retrieval with document diversification",
            "top_docs": top_docs,
            "chunks_per_doc": chunks_per_doc,
            "max_context_chars": max_context_chars,
            "one_pass": True,
        }

    def _build_passages(self) -> list[dict[str, Any]]:
        passages = []
        step = self.corpus.chunk_size - self.corpus.chunk_overlap
        for info in sorted(self.corpus.list_documents(), key=lambda item: item.doc_id):
            document = self.corpus.get_document(info.doc_id)
            if document is None:
                continue
            title = document.get("title", info.doc_id)
            text = document["text"]
            for chunk_id, start in enumerate(range(0, len(text), step)):
                end = min(start + self.corpus.chunk_size, len(text))
                passages.append({
                    "doc_id": info.doc_id,
                    "title": title,
                    "chunk_id": chunk_id,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                })
        if not passages:
            raise ValueError("Cannot build RAG index from an empty corpus")
        return passages

    def _retrieve(self, question: str) -> list[dict[str, Any]]:
        scores = self._bm25.get_scores(_tokenize(question))
        by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item, score in zip(self._passages, scores):
            by_doc[item["doc_id"]].append({**item, "score": float(score)})
        for items in by_doc.values():
            items.sort(key=lambda value: (-value["score"], value["chunk_id"]))

        ranked_docs = sorted(
            by_doc,
            key=lambda doc_id: (-by_doc[doc_id][0]["score"], doc_id),
        )[: self.top_docs]
        selected = []
        total_chars = 0
        for rank in range(self.chunks_per_doc):
            for doc_id in ranked_docs:
                items = by_doc[doc_id]
                if rank >= len(items):
                    continue
                item = items[rank]
                header = (
                    f"[{item['doc_id']} start={item['start']} end={item['end']} "
                    f"title={item['title']!r}]\n"
                )
                size = len(header) + len(item["text"]) + 2
                if selected and total_chars + size > self.max_context_chars:
                    continue
                selected.append(item)
                total_chars += size
        return selected

    @staticmethod
    def _context(passages: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            f"[{item['doc_id']} start={item['start']} end={item['end']} "
            f"title={item['title']!r}]\n{item['text']}"
            for item in passages
        )

    def act(self, observation: str) -> str:
        if self._answered:
            return "SUBMIT: No supported answer CITATIONS: [] EVIDENCE: []"
        question = _extract_question(observation)
        passages = self._retrieve(question)
        context = self._context(passages)
        self._retrieval_metadata = {
            "question": question,
            "context_chars": len(context),
            "retrieved_passages": passages,
        }
        prompt = f"Retrieved passages:\n\n{context}\n\nQuestion: {question}"
        answer = super().act(prompt)
        self._answered = True
        return answer

    def reset(self) -> None:
        super().reset()
        self._answered = False
        self._retrieval_metadata = {}

    def eval_metadata(self) -> dict[str, Any]:
        return self._retrieval_metadata
