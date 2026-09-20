"""Tool preamble injected into the REPL at session start.

Defines search(), read(), passage(), extract(), scan(), aggregate(),
search_within(), and verify() functions that operate on the mounted corpus
directory. Includes memoization so re-running the cumulative script doesn't
repeat expensive file reads.
"""

TOOL_PREAMBLE = '''
import json
import math
import os
import re
from pathlib import Path
from collections import Counter

# Memoization cache — persists across cumulative script re-runs
if "_memo" not in dir():
    _memo = {}

_corpus_env = os.environ.get("CORPUS_DIR", "")
if _corpus_env and Path(_corpus_env).exists():
    CORPUS_DIR = Path(_corpus_env)
elif Path("/workspace/data/corpus").exists():
    CORPUS_DIR = Path("/workspace/data/corpus")
elif Path("/workspace/corpus").exists():
    CORPUS_DIR = Path("/workspace/corpus")
else:
    CORPUS_DIR = Path("data/corpus")

def _load_doc(doc_id: str) -> dict | None:
    """Load a document by ID, with memoization."""
    if doc_id in _memo:
        return _memo[doc_id]
    path = CORPUS_DIR / f"{doc_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        doc = json.load(f)
    _memo[doc_id] = doc
    return doc

def _load_all_docs() -> list[dict]:
    """Load all documents, with memoization."""
    if "_all_docs" in _memo:
        return _memo["_all_docs"]
    docs = []
    for path in sorted(CORPUS_DIR.glob("*.json")):
        with open(path) as f:
            docs.append(json.load(f))
    _memo["_all_docs"] = docs
    return docs

def _get_chunks(chunk_size=500, chunk_overlap=100) -> list[dict]:
    """Build overlapping text chunks from all documents, with memoization."""
    cache_key = f"_chunks_{chunk_size}_{chunk_overlap}"
    if cache_key in _memo:
        return _memo[cache_key]
    chunks = []
    for doc in _load_all_docs():
        text = doc["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append({
                "doc_id": doc["doc_id"],
                "title": doc.get("title", doc["doc_id"]),
                "text": text[start:end],
                "start": start,
            })
            start += chunk_size - chunk_overlap
    _memo[cache_key] = chunks
    return chunks

def _idf_scores() -> dict[str, float]:
    """Compute IDF (inverse document frequency) for all terms."""
    if "_idf" in _memo:
        return _memo["_idf"]
    docs = _load_all_docs()
    n = len(docs)
    df = Counter()
    for doc in docs:
        terms = set(doc["text"].lower().split())
        for t in terms:
            df[t] += 1
    # Smoothed IDF stays positive even for a one-document corpus. The old
    # log(n / (1 + freq)) formula made every term negative when n == 1, so
    # search() discarded every matching result via its score > 0 check.
    idf = {t: math.log((n + 1) / (freq + 1)) + 1.0 for t, freq in df.items()}
    _memo["_idf"] = idf
    return idf

def search(query: str, top_k: int = 5, method: str = "keyword") -> list[dict]:
    """Search over corpus. method: 'keyword' (BM25-like) or 'chunk' (chunk-level TF-IDF).

    'keyword' — scores whole documents by TF-IDF, returns top matches with preview.
    'chunk'   — scores individual 500-char chunks, finds buried facts in long docs.
    """
    query_terms = query.lower().split()
    idf = _idf_scores()

    if method == "chunk":
        chunks = _get_chunks()
        scored = []
        for chunk in chunks:
            text_lower = chunk["text"].lower()
            score = sum(text_lower.count(t) * idf.get(t, 1.0) for t in query_terms)
            if score > 0:
                scored.append({
                    "doc_id": chunk["doc_id"],
                    "title": chunk["title"],
                    "chunk": chunk["text"],
                    "score": round(score, 2),
                    "offset": chunk["start"],
                })
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
    else:
        # Default: document-level TF-IDF search
        results = []
        for doc in _load_all_docs():
            text_lower = doc["text"].lower()
            score = sum(text_lower.count(t) * idf.get(t, 1.0) for t in query_terms)
            if score > 0:
                results.append({
                    "doc_id": doc["doc_id"],
                    "title": doc.get("title", doc["doc_id"]),
                    "chunk": doc["text"][:500],
                    "score": round(score, 2),
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

def read(doc_id: str) -> str:
    """Read full document text by ID."""
    doc = _load_doc(doc_id)
    if doc is None:
        return f"ERROR: Document '{doc_id}' not found"
    return doc["text"]

def passage(doc_id: str, start: int = 0, length: int = 1600) -> dict:
    """Return an exact, bounded passage with stable character offsets."""
    doc = _load_doc(doc_id)
    if doc is None:
        return {"error": f"Document '{doc_id}' not found"}
    if type(start) is not int or type(length) is not int:
        return {"error": "start and length must be integers"}
    if not 0 <= start < len(doc["text"]):
        return {"error": "start must be inside the document"}
    if not 1 <= length <= 3000:
        return {"error": "length must be between 1 and 3000"}
    end = min(start + length, len(doc["text"]))
    return {
        "doc_id": doc_id,
        "start": start,
        "end": end,
        "text": doc["text"][start:end],
    }

def extract(doc_id: str, pattern: str) -> list[str]:
    """Extract text matching a regex pattern from a document."""
    doc = _load_doc(doc_id)
    if doc is None:
        return [f"ERROR: Document '{doc_id}' not found"]
    return re.findall(pattern, doc["text"])

def scan(
    doc_id: str,
    pattern: str,
    max_hits: int = 6,
    context_chars: int = 900,
) -> list[dict]:
    """Return bounded, non-overlapping contexts around full-document regex matches.

    This is the context-preserving counterpart to ``extract()``. It lets an
    agent search for answer-shaped language anywhere in a known document
    without printing the full text or writing its own windowing loop.
    """
    doc = _load_doc(doc_id)
    if doc is None:
        return [{"error": f"Document '{doc_id}' not found"}]
    if not isinstance(pattern, str) or not pattern:
        return [{"error": "pattern must be a non-empty string"}]
    if len(pattern) > 160:
        return [{"error": "pattern must be at most 160 characters"}]
    if type(max_hits) is not int or not 1 <= max_hits <= 10:
        return [{"error": "max_hits must be an integer between 1 and 10"}]
    if type(context_chars) is not int or not 200 <= context_chars <= 1600:
        return [{"error": "context_chars must be an integer between 200 and 1600"}]

    text = doc["text"]
    before = context_chars // 3
    results = []
    try:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            start = max(0, match.start() - before)
            end = min(len(text), start + context_chars)
            # Multiple nearby matches should not spend the bounded result
            # budget on nearly identical evidence.
            if results and start < results[-1]["end"]:
                continue
            results.append({
                "doc_id": doc_id,
                "text": text[start:end],
                "offset": start,
                "end": end,
                "match": match.group(0)[:160],
            })
            if len(results) == max_hits:
                break
    except re.error as error:
        return [{"error": f"Invalid regex: {error}"}]
    return results

def aggregate(doc_ids: list[str], field: str) -> list[dict]:
    """Extract a JSON metadata field across multiple documents."""
    results = []
    for doc_id in doc_ids:
        doc = _load_doc(doc_id)
        if doc is None:
            results.append({"doc_id": doc_id, "error": "not found"})
            continue
        value = doc.get("metadata", {}).get(field)
        results.append({"doc_id": doc_id, field: value})
    return results

try:
    _default_search_within_top_k = int(os.environ.get("ENVOY_SEARCH_WITHIN_TOP_K", "3"))
except ValueError:
    _default_search_within_top_k = 3
_default_search_within_top_k = max(1, _default_search_within_top_k)
_search_within_mode = os.environ.get("ENVOY_SEARCH_WITHIN_MODE", "raw")
if _search_within_mode not in {"raw", "dedupe_merge", "ranked_diverse"}:
    _search_within_mode = "raw"

def _dedupe_merge_windows(windows: list[dict], text: str, top_k: int) -> list[dict]:
    """Return distinct evidence regions while keeping rank and size bounded.

    Ranking happens before this function. Overlapping high-ranked windows are
    merged into one passage instead of consuming several result slots. A merged
    region is capped so a frequent query term cannot expand into an entire paper.
    Selection stops as soon as ``top_k`` distinct regions have been filled. This
    prevents lower-ranked overlaps from rewriting the context around results that
    already won a slot.
    """
    regions = []
    max_passage_chars = 900
    for candidate in windows:
        overlapping = [
            (min(candidate["end"], region["end"])
             - max(candidate["offset"], region["offset"]), index)
            for index, region in enumerate(regions)
            if candidate["offset"] < region["end"]
            and candidate["end"] > region["offset"]
        ]
        if overlapping:
            _, index = max(overlapping)
            region = regions[index]
            start = min(region["offset"], candidate["offset"])
            end = max(region["end"], candidate["end"])
            overlaps_other_region = any(
                other_index != index
                and start < other["end"]
                and end > other["offset"]
                for other_index, other in enumerate(regions)
            )
            if end - start <= max_passage_chars and not overlaps_other_region:
                region.update({
                    "offset": start,
                    "end": end,
                    "text": text[start:end],
                    "score": max(region["score"], candidate["score"]),
                })
            # Even when the bounded region cannot grow, this candidate is a
            # duplicate of evidence already selected and gets no result slot.
            continue
        if len(regions) < top_k:
            regions.append(dict(candidate))
            if len(regions) == top_k:
                break

    regions.sort(key=lambda item: (-item["score"], item["offset"]))
    return [
        {"text": item["text"], "offset": item["offset"], "score": item["score"]}
        for item in regions[:top_k]
    ]

def _ranked_diverse_windows(windows: list[dict], top_k: int) -> list[dict]:
    """Keep the legacy top-three prefix, then add non-overlapping evidence.

    The first three windows are byte-for-byte compatible with raw retrieval.
    Additional slots go only to windows outside every selected region, avoiding
    the repeated sliding-window snippets that made a raw top-eight result large.
    """
    prefix_size = min(3, top_k)
    selected = [dict(item) for item in windows[:prefix_size]]
    for candidate in windows[prefix_size:]:
        if any(
            candidate["offset"] < item["end"]
            and candidate["end"] > item["offset"]
            for item in selected
        ):
            continue
        selected.append(dict(candidate))
        if len(selected) == top_k:
            break
    return [
        {"text": item["text"], "offset": item["offset"], "score": item["score"]}
        for item in selected
    ]

def search_within(doc_id: str, query: str, top_k: int | None = None) -> list[dict]:
    """Search within a document for raw windows or distinct merged passages."""
    doc = _load_doc(doc_id)
    if doc is None:
        return [{"error": f"Document '{doc_id}' not found"}]
    if top_k is None:
        top_k = _default_search_within_top_k
    text = doc["text"]
    query_terms = query.lower().split()
    windows = []
    step = 200
    for start in range(0, len(text), step):
        end = min(start + 500, len(text))
        window = text[start:end]
        score = sum(window.lower().count(t) for t in query_terms)
        if score > 0:
            windows.append({
                "text": window, "offset": start, "end": end, "score": score,
            })
    windows.sort(key=lambda item: (-item["score"], item["offset"]))
    if _search_within_mode == "dedupe_merge":
        return _dedupe_merge_windows(windows, text, top_k)
    if _search_within_mode == "ranked_diverse":
        return _ranked_diverse_windows(windows, top_k)
    return [
        {"text": item["text"], "offset": item["offset"], "score": item["score"]}
        for item in windows[:top_k]
    ]

def verify(doc_id: str, claim: str) -> dict:
    """Check if a claim's keywords appear in a document. Returns bool + matching excerpt."""
    doc = _load_doc(doc_id)
    if doc is None:
        return {"found": False, "error": f"Document '{doc_id}' not found"}
    text_lower = doc["text"].lower()
    claim_terms = claim.lower().split()
    matches = sum(1 for t in claim_terms if t in text_lower)
    ratio = matches / len(claim_terms) if claim_terms else 0
    if ratio < 0.4:
        return {"found": False, "match_ratio": round(ratio, 2)}
    # Find best matching window
    best_score, best_start = 0, 0
    for start in range(0, len(doc["text"]) - 200, 100):
        window = doc["text"][start:start+200].lower()
        score = sum(window.count(t) for t in claim_terms)
        if score > best_score:
            best_score = score
            best_start = start
    excerpt = doc["text"][best_start:best_start+200]
    return {"found": True, "match_ratio": round(ratio, 2), "excerpt": excerpt}

def list_docs() -> list[dict]:
    """List all available documents."""
    return [
        {"doc_id": d["doc_id"], "title": d.get("title", d["doc_id"]), "chars": len(d["text"])}
        for d in _load_all_docs()
    ]

print(
    "Tools loaded: search(), read(), passage(), extract(), scan(), aggregate(), "
    "search_within(), verify(), list_docs()"
)
print(f"Corpus: {len(list_docs())} documents available")
print("TIP: search(q) for doc-level, search(q, method='chunk') for chunk-level search")
'''
