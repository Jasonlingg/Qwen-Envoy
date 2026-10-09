# Query original research papers through MCP

`scripts/research_library_mcp.py` serves one complete, version-pinned public-paper
snapshot. It is separate from `scripts/personal_memory_mcp.py`, which searches
saved Obsidian notes. Neither server writes the vault.

With the MCP dependency installed, start the paper server without a model:

```bash
python -m pip install -e '.[mcp]'
python scripts/research_library_mcp.py \
  --snapshot out/research/weekly-2026-W39-draft-snapshot
```

This is a stdio server for an MCP client, not a web page. In a client that accepts
local-server configuration, use absolute paths for Python, the script, and the
snapshot. The server exposes `search_research_papers` and `get_research_source`
for lexical search and review-bound source opening. It also lists
`investigate_research_question`, which returns `model_not_configured` without
calling a model until Qwen is explicitly configured.

To use a served Qwen checkpoint, first build the network-disabled sandbox image,
then provide all three model flags:

```bash
docker build -t rlm-sandbox .
python scripts/research_library_mcp.py \
  --snapshot out/research/weekly-2026-W39-draft-snapshot \
  --qwen-endpoint http://127.0.0.1:8000/v1 \
  --qwen-model YOUR_SERVED_MODEL_ID \
  --qwen-checkpoint YOUR_EXACT_CHECKPOINT_ID
```

`ENVOY_MODEL_API_KEY` is optional if that endpoint requires authentication.
The configured Qwen worker writes multi-step Python against the frozen paper
tools in Docker. The server rechecks each returned excerpt against the pinned
snapshot, prompt, and `passage-bm25-okapi-v1` search-tool hash, and labels Qwen's
candidate answer `semantic_support: not_reviewed`.
If Docker or the endpoint is unavailable, the response reports that status;
the server does not silently run Qwen-authored code on the host. A connected
host can call `get_research_source` only for papers returned by its preceding
search or investigation review.

The W39 snapshot is a development/demo corpus, not the held-out model-improvement
test. The separate [weekly digest writer](WEEKLY_DIGEST_DEMO.md) previews paper
notes for Obsidian and requires actual human review before publishing them.

On September 28, a real stdio MCP client queried the W39 snapshot: the server
listed all three tools, lexical search returned three paper hits, a review-bound
source open matched its exact excerpt, and the unconfigured Qwen operation
returned `model_not_configured` with zero model requests. This checks the MCP
and source path, not Qwen's live answer quality.
