# arXiv research MCP

An MCP client can discover papers, follow citations, open version-pinned arXiv
text, search passages, and inspect exact offsets. An explicitly configured Qwen
or Nemotron reader can investigate up to five papers in parallel using Python
in Docker. Its answers are proposals: exact quotes establish provenance, not
semantic support. This extends QASPER Agent Studio, a Qwen Envoy case study;
it does not establish that fine-tuned Qwen improved.

## Install and start

```bash
pip install -e '.[mcp,vault]'
# Needed only for model-authored Python:
docker build -f benchmarks/envoybench/Dockerfile -t rlm-sandbox .
claude mcp add arxiv-research -- python "$(pwd)/scripts/arxiv_research_mcp.py" --reader none
```

The default reader is `none`, even if an API key is present. Search, citations,
opening papers, evidence search, and excerpts require no model or GPU. Tools use
stdio; their results carry provenance warnings. The host chooses tool order.
For example: search papers, open selected results, find evidence, then use
get_passage to inspect the surrounding text before making a claim.

For a hosted reader, make `NEBIUS_API_KEY` available in the MCP server's process
environment and register with `--reader nemotron`. The named model defaults to
`nvidia/Nemotron-3-Ultra-550b-a55b`; its weights/revision are not endpoint-attested.
This is an explicit paid-model opt-in, not a dollar-capped service. Each reader
has 12 action steps, up to two format re-asks per action (at most 36 model
requests), and a 1,024-token completion limit per request. Input histories grow;
these limits do not imply a fixed dollar budget. No automatic transport retry
is added. Reader errors return per-paper statuses.

For Qwen v5 with a configured vLLM endpoint (see `scripts/serve_vllm.sh`):

```bash
claude mcp add arxiv-research-qwen -- python "$(pwd)/scripts/arxiv_research_mcp.py" \
  --reader qwen --qwen-endpoint http://127.0.0.1:8000/v1 --qwen-model envoy \
  --qwen-checkpoint jasonlingg/qwen-envoy-qwen3-8b-qasper-sft-v5@27b912a863ff914ad45baa03624d8911dc1e17fb/artifacts/full/checkpoint-50
```

The checkpoint identifier is an operator declaration, not weight attestation.
Qwen receives temperature 0, top-p 1, seed 42 and thinking disabled. The hosted
Nemotron path disables thinking, uses temperature 0 and top-p 1, and does not
claim seeded determinism. Both use harness V2 by default in this new server;
`--harness v1` is available. Existing investigator callers still default to V1.
Policy, investigator and packet use the same effective prompt and hash.

### On-demand Qwen development

On October 17, 2026, the user chose occasional Qwen use and accepted the model
loading delay. Start the GPU inference server for an investigation session,
wait for the endpoint to be ready, then run investigations. Save results locally
before shutting down the session's compute. Retained remote storage can still
incur charges after compute stops.

The current MCP expects an already available endpoint; it does not provision,
start, or stop GPU resources. Automatic startup and idle shutdown are not
implemented. With Qwen offline, the source-only tools remain available, but
Qwen-powered investigation is unavailable. This hosting decision does not
authorize a paid deployment or change the default `--reader none`.

The [paired workflow pilot](implementation/2026-10-17-research-workflow-pilot.md)
prepares five frozen development questions comparing direct host tools with Qwen
delegation. Preparation and the scripted Docker check make no model calls; live
answering and human claim review remain separate steps.

## Tools

| Tool | Purpose | Model calls |
| --- | --- | --- |
| `search_papers(query, since?, until?, limit=10)` | Semantic Scholar relevance, or dated arXiv discovery | None |
| `get_citations(arxiv_id, direction, limit=20)` | References or cited-by papers | None |
| `open_paper(arxiv_id)` | Pin text, coverage, version and section outline | None |
| `find_evidence(query, arxiv_ids, top_k=8)` | Rank exact passages across opened papers | None |
| `get_passage(arxiv_id, start, length)` | Inspect an opened paper; length at most 8,000 | None |
| `investigate(question, arxiv_ids, synthesize=false)` | Verified per-paper reader packets; at most five papers | Explicit reader |

The optional `--reasoner nemotron` enables synthesis when a caller also requests
`synthesize=true`. It receives only the verified packet and untrusted reader
proposals, not full papers. It adds one request with a 1,500-token completion
limit. Quotes outside packet spans are rejected. Reader,
source-opening, packet-verification and reasoner failures preserve other
verified evidence. Synthesis validation checks provenance, not claim support.
Missing request telemetry is reported as unknown rather than zero.

## Storage, source limits and reproducibility

Cache defaults to `~/.envoy/arxiv-library` (`--library-root` overrides it).
Versioned opens are immutable and verified against saved hashes. An unversioned
ID uses the newest **cached** version without network refresh; it is not a claim
that the cached version is latest. Open a specific newer version to ingest it.
Deterministic snapshot views preserve exact stored text and corpus hashes.

Sources must use HTTPS on `arxiv.org`, `export.arxiv.org`, or
`api.semanticscholar.org`, including every redirect. Credentials/non-443 ports
and arbitrary URLs are refused. Redirects are capped at five; default text/API
responses at 2 MB and PDFs at 5 MB. All arXiv requests—including redirects and
PDFs—share one process-wide 3.1-second throttle. Semantic Scholar uses 1.1
seconds and optionally `SEMANTIC_SCHOLAR_API_KEY`. Provider failures return
`source_unavailable`; identical cached discovery requests still work offline.
These restrictions apply to source HTTP; configured model endpoints are a
separate operator-controlled connection (including local vLLM).

Coverage is HTML paragraphs, PDF text if HTML has fewer than five paragraphs,
or abstract only. Extracted text is capped at 250,000 characters; PDFs at 100
pages. Figures, table layout and equations may be omitted or distorted. There
is no OCR. PDF outlines are heuristic. Abstract-only papers cannot support
unreported experimental details. Discovery covers titles/abstracts and bounded
citation pages, not exhaustive full-text literature search. Non-arXiv results
are listed with `openable=false`. Evidence ranking uses the existing lexical
paragraph scorer, not embeddings, a reranker, or a new BM25 quality result.

V2 corpus discovery defaults to five documents; only paper-scoped search and
search_within default to eight windows. Cross-paper find_evidence defaults to
eight total excerpts (maximum 20, up to ten candidate passages per paper), with
identical windows deduplicated. See [the harness diagnostic](HARNESS_V2.md) for
actual fixed-action results and their interpretation.

## Evaluation

Hypothesis: the model-free tools expose pinned, inspectable source text and
report unavailable sources without paid inference. Expected signal: successful
open/search/excerpt round trips with exact offsets and useful per-source errors.
Decision rule: report tool/provenance results separately from semantic claim
review; do not apply the paid five-question answering pass criterion to a
model-free run.

The paid Nebius smoke is skipped: the existing transport has no reliable
all-in <=$0.50 reservation covering input tokenization, accumulated histories,
format re-asks, concurrent readers and provider charges. max_tokens alone is
insufficient. The source-only smoke fixes model-request, model-input-token and
model-output-token budgets at zero and never configures a reader or reasoner.
It does not validate live Nemotron or Qwen answering. Agent-authored claim
review below is separate from quote matching and is not human review.

The [live source report](../reports/arxiv-mcp-live-20261016.json) records 19 MCP
calls and nine source fetches in 20.774 seconds. Dated arXiv discovery succeeded;
Semantic Scholar search and references returned rate-limit statuses. Three
papers opened as HTML: `2402.01030v4`, `2510.03215v2`, and `2511.20639v4`.
All nine finder quotes and nine excerpt samples matched their stored source
text. The unconfigured investigation returned zero requests as expected. Model
requests, input/output tokens and model cost were all zero. The paid five-question
answering evaluation was not run; no answer-quality pass rate is reported.

The small, unblinded **agent-authored** review checked three proposed claims:
CodeAct's Python action representation, C2C's named evaluation datasets, and
LatentMAS's reported 29.1% average reduction in final-agent output text tokens.
They were supported as author proposals/reports in the inspected context.
The review rejected interpreting that final-agent statistic as whole-system
token or API-cost savings. C2C's four setup benchmarks are not exhaustive: its
appendix also evaluates LongBenchV1. This is not independent human review or a
replication of any paper's result. Paper quotes, offsets and rationales are in
the report; long raw responses stay in `out/arxiv-mcp-live-detail.json`, bound by
SHA-256. Public report text is deliberately limited to short source fragments.

Reproduce the free path with:

```bash
python scripts/smoke_arxiv_research_mcp.py
```

Rerunning regenerates source diagnostics and leaves claim review unperformed;
the semantic review must be repeated against the new pinned evidence. Cached
runs may use fewer network fetches. Source availability and versions can change.
