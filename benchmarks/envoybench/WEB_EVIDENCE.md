# Web-to-evidence development pilot

This pilot asks a narrow engineering question: when a public URL is frozen as raw
bytes, can Envoy's current parser retain a short, checked fact, and can its
deterministic paper search return the fact in a top-three passage? It evaluates
the **source-ingestion and retrieval components**, not Qwen, generated answers,
or semantic citation support. It is separate from the QASPER-derived EnvoyBench
model comparison and from user-uploaded papers in Studio.

## Hypothesis and decision rule

The starting hypothesis was that the HTML/PDF path would preserve short source
facts across several real formats, while long or unusual documents might expose
specific parsing or retrieval failures. The expected signals were literal fact
retention, top-three exact-span retrieval, source-title quality, and separate
fetch, parse, and search times on the **same frozen response bytes**.

- If a fetched source cannot be parsed, or a checked fact disappears during
  parsing, improve ingestion before judging the research worker on it.
- If a fact survives but no top-three search result contains it, improve
  retrieval before blaming answer synthesis.
- Only after this development set is stable should a new, independently reviewed
  batch test whether a change generalizes. A quality lift on this set alone is
  an iteration result, not a held-out benchmark result.

## Data and review status

The [case manifest](data/web_evidence_cases_v1.json) has **9 real URLs** from
7 HTML pages and 2 versioned arXiv PDFs, with 18 short target facts. It includes
a simple reference page, technical documentation, two long standards, a NASA
CMS page, an API reference, two papers, and one legacy preformatted RFC page.
The URLs and facts were selected after preliminary inspection of the existing
extractor, including a known parser failure. This is an intentionally small,
failure-seeking **development** batch, not a random or held-out web sample.

The facts were **assistant-checked** against publisher pages/PDFs using a
separate web reader on 2026-10-01. They have not been independently human
reviewed. Their short `expected_text` fields are literal anchors, so scoring
can be replayed without a judge. Literal presence establishes neither the
source's truth nor support for any wider claim. Well-known documents may have
appeared in model pretraining; this pilot makes no unseen-data claim.

`capture` uses the existing bounded public-IP-pinned fetcher. It saves exact
response bytes locally, plus final URL, media type, byte size, fetch time, and
SHA-256. The raw third-party pages/PDFs live under `out/` and are not committed
to the repository. `score` verifies those hashes and the case-manifest hash,
then reads the frozen bytes **without network or model calls**. HTML uses
`web-html-v1`; PDF uses `uploaded-file-v1` / pypdf; the retrieval baseline is
`lexical_paragraph` v1, `top_k=3`. Each retained target has an exact character
offset into the parsed text. A top-three success requires the literal target to
occur in a returned passage. The report records `fetch_error`, `parse_error`,
`fact_missing`, `retrieval_miss`, and `pass` separately.

## Reproduce

From the repository root with the existing viewer/vault dependencies installed:

```bash
python -m benchmarks.envoybench.web_evidence capture \
  --cases benchmarks/envoybench/data/web_evidence_cases_v1.json \
  --output out/envoybench/web-evidence-new-capture

python -m benchmarks.envoybench.web_evidence score \
  --cases benchmarks/envoybench/data/web_evidence_cases_v1.json \
  --capture out/envoybench/web-evidence-new-capture \
  --output out/envoybench/web-evidence-new-capture/report.json
```

Capture refuses to overwrite a directory, and score refuses to overwrite a
report. A fresh fetch may receive revised pages, so compare **raw hashes** before
comparing scores. To replay the October 1 result, run `score` against the saved
`out/envoybench/web-evidence-dev-v1-20261001/` capture, choosing a new output
filename. That replay needs no internet access.

To inspect the report in Studio:

```bash
python -m benchmarks.envoybench.demo --port 8766 \
  --web-evidence-report out/envoybench/web-evidence-dev-v1-20261001/report.json
```

Open `http://127.0.0.1:8766/web-evidence`. This page reads one completed local
report; it does not rerun fetches, search, or models. It is not shown in a blind
model-review session.

## Frozen October 1, 2026 result

The [local report](../../out/envoybench/web-evidence-dev-v1-20261001/report.json)
records case-manifest SHA-256
`cdbf635733ef62e551fba770b99ebe024f54019835bb1ef7f929f8920e91b135`
and capture-manifest SHA-256
`140e40d6b90e04df6f728f0e650d0bb0fc1df43d448208d60f5f33c3733286f4`.
The run used Python 3.11.9 on a Mac15,12 arm64 machine; no model checkpoint,
seed, GPU, or paid model API was used.

| Stage | Observed result |
| --- | ---: |
| Fetch | 9/9 URLs captured |
| Parse | 8/9 sources parsed |
| Literal facts retained, conditional on parsing | 16/16 |
| Fact found in top-three passage, conditional on retention | 8/16 |
| End-to-end evidence hits, all targets | 8/18 |
| HTML title match, conditional on parsing | 5/6 |

The legacy [RFC 3986 HTML](https://www.rfc-editor.org/rfc/rfc3986.html) is a
real parse failure: its preformatted HTML has no ordinary document `<title>`
that the current extractor accepts. The [PEP 257 page](https://peps.python.org/pep-0257/)
parses but its extracted title includes unrelated page-control text. The eight
retrieval misses occur after the target facts survived parsing, particularly on
the long RFC/WCAG text and the PDFs. These are component findings, not evidence
that Qwen answered the questions incorrectly.

Observed local fetch latency had median 0.231 s and nearest-rank p95 0.839 s
over nine requests. Search had median 0.001 s and p95 0.126 s over 16 checked
facts. These are single-run timings, not a controlled performance comparison.
The report records **$0 model API spend** because no model was called;
infrastructure and network costs were not measured.

The next measured change should target the preformatted-HTML parse failure or
the long-document passage ranking, then replay these exact frozen bytes and
check the paired per-fact difference. A later evaluation should add new URLs
and independent review before claiming broader web quality.
