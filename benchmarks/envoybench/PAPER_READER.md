# Live paper reader

Studio has three views: `/papers` for web-fetched or uploaded sources, `/` for the saved
benchmark comparison, and `/model` for the observed v5 training recipe. The
paper reader uses the existing bounded code investigator. These live paper
runs remain separate from the fine-tuned Qwen benchmark comparison.

## Open the local reader

Install the existing viewer/vault extras, then launch:

```bash
python -m pip install -e '.[viewer,vault]'
python -m benchmarks.envoybench.demo --port 8766
```

Open `http://localhost:8766/papers`. Paste an arXiv abstract/PDF link or a
direct HTTPS PDF link and select **Get paper**. The server fetches the PDF,
freezes its original bytes and extracted text, and records the requested and
final URLs. A local PDF upload remains available. Without a model config you can
fetch, upload, inspect, and reopen sources and saved runs. The Run button is disabled. There
are no canned answers or automatic GPU provisioning.
Fetching the same link and bytes again reuses its saved snapshot; revised bytes
create a new snapshot.

Uploads accept UTF-8 Markdown/text or text-based PDF; web links must return PDF
bytes. Sources are limited to 5 MB, 100 PDF pages,
and 250,000 extracted characters. PDF extraction has no OCR and may omit figures,
tables, and equations. The reader reports pages without extractable text. The
**Original PDF** link opens the frozen original; citations highlight positions
in the extracted text reader.

## Connect a model

The model endpoint must already serve `envoy` as the pinned v5 adapter in
`paper-models.example.json`. That file records the base/adapter revisions and
adapter SHA-256; it is an operator declaration, not endpoint attestation. Use
the existing adapter verification/serving procedure before declaring that
identity. There is no fallback to a different model.

```bash
export ENVOY_QWEN_ENDPOINT=http://127.0.0.1:8000/v1
docker build -f benchmarks/envoybench/Dockerfile -t rlm-sandbox .
python -m benchmarks.envoybench.demo --port 8766 \
  --paper-models benchmarks/envoybench/paper-models.example.json
```

The Docker daemon and labeled sandbox image must be available locally. To reach
a remote GPU, use an existing SSH tunnel or HTTPS endpoint. If authentication
is needed, copy the config to a local file and add `api_key_env` naming an
environment variable. Never put credentials in JSON or browser input. Set the
actual serving hardware/runtime in that copy. Changes require a Studio restart.
The readiness indicator checks local configuration and Docker; it does not
probe the remote model. The first run can still report a connection failure.

For a hosted Nebius Token Factory demo, set `NEBIUS_API_KEY` in the local
environment and use `paper-models.nebius.example.json`. It offers Nemotron
Ultra, Lightning, and Nano. These are provider-managed model declarations, not a
substitution for the fine-tuned Qwen checkpoint in the benchmark. The live
reader gives Nemotron an explicit, model-specific `SUBMIT` instruction; each run
records its prompt hash. This does not change the frozen benchmark prompt.
Ultra completed one local integration run, but its citations did not fully
support its answer. Three Lightning attempts failed to submit. Nano is
configured but untested. See the [live validation record](../../release/envoybench-v0.1/LIVE_VALIDATION.md);
these observations do not establish model reliability.
The Nebius example disables thinking per request for concise executable actions.
That is a live-reader decoding choice, not a matched benchmark comparison; the
provider-reasoning field will normally be absent for these runs.

```bash
docker build -f benchmarks/envoybench/Dockerfile -t rlm-sandbox .
python -m benchmarks.envoybench.demo --port 8766 \
  --paper-models benchmarks/envoybench/paper-models.nebius.example.json
```

Keep the completed comparison visible by adding its original arguments:

```bash
python -m benchmarks.envoybench.demo --port 8766 \
  --run-dir out/envoybench/provisional-full-20260930 \
  --review-dir out/envoybench/provisional-review-20260930 \
  --judged-review out/envoybench/codex-reviewed-20260930/review.json \
  --provisional \
  --paper-models benchmarks/envoybench/paper-models.example.json
```

Live model requests consume the configured endpoint's compute. Opening the
reader, fetching or uploading a paper, and inspecting saved runs send no model
requests. Web fetching itself uses ordinary network access.

## What a run saves

The server freezes original bytes and parsed text outside the benchmark under
`out/envoybench/paper-workspace/` (override with `--paper-state-dir`). Fetched
URLs use the existing public-address and redirect checks. Each run
records the question, paper hashes, model declaration, decoding, seed
behavior, prompt/tool hashes, source spans, full tool trace, request usage when
reported, elapsed time, errors, and review status. JSON export is available
beside the saved-run selector. Reading or exporting evidence rechecks its source.

One investigation runs at a time. Limits are 12 program rounds, 60 seconds per
model request, 20 seconds per code action, and a 300-second budget checked
between model requests. In-flight tool execution/cleanup can exceed that last
budget. Restarting Studio marks unfinished runs interrupted; it never retries
them or resumes spending automatically. This is a local trusted-operator tool,
not a multi-user public service.

Model-authored code runs only in Docker with a read-only corpus mount and no container
network access. The investigator discovers source IDs, reads evidence,
and checks exact spans. All accepted answers remain **semantically unreviewed**.
New uploads have no benchmark gold answer and never change the frozen scores.
The product prompts and discovery checks also differ from the frozen known-paper
benchmark; do not pool their measurements as a controlled model comparison.

Training exposure is recorded as **upstream unknown / own fine-tuning not
audited**. File hashes and quotation validity cannot establish unseen training
data, author credibility, or whether a quote supports a claim. The current reader
does not give the model internet tools or automatically audit membership.

Provider-reported reasoning and sampled token logprobs are saved per turn when
the endpoint returns them. They are optional telemetry, not calibrated answer
confidence. A missing field means the endpoint did not provide it. The trace
viewer labels these fields separately from executed actions and tool output.
Failed provider responses retain a safe category and shape-only finish metadata
in the trace. If visible model text fails action parsing, the reader saves at
most 8,000 characters separately as **rejected, unexecuted** text after
redacting configured secrets. It does not store provider error bodies or
execute that rejected text.

## Verification

```bash
python -m pytest -q tests/test_envoybench_paper_api.py \
  tests/test_envoybench_studio_api.py tests/test_envoybench_demo.py \
  tests/test_file_sources.py tests/test_qwen_investigator.py
```

API tests exercise the real importer and investigator with explicitly scripted
test policies. They verify integration, not Qwen quality. A real v5 endpoint
smoke remains a separate check; no speedup is claimed for the prepared
[cache experiment](INFERENCE_ABLATION.md).
