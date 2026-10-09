# Try the local learning-memory loop

> This documents the earlier chat browser slice of the
> [Obsidian second-brain plan](PERSONAL_LEARNING_MEMORY_BUILD_PLAN.md).
> The offline demo returns source passages rather than a generated answer;
> live Nemotron and Qwen use require the explicit model setup below.
> The current question desk is the home page at `/lab`; this chat is at `/chat`.

This is a small runnable product slice using **synthetic** notes. The one-command
launcher uses the ten-note `data/product_memory/linked_demo_vault` with 18
Obsidian wikilinks to show relationships and related notes. The manual CLI
examples below use the unchanged
nine-note `data/product_memory/sample_vault` evaluation fixture. Neither needs a
GPU, account, or API key. They show capture, source review, explicit approval,
append-only Markdown, and later recall. `inspect` and `recall` use a **lexical
paragraph baseline**; they do not run Qwen or Nemotron. Work in a copy so the
committed samples stay intact.

For a quick, judge-safe local test build, install the viewer dependencies once
and launch the bundled **synthetic** vault with one command:

```bash
python -m pip install -e '.[viewer]'
python scripts/launch_sample_demo.py
```

Open the printed `http://127.0.0.1:8765` URL for the question desk, or append
`/chat` to follow the older chat walkthrough below. This launcher accepts no private
vault path or model flags. It copies only the ten linked demo notes into
a temporary directory, keeps snapshot state outside that copied vault, and
binds to loopback. It runs the evidence-only lexical path even if model keys
are present in the environment. Changes made during the demo disappear when
the server exits. The `/chat` page opens in a conversation, with example questions,
source cards, linked notes, and an explicit revision dialog.
Capturing a result also triggers a local check for an earlier decision worth
reviewing. The dated prior/source/proposed timeline does not claim that the
notes contradict each other or write the proposed revision automatically.
To see Obsidian's native graph and backlinks, open the printed temporary
`vault` directory as an Obsidian vault while the server is running. The browser
shows linked neighbors, but it does not open Obsidian for you. The temporary
copy is removed when the launcher exits.

## Open the browser demo

From the repository root, install the viewer extra if needed, then run:

```bash
python -m pip install -e '.[viewer]'
demo_dir=$(mktemp -d)
cp -R data/product_memory/sample_vault "$demo_dir/vault"
python scripts/personal_memory_web.py \
  --vault "$demo_dir/vault" --state-dir "$demo_dir/state" --port 8765
```

Open `http://127.0.0.1:8765/chat`. Ask how the phone-drawer plan changed, open an
exact source or linked note, and read the dates and revision relationships.
Use **Correct this note** to name the earlier record, select passages in the
review dialog, and explicitly approve a new correction. **+ Capture** adds
an observation without rewriting the existing notes. Choose **Result** for a
completed observation, then inspect the earlier decision and new source in the
impact card. **Review proposed revision** opens a scaffold that you must
replace with your own conclusion before saving; the old note stays intact and
the new one links back to it. **Investigate in chat** asks the configured chat
path about the two notes. A plan or ordinary source may surface a related
prior note, but it does not get a correction scaffold. Ask again to see the
approved note alongside the earlier history. The default page identifies
itself as local evidence-only mode and makes no model calls.

The impact check uses Obsidian links and overlapping terms to suggest a note
for review. That is a navigation signal, not a semantic conflict detector.
Capture **Result** only when an observation actually happened; the app cannot
verify that a note labeled Result contains measured data. The local check
works with the ten-note synthetic demo and with new notes that have meaningful
Markdown titles, even though captured files use unique filenames.

When Nemotron is explicitly enabled, the impact card offers **Draft with
Nemotron** for a candidate result. This user-clicked Token Factory request
sends the two selected passages. Its wording is labeled as a model draft,
requires citations to both passages, and still needs source inspection and
explicit human approval. A failed or malformed draft falls back to the local
scaffold. The default local demo never shows this button or spends model
credits.

## Ask through the chat API

The browser uses the same `/api/chat` route shown here. In its default mode it
returns **source passages only** and labels the lexical fallback; it does not
invent a Nemotron answer. For example, while the server above is running:

```bash
curl -s http://127.0.0.1:8765/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"question":"How did my phone-drawer focus plan change?"}' \
  | python -m json.tool
```

The response includes an `evidence` list, source paths, linked neighboring
notes, `review_id`, and `session_id`. Open an exact source with
`GET /api/source/{review_id}/{doc_id}`. Send a subsequent question with the
same `session_id` for bounded in-process conversation history. Chat never
saves a memory; `/api/approve` still requires explicit confirmation and
reviewed evidence IDs. Capturing or approving a
new note resets the history tied to the older snapshot. A response generated
while the vault changes is marked `snapshot_stale` and cannot be approved.

## Query a frozen vault from an MCP host

The local stdio MCP server exposes the same **lexical evidence baseline** to a
larger assistant. It has exactly two read-only tools: `search_memory(query,
top_k)` returns an exact-passage review with a `review_id`, and
`get_memory_source(review_id, doc_id, start, length)` opens up to 8,000
characters of a document returned by that review. The source tool cannot open
an unrelated document or accept a filesystem path. Neither tool writes notes,
calls a model, or sends vault content to an endpoint on its own. The connected
MCP host receives the passages it requests, so use a host you trust with that
vault's contents.

Install the optional SDK, then create a complete snapshot **outside** the
vault. For the bundled nine-note fixture:

```bash
python -m pip install -e '.[mcp]'
demo_dir=$(mktemp -d)
python scripts/personal_memory.py snapshot \
  --vault data/product_memory/sample_vault \
  --output "$demo_dir/snapshot"
python scripts/personal_memory_mcp.py --snapshot "$demo_dir/snapshot"
```

The last command is a stdio server. It waits for an MCP client and does not
print a web address. In an MCP host's local-server configuration, use the
absolute Python and script paths, and point `--snapshot` to an absolute
snapshot path:

```json
{
  "mcpServers": {
    "envoy-memory": {
      "command": "/absolute/path/to/python",
      "args": [
        "/absolute/path/to/rlm-explorer/scripts/personal_memory_mcp.py",
        "--snapshot",
        "/absolute/path/to/snapshot"
      ]
    }
  }
}
```

The server pins the snapshot's corpus hash at startup. Make a **new** snapshot
and restart the server after changing the vault; it does not watch live files.
Its response marks no-match cases as `no_evidence` and warns that exact quotes
establish provenance, not whether a passage supports a conclusion. This
interface does not yet call the Qwen investigator or serve weekly digests. The
focused test `pytest -q tests/test_personal_memory_mcp.py` exercises an actual
stdio client connection as well as review-bound source access.

To call a real NVIDIA Nemotron model on Nebius Token Factory, provide your
Token Factory key and explicitly enable remote use for the **selected** vault:

```bash
export NEBIUS_API_KEY='your-token-factory-key'
python scripts/personal_memory_web.py \
  --vault "$demo_dir/vault" --state-dir "$demo_dir/state-nebius" \
  --port 8765 --enable-nebius
```

The server also reads `NEBIUS_API_KEY` from a repository-root `.env` file if
the variable is absent from its own process environment. `.env` is gitignored;
keep it private (for example, file mode `0600`) and never paste the key into
chat or commit it. This is useful when a separate terminal launches the app.

On September 27, 2026, the live application path returned HTTP 200 through
FastAPI `TestClient` for the synthetic question “How did my phone-drawer focus
plan change?” Nemotron planned the investigation and wrote an answer citing
three source passages. Both Token Factory calls served
`nvidia/Nemotron-3_5-Lightning` with thinking disabled; the recorded result is
in the [synthetic live smoke report](../reports/nebius_live_synthetic_smoke_20260927_final.json).
Qwen was not configured for this run, so retrieval used the lexical paragraph
baseline. This is one synthetic integration case, not a semantic-quality
evaluation or evidence that a private vault works well.

Nemotron plans whether vault investigation is needed, then explains only
independently checked source passages. With no Qwen endpoint, the investigator
is labeled `lexical_paragraph_baseline`. The pinned v5 checkpoint
[failed the September 28 readiness screen](PERSONAL_MEMORY_READINESS_EVAL.md)
and remains off by default. For a controlled Qwen experiment, serve the exact
checkpoint through an OpenAI-compatible endpoint, build the `rlm-sandbox`
Docker image, and add all three flags:

```bash
docker build -t rlm-sandbox .
python scripts/personal_memory_web.py \
  --vault "$demo_dir/vault" --state-dir "$demo_dir/state-models" \
  --port 8765 --enable-nebius \
  --qwen-endpoint http://127.0.0.1:8000/v1 \
  --qwen-model YOUR_SERVED_MODEL_ID \
  --qwen-checkpoint YOUR_EXACT_CHECKPOINT_ID
```

Qwen-authored Python is allowed only in the network-disabled Docker sandbox.
An unavailable sandbox or model is reported and causes lexical fallback. A
source quote is checked against the frozen vault snapshot before Nemotron
receives it; this does not prove that the quote supports the answer. Do not
point a private vault at a hosted model endpoint unless you intend to send its
selected excerpts to that service. `model_requests_attempted` counts model
requests, including failed requests that may have sent vault context; it is
not a billing record. After a successful Nemotron answer, `answer_served_model`
reports the provider's model identity when available, and `answer_usage`
reports tokens for the answer call; the routing call is not included in that
usage field.

## Repeat the flow from the command line

From the repository root, with the project dependencies installed:

```bash
demo_dir=$(mktemp -d)
cp -R data/product_memory/sample_vault "$demo_dir/vault"

python scripts/personal_memory.py capture \
  --vault "$demo_dir/vault" --kind attempt --effective-date 2026-09-25 \
  --title 'One more focus session' \
  --text 'I wrote a clear next action before studying and muted notifications; starting felt easier in one session.'

python scripts/personal_memory.py snapshot \
  --vault "$demo_dir/vault" --output "$demo_dir/snapshot-1"

python scripts/personal_memory.py inspect \
  --snapshot "$demo_dir/snapshot-1" \
  --query 'correction starting friction phone next action' \
  --output "$demo_dir/review.json"

python -m json.tool "$demo_dir/review.json"
```

Inspect the quotations and `record` fields in `review.json`. With this frozen
fixture, E1 is the July correction, E2 the later decision, and E4 the new
attempt. The June plan remains available as E3. The `status` field is
`no_evidence` when lexical search finds no match; that result cannot be
approved as a sourced lesson.

After checking those passages, explicitly approve a *new* correction. The
old June note remains unchanged; `--supersedes` accepts its stable `record_id`:

```bash
python scripts/personal_memory.py approve \
  --vault "$demo_dir/vault" --snapshot "$demo_dir/snapshot-1" \
  --review "$demo_dir/review.json" \
  --title 'Focus lesson after another session' --kind correction \
  --effective-date 2026-09-25 \
  --supersedes life-focus-plan-2026-06-10 \
  --evidence E1 E2 E4 --confirm \
  --text 'The June phone drawer idea was incomplete. My July notes favored a written next action. In one September session, I combined that with muted notifications and starting felt easier. I need more comparable sessions before claiming a cause.'

python scripts/personal_memory.py snapshot \
  --vault "$demo_dir/vault" --output "$demo_dir/snapshot-2"

python scripts/personal_memory.py recall \
  --snapshot "$demo_dir/snapshot-2" \
  --query 'June phone drawer incomplete September muted notifications'
```

The approved note appears under `Learning Memory/` in the copied vault. Its
frontmatter records the approval, effective date, source-snapshot hash, and
superseded document. It links the exact reviewed passages and says that their
*semantic* support was not automatically judged. The second snapshot includes
the new note, so the final `recall` returns it alongside the earlier history.

To inspect a **real saved** code-execution run, use `review-code-exec` with
that run's transcript and its matching frozen snapshot. This verifies the
model-cited spans without running model code or claiming that the model's
answer is correct:

```bash
python scripts/personal_memory.py review-code-exec \
  --snapshot /path/to/matching-snapshot \
  --transcript /path/to/saved-run.json \
  --question-id QUESTION_ID --output /path/to/reviewed-packet.json
```

If the transcript contains no evidence, the command reports `no_evidence` or
`unsupported_answer` with an empty passage list. It never invents a quote.
The source files and test cases are in `src/product/memory.py` and
`tests/test_personal_memory.py`. This CLI uses local lexical retrieval; the
separate browser path above can call Nemotron and the Qwen investigator when
explicitly configured.

The separate [personal-memory readiness evaluation](PERSONAL_MEMORY_READINESS_EVAL.md)
defines how to judge whether Qwen's tool use and evidence quality are good
enough to take over the lexical investigation step. Its synthetic questions
are development cases, not a claim of personal-vault performance. That
evaluation's `bm25_passage_baseline` is a separate retrieval implementation
from this demo's `lexical_paragraph_baseline`.
