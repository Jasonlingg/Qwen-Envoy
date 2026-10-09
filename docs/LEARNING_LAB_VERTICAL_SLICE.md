# Question-centered learning lab: first working slice

The `/lab` page is the app's home. It adds a dated research question, a
user-stated current view, frozen web sources or file attachments, source-backed
revision proposals, and an inspectable timeline. A proposal does not change the current view.
The user explicitly accepts it in the working thread index. Both imported
sources and proposed revisions are staged as drafts under Obsidian `_inbox/`;
only the user can review and move them into `library/`.

This is a product workflow slice, not a trained-Qwen improvement result. It
uses lexical passages for the URL preview. When explicitly configured, Nemotron
can suggest wording from the saved view and selected frozen passages; the user
must edit and review the suggestion. No model call occurs by default. The
existing public-paper Qwen benchmark and held-out set are unchanged.

If a Qwen checkpoint endpoint is configured, the lab offers **Investigate
reviewed notes with Qwen**. The existing bounded Python worker searches the
frozen vault in Docker, and the host rechecks its exact note spans before the
page lets you select them for a revision. Its full run record goes under the
state directory. An unavailable sandbox or model is shown as unavailable; the
lab does not pretend that a lexical match was a Qwen result. This path has a
fake-worker integration test, not a live checkpoint quality result.

## Try it locally

Use a copy of the sample vault so the demo does not alter repository fixtures:

```bash
python -m pip install -e '.[viewer]'
mkdir -p out/learning-lab
cp -R data/product_memory/sample_vault out/learning-lab/vault
python scripts/personal_memory_web.py \
  --vault out/learning-lab/vault \
  --state-dir out/learning-lab/state
```

Open `http://127.0.0.1:8765/lab`. The older freeform chat remains at `/chat`.
Create a question and record the view you currently hold. Import a public HTML
URL, such as a paper's arXiv abstract page or a technical blog post, or attach
a local `.md`, `.txt`, or text-extractable `.pdf` file up to 5 MB. The host
freezes the original bytes and parsed text. Web snapshots retain the URL,
retrieval time, hashes, and extraction coverage. File snapshots retain the
filename, upload time, hashes, and PDF page offsets. The source draft appears
in `_inbox/`; the page cannot modify your
reviewed library. Select a returned passage, mark whether you think it
supports or challenges your current view (or is context/unclear), and, if
relevant, search for an earlier experiment note in the selected vault. This
mark is your interpretation, not a verified support judgment. Write a possible
revised view and why you are reconsidering, then stage it. The draft includes the exact
source and vault passages, snapshot identities, and offsets. Accepting the
proposal changes the **working thread timeline only**. The earlier view remains visible.

Acceptance verifies the exact staged Markdown bytes. If you edit or move a
proposal draft before clicking **Accept as current view**, acceptance rejects
it; stage a fresh proposal from the edited view. After acceptance, you can
review and edit the inbox draft for Obsidian promotion. Selected note passages
are tied to a vault snapshot, so a vault refresh requires selecting them again.

For the source or revision to become searchable in the Obsidian library, inspect
the original URL or download the frozen attachment and review its cited excerpt.
Edit the draft's frontmatter from
`review_status: agent_authored_draft` to `review_status: user_reviewed`, and move
the file from `_inbox/` to `library/` yourself. Then press **Refresh vault** in
the lab page. The vault importer excludes `_inbox/` and continues to skip an
agent-authored draft moved without that review edit. The main chat can now find
the promoted note; its citations still indicate provenance, not automatic
semantic support.

The app accepts public HTTP(S) HTML pages on default ports. It rejects local or
private targets, unsafe redirects, non-HTML responses, and responses larger
than 2 MB. Connections use a validated public IP with the original hostname
for TLS verification. The local web server also rejects foreign Host headers
and cross-origin writes. Frozen source snapshots and the working thread index
live under the selected state directory, outside Obsidian. A page changed online is a new
snapshot rather than a silent change to an earlier citation.

An empty vault can start with a question and an attachment. Reviewed-note
search and chat become available after you promote a source draft and refresh
the vault. The source preview selects lexical passages from one frozen source;
PDF extraction omits images and scanned pages, and the app does not search the
whole web.

Optional Nemotron suggestion is exposed only when the app starts with
`--enable-nebius` and a configured `NEBIUS_API_KEY`; clicking the suggestion
button makes a Token Factory call. Qwen can be configured independently with
`--qwen-endpoint`, `--qwen-model`, and `--qwen-checkpoint`, plus the local
`rlm-sandbox` Docker image; the app does not start a model server for you. A
reviewed base-versus-trained comparison remains separate work described in
[the public-paper development runbook](RESEARCH_LIBRARY_DEV_QWEN_RUNBOOK.md).

Focused verification:

```bash
python -m pytest tests/test_web_sources.py tests/test_file_sources.py \
  tests/test_learning_thread.py \
  tests/test_learning_lab_web.py tests/test_personal_memory_web.py -q
```
