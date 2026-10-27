# Qwen Envoy Research MCP — hackathon submission draft

**Track:** Best Apps and Agents. **Status:** working local test build and a
[published saved-run inspector](https://jasonlingg.github.io/Qwen-Envoy/research-mcp/);
the video and Devpost entry remain open. The [official rules](https://nebiusglobalaihackathon.devpost.com/rules)
list the deadline as **October 30, 2026, 10:00 a.m. PDT**. Target an internal
finish by October 28 to leave time for recording and submission.

## Submission text

ML engineers repeatedly investigate the same question across papers, runs and
assistant sessions, then lose the evidence behind a decision. Qwen Envoy Research
MCP is a personal research-to-experiment loop. An MCP host searches public papers,
opens versioned source text, records exact passages and uncertainties, freezes an
agreed experiment plan, attaches measured results, and recalls a related attempt
in a later session. Local records are immutable and content-addressed. The MCP
checks that paper spans and selected run artifacts match the saved source bytes;
these checks do not decide whether a claim is true.

An optional NVIDIA **Nemotron 3.5 Lightning** step runs on **Nebius Token Factory**
to explain one selected public-paper research note. It receives the selected
question, findings and uncertainties with at most five verified spans, and no
local run artifact values. Use this opt-in tool only on notes safe to disclose
to Nebius. One capped request returns an explanation
with claim-to-evidence IDs; a reviewer still judges semantic support. Qwen remains
part of a separate agent-training case study. An experimental Qwen passage-reader
tool can be configured separately, but this demo does not invoke it and no Qwen
quality gain is claimed here.

The distinguishing workflow is the handoff from **research to a measured next
decision**, not an isolated paper summary. The host can find a prior experiment
by local word overlap, inspect its actual result and unresolved questions, and
avoid treating a cited paper as proof that a method worked in this workload.
An optional [cross-paper justification packet](RESEARCH_JUSTIFICATIONS.md) now
requires a simpler baseline, at least two source-verified papers, a non-supporting
comparison, transfer gaps and a matched test rule. The relations remain
caller-authored and need claim-level review; the packet is not a consensus vote.

## Significant updates in this submission period

This repository added the versioned arXiv MCP, source receipts, experiment
briefs, frozen plans, artifact-bound result records, restart-safe research
notes, cross-experiment recall, and the opt-in Nemotron Token Factory
explanation path. The Research MCP is the product submitted here. QASPER Agent
Studio documents the separate Qwen training and evaluation research.

## Recorded demo evidence, October 26

- **Live source smoke:** 19 actual MCP calls; three versioned public papers
  opened; 9/9 selected passage spans matched source text. No model requests.
- **Completed record replay:** 16 MCP calls across two stdio processes; prior
  research, one artifact-bound result, related-work lookup after restart,
  idempotent retries and conflict rejection. The result was **inconclusive**
  on latency, and the replay made no fresh source or model requests.
- **Decision-memory extension:** the current 20-call stdio replay also saves a
  separate, agent-proposed decision to defer a latency-win claim. After restart,
  the MCP finds that decision thread and checks that it still points to the
  exact saved result and research note. This is a record-integrity check, not
  evidence that the recommendation is useful or human-approved.
- **Research-to-Nemotron live path:** eight MCP calls; CodeAct paper opened,
  exact passage and research note saved, related record recalled, and **one**
  Nemotron 3.5 Lightning Token Factory request submitted successfully. Usage was
  **755 input and 273 output tokens**. At the [listed rates](https://nebius.com/services/token-factory/models/nvidia-nemotron-models-inference)
  of $0.06 and $0.24 per million tokens, the calculated charge for this successful
  request is **$0.00011082**, excluding any account adjustments. A preceding
  600-output-token attempt hit its cap; its usage was not saved, so total provider
  billing must be checked in Nebius. Both calls had one-request/no-retry and
  output-token limits, and the second preflight estimate with 10× pricing
  headroom was below $0.01.

The live output's sole claim says CodeAct's executable Python actions can adapt
after observations and outperform predefined JSON formats on API-Bank. **Assistant
claim review:** the selected passage directly supports the executable-action and
revision description, and says the approach beat widely used alternatives on
API-Bank and a new benchmark. It does **not** identify every compared alternative
in that benchmark sentence, so the specific “outperforming predefined JSON
formats” clause is stronger than this excerpt alone establishes. The explanation
correctly marks transfer to our document-research workload as untested. This is
an assistant review of one claim, **not independent human validation**. The
judge-facing wording above uses the narrower supported description.

## Nebius/NVIDIA feedback for the submission form

Nemotron 3.5 Lightning on Token Factory worked as an optional explanation step
over a small, verified public-paper packet. The successful request reported 755
input and 273 output tokens. A preceding 600-output-token attempt reached its
limit without a usable explanation; allowing up to 1,800 output tokens produced
the saved response. The model still overstated one comparison relative to the
selected passage, so the application keeps claim review separate from citation
integrity. For a personal MCP, a provider-enforced hard spend cap would make it
safer to give judges public access to live inference without exposing an
unrestricted key. This is feedback from two attempts, not a reliability or
latency benchmark.

## Reproduce the local test build

Install the repository's `mcp` and `vault` extras per the
[full project guide](../PROJECT_GUIDE.md).
For a no-model research-and-memory run using an existing public-paper cache:

```sh
python scripts/demo_hackathon_research_mcp.py \
  --library-root ~/.envoy/arxiv-library \
  --output out/my-research-demo
```

The first run may fetch public arXiv text. It writes `report.json` with MCP calls,
the exact selected passage, saved record ID and related-work result. To run the
optional NVIDIA step, put `NEBIUS_API_KEY` in the environment or ignored `.env`,
use a **new** output directory and append `--live-nebius`. That opt-in path makes
one bounded paid call. It does not activate a GPU.

On macOS Python installations without a configured CA bundle, a cold arXiv
fetch may report `CERTIFICATE_VERIFY_FAILED`. Point `SSL_CERT_FILE` at a trusted
bundle such as `$(python -c 'import certifi; print(certifi.where())')`, then use a
new output directory for the retry. Do not disable TLS verification.

A separate complete result-and-restart demonstration is:

```sh
python scripts/demo_experiment_mode.py --output out/my-completed-experiment-demo
```

### Clean-checkout verification, October 26

A fresh public clone installed `.[mcp,vault]` in a new Python 3.13 virtual
environment. With a trusted CA bundle configured on this macOS host, the
no-key research demo fetched CodeAct into a cold cache and completed **7/7 MCP
calls**, with **zero model requests** and no tool errors. The independent
experiment-record demo completed **20 MCP calls across two processes** and
preserved an inconclusive result and decision thread; its deliberate conflicting
write checks were rejected as expected. It made **zero model requests and GPU
activations**. An initial cold fetch without a configured CA bundle returned
`CERTIFICATE_VERIFY_FAILED`; the demo now exposes the underlying source error
instead of only saying the paper was unavailable. These checks do not repeat
the historical paid Nemotron call.

## Video outline, under three minutes

1. **Problem (15 s):** Show a real ML question and the lost-context failure.
2. **Research (45 s):** Run the no-key MCP source demo, open CodeAct, inspect
   the exact passage and saved research note, and show its uncertainty.
3. **Nebius (30 s):** Show the **saved response from the earlier live** Nemotron
   Token Factory call, including model ID, token usage and claim-level caveat.
   Label it recorded; do not imply the public inspector makes a new call.
4. **Memory (45 s):** Run the record demo, show its second-process resume,
   retrieve the inconclusive measured outcome and inspect the next test.
5. **Boundaries (20 s):** No Qwen improvement or Codex-web productivity win is
   claimed. Explain the intended personal use and what remains manual.

## Submission gates still open

- Publish a public YouTube video under three minutes. The research MCP inspector
  is live at `https://jasonlingg.github.io/Qwen-Envoy/research-mcp/`; it displays
  saved output and does not make live calls. The repository's local test build
  is the working MCP route. The existing QASPER Studio is a separate interface.
- The [public repository](https://github.com/Jasonlingg/Qwen-Envoy) currently
  displays an **MIT license** in GitHub's About panel. The clean-clone no-key
  path above works; keep it available free through
  the judging period; never expose the Nebius API key.
- The optional live Nemotron path requires a private Nebius key. The public
  inspector shows a recorded response and cannot make a new model request.
  Judge access to the live path, and whether the current test build fully meets
  the event's platform-access expectation, remain unresolved. Do not expose an
  unrestricted paid key or describe the static page as live inference.
- Enter the track, text, repository URL, demo URL, video URL, and honest Nebius
  feedback on Devpost. Explain the substantial new work above.
- Obtain independent human review before claiming any semantic answer-quality
  improvement. The current live explanation is one illustrative case, not a
  paired benchmark against ordinary web browsing.
