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
an optional code-execution reader, and no Qwen quality gain is claimed here.

The distinguishing workflow is the handoff from **research to a measured next
decision**, not an isolated paper summary. The host can find a prior experiment
by local word overlap, inspect its actual result and unresolved questions, and
avoid treating a cited paper as proof that a method worked in this workload.
An optional [cross-paper justification packet](RESEARCH_JUSTIFICATIONS.md) now
requires a simpler baseline, at least two source-verified papers, a non-supporting
comparison, transfer gaps and a matched test rule. The relations remain
caller-authored and need claim-level review; the packet is not a consensus vote.

## Significant updates in this submission period

The pre-existing Qwen document agent was extended with the versioned arXiv MCP,
source receipts, experiment briefs, frozen plans, artifact-bound result records,
restart-safe research notes, cross-experiment recall, and the opt-in Nemotron
Token Factory explanation path. The new product path addresses a personal ML
engineering workflow; the older QASPER Agent Studio remains a separate saved
model-evaluation case study.

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
one bounded paid call. It does not activate a GPU. A separate complete
result-and-restart demonstration is:

```sh
python scripts/demo_experiment_mode.py --output out/my-completed-experiment-demo
```

## Video outline, under three minutes

1. **Problem (15 s):** Show a real ML question and the lost-context failure.
2. **Research (45 s):** Open CodeAct through MCP, inspect exact passage and the
   saved research note. Point out the source-span check and uncertainty.
3. **Nebius (30 s):** Invoke the opt-in Nemotron tool, show model ID, token usage,
   explanation and the claim-level caveat. State that the earlier capped attempt
   failed and was fixed.
4. **Memory (45 s):** Restart the MCP host, ask a related question, retrieve the
   record, show its measured outcome is inconclusive, and inspect the next test.
5. **Boundaries (20 s):** No Qwen improvement or Codex-web productivity win is
   claimed. Explain the intended personal use and what remains manual.

## Submission gates still open

- Publish a public YouTube video under three minutes. The research MCP inspector
  is live at `https://jasonlingg.github.io/Qwen-Envoy/research-mcp/`; it displays
  saved output and does not make live calls. The repository's local test build
  is the working MCP route. The existing QASPER Studio is a separate interface.
- The [public repository](https://github.com/Jasonlingg/Qwen-Envoy) currently
  displays an **MIT license** in GitHub's About panel. Verify setup from a clean
  clone. Keep a no-key source-and-memory path available free through
  the judging period; never expose the Nebius API key.
- Enter the track, text, repository URL, demo URL, video URL, and honest Nebius
  feedback on Devpost. Explain the substantial new work above.
- Obtain independent human review before claiming any semantic answer-quality
  improvement. The current live explanation is one illustrative case, not a
  paired benchmark against ordinary web browsing.
