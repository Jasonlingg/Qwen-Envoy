# Nebius x NVIDIA hackathon build: an evidence-backed AI research companion

> **Scope update (September 26):** The current cross-domain Personal AI product
> and dated delivery plan are in [the personal learning memory build plan](PERSONAL_LEARNING_MEMORY_BUILD_PLAN.md).
> This document preserves the earlier AI-research slice and initial Nebius
> bridge research. Follow the newer plan for product scope, schedule, and the
> account-specific Qwen deployment gate.

**Decision (September 26, 2026):** Adapt Envoy for the **Personal AI** track as a
working weekly AI-research learning loop. This is a product build around the
existing Qwen code-execution research worker, not a claim that the current Qwen
checkpoint has passed the personal-vault or new evidence-search gate. The new
NVIDIA model must have a real runtime role: Nemotron on Nebius Token Factory
turns checked evidence packets into explanations and weekly notes. Qwen remains
the bounded source-finding worker and a separately measured model-improvement
experiment.

## Rules and resource boundary

The [official rules](https://nebiusglobalaihackathon.devpost.com/rules) require a
working app that makes a **runtime** Token Factory inference call or runs on
Nebius AI Cloud, and uses at least one NVIDIA open-source model. Qwen alone does
not satisfy the latter requirement. The Personal AI track fits the vault,
persistent notes, read-only MCP access, and a user-controlled weekly workflow.
The Coding and Agentic Engineering track instead emphasizes developer tools and
Token Factory Sandboxes; Envoy's Python paper-search actions are not that product.

Submission closes **October 30, 2026 at 10:00 AM Pacific**. The existing project
must be significantly updated after August 26; identify the new Nebius/Nemotron
runtime integration, weekly loop, vault/MCP product path, and reviewed user flow
in the submission. Required artifacts are a public, licensed repository and
setup README, a working demo URL/test build, a public YouTube demo under three
minutes, a project description and track selection, and feedback. The demo must
remain available for free judging through December 15. The repository already
has an MIT `LICENSE`; verify the public page recognizes it. The rules exclude
Quebec residents, so entrant eligibility depends on actual residence.

The [hackathon resources](https://nebiusglobalaihackathon.devpost.com/resources)
offer $25 Token Factory inference credit with activation code
`NEBIUS-DEVPOST-GLOBAL26` and another $25 Token Factory credit through the
[Builders Program](https://dev.nebius.com/builders). These are **not promised AI
Cloud GPU training credits**. [Program terms](https://nebius.com/builders-terms-and-conditions)
say the Builder credit expires after 90 days and is for experimentation; make
sure the demo's runtime billing and availability are covered through judging.
The hackathon [resources page](https://nebiusglobalaihackathon.devpost.com/resources)
also says in-person Builders & Brews attendees may unlock additional AI Cloud
credits; the Toronto event is listed for September 29. Treat that as an
opportunity to check, not as compute already in the account.
Nebius [lists Nemotron 3.5 Lightning](https://nebius.com/services/token-factory/models/nvidia-nemotron-models-inference)
at $0.06/M input and $0.24/M output tokens, with an OpenAI-compatible public
endpoint. Start with Lightning and meter real calls. Super is an optional
quality comparator, not an assumed improvement. No new Runpod RL job is in the
hackathon critical path.

Nebius [lists Qwen3-8B for LoRA fine-tuning](https://docs.tokenfactory.nebius.com/post-training/models),
but its current [deployment overview](https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/overview)
says public serverless endpoints serve base models and eligible custom weights
use dedicated endpoints billed per GPU-hour. An older serverless custom-LoRA
guide now returns 404. Hosting the existing adapter therefore requires an
account-specific compatibility and cost check before it is promised in a demo.
A live Nemotron call is the required NVIDIA component either way.
Token Factory also lists self-service Qwen supervised fine-tuning, but the
hackathon credits' applicability and job price must be checked in the account;
the public offer does not establish a funded RL training run.

## Demo contract

Choose one historical week and one topic, initially tool-using research agents.
Freeze a small public-paper snapshot with source hashes and human-reviewed
anchors. The user sees three recommended papers, opens one, asks a follow-up
question, inspects exact source passages, and saves a weekly Markdown digest to
an Obsidian-compatible vault. A larger assistant can later ask a read-only MCP
tool for the saved digest and source evidence. The public demo uses only public
papers and synthetic/example notes. For a real personal vault, keep the index
local and send only the selected excerpts the user has chosen to process; do
not upload an entire private vault as a default.

```text
frozen public papers / opt-in vault snapshot
  -> deterministic retrieval and bounded Qwen Python search
  -> exact-span validation and small evidence packet
  -> Nemotron 3.5 Lightning on Nebius Token Factory (live)
  -> claim-linked explanation and weekly digest
  -> Obsidian Markdown + read-only MCP retrieval
```

The existing code-execution harness, recorded Qwen traces, corpus import,
OpenAI-compatible policy, evidence-packet explainer, and vault export are useful
components. The older JSON-action research agent is paused; the main demo
handoff must consume the **code-execution** result shape. Quotation/offset checks
prove provenance, not semantic support. Human review of key digest claims is
required before presenting the example as accurate. A deterministic retriever
is a useful live fallback when no Qwen endpoint is available, but recorded Qwen
actions must be labeled as recorded.

The first bridge is implemented in `src/research/code_exec_packet.py` and
`scripts/explain_code_exec_nebius.py`. Inspect the exact packet from a historical
development run without a key or billable call:

```bash
python scripts/explain_code_exec_nebius.py \
  --transcript out/research/learning-loop-model-sweep-20260926/v5_3.json \
  --question-id learn_01_react_loop \
  --corpus out/research/starter-2026-09-12/corpus \
  --output /tmp/envoy-nebius-packet.json --packet-only
```

Once `NEBIUS_API_KEY` is configured in a local environment or secret manager,
omit `--packet-only` and use a new output path to call Nemotron 3.5 Lightning
on Token Factory. The script saves JSON and Markdown for review. This historical
question was already used in development and is **not** a held-out demo result.
The bridge has been verified offline; a live Nebius call, response-format
behavior, token metering, and human semantic review remain to be done.

## Bounded build sequence

1. **Access and meter:** join the hackathon, claim credits, create a Token
   Factory API key outside Git, make one Lightning chat-completions call, and
   record actual tokens, latency, and charge. Do not print or commit the key.
2. **Evidence handoff:** convert active code-execution result spans into a
   bounded, source-materialized packet. Reject nonexistent document IDs,
   invalid offsets, duplicate spans, and packets with no usable evidence. Test
   this without a model call.
3. **Live NVIDIA step:** have Nemotron explain only that packet. Validate JSON
   and evidence IDs mechanically; review semantic support separately. Save model
   ID, endpoint, prompt version, source snapshot hash, token usage, and cost.
4. **Product slice:** make the one-week shortlist, one deep explanation, and one
   Obsidian digest actually work. Expose saved notes/evidence through read-only
   MCP. Add a simple UI or CLI path that a judge can run without a private vault.
5. **Review and submit:** review the example claims, compare direct retrieval,
   base Qwen, and v5 only where a live endpoint is available, measure cost and
   failures, and record the limits. Provide the demo URL, README, public repo,
   short YouTube video, and substantial-update explanation before the deadline.

The hackathon product has its own acceptance check: a new user can run or open
the demo, understand why a paper was recommended, inspect supporting passages,
save a useful digest, and retrieve the same evidence later. The separate Envoy
research claim still requires trained Qwen to beat base Qwen on held-out
supported-answer quality with acceptable execution failures, latency, and cost.
Do not let a working Nemotron digest or a hackathon submission stand in for that
model-improvement result.
