# Envoy chat-first second brain: build plan

**Planning baseline:** September 26, 2026. **Target:** a working Personal AI
hackathon submission by October 29, ahead of the [October 30, 2026, 10:00 a.m.
Pacific deadline](https://nebiusglobalaihackathon.devpost.com/rules). This is the
current broader product direction requested by the user. The earlier
[AI-research-specific hackathon plan](NEBIUS_HACKATHON_2026.md) remains useful
for its service research and evidence bridge, but this document sets the
product scope and schedule. **September 26 direction update:** the main experience
is a conversation about what the user has learned and how their understanding
changed. The existing form-based [local memory demo](PERSONAL_MEMORY_LOCAL_DEMO.md)
is a reusable storage/evidence prototype, not the intended interface. Defer
interface polish until the chat and linked-vault behavior work. The
[weekly research radar](WEEKLY_RESEARCH_RADAR.md) is one source and eventual
workflow, not the entire product. The [second-brain landscape and positioning
review](SECOND_BRAIN_LANDSCAPE_AND_POSITIONING_20260926.md) narrows the initial
product to an evidence-linked learning and application history: generic
vault chat, graphs, and spaced quizzes are already common.

## Product and honest claim

Build a **chat-first lifelong-learning assistant** over an Obsidian vault,
starting with research and technical learning while allowing life lessons and
homework in the same vault. A person can ask what they previously
believed, what evidence or attempt changed their mind, how two ideas connect,
what remains uncertain, and what to revisit or try next. The system traces
answers to original dated sources, distinguishes user observations from
assistant inferences, and proposes updates for approval. The vault is the
durable source of truth; its ordinary internal links, backlinks, and native
graph let the person explore the connected history. The chat can eventually
live in an Obsidian side panel, but a small chat surface is sufficient for the
first working loop. A separate custom graph UI is not on the critical path.

The strongest first demo uses a research learning thread close to Qwen's
training domain: “What did I expect the trained search worker to improve,
what did the base-versus-v5 test actually show, and what should I test next?”
The answer cites dated hypotheses, source papers, executed results, and a
revised decision in Obsidian. The user can approve a linked revision; a later
chat retrieves it alongside its history. Life and homework threads show the
schema's breadth, but do not imply Qwen has passed those domains. Public demo
data will be synthetic or public; private vault imports require explicit
folder selection. Selected excerpts sent to Nebius must be visible to the user.

This is an integration and measured-worker project, not a claim that a generic
AI memory bank is new. [Dria mem-agent](https://huggingface.co/blog/driaforall/mem-agent-blog)
already trains Qwen for Python-tool memory work behind MCP;
[Hindsight](https://github.com/vectorize-io/hindsight) already has
retain/recall/reflect and an [Obsidian plugin](https://github.com/vectorize-io/hindsight-obsidian).
The differentiated test is whether a bounded Qwen evidence worker improves
dated, source-supported learning recall and downstream Nemotron answers over
ordinary retrieval, with user-approved changes and a reproducible comparison.

## Two outcomes with separate acceptance gates

**Hackathon product gate, due October 29.** A judge can open a working URL or
test build, ask a question in chat, follow connected notes and dated revisions
in Obsidian, inspect cited source passages, approve a proposed update, and
later recall that update. They can see which model did each step.
Nemotron makes a real runtime call through Nebius Token Factory. Qwen performs
a real bounded investigation in the demonstrated workflow; a recorded replay
is labeled as a replay and cannot stand in for a live worker. The public
repository has an open-source license, setup README, and a reproducible public
sample. A public video under three minutes and the required feedback and
significant-update description complete the submission. Free judge access must
remain available through December 15. These are [official rules](https://nebiusglobalaihackathon.devpost.com/rules),
not optional polish.

**Envoy research/product gate, potentially after the hackathon.** A useful
weekly AI-research digest lands in Obsidian, and a larger assistant can query
the accumulated evidence through MCP. On the same held-out personal-vault
code-execution tasks, trained Qwen beats base Qwen on human-reviewed,
source-supported answers without unacceptable execution, latency, or cost
regressions. A good hackathon demo does not establish that model result.

## One job per component

| Component | Responsibility | Boundary |
| --- | --- | --- |
| Opt-in vault importer | Freeze chosen Markdown/PDF sources with IDs, hashes, dates, and visibility scope | Preserve originals; no automatic whole-vault upload |
| Search and link index | Cheap lexical retrieval plus deterministic Obsidian links, backlinks, dated revision edges, and one-hop candidate expansion | Candidate provider, not the answerer; no graph database needed initially |
| **Qwen memory investigator** | Write Python using search/read/extract over the allowed snapshot; follow candidate connections and return exact passages, possible outdated/conflicting records, or an explicit no-evidence result | Read-only, bounded turns; cannot edit user memory or assert a life lesson as fact |
| Deterministic verifier | Check allowed source IDs, exact quote offsets, snapshot hash, duplicates, and budget | Provenance validation does not prove semantic support |
| **Nemotron chat coordinator** | Decide when to investigate, give a conversational explanation, acknowledge uncertainty, and suggest a question or application when useful | Live NVIDIA model on Nebius Token Factory; cite packet IDs; cannot turn an unverified guess into a memory |
| User-approved writer | Save a new learning/revision/decision note with actual Obsidian links to concepts, sources, and superseded notes | Append revision history; do not silently replace old beliefs |
| Read-only MCP interface | Let another assistant search and inspect approved memories and evidence | Write operations remain separate and explicit |

The memory record needs at least: stable ID, kind (source/attempt/concept/
learning/revision/decision), source path and hash, source type, captured date,
effective date if known, review status, visibility scope, and
supersedes/derived-from relationships. Source notes preserve what was said or
observed; concept notes state a current, revisable understanding; dated
learning/revision notes say what changed and why. **Real `[[internal links]]`
must connect these files** for Obsidian's graph and backlinks; frontmatter IDs
alone are not graph edges. A small deterministic index can retain typed
relationships such as “supersedes,” which the native graph does not label.
Paper claims, homework attempts, and user beliefs remain distinct records.
Qwen's packet must include time and source type, a no-match path, and uncertainty.
The current [packet code](../src/research/code_exec_packet.py) validates spans
but requires a proposed answer and nonempty evidence, so this contract requires
an implementation change.

The learning loop is a product hypothesis, not just a storage workflow: chat
can ask the user to recall an idea **before** showing the source, then give
source-linked feedback, revisit it later, and record whether they applied it.
[Retrieval practice](https://doi.org/10.1126/science.1152408),
[spacing](https://pubmed.ncbi.nlm.nih.gov/16719566/), and
[corrective feedback](https://pubmed.ncbi.nlm.nih.gov/18605878/) support those
ingredients in human-learning research. They do not establish that this
particular AI assistant improves learning. Measure later recall and useful
application, not note count or graph size; keep prompts optional and ask the
user which subjects they want to revisit.

The existing [vault importer/exporter](../src/research/vault.py),
[code-execution environment](../src/env/document_env.py),
[evaluation harness](../src/eval/harness.py),
[replay viewer](../scripts/viewer.py), and
[offline-tested Nemotron bridge](../scripts/explain_code_exec_nebius.py) are
reusable. A [localhost product slice](PERSONAL_MEMORY_LOCAL_DEMO.md) now
provides capture, dated source review, explicit append-only approval, and later
recall over a synthetic vault, with lexical retrieval clearly labeled. The
chat-first browser now shows a source-triggered prior/result/revision timeline
after a capture, while a separate explicit button can request a Nemotron draft
when configured. This local candidate check does not establish semantic
contradiction, and no model draft is saved without user approval. The
[readiness screen](PERSONAL_MEMORY_READINESS_EVAL.md) fixes ten named questions
and a base/v5 comparison protocol. The browser demo has one verified live
Nebius/Nemotron synthetic smoke path, and a separate local read-only MCP bridge
now serves pinned-snapshot search and source inspection. A manual weekly writer,
[historical replay](../data/research/weekly_digest_2025_w32_agent_draft.json), and
[current-week agent-authored draft](../data/research/weekly_digest_2026_w39_agent_draft.json)
exist, but no human-reviewed digest has been published. There is still no live
Qwen run on the personal-memory fixture, deployed build, or
substantive personal-vault confirmation. The configured personal vault has only
a Welcome note; the existing viewer remains an evaluation UI. The nine-note
readiness fixture remains hash-locked and has no visible Obsidian links. A
separate ten-note synthetic demo vault adds 18 real wikilinks and a cross-domain
concept note, with corpus hash
`92186559398144adf01851599b0057a616b0b044df93c8fde4c7fb7bb8180a41`.
Its observations cannot be substituted for results on the locked fixture.

For a public live Qwen demo, code execution must run inside an isolated
container or equivalent sandbox and fail closed if it is absent. The current
[REPL](../src/env/repl.py) can fall back to a local Python subprocess; that is
unsuitable for a public endpoint executing model-authored code.

## Scope by delivery level

**October demo:** one public sample vault with substantive notes in all three
domains; linked concept/source/revision notes visible in Obsidian's native
graph; one dated correction; one newer decision that supersedes an older one;
one AI-paper item; a live Nemotron/Qwen investigation in chat; source inspection
and Obsidian deep links; user approval and Markdown save; later recall;
read-only MCP; a short optional “revisit and apply” conversation. Reuse the
existing backend, but replace the form-led journey with a minimal chat. Limit
connectors to selected local Markdown/PDF and the public sample.

**After the deadline:** unattended paper discovery and deduplication, recurring
weekly research ranking/digest over real new papers, broader vault import,
stronger privacy controls, more polished always-on hosting, and a convincing
held-out Qwen training result. These remain part of the full Envoy goal; they
are not implied by the October demo.

## Dated execution plan

The estimate assumes one developer, about 30–40 focused hours per week,
existing repo components, and no guaranteed GPU training credit. Dates are
delivery targets, with October 30 reserved for submission contingency rather
than feature work.

| Dates | Work and concrete deliverable | Effort | Exit test |
| --- | --- | ---: | --- |
| **Sep 28–Oct 2** | Lock the one-story chat demo and source/concept/revision schema; create linked public notes across life, homework, and research; claim/check Nebius access; make one metered live Nemotron call; test Qwen serving routes and record price | 18–26 h | Named sample sources and questions; visible Obsidian links; live Nemotron response; an affordable live Qwen path chosen by Oct 2 |
| **Oct 5–9** | Index vault links and dated revisions; extend Qwen's read-only packet contract for dates, conflicts, no-match, and allowed folders; wire Qwen endpoint; verify exact spans; make public code execution fail closed to isolation | 28–38 h | A live Qwen episode returns an inspectable packet from the linked sample, including a correct abstention case |
| **Oct 12–16** | Connect Nemotron to packets; build chat turns, proposed linked revision, explicit approval and Markdown save; add read-only MCP | 34–44 h | A chat answer cites old/new linked notes; an approved revision is visible in Obsidian and retrievable later |
| **Oct 19–23** | Run paired baseline checks and human source-support review; fix observed product failures; test privacy scope, temporal conflicts, bad quotes, and deployment | 28–38 h | Same named questions run through simple search, base Qwen, and v5 where available; no unsupported demo claims |
| **Oct 26–29** | Deploy or package a judge-ready test build; test from a fresh checkout; write README, method/evidence report, short video, Devpost text, feedback, and significant-update account | 22–30 h | Another person can follow setup, open the demo, and complete the story without private data |

**Total October estimate: 130–176 focused hours (roughly 17–22 full
developer-days).** Some tasks overlap; the calendar is tight because there
are about 24 normal weekdays before submission. At 35–40 hours per week, a
limited working demo is plausible by October 29 if model access is resolved in
week one. At 15 hours per week, the same work takes roughly 9–12 weeks and
misses this deadline. A broader polished personal assistant plus the full
weekly radar and proven model improvement likely takes another **6–10
full-time weeks**, depending mainly on real data, compute, and hosting.

## Decisions that prevent wasted work

1. **Qwen serving, by October 2.** Nemotron's public Token Factory path is
   documented. Current [Nebius deployment docs](https://docs.tokenfactory.nebius.com/ai-models-inference/dedicated-endpoints/overview)
   say public serverless endpoints serve base models, while eligible custom
   weights require dedicated, GPU-hour-billed endpoints. An older serverless
   LoRA guide is stale. Check the account catalog, current adapter compatibility,
   cost, and a real Qwen response before designing around hosted v5. If that
   route is unavailable, evaluate a public base-Qwen API or a packaged local
   quantized worker; neither is silently presented as the trained v5 adapter.
   Keep an explicit spend cap, and ensure judge access lasts through December 15.
2. **Data, by October 5.** The real vault currently lacks substantive notes.
   The public sample can prove the app works but cannot prove personal
   usefulness or trained-model transfer. If the user opts in to a separate real
   vault slice, keep it out of the public repo and judge fixture.
3. **Safety and trust, before public access.** Run Qwen-generated Python only
   in isolation, enforce folder permissions in tools as well as prompts, and
   verify excerpts before Nemotron sees them. Show citations and uncertainty.
   Require user approval for durable changes.
4. **Training, only after the baseline.** The existing v5 checkpoint was the
   best development candidate but had zero fully supported passes on the
   eight-question paper sweep; that is not a personal-memory result.
   [Dria's mem-agent](https://huggingface.co/blog/driaforall/mem-agent-blog)
   shows a trained Qwen memory worker can improve on its own small benchmark,
   but used iterative RL on an eight-H100 node. Do not put new RL on the
   hackathon critical path or reuse the MuSiQue reward as a memory reward.

## Qwen experiment after the task is real

**Hypothesis:** a trained code-execution Qwen will find more complete,
time-correct evidence for personal learning questions than base Qwen, while
staying within the same tool and time budget. First freeze a substantive
cross-domain corpus and named questions. Include multi-note joins, outdated
beliefs, user corrections, and answer-absent cases. Split by underlying
episode/source cluster so paraphrases cannot leak into training.

Run ordinary lexical/hybrid retrieval, base Qwen3-8B, and v5 on identical
corpus revisions, prompts, search index, tool budget, decoding, and hardware.
Keep Nemotron's model and prompt fixed when judging downstream answers.
Primary score: blind human review of whether every material final claim is
supported and correct for the question's date. Report evidence recall at five,
correct abstention, execution failures, steps, latency, token usage, and cost.
The current eight-paper development sweep is consumed development evidence,
not a held-out result.

Before any training job, name the recurring tool-policy error, record a
predeclared minimum improvement on a held-out set, and confirm compute. If the
failure is teachable, build complete, executable, source-checked trajectories
with explicit no-evidence examples and run **one** action-masked LoRA/QLoRA
SFT pilot from a pinned base model. Inspect paired cases and promote only if
held-out supported-answer quality improves with acceptable cost and errors.
If plain retrieval matches Qwen, improve the product/search index instead of
training. [ARK](https://aclanthology.org/2026.acl-long.714/) is relevant
evidence for trajectory-trained Qwen retrieval, but its data and H100 costs
do not transfer directly to this vault.

This benchmark/baseline work needs approximately **1–2 focused weeks** once a
real corpus and reviewers exist. A single data-and-SFT pilot plus held-out
analysis adds roughly **one further week**, conditional on an affordable GPU.
Those are work estimates, not a promise of a positive result by October 30.

## Cost and submission constraints

The hackathon advertises [$25 in Token Factory credits](https://nebiusglobalaihackathon.devpost.com/resources)
and another $25 through Builders. They are not guaranteed AI Cloud GPU
training credits. Public [Nemotron 3.5 Lightning pricing](https://nebius.com/services/token-factory/models/nvidia-nemotron-models-inference)
is low per token; actual calls and usage still need metering. Qwen serving,
custom-weight deployment, and any training bill remain unknown until account
verification. Do not turn on an hourly dedicated endpoint without a budget and
shutdown plan. The demo and its access must survive the December judging
window, not just the submission day.

The final submission requires a working URL/test build, public licensed source
and setup instructions, public video under three minutes, model/platform usage
description, Nebius feedback, and an explanation of the significant changes
since August 26. Publish only the synthetic/public fixture and redacted run
artifacts.
