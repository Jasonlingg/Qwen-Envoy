# Question desk: product direction

## The job

Help someone keep working on a question after the first answer. A user records
what they currently think, attaches a paper, report, or their own result,
inspects exact passages, decides whether the evidence changes their view, and
keeps the reason and remaining uncertainty. When new research arrives, the
system should bring it back to that question instead of starting another
isolated chat.

The primary screen should be a **question desk**: active questions, incoming
evidence to review, a dated current view, and the history of decisions. Chat is
an action inside a question for exploring its sources. Qwen is the bounded
code-execution investigator over frozen sources; Nemotron can help explain or
draft a possible revision. Neither model gets to declare that a real quote
supports a conclusion or silently publish a view into Obsidian.

## Why this is a product hypothesis, not a novelty claim

Uploading files and chatting with citations are standard. [Gemini Notebook](https://support.google.com/gemininotebook/answer/16215270) accepts many file
types and URLs. [Elicit Research Agent](https://elicit.com/solutions/research-agent)
supports uploaded documents and projects that build over time; [Elicit Alerts](https://elicit.com/solutions/alerts) already follow research questions. [Readwise](https://docs.readwise.io/readwise/docs/exporting-highlights/obsidian) can sync
source material to Obsidian. The proposed wedge is the combination of a
personal view **recorded before** new evidence, explicit evidence-to-view
review, a dated explanation of why that view changed, and a reviewed trail in
the user's own vault. We should test whether this workflow helps people make
better research decisions; its individual ingredients are not unique.

## One concrete journey

Question: “When does multi-step retrieval help my research library?” The user
records “I expect the iterative worker to beat the cheap baseline.” They
attach a PDF paper and a local pilot report, plus a web source. The current
host retrieves passages from these frozen sources and verifies their exact
offsets. A future Qwen worker could investigate the same material under the
code-execution protocol; today its optional investigation only reads the
reviewed vault snapshot. The user marks one pilot passage as challenging the earlier
view, reads the paper limitation, and accepts a narrower view: “Our small
pilot favors one-pass retrieval; a larger held-out test remains open.” The
question page keeps both dated views and the source receipts. A future weekly
paper should re-open this question only if it bears on the unresolved test.

The same flow works for a homework concept, a workplace experiment, or a life
lesson, but source strength and privacy differ by domain. An uploaded personal
note is not automatically a scientific result.

## Implementation sequence

1. **Current slice:** Start at `/lab`, keep `/chat` as a secondary tool, and accept
   bounded `.md`, `.txt`, and text-extractable `.pdf` attachments alongside URLs.
   Freeze originals outside the vault, verify exact quote spans, and stage only
   unreviewed Markdown under `_inbox/`.
2. **Current slice:** Let the user mark each passage as supporting, challenging,
   adding context, or unresolved. Preserve that *judgment* separately from the
   exact quotation receipt. Show “before / possible after / why” at acceptance.
3. **Next:** Route weekly radar candidates to active questions and show a small
   “new evidence to revisit” queue. Do not invent urgency or silently update a
   view. Publish reviewed notes to the Obsidian library through the existing
   manual boundary and expose them through MCP.
4. **Model gate:** Compare base and trained Qwen on the same held-out frozen
   code-execution questions before saying the fine-tuned worker improved this
   product. Record answer support, failures, latency, and cost.

## Testable bet

**Hypothesis:** a question desk with a dated prior view and source receipts
helps a user recover what changed and choose a next experiment more reliably
than a chat transcript of the same materials.

**Signal:** across three genuine questions, the user can find the prior view,
the passage that caused a revision, the remaining uncertainty, and the next
check without rereading the full conversation. Reviewers can trace each
material claim to an inspectable source and distinguish a personal result from
an external claim.

**Decision rule:** if the question page is merely a slower route to asking the
same chat question, simplify the interface and reconsider the product wedge.
Do not count successful file parsing, real quote spans, or a working model
endpoint as proof that the learning workflow is useful.
