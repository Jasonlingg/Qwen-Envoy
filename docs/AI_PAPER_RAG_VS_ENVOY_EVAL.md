# AI-paper RAG versus Envoy evaluation

## Pre-registered comparison

This development experiment asks whether iterative code execution earns its cost over a strong
one-pass RAG baseline. It uses the frozen ten-question AI-paper pilot and six-paper corpus. Those
papers, questions, answers, and grader notes were excluded from training. The locked 40-question
test set remains untouched.

Both systems use `Qwen/Qwen3-8B`, the same targeted QASPER checkpoint, greedy decoding, seed 42,
and the same frozen corpus. The one-pass baseline uses diversified BM25 passage retrieval: four
candidate papers, up to three 512-character passages per paper, and a 12,000-character context
limit. Envoy may write Python and search iteratively for up to 15 steps. It receives four bounded
verifier recoveries before an unresolved answer is escalated. No host model is part of this
comparison.

The hypothesis is that Envoy will help on the four questions requiring two papers and the two
questions requiring careful refusal or limitation, while one-pass RAG will be faster and may be
equally good on direct single-paper questions.

Human blind review is the primary score. A response passes only when it answers every part, its
major claims are supported by submitted evidence, and its limitations are honest. Automatic
answer overlap is diagnostic. We also record required-document recall, valid-evidence rate,
execution errors, steps, and wall time.

Decision rule:

- Keep iterative exploration as the default only if it has a higher supported-answer rate overall
  and gains at least two paired verdicts on the six multi-paper/epistemic questions.
- If the systems tie overall, use one-pass RAG by default and dispatch Envoy only for the hard
  question classes where it wins.
- If RAG wins overall and Envoy has no clear hard-subset advantage, make RAG the product path.

This pilot has only ten questions. Passing it justifies a larger paper-disjoint comparison; it is
not a final performance claim.

## Result — 2026-09-20

The one-pass RAG baseline won this pilot. Iterative Envoy is not the product default under the
pre-registered rule.

| Metric | One-pass RAG | Iterative Envoy |
|---|---:|---:|
| Blind pass / partial / fail | **1 / 5 / 4** | **0 / 0 / 10** |
| Mean blind-review score (0–2) | **0.70** | **0.00** |
| Paired wins / ties | **6 wins** | **0 wins / 4 ties** |
| Multi-paper and epistemic subset | **4 wins / 2 ties** | **0 wins** |
| Automatic outcome reward | **0.397** | **0.000** |
| Submission rate | **100%** | **0%** |
| Required-document recall | **76.9%** | **0%** |
| Questions with valid evidence | **70%** | **0%** |
| Average steps | **1.0** | **15.0** |
| Average wall time | **8.15 s** | **31.84 s** |

The failure occurred before research synthesis. Envoy made 115 `search_within()` calls with
invented document IDs; all 115 returned “document not found.” It called `list_docs()` once across
the entire evaluation, never recovered a useful document, and either escalated or exhausted the
step budget. The evidence verifier correctly rejected unsupported submissions, but it could not
repair failed corpus discovery.

The behavior is consistent with a protocol mismatch: the targeted QASPER checkpoint practiced
working inside a known paper, while this environment requires the policy to discover valid paper
IDs before calling `search_within()`. This experiment does not isolate whether that behavior comes
from SFT, the prompt, or the tool interface because it did not include a current-prompt base-model
Envoy arm.

One-pass RAG avoided that failure by retrieving passages before generation, but its absolute
quality is still weak: only one answer fully passed blind review. It often found relevant evidence
yet omitted a requested comparison, qualification, or experimental detail. The result supports a
RAG-first product path today; it does not establish that the current RAG system is production
ready.

Before more fine-tuning, the next diagnostic should make corpus discovery reliable and rerun this
same pilot with three arms: one-pass RAG, base Qwen3-8B Envoy, and the targeted-SFT Envoy. That
comparison can separate a harness problem from an SFT regression. The locked 40-question test set
remains untouched.

Artifacts are in `out/research/rag-vs-envoy-ai-pilot-v1/`. The two manifests record the exact
checkpoint, question and corpus hashes, decoding settings, seed, reward version, software, and
NVIDIA A40 hardware. Human judgments were completed before opening the blind assignment key.

## Paper-ID intervention follow-up — pre-registered 2026-09-20

The first comparison isolated a document-discovery failure: Envoy never reached valid paper text.
This follow-up keeps the checkpoint, original questions, corpus, decoding, verifier, seed, and
15-step budget fixed while changing only the candidate IDs shown with each question.

- **Oracle-ID diagnostic:** show the gold-required paper IDs. For the snapshot-level unanswerable
  question, show the complete six-paper catalog. This is diagnostic and cannot be reported as a
  deployable product score.
- **Retrieved-ID system:** show the top four documents from the same label-free BM25 passage
  retriever used by the one-pass RAG baseline. Distractors remain in the candidate set.

Hypothesis: valid candidate IDs will remove the failed-discovery loop and allow the targeted
QASPER checkpoint to exercise its known-paper evidence behavior. The expected operational signal
is at least 70% submission rate and valid evidence on at least 60% of questions in the oracle arm.

Decision rule:

- If the oracle arm misses either operational threshold, document discovery is not the only major
  failure; stop adding training data and inspect reasoning/tool behavior.
- If oracle passes but retrieved IDs do not, improve candidate retrieval before changing SFT.
- Treat the hybrid as a viable replacement for one-pass RAG only if the retrieved-ID arm exceeds
  RAG's 0.70 mean blind-review score and wins more paired questions than it loses.

## Paper-ID intervention result — 2026-09-20

Valid IDs fixed the catastrophic navigation failure. They did not make Envoy better than one-pass
RAG on semantic answer quality.

| Metric | One-pass RAG | Envoy + oracle IDs | Envoy + retrieved IDs |
|---|---:|---:|---:|
| Blind mean score (0–2) | **0.70** | 0.50 | 0.60 |
| Blind pass / partial / fail | **1 / 5 / 4** | 0 / 5 / 5 | 1 / 4 / 5 |
| Submission rate | 100% | **90%** | **90%** |
| Questions with valid evidence | 70% | 70% | **90%** |
| Required-document citation recall | 76.9% | **92.3%** | 61.5% |
| Average steps | **1.0** | 4.2 | 4.3 |
| Average wall time | **8.15 s** | 21.42 s | 29.89 s |

The label-free BM25 candidate retriever placed 12 of the 13 required papers in its top-four
candidate sets, or 92.3% candidate recall. Once those IDs were present, Qwen used valid
`search_within()` and `read()` calls instead of inventing IDs. Oracle-ID Envoy passed the
pre-registered operational gate: submission rate rose from 0% to 90%, and 70% of questions had
valid evidence. Document discovery therefore caused the original total collapse.

Discovery was not the only problem. Oracle-ID Envoy lost its paired semantic comparison to RAG
3–4 with three ties. Retrieved-ID Envoy lost 1–2 with seven ties and scored 0.60 rather than the
required greater-than-0.70. On the six hard questions, retrieved-ID Envoy recorded no paired wins,
RAG recorded two, and four tied.

The remaining failures occur after retrieval: incomplete multi-part answers, a false current-best
claim even with the complete six-paper catalog, an incorrect statement about original RAG's
updateable knowledge, distraction by irrelevant candidate papers, and submissions that the
evidence verifier rejected. Under the pre-registered rule, one-pass RAG remains the default. More
known-paper SFT would not target these demonstrated failures.

Artifacts are in `out/research/ai-paper-id-diagnostics-v1/`. The GPU runs used
`Qwen/Qwen3-8B`, adapter `jasonlingg/qwen-envoy-qwen3-8b-qasper-targeted-sft-v1`, greedy decoding,
seed 42, outcome-v1, and one NVIDIA RTX 4090. The locked 40-question test set remained untouched.

## Base-versus-SFT retrieved-ID diagnostic — pre-registered 2026-09-20

This final training diagnostic asks whether targeted QASPER SFT improves Qwen after candidate-paper
discovery is held fixed. It compares base `Qwen/Qwen3-8B` with the existing retrieved-ID SFT run.
Both arms use the same frozen ten questions, top-four BM25 candidate IDs, six-paper corpus, code
prompt, greedy decoding, seed 42, evidence verifier, four recovery turns, and 15-step budget.

Blind semantic review is primary. Keep the SFT adapter only if it has a higher mean review score,
wins more paired questions than it loses, and does not regress submission rate or
questions-with-valid-evidence rate. A tie or loss means the adapter has not justified its product
complexity. No new training follows this diagnostic; its result freezes the model choice before
the product demo and final held-out evaluation.

## Base-versus-SFT retrieved-ID result — 2026-09-20

Targeted SFT improved Qwen's behavior under the same retrieved-ID harness and passed the
pre-registered adapter-retention rule. The improvement is real but small: it establishes that the
training was useful, not that the resulting agent is ready or better than one-pass RAG.

| Metric | Base Qwen3-8B | Targeted-SFT Envoy |
|---|---:|---:|
| Blind mean score (0–2) | 0.40 | **0.60** |
| Blind pass / partial / fail | 1 / 2 / 7 | **1 / 4 / 5** |
| Paired wins / losses / ties | 0 / 2 / 8 | **2 / 0 / 8** |
| Automatic outcome reward | 0.194 | **0.336** |
| Submission rate | 50% | **90%** |
| Questions with valid evidence | 50% | **90%** |
| Required-document citation recall | 46.2% | **61.5%** |
| Execution-error step rate | 14.1% | **0%** |
| Average steps | 12.8 | **4.3** |
| Average wall time | 40.78 s | **29.89 s** |

The adapter improved protocol use most clearly. Base Qwen often exhausted all 15 steps without a
submission and made invalid or repeated calls. The trained model usually searched the supplied
candidate papers, cited valid spans, and submitted in a few turns. It also converted two base
failures into partial answers without losing any paired question. These effects remained after
candidate discovery, prompts, tools, decoding, questions, and corpus were held fixed, so they are
evidence of an SFT effect rather than another harness change.

The semantic ceiling remains low. Both systems produced only one fully passing answer out of ten,
and SFT's 0.60 blind score remains below one-pass RAG's 0.70. Training taught Qwen to operate the
harness more reliably, but it did not reliably teach complete comparison, calibrated abstention,
or careful synthesis. Keep the adapter for the iterative Envoy arm; keep one-pass RAG as the
product default. Do not spend more training compute on this data recipe before a larger held-out
evaluation demonstrates a specific remaining gap worth targeting.

Artifacts are in `out/research/ai-paper-id-diagnostics-v1/blind-base-vs-sft-retrieved/`. The runs
used the same Qwen3-8B base revision, top-four BM25 candidates, greedy decoding, seed 42,
outcome-v1 reward, and one NVIDIA RTX 4090. Human judgments were completed before opening the
blind assignment key. The locked 40-question test set remained untouched.
