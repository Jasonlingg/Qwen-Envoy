# Implementation contract for harness v2 and arXiv MCP

The user's October 16 Native workflow supersedes the draft plans' unavailable
superpowers execution subskills. Implement A, independently review and fix A,
then implement B, independently review and fix B. Preserve the original drafts.

## Corrections applied before implementation

- Recorded model Python is untrusted: replay uses Docker only, never host
  LocalREPL. Match recorded V1 observations/action outcomes before any V2
  comparison; publish per-question/action evidence. A mismatch blocks comparison.
  Fixed-action replay cannot measure Nemotron's response to new feedback.
- Preserve the V1 preamble bytes and pin SHA-256
  `233adfef3cd72c8b8221e45dcaa58c76809f36bd717d1d342514a7c332c3dd71`.
  V2 stays opt-in; ordinary discovery search defaults to five. Only paper-scoped
  search/search_within defaults to eight. Accept both positional paper/query
  orders, reject scoped search before discovery, count it as source inspection.
- Test investigator budgets must be at least two steps. Literal submit conversion
  is AST-only; free format retries are capped at two per action.
- Construct one effective V2 prompt for the model policy, investigator and packet
  hash; V2 Nemotron instructions and reminders must permit literal submit().
- Validate HTTPS host, credentials and port before every redirect request,
  including PDF; cap downloaded bytes. Share one process-wide arXiv throttle
  across discovery, metadata, HTML, PDF and redirects.
- Isolate each paper reader failure and reasoner failure. Preserve all other
  verified spans. Exact quotes prove provenance, not semantic support; any live
  evaluation includes explicit claim-level semantic review, labeled as agent
  review, never independent human review.
- No paid smoke unless a reliable <=$0.50 cap can be enforced with hard request,
  input-token and output-token bounds (including retries and concurrency).
  Otherwise skip paid calls and document the reason. Free source checks can run.

Hypothesis for the replay: V2 accepts more of the same recorded tool calls.
Expected signal: fewer signature/syntax failures, with V1 reproduction first.
Decision rule: report comparison only after V1 matches; infer no model quality
or new-feedback behavior from replay. No training or model-improvement claim.
