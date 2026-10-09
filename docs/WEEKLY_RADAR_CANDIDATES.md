# Weekly radar candidate collector

This CPU stage collects arXiv metadata for a pinned topic profile. It creates a
version-aware candidate store for later ranking and human review. It does not
judge relevance, download full papers, write a digest, or prove that Qwen has
improved.

Run the synthetic, offline demonstration:

```bash
python scripts/weekly_radar_candidates.py --demo
```

For a live, narrow topic and a recent window:

```bash
python scripts/weekly_radar_candidates.py \
  --topic tool-using-agents --since 2026-09-21 --until 2026-09-27 \
  --state-dir out/research/weekly-radar-candidates
```

The first successful run requires `--since`. Later runs omit it and overlap
the previous cursor by two days. A per-topic file lock covers fetch, merge, and
atomic commit. Retrying a window cannot duplicate a version or move the cursor
back. Each store records the topic-profile hash, query URLs, timestamps,
fetched counts, update scan bound, and every version observed. The store uses
`weekly-radar-candidates-v2`; it refuses to reuse a v1 cursor, which only
covered submissions. Use a new state directory or explicitly rebuild it.

The [arXiv API manual](https://info.arxiv.org/help/api/user-manual.html)
documents `submittedDate` as its sole date search filter. `lastUpdatedDate`
is a sort key, not a query filter. The collector therefore makes two passes:
one submission-date query and one descending last-update scan. The latter
filters the Atom `<updated>` date locally and scans until it sees an earlier
date or exhausts the result set. It checks result ordering and paging metadata.
If the result cap (`--limit`, default 100) or update scan cap
(`--update-scan-limit`, default 1000) is reached before coverage is established,
the run fails and leaves the cursor and store unchanged. Narrow the query or
date window before retrying. API calls are serial and at least 3.1 seconds
apart, following [arXiv's API rate limit](https://info.arxiv.org/help/api/tou.html).
Run one collector at a time across topics to respect the account-wide limit.

The update pass sees the latest version currently visible in search. A paper
revised twice between runs may have an intermediate version that the search
results no longer expose. Replaying an old week may also exceed the scan bound
because newer updates come first. Treat historical replay as incomplete unless
the scan reaches the requested window; for broad or long-range harvesting use
arXiv's [OAI-PMH metadata interface](https://info.arxiv.org/help/oa/index.html)
or an archived feed. Search matching and title-based deduplication are candidate
heuristics, not guarantees of literature completeness or paper identity.

Offline fixture mode accepts a JSON object with separate `submitted` and
`updated` response objects. Each needs a `papers` array of versioned arXiv
records with `arxiv_id`, `title`, `abstract`, `submitted`, `updated`, and
`source_url`. Mark incomplete responses with `"status": "partial"`; the cursor
will not advance.
