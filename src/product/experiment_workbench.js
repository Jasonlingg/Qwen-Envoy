const $ = (id) => document.getElementById(id);
const results = $("results"), detail = $("detail"), message = $("message");
const plain = (text) => text.replaceAll("**", "").replaceAll("`", "");

function element(tag, text, className) {
  const item = document.createElement(tag);
  if (text !== undefined && text !== null) item.textContent = String(text);
  if (className) item.className = className;
  return item;
}
function add(parent, tag, text, className) {
  const item = element(tag, text, className);
  parent.append(item);
  return item;
}
function clear(parent) { parent.replaceChildren(); }
function status(text, error=false) { message.textContent = text; message.className = error ? "error" : ""; }
async function api(path, body) {
  const options = body === undefined ? {} : {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)};
  const response = await fetch(path, options);
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail || `HTTP ${response.status}`);
  return value;
}
async function guarded(action) {
  try { await action(); } catch (error) { status(error.message || String(error), true); }
}
function card(parent, title, description, kind, click) {
  const button = add(parent, "button", null, "card");
  add(button, "span", kind, "tag");
  add(button, "strong", title);
  add(button, "small", description);
  button.addEventListener("click", () => guarded(click));
}
function section(title) {
  const box = add(detail, "section", null, "section");
  add(box, "h3", title);
  return box;
}
function paragraph(parent, title, text) {
  if (!text) return;
  add(parent, "strong", title);
  add(parent, "p", text, "compact");
}
function input(parent, title, type, value="") {
  const label = add(parent, "label", title);
  const field = add(label, type === "textarea" ? "textarea" : "input");
  if (type !== "textarea") field.type = type;
  field.value = value;
  return field;
}
function select(parent, title, options) {
  const label = add(parent, "label", title), field = add(label, "select");
  options.forEach(([value, text]) => {
    const option = add(field, "option", text); option.value = value;
  });
  return field;
}
function button(parent, text, click) {
  const item = add(parent, "button", text, "secondary");
  item.type = "button";
  item.addEventListener("click", () => guarded(click));
  return item;
}
async function search() {
  const query = $("query").value.trim();
  if (query && query.length < 3) { status("Enter at least three characters to search.", true); return; }
  status(query ? `Searching “${query}”…` : "Showing the curated atlas and local records.");
  const encoded = encodeURIComponent(query);
  const [atlas, experiments, decisions] = await Promise.all([
    api(`/api/atlas${query ? `?q=${encoded}` : ""}`),
    api(`/api/experiments${query ? `?q=${encoded}` : ""}`),
    query ? api(`/api/decisions?q=${encoded}`) : Promise.resolve({matches:[]})
  ]);
  clear(results);
  add(results, "h2", "Past attempts");
  for (const row of atlas.matches) card(results, plain(row.attempt), plain(row.observation), `Atlas · ${row.atlas_id}`, () => openAtlas(row.atlas_id));
  if (!atlas.matches.length) add(results, "p", "No matching curated attempts.", "muted");
  add(results, "h2", "Saved experiments");
  for (const row of (experiments.matches || experiments.experiments)) {
    card(results, row.question, `${row.latest_assessment || row.status} · ${row.result_count} result(s)`, "Record", () => openExperiment(row.experiment_id));
  }
  if (!(experiments.matches || experiments.experiments).length) add(results, "p", "No saved experiment found in this store.", "muted");
  if (query) {
    add(results, "h2", "Decisions");
    for (const row of decisions.matches) card(results, row.question, row.latest_decision || "No decision yet", "Thread", () => openThread(row.thread_id));
  }
  status(`${atlas.matches.length} curated attempt(s), ${(experiments.matches || experiments.experiments).length} saved experiment(s), ${decisions.matches.length} decision thread(s).`);
}
async function openAtlas(id) {
  const item = await api(`/api/atlas/${encodeURIComponent(id)}`);
  clear(detail);
  add(detail, "span", `Historical attempt · ${id}`, "tag");
  add(detail, "h2", plain(item.attempt));
  paragraph(detail, "Saved observation", plain(item.observation));
  paragraph(detail, "Recorded decision / next question", plain(item.decision));
  const box = section("Primary artifacts");
  add(box, "p", "Open the artifact excerpt before relying on this summary. Its hash verifies the file, not the claim.", "muted compact");
  item.sources.forEach((source, index) => {
    button(box, `${source.label} · ${source.path}`, () => openSource(id, index, 0));
  });
}
async function openSource(id, index, start) {
  const item = await api(`/api/atlas/${encodeURIComponent(id)}/source/${index}?start=${start}`);
  const existing = $("source-view"); if (existing) existing.remove();
  const box = section(`${item.label} · saved source excerpt`); box.id = "source-view";
  add(box, "p", `${item.path} · SHA-256 ${item.sha256} · byte offset ${item.start}`, "muted compact");
  add(box, "pre", item.text);
  if (item.truncated) button(box, "Read next excerpt", () => openSource(id, index, start + 4000));
}
async function openExperiment(id) {
  const item = await api(`/api/experiments/${encodeURIComponent(id)}`);
  clear(detail);
  add(detail, "span", `Saved ${item.status} experiment`, "tag");
  add(detail, "h2", item.brief.question);
  add(detail, "p", `ID ${id} · Revision ${item.revision} · ${item.results.length} result(s) · ${item.research_notes.length} research note(s)`, "muted compact");
  paragraph(detail, "Hypothesis", item.brief.hypothesis);
  paragraph(detail, "Expected signal", item.brief.expected_signal);
  if (item.results.length) {
    const box = section("Measured runs");
    for (const run of item.results) button(box, `${run.run_id} · ${run.overall} · ${run.data_kind}`, async () => {
      const full = await api(`/api/experiments/${encodeURIComponent(id)}?run_id=${encodeURIComponent(run.run_id)}`);
      const old = $("run-view"); if (old) old.remove();
      const view = section("Selected result"); view.id = "run-view";
      paragraph(view, "Interpretation", full.selected_result.result.interpretation);
      paragraph(view, "Assessment", `${full.selected_result.assessment.overall} · ${full.selected_result.assessment.effective_timing}`);
      for (const arm of ["baseline", "candidate"]) {
        const values = full.selected_result.result[arm].measurements;
        add(view, "strong", `${arm[0].toUpperCase()}${arm.slice(1)} measurements`);
        add(view, "pre", values.map(metric => `${metric.metric_id}: ${metric.value ?? metric.unknown_reason} ${metric.unit} (${metric.coverage})`).join("\n"));
      }
      add(view, "strong", "Frozen-plan criteria");
      add(view, "pre", JSON.stringify(full.selected_result.assessment.criteria, null, 2));
      add(view, "p", `${full.selected_result.evidence_receipts.length} source receipt(s) · ${full.selected_result.assessment.warnings.length} assessment warning(s)`, "muted compact");
      add(view, "p", `Result SHA-256 ${full.selected_result.sha256}`, "muted compact");
    });
  }
  if (item.research_notes.length) {
    const box = section("Research notes");
    for (const note of item.research_notes) button(box, note.question, async () => {
      const full = await api(`/api/experiments/${encodeURIComponent(id)}?research_note_id=${encodeURIComponent(note.note_id)}`);
      const old = $("research-view"); if (old) old.remove();
      const view = section("Selected research note"); view.id = "research-view";
      for (const finding of full.selected_research_note.note.findings) paragraph(view, `Claim · ${finding.reference_ids.join(", ")}`, finding.claim);
      paragraph(view, "Uncertainties", full.selected_research_note.note.uncertainties.join("; "));
      add(view, "p", `Note SHA-256 ${full.selected_research_note.sha256} · Semantic support ${full.selected_research_note.note.semantic_support}`, "muted compact");
    });
  }
  const box = section("Make a decision from this record");
  add(box, "p", "Your decision is appended with hashes of the selected experiment and run. It will still be here after you restart the server.", "muted compact");
  const form = add(box, "div", null, "stack");
  const question = input(form, "Decision question", "text", item.brief.question);
  button(form, "Create decision thread", async () => {
    const thread = await api("/api/decisions", {experiment_id:id, question:question.value});
    await openThread(thread.thread_id);
    status("Decision thread created. Record your choice below.");
  });
}
async function openThread(id) {
  const thread = await api(`/api/decisions/${encodeURIComponent(id)}`);
  clear(detail);
  add(detail, "span", `Decision thread · revision ${thread.revision}`, "tag");
  add(detail, "h2", thread.question);
  add(detail, "p", `ID ${thread.thread_id} · linked experiment(s): ${thread.experiment_ids.join(", ")}`, "muted compact");
  for (const event of thread.decisions) {
    const box = section(`${event.choice.outcome.toUpperCase()} · ${event.recorded_at}`);
    paragraph(box, "Rationale", event.choice.rationale);
    paragraph(box, "Revisit when", event.choice.revisit_when);
    add(box, "p", `${event.choice.author_kind} · ${event.basis_receipts.length} source-bound basis receipt(s)`, "muted compact");
  }
  const experimentId = thread.experiment_ids[0];
  const experiment = await api(`/api/experiments/${encodeURIComponent(experimentId)}`);
  const box = section("Record your next decision");
  const form = add(box, "div", null, "stack");
  const outcome = select(form, "Outcome", [["defer","Defer"],["adopt","Adopt"],["reject","Reject"]]);
  const run = select(form, "Measured run used as basis", [["","No run — experiment brief only"], ...experiment.results.map(row => [row.run_id, `${row.run_id} · ${row.overall}`])]);
  const note = select(form, "Research note used as basis", [["","No research note"], ...experiment.research_notes.map(row => [row.note_id, row.note_id])]);
  const rationale = input(form, "Why?", "textarea");
  const revisit = input(form, "Revisit when (required for defer)", "textarea");
  const authorLabel = add(form, "label");
  const userAuthored = add(authorLabel, "input"); userAuthored.type = "checkbox";
  add(authorLabel, "span", "I personally wrote and stand behind this decision");
  button(form, "Save decision", async () => {
    const saved = await api("/api/decisions/record", {thread_id:id, expected_revision:thread.revision,
      experiment_id:experimentId, run_id:run.value || null, research_note_id:note.value || null,
      outcome:outcome.value, rationale:rationale.value, revisit_when:revisit.value || null,
      author_kind:userAuthored.checked ? "user_reported" : "agent_proposed"});
    await openThread(saved.thread_id);
    status("Decision saved. Search for this question after restarting to retrieve it.");
  });
  add(box, "p", thread.warning, "muted compact");
}
$("search-form").addEventListener("submit", event => {event.preventDefault(); guarded(search);});
guarded(search);
