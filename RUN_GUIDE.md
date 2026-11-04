# ragengine: run guide

## The one command

```bash
python run.py go <folder-with-json-reports>        # Windows: py run.py go <folder>
```

That is all. On the first run it creates `.venv`, installs every package, writes `.env`, downloads the embedding and reranker weights, starts Ollama and pulls the chat model. Then it surveys, validates, ingests and crosschecks the folder, runs the smoke questions and opens the web console at http://localhost:8077. On later runs it skips whatever is already done (unchanged reports are not re-embedded). If anything fails, it prints the reason and the fix; run the same command again after fixing it.

**What it cannot do for you** (install these once, beforehand):
1. **Python 3.10 to 3.12** (3.13 works where PyTorch has wheels).
2. **The Ollama application**, from https://ollama.com/download. `go` starts it if it is installed but not running.
3. **Internet on the first run**: about 8 to 10 GB (PyTorch, ~2.4 GB of embedding/reranker weights, ~5 GB for qwen3:8b). After that, everything runs offline.

`python run.py go` with no folder prepares everything and starts the console on whatever is already indexed.


This guide takes you from a fresh unzip to asking questions about a folder of CERT-In audit JSON in the web console. Every step uses `run.py`, a launcher that only needs plain Python. On Windows, use `py run.py ...` or `run.bat ...`. On macOS/Linux, use `python3 run.py ...` or `./run.sh ...`.

> **Do Part A tonight, on a good connection.** It downloads about 8 to 10 GB (PyTorch, the two Qwen3 weight folders, the Ollama chat model). Part B, tomorrow, needs no internet at all.

---

## Already set up with the previous version? Upgrade in 2 minutes

You do not need to redo setup. Keep `.venv/`, `models/`, `.env` and `var/`, and replace only the code:

1. Stop the server (Ctrl-C).
2. Unzip the new version somewhere, then copy everything **except** `.venv/`, `models/`, `.env` and `var/` over your existing folder (overwrite when asked).
3. Run:
   ```bash
   python run.py selftest          # should end with "passed"
   python run.py ingest <your-json-folder>   # reprocesses every report once (the ingest logic changed)
   python run.py doctor
   ```

Nothing in `.env` needs to change. One optional new setting, `RAGAUDIT_RETRIEVAL__RERANK_K` (default 24), controls how many passages the reranker scores per search. Lower it to 12 if searches feel slow on CPU.

---

## Demo-day checklist (morning of the demo)

1. **Fresh index for their reports:** `python run.py reset --yes`, so leftovers from test data don't appear in counts.
2. **Readiness:** `python run.py doctor`. Every line should be `[ OK ]`.
3. **Their folder:** `python run.py survey <folder>`. If `advice` asks for an alias, add it (Part B1, 2 minutes).
4. **Build:** `python run.py go <folder>`. This ingests, crosschecks, runs the smoke questions and starts the console.
   - Many reports quarantined and you need them in the demo? `python run.py ingest <folder> --no-quarantine`. This re-processes only the quarantined ones. Say in the demo that the gate flagged them, and why (`Which reports failed validation and why?`).
5. **Open http://localhost:8077** a minute before you present. The server loads all models in the background at startup, and the top-right pills turn green when ready. Ollama keeps the chat model loaded for 60 minutes after each question, so pauses won't cause a cold start.
6. **Demo order that shows the strengths:** the sidebar questions top to bottom. Start with corpus and count questions (instant, exact), then the narrative ones (cited quotes; click a citation), then "Was the HR database encrypted at rest?" (it should decline rather than invent), then compare, repeats and validation.
7. **If something is slow or odd mid-demo:** counts and lists never depend on the chat model. If the model misbehaves, the system falls back to the deterministic path automatically (flag `agent_fallback:single_pass`).

---

## Switching models (one file: `.env`)

Everything model-related lives in the **MODELS** section at the top of `.env`, with ready-made presets you uncomment. Nothing else needs editing.

| To change | Edit in `.env` | Then run |
|---|---|---|
| Chat model (Ollama) | `RAGAUDIT_GENERATION__MODEL_NAME=qwen3:4b` | `python run.py ollama` |
| Embedding model | the `RAGAUDIT_EMBEDDING__REPO_ID` + `__MODEL_PATH` pair (presets: Qwen3-Embedding, bge-large-en-v1.5, bge-m3, multilingual-e5-large) | `python run.py models`, then `python run.py reset --yes` and `python run.py ingest <folder>` |
| Reranker | the `RAGAUDIT_RERANKER__REPO_ID` + `__MODEL_PATH` pair, or `RAGAUDIT_RERANKER__ENABLED=false` | `python run.py models` (no re-ingest) |

`python run.py doctor` prints the active models at the top.

What the engine adapts to automatically:
- **Embedding pooling and prompt format** by family: Qwen3 (last token, "Instruct/Query" prompt), BGE v1.5 (CLS, query prefix), BGE-M3 (CLS, no prefix), E5 ("query:"/"passage:"), others (mean pooling). Override with `RAGAUDIT_EMBEDDING__POOLING` / `__QUERY_TEMPLATE` if needed.
- **Embedding dimension** is read from the model; there is no dimension to set.
- **The `think` option** is sent only to models that support it (qwen3, deepseek-r1, gpt-oss and similar), so llama/mistral-family chat models work too. The chat model must support tool calling.
- **Safety:** if you change the embedding model but forget to re-ingest, every command stops with "the vector index was built with embedding setup X... run python run.py reset --yes" instead of silently mixing vectors from two models.

---

## Part A: one-time setup (tonight)

### A1. Install the two prerequisites

1. **Python 3.10, 3.11 or 3.12.** Check with `python --version`. On Windows, tick "Add python.exe to PATH" in the installer. 3.13 works too if PyTorch has a wheel for your platform.
2. **Ollama** (runs the chat model locally). Download it from https://ollama.com/download. On Linux: `curl -fsSL https://ollama.com/install.sh | sh`. Use a recent version (the code uses tool calling and the `think` switch). On Windows/macOS, start the Ollama app once so it runs in the background.

### A2. Unzip and run it once on the example report

```bash
cd ragengine                     # the folder containing run.py
python run.py go examples        # Windows: py run.py go examples
```

The first run installs and downloads everything (below), then ingests `examples/standard_template_report.json` and opens the console. Leave it running overnight if your connection is slow; if it stops, run the same command again and it continues. (`python run.py setup` does only the install/download part, if you prefer to do that separately.)

What the first run installs, in order:

| Step | What happens | Size / time |
|---|---|---|
| 1 | Checks your Python version | instant |
| 2 | Creates `.venv/` and installs `requirements.txt` (PyTorch, transformers, FastAPI, ...) | 1 to 3 GB, 5 to 15 min |
| 3 | Copies `.env.example` to `.env`, and switches the device to `cuda`/`mps` if it finds a GPU | instant |
| 4 | Downloads `Qwen/Qwen3-Embedding-0.6B` and `Qwen/Qwen3-Reranker-0.6B` into `models/` | about 2.4 GB |
| 5 | Starts Ollama if needed and runs `ollama pull qwen3:8b` | about 5 GB |

Options:
- `python run.py setup --cpu`: installs the CPU-only PyTorch build. On Linux without an NVIDIA GPU this is a much smaller download.
- `python run.py setup --no-models`: skips step 4. Run `python run.py models` later.

Each step can be re-run on its own: `python run.py models` (weights only) and `python run.py ollama` (chat model only). Setup can also be re-run safely; it skips whatever is already done.

### A3. Pick the chat model for your machine

Edit the MODELS section at the top of `.env` (created in A2; see "Switching models" above):

| Your machine | `RAGAUDIT_GENERATION__MODEL_NAME` | Notes |
|---|---|---|
| NVIDIA GPU with 8 GB+ VRAM, or Apple Silicon with 16 GB+ | `qwen3:8b` (default) | best answers |
| Laptop, CPU only, 16 GB RAM | `qwen3:4b` | good balance; also set `RAGAUDIT_AGENT__DEADLINE_S=180` |
| Weak machine | `qwen3:1.7b` | counts and lookups are fine (no model maths involved); open questions are weaker |

If you change the model, run `python run.py ollama` to pull it.

### A4. Check everything, then rehearse on the bundled sample

```bash
python run.py doctor        # every line should be [ OK ]; each [FAIL] prints its fix
python run.py selftest      # test suite with fake models; proves the code works on your machine
python run.py go sample_data
```

`go sample_data` runs the whole pipeline on three fictional reports in `sample_data/`, then opens the web console at **http://localhost:8077**. One sample report is deliberately broken and should show as **quarantined**; that is expected. If this works, tomorrow is the same command pointed at your folder. Stop the server with Ctrl-C, then clear the sample:

```bash
python run.py reset --yes
```

---

## Part B: tomorrow, with the CERT-In JSON

Put the JSON files in one folder. Sub-folders are fine, because the search is recursive. The examples below use `D:\cert_in_json` (Windows) or `~/cert_in_json` (macOS/Linux). Paths may be relative to where you run the command.

### The one-command version

```bash
python run.py go ~/cert_in_json
```

This installs or downloads anything still missing, then runs B1 to B6 below in order and finishes by starting the web console. If a step cannot be fixed automatically (no internet, Ollama not installed), it stops with the reason and the fix. To run the pipeline without starting the server, add `--no-serve`.

**For a large or unfamiliar corpus, run the steps one at a time the first time**, because B1 may show that you need a one-line alias change before ingesting.

### B1. Survey: does the engine understand your JSON? (writes nothing)

```bash
python run.py survey ~/cert_in_json
python run.py survey ~/cert_in_json --sample 50     # faster first look
```

What to read in the output:

- **`findings_array_key_used`**: which key held the list of findings in each file. Files listed under `files_without_findings_array` will be quarantined.
- **`canonical_field_coverage`**: for each field the engine understands (`title`, `severity`, `status`, `observation`, `recommendation`, ...), how many findings it was found in. Low coverage on `title`, `severity`, `observation` or `recommendation` means your files spell it differently.
- **`unmapped_finding_keys`**: per-finding keys the engine is ignoring. The missing spelling is usually here (for example `vuln_name`, `risk_level_rating`, `remediation_steps`).
- **`severity_values` / `status_values`**: each raw value and what it maps to, such as `Moderate -> Medium`. Anything mapping to `Unknown` drops out of severity/status filters.
- **`advice`**: plain-language next steps.

**If you need to add a spelling** (about 2 minutes, then survey again):

| Problem | File and place to edit |
|---|---|
| Findings array under an unknown key | `ragaudit/ingest/normalize.py`, `FINDINGS_ARRAY_ALIASES` |
| A field such as title or recommendation under an unknown key | `ragaudit/ingest/normalize.py`, `FIELD_ALIASES["title"]` (and so on); add the key to the tuple |
| A severity word maps to Unknown (for example `"P1"`, `"Severe-High"`) | `ragaudit/domain/models.py`, `Severity.parse` `aliases` dict, lowercase keys |
| A status word maps to Unknown (for example `"Not Fixed"`, `"Patched"`) | `ragaudit/domain/models.py`, `FindingStatus.parse` `aliases` dict, lowercase keys |

The validator reads the same alias tables, so one edit covers validation and ingest. Example: to treat `"not fixed"` as Open, add `"not fixed": cls.OPEN,` to the `FindingStatus.parse` aliases.

### B2. Validate: what will be quarantined? (writes nothing)

```bash
python run.py validate ~/cert_in_json --quiet --only-failing
python run.py validate ~/cert_in_json --json > validate_report.json   # full detail
```

The end of the output lists the **most frequent checks** and the **quarantine percentage**. How to read the checks:

| Check | Meaning | What to do |
|---|---|---|
| `findings_array_present` | no findings list under any known key | add the key (B1) |
| `title_not_a_url` (findings) | URLs in the detailed findings' title fields (columns shifted during PDF extraction) | extraction defect; blocking |
| `title_not_a_url` (summary table), `total_matches_array`, `severity_counts_sum`, `summary_detail_ids_match` | the executive summary is misaligned or disagrees with the detailed findings | **warning only**: answers come from the detailed findings, so the report is still indexed (your `2-report.json` is this case) |
| `title_not_a_methodology_phase` | methodology table parsed as findings, so every later row is shifted (your `4-report.json` is this case) | extraction defect; blocking |
| `narrative_fields_present` | every finding has empty text | check the observation alias (B1) |
| warnings (`severity_variation`, `finding_ids_unique`, ...) | indexed anyway, flagged | fine for testing |

**Quarantine is the safety feature, not a failure.** It stops the assistant from answering from corrupted data. A first-batch quarantine rate of 20 to 50 percent on real extracted PDFs is normal. Only if you deliberately want to index everything for a test, set `RAGAUDIT_INGEST__FAIL_ON_VALIDATION_ERROR=false` in `.env`, and remember that counts from those reports can be wrong.

### B3. Ingest: build the index

```bash
python run.py ingest ~/cert_in_json
```

You get one progress line per file:

```
[37/412] indexed     CERTIN-2024-0137            14 findings, 16 chunks   (elapsed 95s, eta 961s)
[38/412] quarantined CERTIN-2024-0138            [blocking] total_matches_array at ...
```

- **Ctrl-C is safe.** Progress is saved every 10 reports and on exit. Run the same command again and it continues where it stopped; unchanged files are skipped.
- **A crash or power cut is also safe now.** On the next run, any report whose vectors were lost is detected and re-embedded automatically.
- **Speed:** the ETA after the first 20 files is a good estimate. Embedding on CPU is the slow part. With a GPU (`RAGAUDIT_EMBEDDING__DEVICE=cuda` in `.env`), it is many times faster.
- **Same file name in two folders:** the second one is stored as `NAME__FOLDER` and a `note:` line says so. Nothing is overwritten.
- **Statuses:** `indexed` (searchable), `quarantined` (recorded with its reasons, never cited), `skipped` (unchanged since last run), `failed` (unreadable file; the error is printed in the final summary).

To start completely fresh: `python run.py reset --yes`, then ingest again.

### B4. Crosscheck: are the stored numbers right?

```bash
python run.py crosscheck ~/cert_in_json
```

This recounts the findings in every raw file, with no model involved, and compares them with what was stored, matched by file path. It also checks that every indexed report has all its vectors. The expected last line is `OK: stored counts match the raw files.` Problem lines:

- `COUNT_DIFF`: a normalisation bug; tell me which file.
- `VECTORS`: run ingest again to repair.
- `NOT_INGESTED`: the file was never ingested (or was skipped by name; see the note printed at the start of ingest).

### B5. Smoke test from the terminal

```bash
python run.py smoke
python run.py doctor
```

`smoke` asks a fixed set of questions and prints each answer, its sources and the agent's tool calls. Look for:
- counts answered instantly with the flag `agent_shortcut:count_findings` (exact SQL, no model maths);
- narrative answers with a `Sources:` list;
- "Was the database server patched?" answered with "not covered" rather than a guess.

### B6. Web console

```bash
python run.py serve           # then open http://localhost:8077
```

The top bar shows the index size and whether the LLM is reachable (these fill in a few seconds after the first page load, while the embedding model loads). The scope dropdown restricts answers to one report. Each answer shows citations, any charts or tables, and the tool trace. Stop it with Ctrl-C. The API docs are at http://localhost:8077/docs.

To let a teammate on the same network use it, set `RAGAUDIT_API__API_KEY=<long random string>` in `.env`, run `python run.py serve --host 0.0.0.0`, and share the key. The console asks for it once. Without a key, keep the default `127.0.0.1`.

---

## Part C: what to test, and how to judge it

Ask these in the console or with `python run.py ask "<question>" --trace`. Then check each answer against the raw JSON or directly in SQL (C2).

| # | Question | A good answer | Red flag |
|---|---|---|---|
| 1 | How many findings are there in total? | exact number, instant | any number different from crosscheck |
| 2 | How many high-severity findings are still open? | exact; matches the SQL in C2 | count differs from SQL |
| 3 | Are all critical findings closed? | Yes/No plus counterexamples | "Yes" while SQL shows open criticals |
| 4 | Break down the findings by severity and status | stacked bar chart | numbers that do not sum to the total |
| 5 | Which findings cite CWE-79? | list of the right findings | includes CWE-791 etc. (fixed, but check) |
| 6 | What remediation was recommended for <real finding title>? | quote plus citation | quote not found verbatim in the JSON |
| 7 | Summarise the <organisation> report | prose with several citations | content from another report |
| 8 | Compare the reports by number of open findings | chart plus comparison table | |
| 9 | Which findings repeat across reports? | grouped by CWE/CVE/title | |
| 10 | Which reports failed validation and why? | the quarantined list with reasons | |
| 11 | Was the HR database encrypted? (something not covered) | "the reports do not say" | a confident invented answer |
| 12 | Tell me more about the second one (after #5) | resolves "the second one" | answers about something else |

### C1. Read the flags under each answer

| Flag | Meaning |
|---|---|
| `agent_shortcut:<tool>` | answered from SQL with no generation step: exact |
| `catalog_lookup` | a question about the reports themselves (how many, which), answered from the catalogue, no model |
| `agent_fallback:single_pass` | the chat model answered without using any tool, so the system re-answered through the deterministic path |
| `citation_not_in_evidence:<id>` | the model cited a passage it never saw; the citation was **dropped** |
| `quote_not_in_chunk:<id>` | the quote was not in that passage; **dropped** |
| `agent_forced_abstention` | no citation survived verification, so the system refused to answer |
| `agent_deadline:partial`, `agent_max_steps:partial` | ran out of time or steps; answer built from partial evidence |
| `prompt_injection_like_text_in_evidence` | a report contains text resembling instructions to the model; treated as data |

Dropped citations prove verification is working. They are model mistakes that were caught, not engine bugs.

### C2. Verify counts by hand (no model involved)

```bash
python run.py findings --severity High --status Open
python run.py findings --cwe CWE-79
```

Or open the SQLite file directly (`var/catalog.sqlite3`; on Windows, DB Browser for SQLite works):

```sql
SELECT severity, status, COUNT(*) FROM findings GROUP BY 1, 2;
SELECT report_id, ingest_status, ingest_path FROM reports WHERE ingest_status = 'quarantined';
SELECT status, COUNT(*) FROM findings GROUP BY 1;   -- many 'Unknown'? add status aliases (B1)
```

### C3. Measure it properly (optional, for a write-up)

Write 50 to 100 questions with known answers in the format of `eval/sample_questions.json`:

```json
[
  {"id": "Q1", "text": "How many high-severity findings are open?", "expected_route": "structured"},
  {"id": "Q2", "text": "Are all findings closed?", "gold_answer": "No"},
  {"id": "Q3", "text": "What was recommended for OBS-03?", "report_id": "CERTIN-2024-0137",
   "gold_chunk_ids": ["<chunk_id from: SELECT chunk_id, text FROM chunks WHERE finding_id='OBS-03'>"]}
]
```

```bash
python run.py eval my_questions.json
```

The results go to `var/eval/<run>/` (`metrics.json`, `questions.csv`). Report the **confidence intervals**, not just the point accuracy. The harness itself warns that below 100 questions every number is directional.

---

## Part D: troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `doctor`: Ollama not reachable | Ollama app/service not running | start the Ollama app, or run `ollama serve` in another terminal |
| `doctor`: chat model not pulled | | `python run.py ollama` |
| `LLM rejected the request (400) ...` | model name wrong, or an old Ollama | check `RAGAUDIT_GENERATION__MODEL_NAME` against `ollama list`; update Ollama |
| `model weights not found at models/...` | weights not downloaded | `python run.py models`; offline: copy the two folders into `models/` |
| `could not download Qwen/...` | Hugging Face blocked on this network | download on another network with `huggingface-cli download Qwen/Qwen3-Embedding-0.6B --local-dir Qwen3-Embedding-0.6B` (same for the reranker) and copy into `models/` |
| `failed to load reranker` | old transformers, or a corrupt download | `.venv` pip: `pip install -U transformers`; or set `RAGAUDIT_RERANKER__ENABLED=false` |
| Answers end with `agent_deadline:partial` | slow CPU | `RAGAUDIT_AGENT__DEADLINE_S=180` and `RAGAUDIT_GENERATION__TIMEOUT_S=240`, or use a smaller model |
| Every answer says the LLM backend is unavailable | Ollama crashed or ran out of memory | check Ollama; use `qwen3:4b` |
| `stored index has dimension ... but the configured embedding model produces ...` | embedding model changed after ingest | `python run.py reset --yes` and ingest again |
| Ingest very slow | CPU embedding | GPU device in `.env`; for testing, ingest one sub-folder first |
| Many reports quarantined with `findings_array_present` | unknown findings key | B1: add the alias |
| Counts look low | severity/status mapping to Unknown | B1: `severity_values` / `status_values`; add aliases |
| `Address already in use` on serve | port 8077 taken | `python run.py serve --port 8090` |
| Windows: `python` not found | | use `py run.py ...` or `run.bat ...` |

Logs: set `RAGAUDIT_LOG_LEVEL=INFO` in `.env` for more detail (or `DEBUG`).

---

## Part E: input format the engine expects

**Your Standard Template v2 schema is supported field for field.** `examples/standard_template_report.json` is a complete, fictional report in exactly that schema. `survey` shows every field mapped and none ignored. What the engine does with the template-specific parts:
- `document_control` (sign-off objects), `auditing_team`, `report_meta` and `executive_summary.narrative` are indexed, so "Who reviewed the report?", "What certifications does the team hold?" and "When was it released?" get cited answers.
- Per finding: `impact_level`, `likelihood_level`, `reported_epss_score`, `observation_date`, `severity_rationale`, `regulatory_references`, `likely_root_cause` and structured `proof_of_concept` entries are searchable. `cve_id`, `cwe` and `affected_asset` lists are queryable exactly.
- Status `Partially Remediated` is its own status (counts and filters use it).
- Template rules checked as **warnings** (never quarantine): a Closed/Partially Remediated finding with no Retest & Closure entry or retest PoC ("treated as unverified"); reviewer missing or the same as the preparer; summary totals or table disagreeing with the detailed findings. "Which reports failed validation and why?" lists these too.

The older layout below (your `2-report.json` style) also still works.

One report per `.json` file, as a JSON object, produced by your PDF-to-JSON extractor. Only the findings list is required; everything else improves answers. Accepted spellings are listed in `ragaudit/ingest/normalize.py`.

```json
{
  "report_meta": {"report_id": "CERTIN-2024-0137", "report_title": "...", "classification": "Confidential",
                  "report_release_date": "2024-06-01", "audit_period": {"from": "...", "to": "..."}},
  "executive_summary": {"total_findings": 2, "severity_count": {"High": 1, "Low": 1},
                        "summary_table": [{"finding_id": "OBS-01", "title": "...", "severity": "High"}]},
  "detailed_observations": [
    {"finding_id": "OBS-01", "title": "Broken access control", "severity": "High", "status": "Open",
     "cvss_score": "8.1", "cwe": "CWE-284", "affected_asset": "https://...",
     "observation_description": "...", "impact": "...", "recommendation": "...",
     "proof_of_concept_steps": ["step 1", "[IMAGE_PAGE_7_FIG_1]"]}
  ],
  "engagement_scope": "...", "methodology": {"standards_referred": ["OWASP"]}, "conclusion": "..."
}
```

**Layouts accepted** (all checked by `survey`, `validate`, `ingest` and `crosscheck`):

| File contains | Handled as |
|---|---|
| one report object (above) | one report |
| a report wrapped in an object, e.g. `{"report": {"report_meta": ..., "findings": [...]}}` | unwrapped, then one report |
| findings under an unknown key, e.g. `"issues_identified": [...]` | found by shape (rows with a title/severity field); the summary table is never mistaken for it |
| a list of report objects `[{...}, {...}]` | one report each (ids `FILE-1`, `FILE-2` if they have none) |
| a bare list of findings `[{"title": ..., "severity": ...}, ...]` | one report named after the file |

- `report_id` comes from `report_meta.report_id` or `document_control.document_id`, otherwise from the file name. If a file's id changes later (for example, you add a `report_id`), re-ingesting replaces the old copy instead of duplicating it.
- A report with no title anywhere is titled after its file name.
- `executive_summary` totals are optional, but when present they must match the list, or the report is quarantined.
- Figure images (optional): put `IMAGE_PAGE_7_FIG_1.png` in `var/figures/<report_id>/`, then re-ingest that report with `--force`.

See `sample_data/` for three complete examples, including one with vendor spellings (`vulnerabilities`, `risk_rating`, `remediation_status`) and one deliberately broken file.

---

## Command reference

| Command | Writes? | Purpose |
|---|---|---|
| `python run.py setup [--cpu] [--no-models]` | yes | install/download only (`go` does this automatically) |
| `python run.py models` / `ollama` | yes | download weights / pull the chat model |
| `python run.py doctor` | no | readiness, with a fix for each problem |
| `python run.py go [<folder>] [--no-serve] [--force]` | yes | **the one command**: installs/downloads anything missing, then the whole pipeline, then the server (no folder: just prepare and serve) |
| `python run.py survey <folder> [--sample N] [--json]` | no | JSON shape and field coverage |
| `python run.py validate <folder> [--quiet] [--only-failing] [--json]` | no | quality gate dry run |
| `python run.py ingest <folder> [--force] [--verbose] [--include-all]` | yes | build or update the index |
| `python run.py crosscheck <folder>` | no | stored counts vs raw files |
| `python run.py smoke` | no | fixed questions through the full stack |
| `python run.py ask "<q>" [--report ID] [--trace] [--json]` | no | one question |
| `python run.py chat [--report ID] [--trace]` | no | multi-turn chat in the terminal |
| `python run.py findings [--severity X] [--status Y] [--cwe Z] [--report ID]` | no | direct SQL query |
| `python run.py serve [--port 8077] [--host 127.0.0.1]` | no | web console and API |
| `python run.py status` | no | component readiness as JSON |
| `python run.py eval [questions.json]` | yes (var/eval) | evaluation with confidence intervals |
| `python run.py reset --yes` | deletes | wipe the catalogue and index |
| `python run.py selftest` | no | unit and integration tests with fake models |
| `python run.py demo` | yes | ingest `sample_data` and run the scripted walkthrough |

`--include-all` matters only if your report files are literally named `results.json`, `metrics.json`, `questions.json` and so on. Those exact names are skipped by default, and the skip is printed.
