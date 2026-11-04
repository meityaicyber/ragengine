# Changes in 0.3.4 (one command)

- `python run.py go <folder>` now prepares everything itself: on a machine with only Python, it creates `.venv`, installs packages, writes `.env`, downloads the configured model weights, starts Ollama if installed and pulls the chat model, then runs the pipeline and the console. Re-running continues where it stopped. Only installing Python and the Ollama app cannot be automated.
- Download and setup failures print a one-line reason and the fix instead of a traceback.

# Changes in 0.3.3 (Standard Template v2 schema)

- Full support for the team's Standard Template v2 JSON schema (`examples/standard_template_report.json`): sign-off objects (`prepared_by: {name, auditing_team_sr_no}`), `executive_summary.narrative`, `document_control`, `auditing_team` and `report_meta` are indexed as citable sections; finding fields `impact_level`, `likelihood_level`, `reported_epss_score`, `observation_date`, `severity_rationale`, `regulatory_references` are searchable.
- `Partially Remediated` is a first-class status (it was folded into "In Progress").
- New template-compliance warnings: unverified closures (Closed/Partially Remediated with no retest entry), reviewer missing or same as preparer. Shown in "Which reports failed validation and why?".
- Regression tests on the template schema (66 tests).

# Changes in 0.3.2 (agentic run with real prompts; one-file model switching)

Found by running the agent on 2-report.json and answering the engine's actual prompts by hand, one request at a time:

- **Tool results were cut at 600 characters.** During the tool loop the model never saw any recommendation or root cause, even through `get_finding_text`, whose description promises the full text. `get_finding_text` now returns the whole finding; search results show 1,400 characters per passage (6 passages by default).
- **The web console never remembered a conversation.** `SessionStore` defines `__len__`, so an empty store is falsy, and `sessions or SessionStore()` gave every request a new private store. Follow-ups ("what was recommended for it?") had no history and the query-rewrite step never ran. Fixed (`is not None`); a follow-up now reaches the agent with the previous turn.
- **Abstentions showed unrelated figure placeholders.** Figures are now attached only for findings the answer actually cites.
- **The final-answer prompt is fitted to the context window** (Ollama silently truncates over-long prompts from the front).

Model switching from one file (`.env`):

- MODELS section at the top of `.env.example` with presets (chat: qwen3 sizes, llama3.1, mistral-nemo; embedding: Qwen3, bge-large-en-v1.5, bge-m3, multilingual-e5-large; reranker: Qwen3, bge-reranker-v2-m3).
- `run.py models` downloads whatever repo `.env` names (previously it always downloaded Qwen); `doctor` shows the active models.
- The embedder picks pooling and query/passage format by model family; the dimension is read from the model.
- The vector index records which embedding setup built it, and refuses to load under a different one with the exact fix ("reset, then ingest").
- `think` is sent only to models that support it, so non-Qwen chat models work.
- CLI configuration errors print a one-line message with the fix instead of a traceback.
- 64 tests.

# Changes in 0.3.1 (tested on the team's real 2-report.json / 4-report.json)

- **Quality gate policy:** a report is blocked only when its *detailed findings* are corrupt. Executive-summary problems (misaligned summary table, totals or ids disagreeing with the details) are now warnings, because answers never come from the summary. `2-report.json` was previously quarantined for its summary table even though all 7 detailed findings are clean; it is now indexed. `4-report.json` (methodology phases parsed as OBS-01 to OBS-04, rows shifted) is still quarantined, correctly.
- **CWE/CVE ids are stored canonical**: "CWE-284: Improper Access Control" and "CWE-639 / CWE-862 Branch Dispatcher" become `CWE-284`, `CWE-639`, `CWE-862`, so top-CWE counts and repeat detection are right.
- **Format drift tolerance:** keys are normalised ("Finding ID", "Risk-Rating", "Detailed Observations" all work); severities like "4 - High", "High Risk", "CRITICAL (9.8)"; statuses like "Not Fixed" (Open), "Partially Remediated" (In Progress), "Closed - verified"; CVSS like "8.1 (High)".
- **`likely_root_cause` and `reference`** are now indexed, searchable and citable.
- Regression tests on the two real reports plus a "drifted extractor" copy (53 tests).
- Wording: "CWE values", "matching CWE-284".
- Server warm-up uses FastAPI's lifespan API (the deprecated startup hook is gone).

# Changes in 0.3.0 (demo hardening)

Found while testing with real reports and qwen3:8b: questions about the corpus itself ("How many audit reports are indexed?", "What types of information are available for each finding?") always abstained with "The report does not say".

## 0.3.5

- `show_chart` now renders two-level aggregates (severity by status) as a stacked bar and can chart `top_values` results; previously it returned an error when the model asked to re-chart them.

## Fixed

1. **Corpus-level questions abstained.** Only eight query tools' results reached the final answer, and only five could use the deterministic template. The model called the right tool (`list_reports`, `describe_schema`), and its result was then discarded. Now every non-search tool's result reaches the final answer, and corpus-level tools (`list_reports`, `corpus_stats`, `describe_schema`, `get_report_outline`, `find_report`, `report_quality`, `find_data_gaps`, `compare_reports`, `find_repeat_findings`, `describe_capabilities`, `query_findings`) answer through exact templates with no extra model call.
2. **"How many reports" style questions** are answered straight from the catalogue (flag `catalog_lookup`), including how many were quarantined.
3. **The final-answer prompt forbade stating counts.** It reused the single-pass prompt ("never state a total"), contradicting the exact SQL results it was given. The agent now has its own final-answer prompt that treats tool results as exact facts and passages as citable evidence.
4. **A model that answers without calling tools** (or whose calls all fail) now falls back to the deterministic single-pass path instead of abstaining (flag `agent_fallback:single_pass`).
5. **The agent prompt now says which tool fits which question**, and that it must always call one.
6. **Findings under nested or unknown keys, lists of reports, and bare finding lists** are now ingested instead of quarantined or failed. This is shared by survey, validate, ingest and crosscheck.
7. **A report re-ingested under a new id left its old copy behind**, double-counting findings. It is now replaced.
8. **Reports quarantined by older logic stayed quarantined forever** (the file was unchanged). Each report records the ingest-logic version, and older ones are re-processed.
9. `list_reports` ignored ingest status; quarantined reports are now marked, in answers and in the console's scope menu (where they are disabled).
10. Repeat findings that matched on both title and CWE were reported twice; they are now merged.
11. A quarantine reason no longer prints twice.
12. Reports with no title fall back to their file name.

## Added

- Ollama `keep_alive` (60 minutes) so the model is not unloaded between demo questions, and background warm-up of all models when the server starts.
- `retrieval.rerank_k` (24): the reranker scores the top 24 fused candidates instead of all 60, a large speed-up on CPU.
- `ingest --no-quarantine`: index flagged reports anyway, re-processing only those.
- Tables in the console for report lists and finding lists.
- Sidebar examples that work on any corpus (the old ones referenced the sample data).
- Tool results sent to Ollama carry `tool_name` (newer Ollama) as well as `name`.
- 50 tests (previously 34).

# Changes in 0.2.0

Every fix below is covered by a test in `tests/` (`python run.py selftest`).

## Bugs fixed

1. **Ingest was not actually resumable.** The vector index was written to disk only once, at the end of `ingest`. An interrupted run left reports in SQLite whose vectors were never saved, and the next run skipped them (unchanged hash), so semantic search could never find them. Now:
   - the index is checkpointed every `ingest.checkpoint_every` (10) reports, and on exit including Ctrl-C;
   - a report counts as up to date only if its hash is unchanged **and** every chunk has a vector, so lost vectors are re-embedded automatically;
   - embedding runs before any write, so an embedder failure leaves the previous version intact;
   - index files are saved atomically, and a mismatched pair self-heals on the next ingest.
2. **Embedding pooling was wrong for Qwen3-Embedding.** Its tokenizer pads on the left; the code assumed right padding, so in any batch with uneven lengths the shorter texts were pooled from the wrong token, silently degrading retrieval. It now uses the model card's `last_token_pool`.
3. **Validator and normaliser disagreed on field spellings** (for example `observations`). The validator now imports the normaliser's alias tables, so one edit covers both. More common spellings were also added.
4. **Duplicate finding ids crashed the whole report** (primary-key violation). Repeats are now stored as `ID#2`, `ID#3` with a validation warning.
5. **Report id collisions overwrote reports.** Two files deriving the same id (same file name in different folders, or a reused `report_id`) now get distinct ids (`ID__FOLDER`). Mapping is stable across runs via the new `reports.ingest_path` column, which is added automatically to existing databases.
6. **CWE/CVE matching was a substring match**: `CWE-28` also matched `CWE-284`, inflating exact counts. It now matches whole identifiers. Asking for a CWE and a CVE together previously dropped the CVE filter.
7. **`GET /findings` returned the page length as `count`.** It now returns the true total plus `returned`.
8. **Report discovery silently skipped real reports** whose name merely contained `results`, `metrics`, and so on. It now skips exact names only and prints what it skipped (`--include-all` overrides).
9. **UTF-8 BOM files** (common from Windows tools) failed with a JSON error. They now load.
10. **A top-level JSON array** crashed with an `AttributeError`. It now fails with a clear message.
11. **Reranker/embedder used different dtype keywords** (`dtype` vs `torch_dtype`), so one was wrong on any given transformers version. Both now use the right one for the installed version.
12. **Counts loaded every row to `len()` them.** They now use `SELECT COUNT(DISTINCT ...)`.
13. **Concurrent API requests shared one SQLite connection across threads.** `/ask` and `/findings` are now serialised.
14. Agent pre-resolution only recognised `OBS-nn` ids and ran one query per report per question. It now matches any id-shaped token (`V-4`, `VAPT-03`) against stored ids, in one query.
15. Grammar in templated count answers ("1 finding match").

## Removed

- `ingest.workers` and `ingest.contextualize` settings: they were never wired to anything.

## Added

- `run.py` (plus `run.sh` / `run.bat`): setup, doctor, go, survey, validate, ingest, crosscheck, smoke, serve, eval, reset, selftest.
- `ragaudit survey`, `crosscheck`, `reset` CLI commands. `ingest` prints progress with an ETA. `ask --trace` prints the agent's tool calls.
- `requirements.txt`, `pyproject.toml`, `.env.example` (settings are now read from `.env`).
- Optional API key (`RAGAUDIT_API__API_KEY`); the web console asks for it.
- `sample_data/` (three fictional reports, one deliberately broken) and `eval/sample_questions.json`.
- `tests/`: 34 tests running the real pipeline with a fake embedder and a scripted fake LLM.
- `serve` defaults to port 8077 (matching the docs).
