# ragengine

An agentic question-answering engine for cybersecurity audit reports. Ask a
question in plain language; the engine works out what it needs, queries the
reports, and answers with a citation behind every claim — as prose, tables,
or charts, whichever fits.

Runs fully air-gapped: local models through Ollama and locally staged
embedding/reranker weights. Nothing leaves the machine.

---

## How it answers

Every question except small talk goes through a tool-calling agent:

```
question -> pre-resolve entities -> agent loop (tools) -> answer
                                                        |
                         one structured tool answered?  |-- yes -> deterministic text, no extra model call
                                                        '-- no  -> one schema-constrained finalize call,
                                                                   citations verified against evidence
```

Three properties hold throughout:

- **Numbers are exact.** Counts, totals and breakdowns come from SQL over the
  extracted fields, never from a model reading a sample of passages. A count
  answered this way skips the finalize call entirely.
- **Citations are verified.** Every quote is checked against the text the
  model was actually shown. Citations to passages it never saw, or quotes
  lifted from one finding and attributed to another, are dropped and flagged.
  A narrative answer left with no verified citation abstains instead of
  guessing.
- **Scope is enforced.** A report named in the question ("in the Tatva
  report…") or chosen in the UI bounds every tool call, so evidence never
  leaks across reports.

## Tools

23 tools in seven groups. The model sees the core groups by default and
unlocks the rest on demand, so a small model is not asked to choose among all
of them at once.

| Group | Tools | Default |
|---|---|---|
| navigate | `list_reports`, `find_report`, `get_report_outline`, `corpus_stats`, `describe_schema` | on |
| query | `query_findings`, `count_findings`, `aggregate_findings`, `distinct_values`, `top_values`, `get_finding` | on |
| search | `search_reports`, `keyword_search`, `get_finding_text` | on |
| render | `show_chart`, `show_table`, `show_note` | on |
| analyse | `compare_reports`, `find_repeat_findings`, `compute` | unlock |
| quality | `report_quality`, `find_data_gaps` | unlock |
| meta | `describe_capabilities` | unlock |

`compute` does arithmetic deterministically over earlier tool results (no
`eval`), and every computed figure carries the inputs it came from. Render
tools reference earlier results by step rather than accepting typed data, so
a chart cannot carry a number the model invented.

The agent loop stops on a wall-clock deadline, a prompt-token ceiling,
repeated identical calls, steps that add no new evidence, a search cap, too
many tool errors, or a step cap — whichever comes first.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) with a chat model that supports tool calling
  (`qwen3:8b` recommended; `qwen3:1.7b` works for simple questions)
- Locally staged weights for the embedder and reranker
  (`Qwen3-Embedding-0.6B`, `Qwen3-Reranker-0.6B`)

```bash
pip install "pydantic>=2.6" "pydantic-settings>=2.2" numpy requests fastapi uvicorn \
            sentence-transformers transformers torch
pip install faiss-cpu      # optional: HNSW index for large corpora
```

## Running

```bash
ollama serve
ollama pull qwen3:8b

# Ingest extracted report JSON (validates, normalises, chunks, embeds, indexes)
python -m ragaudit.cli ingest path/to/reports

# Web console with charts, citations and a tool-call trace
python -m ragaudit.cli serve --port 8077      # then open http://localhost:8077
```

Other commands:

| Command | Purpose |
|---|---|
| `validate <path>` | Check extraction quality; writes nothing |
| `ask "<question>"` | One question from the terminal |
| `chat` | Interactive multi-turn session |
| `findings --severity High` | Query the structured store directly |
| `route "<question>"` | Show how a question would be handled |
| `status` | Readiness of every component |
| `evaluate --questions q.json` | Run an evaluation set |

## API

| Endpoint | |
|---|---|
| `POST /ask` | Answer with citations, blocks (charts/tables) and the tool-call trace |
| `POST /ask/stream` | Same, as Server-Sent Events: `status` → `step` → `answer` |
| `GET /findings` | Exact structured query, no model involved |
| `GET /reports`, `/stats` | Catalogue |
| `GET /figures/{report_id}/{token}` | A staged figure image |
| `GET /healthz`, `/readyz` | Liveness, and whether a question can actually be served |

## Configuration

Every setting is overridable by environment variable, prefix `RAGAUDIT_`,
nested with `__`:

```bash
RAGAUDIT_GENERATION__MODEL_NAME=qwen3:8b
RAGAUDIT_GENERATION__BASE_URL=http://localhost:11434
RAGAUDIT_EMBEDDING__MODEL_PATH=models/Qwen3-Embedding-0.6B
RAGAUDIT_EMBEDDING__DEVICE=cuda
RAGAUDIT_RERANKER__ENABLED=true
RAGAUDIT_RETRIEVAL__BACKEND=faiss
RAGAUDIT_AGENT__ENABLED=true        # false restores the single-pass router
```

## Ingest and validation

Reports arrive as JSON from an upstream extractor. Before anything is indexed,
a validation gate checks the extraction itself: column misalignment (a URL in
a title field), table headers parsed as findings, methodology phases parsed
as findings, summary rows with no matching finding, and constant-filled
severities. A report that fails a blocking check is quarantined rather than
indexed, so the engine never answers from corrupt evidence.

Field names are normalised across vendors (`detailed_observations` /
`vulnerabilities`, and so on), each finding becomes its own chunk with
character offsets carried from construction, and figure placeholders are kept
and joined to their findings rather than dropped.

## Layout

```
ragaudit/
  agent/      tool registry, tools, agent loop, evidence ledger, compute, render blocks
  chat/       chat service and sessions
  ingest/     validation gate, normalisation, chunking, ingest pipeline
  retrieve/   embedder + reranker, hybrid retrieval (RRF), router, structured answers
  generate/   Ollama client (tool calling), prompts, in-context examples
  store/      SQLite catalogue with FTS5, vector index (numpy / FAISS)
  api/        FastAPI server and the web console
  eval/       metrics with bootstrap intervals and McNemar's test
```

## Known limitations

- Some aggregate paths still load matching rows to count them, and the
  default vector backend is an exact linear scan. Both are fine at hundreds of
  reports; use `faiss` and expect a scalability pass before thousands.
- Exporting results (CSV) behind a confirmation prompt is designed but not
  yet implemented.
- Figures display only when image files have been staged next to the index;
  otherwise they are listed as referenced-but-missing.
- On CPU an agent run takes tens of seconds. Single-tool counts return in
  about a second.
