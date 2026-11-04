"""End to end through ChatService, the agent loop and the HTTP API (fake models)."""

from __future__ import annotations

import pytest
from conftest import FakeLLM, make_app

from ragaudit.corpus_tools import crosscheck, survey


def ask(app, question, **kw):
    answer, _ = app.build_chat().ask(question, **kw)
    return answer


def test_greeting_never_reaches_model(app):
    before = len(app.llm.calls)
    answer = ask(app, "hello")
    assert answer.route == "chitchat"
    assert len(app.llm.calls) == before


def test_count_is_exact_and_skips_finalize(app):
    # High + Open across both indexed reports: OBS-01 (web) and V-4 (api).
    # V-2 is High but "Resolved" -> Closed. The fake model passes lowercase
    # arguments; coercion must still produce the right filter.
    answer = ask(app, "How many high-severity findings are still open?")
    assert answer.text.startswith("2 findings match")
    assert answer.route == "structured"
    assert "agent_shortcut:count_findings" in answer.review_flags
    assert not any(c["schema"] for c in app.llm.calls)       # no finalize call


def test_aggregate_produces_chart_block(app):
    answer = ask(app, "Break down the findings by severity")
    charts = [b for b in answer.blocks if b["kind"] == "chart"]
    assert charts and sum(charts[0]["series"][0]["values"]) == 9


def test_narrative_answer_has_verified_citation(app):
    answer = ask(app, "What remediation was recommended for the token API issue?")
    assert not answer.abstained
    assert answer.citations, answer.review_flags
    assert all(c.chunk_id for c in answer.citations)
    assert any(s["tool"] == "search_reports" for s in answer.steps)


def test_scope_keeps_evidence_in_one_report(app):
    answer = ask(app, "What remediation was recommended for the token API issue?",
                 report_ids=["SAMPLE-API-002"])
    assert answer.citations
    assert {c.report_id for c in answer.citations} == {"SAMPLE-API-002"}


def test_fabricated_citation_is_dropped_and_answer_abstains(tmp_path, corpus):
    app = make_app(tmp_path, llm=FakeLLM(liar=True))
    app.build_ingest().ingest_paths(sorted(corpus.glob("*.json")))
    answer = ask(app, "Was the database server patched?")
    assert answer.abstained
    assert any(f.startswith("citation_not_in_evidence") for f in answer.review_flags)
    assert "2019" not in answer.text
    app.close()


def test_non_obs_finding_ids_are_pre_resolved(app):
    from ragaudit.chat.service import _pre_resolve

    resolved = _pre_resolve("Tell me about V-4 and OBS-01", app.catalog)
    assert resolved["finding_ids"] == ["OBS-01", "V-4"]


def test_survey_and_crosscheck(app, corpus):
    s = survey(corpus)
    assert s["files"] == 3 and s["findings_total"] == 12
    assert s["findings_array_key_used"] == {"detailed_observations": 2, "vulnerabilities": 1}
    app.build_ingest()._checkpoint()
    result = crosscheck(corpus, app.settings.catalog_path, app.settings.index_dir)
    assert result["ok"], result["problems"]
    assert result["matched"] == 2 and result["quarantined_not_compared"] == 1


# -- HTTP API -----------------------------------------------------------------

@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    from ragaudit.api.server import create_app

    return TestClient(create_app(app))


def test_api_end_to_end(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz")
    assert ready.status_code == 200 and ready.json()["ready"] is True
    assert len(client.get("/reports").json()["reports"]) == 3
    assert "Audit Report Assistant" in client.get("/").text

    r = client.post("/ask", json={"question": "How many high-severity findings are still open?"})
    assert r.status_code == 200 and r.json()["answer"].startswith("2 findings")

    f = client.get("/findings", params={"severity": "High", "limit": 1}).json()
    assert f["count"] == 4 and f["returned"] == 1       # true total, not page length


def test_api_key_enforced_when_set(app, client):
    app.settings.api.api_key = "s3cret"
    try:
        assert client.get("/healthz").status_code == 200    # liveness stays open
        assert client.get("/").status_code == 200           # console page loads, then asks
        assert client.get("/reports").status_code == 401
        assert client.get("/reports", headers={"X-API-Key": "wrong"}).status_code == 401
        assert client.get("/reports", headers={"X-API-Key": "s3cret"}).status_code == 200
        assert client.get("/stats?key=s3cret").status_code == 200
    finally:
        app.settings.api.api_key = ""


def test_stream_endpoint(client):
    with client.stream("POST", "/ask/stream", json={"question": "Break down the findings by severity"}) as r:
        body = "".join(r.iter_text())
    assert "event: status" in body and "event: answer" in body


def test_finding_tools_use_sql_lookup_and_tolerate_case(app):
    from ragaudit.agent.evidence import EvidenceLedger
    from ragaudit.agent.registry import ToolContext
    from ragaudit.agent.tools import build_registry

    reg = build_registry()
    ctx = ToolContext(catalog=app.catalog, retriever=app.build_retriever(), settings=app.settings,
                      ledger=EvidenceLedger())
    text = reg.dispatch("get_finding_text", {"finding_id": "obs-03"}, ctx)
    assert text.error is None and text.content["count"] >= 1
    assert len(ctx.ledger) == text.content["count"]          # shown text became citable
    detail = reg.dispatch("get_finding", {"finding_id": "v-4"}, ctx)
    assert detail.error is None and detail.content["finding"]["report_id"] == "SAMPLE-API-002"
    missing = reg.dispatch("get_finding", {"finding_id": "NOPE-1"}, ctx)
    assert missing.error and "no finding" in missing.error


# -- corpus-level questions and fallbacks (the "The report does not say" bug) --

def test_how_many_reports_is_answered_exactly_by_the_agent(app):
    # The model decides (no keyword bypass); list_reports answers it through
    # the deterministic template, with no finalize call.
    answer = ask(app, "How many audit reports are indexed?")
    assert answer.text.startswith("3 report(s) are in the catalogue: 2 indexed")
    assert "quarantined" in answer.text and not answer.abstained
    assert "agent_shortcut:list_reports" in answer.review_flags
    assert any(b["kind"] == "table" for b in answer.blocks)
    assert not any(c["schema"] for c in app.llm.calls)       # no finalize call


@pytest.mark.parametrize("question,starts", [
    ("What types of information are available for each finding?", "Each finding can carry"),
    ("Which reports failed validation?", "1 of 3 report(s) were quarantined"),
    ("Which findings repeat across reports?", "1 issue recurs across reports"),  # title + CWE-284, merged
    ("Compare the reports", "Report comparison"),
    ("Show corpus stats", "3 report(s) in the catalogue"),
    ("Which findings cite CWE-284?", "2 findings matching CWE-284"),
])
def test_corpus_level_tools_answer_instead_of_abstaining(app, question, starts):
    answer = ask(app, question)
    assert not answer.abstained, answer.text
    assert answer.text.startswith(starts), answer.text
    assert any(f.startswith("agent_shortcut:") for f in answer.review_flags)


def test_tool_free_model_falls_back_to_data(tmp_path, corpus):
    app = make_app(tmp_path, llm=FakeLLM(lazy=True))
    app.build_ingest().ingest_paths(sorted(corpus.glob("*.json")))
    count = ask(app, "How many high-severity findings are still open?")
    assert "12" not in count.text and count.text.startswith("2 finding")
    assert "agent_fallback:single_pass" in count.review_flags
    narrative = ask(app, "What remediation was recommended for the token API issue?")
    assert "agent_fallback:single_pass" in narrative.review_flags
    app.close()


def test_finalize_receives_navigation_results(app):
    """Two non-search tools -> no shortcut -> both results must reach finalize."""
    from ragaudit.agent.loop import _is_structured, _summarise_structured
    from ragaudit.generate.llm import ToolCall

    assert _is_structured("list_reports") and _is_structured("describe_schema")
    assert not _is_structured("search_reports")
    summary = _summarise_structured("", ToolCall("list_reports", {}),
                                    {"reports": [{"report_id": "R1", "title": "T"}], "count": 1})
    assert "R1" in summary
