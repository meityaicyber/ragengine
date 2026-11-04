"""Regression tests on the team's real extracted reports (tests/fixtures).

2-report.json: clean detailed findings, misaligned executive-summary table.
4-report.json: methodology phases parsed as findings, rows shifted.
Plus a "drifted" copy of 2-report imitating a different extractor's output.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from conftest import make_app, write_report

from ragaudit.corpus_tools import crosscheck

FIX = Path(__file__).parent / "fixtures"

# The real reports are confidential and are not shipped in shared copies.
if not (FIX / "2-report.json").exists() or not (FIX / "4-report.json").exists():
    import pytest

    pytest.skip("confidential report fixtures not present", allow_module_level=True)


def load(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _copy(tmp_path, *names):
    import shutil

    folder = tmp_path / "reports"
    folder.mkdir(exist_ok=True)
    for n in names:
        shutil.copy(FIX / n, folder / n)
    return folder


def test_real_reports_ingest_as_expected(tmp_path):
    folder = _copy(tmp_path, "2-report.json", "4-report.json")
    app = make_app(tmp_path)
    out = {o.report_id: o for o in app.build_ingest().ingest_paths(
        [folder / "2-report.json", folder / "4-report.json"]).outcomes}
    good, bad = out["2-REPORT"], out["4-REPORT"]
    # Summary-table misalignment no longer quarantines a report whose details are clean.
    assert good.status == "indexed" and good.findings == 7
    assert any("title_not_a_url" in w for w in good.warnings)
    # Genuinely corrupted details are still quarantined, with the real reason.
    assert bad.status == "quarantined"
    assert any("title_not_a_methodology_phase" in b for b in bad.blocking)

    findings = {f.finding_id: f for f in app.catalog.findings(report_ids=["2-REPORT"])}
    assert findings["OBS-01"].title == "Broken Access Control"
    assert sorted(findings["OBS-06"].cwe_ids) == ["CWE-200", "CWE-209"]   # bare canonical ids
    assert app.catalog.count_findings(severities=["Medium"]) == 4
    assert app.catalog.count_findings(cwe="CWE-284") == 3
    # Root cause is now searchable text.
    texts = " ".join(r["text"] for r in app.catalog.all_chunks(["2-REPORT"]))
    assert "Root cause: Missing server-side validation" in texts
    result = crosscheck(folder, app.settings.catalog_path, app.settings.index_dir)
    assert result["ok"], result
    app.close()


def _drift(data):
    """Imitate another extractor: Title Case keys, wrapper object, odd values."""
    d = copy.deepcopy(data)
    sev = {"High": "4 - High", "Medium": "Medium Risk", "Low": "LOW (2.1)"}
    rows = []
    for r in d["detailed_observations"]:
        rows.append({
            "Finding ID": r["finding_id"], "Vulnerability Name": r["title"],
            "Risk Rating": sev[r["severity"]], "Current Status": "Not Fixed",
            "CVSS Score": "8.1 (High)", "CWE ID": " / ".join(r["cwe"]),
            "Affected URLs": r["affected_asset"], "Description": r["observation_description"],
            "Business Impact": r["impact"], "Recommended Solution": r["recommendation"],
            "Root Cause": r["likely_root_cause"],
        })
    return {"Audit Report": {"Report Meta": {"Report ID": "TATVA-2026-01",
                                              "Report Title": d["report_meta"]["report_title"]},
                             "Vulnerability Details": rows}}


def test_drifted_extractor_format_still_ingests(tmp_path):
    path = write_report(tmp_path / "d" / "tatva_v2.json", _drift(load("2-report.json")))
    app = make_app(tmp_path)
    outcome = app.build_ingest().ingest_file(path)
    assert outcome.status == "indexed", outcome.blocking
    assert outcome.report_id == "TATVA-2026-01" and outcome.findings == 7
    fs = {f.finding_id: f for f in app.catalog.findings()}
    assert fs["OBS-01"].severity.value == "High"          # "4 - High"
    assert fs["OBS-02"].severity.value == "Medium"        # "Medium Risk"
    assert fs["OBS-07"].severity.value == "Low"           # "LOW (2.1)"
    assert fs["OBS-01"].status.value == "Open"            # "Not Fixed" is NOT Closed
    assert fs["OBS-01"].cvss_score == 8.1                 # "8.1 (High)"
    assert sorted(fs["OBS-06"].cwe_ids) == ["CWE-200", "CWE-209"]
    assert app.catalog.get_report_meta("TATVA-2026-01").title == "Tatva API Security Testing Report"
    app.close()


def test_severity_and_status_parsing_variants():
    from ragaudit.domain.models import FindingStatus as S
    from ragaudit.domain.models import Severity as V

    assert V.parse("CRITICAL (9.8)") is V.CRITICAL and V.parse("Sev: Medium") is V.MEDIUM
    assert V.parse("Informational") is V.INFORMATIONAL and V.parse("") is V.UNKNOWN
    for raw, want in (("Not Fixed", S.OPEN), ("Not yet remediated", S.OPEN), ("Pending", S.OPEN),
                      ("Partially Remediated", S.PARTIALLY_REMEDIATED), ("Closed - verified on retest", S.CLOSED),
                      ("Patched", S.CLOSED), ("Risk Accepted by management", S.RISK_ACCEPTED)):
        assert S.parse(raw) is want, raw


# -- bugs found by running the agent on 2-report.json with real prompts -------------

def _app_with_2report(tmp_path, llm=None):
    app = make_app(tmp_path, llm=llm) if llm else make_app(tmp_path)
    app.build_ingest().ingest_paths([FIX / "2-report.json"])
    return app


def test_get_finding_text_shows_the_whole_finding(tmp_path):
    from ragaudit.agent.evidence import EvidenceLedger
    from ragaudit.agent.registry import ToolContext
    from ragaudit.agent.tools import build_registry

    app = _app_with_2report(tmp_path)
    ctx = ToolContext(catalog=app.catalog, retriever=app.build_retriever(), settings=app.settings,
                      ledger=EvidenceLedger())
    out = build_registry().dispatch("get_finding_text", {"finding_id": "OBS-01"}, ctx).content
    text = out["passages"][0]["text"]
    assert not out["passages"][0]["truncated"]
    assert "Recommendation:" in text and "Root cause: Missing server-side validation" in text
    search = build_registry().dispatch("search_reports", {"query": "access control"}, ctx).content
    assert search["count"] <= 6
    app.close()


def test_web_console_remembers_the_conversation(tmp_path):
    from fastapi.testclient import TestClient

    from conftest import FakeLLM
    from ragaudit.api.server import create_app

    seen = []

    class Spy(FakeLLM):
        def chat(self, messages, **kw):
            if kw.get("tools"):
                seen.append([m["role"] for m in messages])
            return super().chat(messages, **kw)

    app = _app_with_2report(tmp_path, llm=Spy())
    client = TestClient(create_app(app))
    first = client.post("/ask", json={"question": "What was the likely root cause of OBS-01?"}).json()
    second = client.post("/ask", json={"question": "What was recommended for it?",
                                       "session_id": first["session_id"]}).json()
    assert second["session_id"] == first["session_id"]           # same session kept
    assert seen[-2][:4] == ["system", "user", "assistant", "user"]  # history reached the agent
    app.close()


def test_abstention_shows_no_unrelated_figures(tmp_path):
    from conftest import FakeLLM

    app = _app_with_2report(tmp_path, llm=FakeLLM(liar=True))     # fabricates -> forced abstention
    answer, _ = app.build_chat().ask("Was the HR database encrypted at rest?")
    assert answer.abstained and answer.figures == []
    app.close()


# -- the team's Standard Template v2 JSON schema ----------------------------------

def test_standard_template_schema_is_fully_used(tmp_path):
    folder = _copy(tmp_path, "standard_template_report.json")
    app = make_app(tmp_path)
    out = app.build_ingest().ingest_file(folder / "standard_template_report.json")
    assert out.status == "indexed" and out.findings == 4
    # Closed without retest evidence is flagged (template: "treated as unverified"), not blocked.
    assert any("closure_verified" in w and "OBS-03" in w for w in out.warnings)
    assert not any("OBS-02" in w and "closure" in w for w in out.warnings)   # has a retest entry

    meta = app.catalog.get_report_meta("SARUS-PAY-2026-01")
    assert meta.auditor == "Rohan Iyer"                    # sign-off object -> name
    assert meta.release_date == "2026-03-20"
    assert app.catalog.count_findings(statuses=["Partially Remediated"]) == 1
    assert app.catalog.count_findings(cve="CVE-2020-11023") == 1

    chunks = {r["field_path"]: r["text"] for r in app.catalog.all_chunks(["SARUS-PAY-2026-01"])}
    assert "Meera Nair" in chunks["document_control"]          # who reviewed it
    assert "OSCP" in chunks["auditing_team"]                   # certifications
    assert "critical SQL injection" in chunks["executive_summary"]
    assert "summary table" not in chunks["executive_summary"].lower()
    assert "Follow-up Audit Report" in chunks["report_meta"]
    obs2 = chunks["findings[OBS-02]"]
    assert "EPSS: 0.27" in obs2 and "Impact level: Moderate" in obs2 and "Repeat finding" in obs2
    obs1 = chunks["findings[OBS-01]"]
    assert "Severity rationale: Unauthenticated attackers" in obs1
    assert "Regulatory references: RBI Master Direction" in obs1
    result = crosscheck(folder, app.settings.catalog_path, app.settings.index_dir)
    assert result["ok"], result
    app.close()


def test_reviewer_same_as_preparer_is_warned():
    from ragaudit.ingest.validate import validate_report

    data = load("standard_template_report.json")
    data["document_control"]["reviewed_by"] = {"name": "Rohan Iyer", "auditing_team_sr_no": 2}
    issues = [str(i) for i in validate_report(data, "X").warnings]
    assert any("reviewer_distinct" in i for i in issues)
