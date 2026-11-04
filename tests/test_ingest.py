"""Ingest, validation and storage fixes."""

from __future__ import annotations

import json

from conftest import FakeEmbedder, finding, make_app, mini_report, write_report

from ragaudit.ingest.pipeline import discover_reports
from ragaudit.ingest.validate import validate_report
from ragaudit.store.vectors import open_index


def test_sample_corpus_ingests(app):
    stats = app.catalog.stats()
    assert stats["reports_indexed"] == 2
    assert stats["reports_quarantined"] == 1          # the deliberately broken sample
    assert stats["findings"] == 9                     # 5 + 4
    for rid in app.catalog.report_ids(status="indexed"):
        assert app.index.vector_count(rid) == len(app.catalog.chunk_ids(rid)) > 0


def test_vendor_spellings_normalise(app):
    # "Moderate" -> Medium, "Resolved" -> Closed, "WIP" -> In Progress
    by_id = {f.finding_id: f for f in app.catalog.findings(report_ids=["SAMPLE-API-002"])}
    assert by_id["V-3"].severity.value == "Medium"
    assert by_id["V-2"].status.value == "Closed"
    assert by_id["V-3"].status.value == "In Progress"
    assert by_id["V-2"].cve_ids == ["CVE-2015-9235"]


def test_reingest_unchanged_is_skipped(app, corpus):
    report = app.build_ingest().ingest_paths(sorted(corpus.glob("*.json")))
    assert report.count("skipped") == 3


def test_crash_before_index_save_self_heals(tmp_path, corpus):
    """Old bug: SQL committed, vectors never saved, next run skipped the report."""
    app1 = make_app(tmp_path)
    pipe = app1.build_ingest()
    pipe.checkpoint_dir = None                          # simulate dying before any save
    pipe.ingest_file(corpus / "sample_web_portal.json")
    assert app1.catalog.stats()["findings"] == 5
    app1.close()                                        # index never written to disk

    app2 = make_app(tmp_path)                           # fresh process, empty index on disk
    assert app2.index.vector_count("SAMPLE-WEB-001") == 0
    outcome = app2.build_ingest().ingest_file(corpus / "sample_web_portal.json")
    assert outcome.status == "indexed"                  # re-embedded, not "skipped"
    assert app2.index.vector_count("SAMPLE-WEB-001") == outcome.chunks
    app2.close()


def test_ctrl_c_saves_progress_and_resumes(tmp_path, corpus):
    paths = sorted(corpus.glob("*.json"))
    good = [p for p in paths if "broken" not in p.name]
    app1 = make_app(tmp_path, embedder=FakeEmbedder(fail_on_call=2, exc=KeyboardInterrupt))
    report = app1.build_ingest().ingest_paths(good)
    assert report.interrupted
    assert report.count("indexed") == 1
    app1.close()

    on_disk = open_index(tmp_path / "var" / "index", dimension=1024)
    assert len(on_disk) > 0                             # first report's vectors were checkpointed

    app2 = make_app(tmp_path)
    report2 = app2.build_ingest().ingest_paths(good)
    assert report2.count("skipped") == 1 and report2.count("indexed") == 1
    app2.close()


def test_embedding_failure_keeps_previous_version(tmp_path, corpus):
    app = make_app(tmp_path, embedder=FakeEmbedder(fail_on_call=1))
    report = app.build_ingest().ingest_paths([corpus / "sample_web_portal.json"])
    assert report.count("failed") == 1
    assert app.catalog.stats()["reports"] == 0          # nothing half-written
    app.close()


def test_duplicate_finding_ids_no_longer_fail(tmp_path):
    path = write_report(tmp_path / "d" / "dup.json",
                        mini_report([finding("OBS-01"), finding("OBS-01", "Other"), finding("OBS-02")]))
    app = make_app(tmp_path)
    outcome = app.build_ingest().ingest_file(path)
    assert outcome.status == "indexed"
    ids = sorted(f.finding_id for f in app.catalog.findings())
    assert ids == ["OBS-01", "OBS-01#2", "OBS-02"]
    assert any("finding_ids_unique" in w for w in outcome.warnings)
    app.close()


def test_observations_key_accepted_by_validator():
    data = {"report_meta": {"report_title": "x", "classification": "y"},
            "observations": [finding("A-1")]}
    assert validate_report(data, "X").ok


def test_same_filename_in_two_folders_does_not_overwrite(tmp_path):
    a = write_report(tmp_path / "d" / "vendorA" / "report.json", mini_report([finding("F-1")]))
    b = write_report(tmp_path / "d" / "vendorB" / "report.json",
                     mini_report([finding("F-1"), finding("F-2")]))
    app = make_app(tmp_path)
    report = app.build_ingest().ingest_paths([a, b])
    ids = sorted(o.report_id for o in report.outcomes)
    assert ids == ["REPORT", "REPORT__VENDORB"]
    assert app.catalog.stats()["findings"] == 3
    # Stable across runs: re-running maps each file back to its own id.
    again = app.build_ingest().ingest_paths([a, b])
    assert sorted(o.report_id for o in again.outcomes) == ids
    assert again.count("skipped") == 2
    app.close()


def test_utf8_bom_file_ingests(tmp_path):
    path = tmp_path / "d" / "bom.json"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(mini_report([finding("B-1")])).encode())
    app = make_app(tmp_path)
    assert app.build_ingest().ingest_file(path).status == "indexed"
    app.close()


def test_non_object_json_fails_clearly(tmp_path):
    path = tmp_path / "d" / "list.json"
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2, 3]")
    app = make_app(tmp_path)
    outcome = app.build_ingest().ingest_paths([path]).outcomes[0]
    assert outcome.status == "failed" and "holds neither" in outcome.error
    app.close()


def test_discovery_only_skips_exact_names(tmp_path):
    for name in ("vapt_results_2024.json", "metrics.json", "q3_metrics_audit.json", "results.json"):
        write_report(tmp_path / name, {})
    skipped = []
    found = {p.name for p in discover_reports(tmp_path, skipped=skipped)}
    assert found == {"vapt_results_2024.json", "q3_metrics_audit.json"}
    assert {p.name for p in skipped} == {"metrics.json", "results.json"}


def test_cwe_match_is_whole_identifier(tmp_path):
    path = write_report(tmp_path / "d" / "cwe.json", mini_report(
        [finding("C-1", cwe="CWE-284"), finding("C-2", cwe="CWE-28"), finding("C-3", cwe="CWE-284: Improper Access Control")]))
    app = make_app(tmp_path)
    app.build_ingest().ingest_file(path)
    assert {f.finding_id for f in app.catalog.findings(cwe="CWE-284")} == {"C-1", "C-3"}
    assert {f.finding_id for f in app.catalog.findings(cwe="CWE-28")} == {"C-2"}
    assert {f.finding_id for f in app.catalog.findings(cwe="284")} == {"C-1", "C-3"}
    app.close()


def test_sql_count_matches_list(app):
    for kwargs in ({}, {"severities": ["High"]}, {"statuses": ["Open"]}, {"cwe": "CWE-284"},
                   {"severities": ["High"], "statuses": ["Open"]}):
        assert app.catalog.count_findings(**kwargs) == len(app.catalog.findings(**kwargs))


def test_old_catalog_is_migrated(tmp_path):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE reports (report_id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '')")
    conn.commit()
    conn.close()
    from ragaudit.store.catalog import Catalog

    cat = Catalog(db)
    cols = {r["name"] for r in cat._conn.execute("PRAGMA table_info(reports)")}
    assert "ingest_path" in cols
    cat.close()


# -- layouts other than "one report object per file" -----------------------------

def test_nested_findings_array_is_found(tmp_path):
    data = {"report": {"report_meta": {"report_id": "NEST-1", "report_title": "Nested"},
                       "vulnerabilities": [finding("N-1"), finding("N-2")]}}
    path = write_report(tmp_path / "d" / "nested.json", data)
    app = make_app(tmp_path)
    outcome = app.build_ingest().ingest_file(path)
    assert outcome.status == "indexed" and outcome.findings == 2
    assert outcome.report_id == "NEST-1"                          # wrapper metadata is used
    assert app.catalog.get_report_meta("NEST-1").title == "Nested"
    app.close()


def test_unknown_findings_key_found_by_shape(tmp_path):
    data = {"meta": {"name": "x"}, "issues_identified": [finding("U-1"), finding("U-2"), finding("U-3")],
            "executive_summary": {"summary_table": [{"title": "U-1", "severity": "High"}]}}
    path = write_report(tmp_path / "d" / "shape.json", data)
    app = make_app(tmp_path)
    outcome = app.build_ingest().ingest_file(path)
    assert outcome.status == "indexed" and outcome.findings == 3     # not the summary table
    app.close()


def test_bare_list_of_findings_is_one_report(tmp_path):
    path = write_report(tmp_path / "d" / "bare.json", [finding("B-1"), finding("B-2")])
    app = make_app(tmp_path)
    outcome = app.build_ingest().ingest_file(path)
    assert outcome.status == "indexed" and outcome.findings == 2 and outcome.report_id == "BARE"
    app.close()


def test_file_with_several_reports_is_split(tmp_path):
    from ragaudit.corpus_tools import crosscheck

    reports = [mini_report([finding("A-1")], report_id="MULTI-A"),
               mini_report([finding("B-1"), finding("B-2")]),            # no id of its own
               mini_report([finding("C-1")])]
    path = write_report(tmp_path / "d" / "bundle.json", reports)
    app = make_app(tmp_path)
    out = app.build_ingest().ingest_paths([path])
    assert sorted(o.report_id for o in out.outcomes) == ["BUNDLE-2", "BUNDLE-3", "MULTI-A"]
    assert app.catalog.stats()["findings"] == 4
    again = app.build_ingest().ingest_paths([path])
    assert again.count("skipped") == 3                                    # stable ids on re-run
    result = crosscheck(tmp_path / "d", app.settings.catalog_path, app.settings.index_dir)
    assert result["ok"] and result["matched"] == 3, result
    app.close()


def test_no_quarantine_reingests_only_quarantined(app, corpus):
    pipe = app.build_ingest()
    pipe.fail_on_validation_error = False
    out = pipe.ingest_paths(sorted(corpus.glob("*.json")))
    assert out.count("skipped") == 2 and out.count("indexed") == 1
    assert app.catalog.report_status("SAMPLE-BROKEN-003") == "indexed"


def test_changed_report_id_replaces_old_copy(tmp_path):
    path = write_report(tmp_path / "d" / "r.json", mini_report([finding("X-1"), finding("X-2")]))
    app = make_app(tmp_path)
    assert app.build_ingest().ingest_file(path).report_id == "R"
    write_report(path, mini_report([finding("X-1"), finding("X-2")], report_id="REAL-ID"))
    outcome = app.build_ingest().ingest_file(path)
    assert outcome.report_id == "REAL-ID" and "replaces" in outcome.note
    assert app.catalog.report_ids() == ["REAL-ID"]
    assert app.catalog.stats()["findings"] == 2                    # not 4
    assert app.index.vector_count("R") == 0
    app.close()


def test_title_falls_back_to_file_name(tmp_path):
    path = write_report(tmp_path / "d" / "untitled_vapt_2024.json", {"findings": [finding("T-1")]})
    app = make_app(tmp_path)
    rid = app.build_ingest().ingest_file(path).report_id
    assert app.catalog.get_report_meta(rid).title == "untitled vapt 2024"
    app.close()
