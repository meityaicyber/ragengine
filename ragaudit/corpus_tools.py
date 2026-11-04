"""Corpus inspection: look at a folder of real JSON before and after ingest.

``survey``     What do these files actually look like? Which findings-array key
               do they use, which per-finding keys are mapped to canonical
               fields and which are ignored, and do severity/status values
               parse (or fall into "Unknown")? Run this BEFORE ingesting an
               unfamiliar corpus; it tells you which alias to add.

``crosscheck`` After ingest: recount findings straight from each raw file and
               compare with what the catalogue stored, matched by file path.
               Any mismatch is a normalisation or ingest bug, never a model
               error, because no model is involved on either side.

Both use only the standard library plus the alias tables, and write nothing.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from ragaudit.domain.models import FindingStatus, Severity
from ragaudit.ingest.normalize import FIELD_ALIASES, FINDINGS_ARRAY_ALIASES, expand_reports, find_findings_array
from ragaudit.ingest.pipeline import derive_report_id, discover_reports, load_json

_ALL_ALIASES = {alias for aliases in FIELD_ALIASES.values() for alias in aliases}


def _first(row: dict[str, Any], canonical: str) -> Any:
    for key in FIELD_ALIASES[canonical]:
        if row.get(key) not in (None, "", [], {}):
            return row[key]
    return None


def survey(root: Path, *, include_all: bool = False, sample: int = 0) -> dict[str, Any]:
    skipped: list[Path] = []
    paths = discover_reports(Path(root), include_all=include_all, skipped=skipped)
    if sample:
        paths = paths[:sample]

    top_types: Counter = Counter()
    top_keys: Counter = Counter()
    array_keys: Counter = Counter()
    row_keys: Counter = Counter()
    unmapped_row_keys: Counter = Counter()
    canonical_hits: Counter = Counter()
    severity_raw: Counter = Counter()
    status_raw: Counter = Counter()
    unreadable: list[str] = []
    no_array: list[str] = []
    layouts: Counter = Counter()
    ids: dict[str, list[str]] = {}
    findings_total = 0

    for path in paths:
        try:
            data = load_json(path)
        except Exception as exc:
            unreadable.append(f"{path}: {type(exc).__name__}: {exc}"[:300])
            continue
        top_types[type(data).__name__] += 1
        reports = expand_reports(data)
        if not reports:
            no_array.append(f"{path} (top-level {type(data).__name__} is not a report)")
            continue
        if isinstance(data, list):
            layouts[f"list holding {len(reports)} report(s)" if reports[0][0] else "bare list of findings"] += 1
        for suffix, report in reports:
            top_keys.update(report.keys())
            ids.setdefault(derive_report_id(report, path) + suffix, []).append(str(path) + suffix)
            found, rows = find_findings_array(report)
            if not found:
                no_array.append(str(path) + suffix)
                continue
            array_keys[found] += 1
            findings_total += len(rows)
            for row in rows:
                row_keys.update(row.keys())
                unmapped_row_keys.update(k for k in row.keys() if k not in _ALL_ALIASES)
                for canonical in FIELD_ALIASES:
                    if _first(row, canonical) is not None:
                        canonical_hits[canonical] += 1
                sev = _first(row, "severity")
                severity_raw[f"{sev!s:.40} -> {Severity.parse(sev).value}"] += 1
                st = _first(row, "status")
                status_raw[f"{st!s:.40} -> {FindingStatus.parse(st).value}"] += 1

    files = len(paths)
    collisions = {k: v for k, v in ids.items() if len(v) > 1}
    coverage = {
        c: f"{canonical_hits[c]}/{findings_total}" for c in FIELD_ALIASES
    }
    advice: list[str] = []
    if no_array:
        candidates = [k for k, _ in top_keys.most_common(40) if k not in FINDINGS_ARRAY_ALIASES]
        advice.append(
            f"{len(no_array)} file(s) have no recognised findings array. If the findings live "
            f"under one of these top-level keys {candidates[:12]}, add it to "
            "FINDINGS_ARRAY_ALIASES in ragaudit/ingest/normalize.py"
        )
    if findings_total:
        for canonical in ("title", "severity", "status", "observation", "recommendation"):
            if canonical_hits[canonical] < findings_total * 0.5:
                advice.append(
                    f"'{canonical}' is found in only {canonical_hits[canonical]}/{findings_total} findings. "
                    f"Check the unmapped keys below and add the right one to FIELD_ALIASES['{canonical}']"
                )
        unknown_sev = sum(n for k, n in severity_raw.items() if k.endswith("-> Unknown"))
        if unknown_sev > findings_total * 0.1:
            advice.append(
                f"{unknown_sev} findings have a severity that parses as Unknown; see severity_values. "
                "Add spellings to Severity.parse aliases in ragaudit/domain/models.py"
            )
        unknown_st = sum(n for k, n in status_raw.items() if k.endswith("-> Unknown"))
        if unknown_st > findings_total * 0.1:
            advice.append(
                f"{unknown_st} findings have a status that parses as Unknown; see status_values. "
                "Add spellings to FindingStatus.parse aliases in ragaudit/domain/models.py "
                "(a report with no status field at all will legitimately show Unknown)"
            )
    if collisions:
        advice.append(
            f"{len(collisions)} report id(s) are shared by several files; ingest will store the "
            "extra copies under suffixed ids (ID__FOLDER). That is safe, but check they are not duplicates."
        )
    if unreadable:
        advice.append(f"{len(unreadable)} file(s) are not valid JSON; they will show as 'failed' at ingest.")
    if not advice and files:
        advice.append("Field names look covered. Next: validate, then ingest.")

    return {
        "folder": str(root),
        "files": files,
        "skipped_by_name": [str(p) for p in skipped],
        "unreadable": unreadable,
        "top_level_types": dict(top_types),
        "findings_array_key_used": dict(array_keys),
        "non_object_files": dict(layouts),
        "files_without_findings_array": no_array[:20],
        "findings_total": findings_total,
        "canonical_field_coverage": coverage,
        "unmapped_finding_keys (ignored by ingest)": dict(unmapped_row_keys.most_common(25)),
        "severity_values": dict(severity_raw.most_common(15)),
        "status_values": dict(status_raw.most_common(15)),
        "top_level_keys": dict(top_keys.most_common(30)),
        "report_id_collisions": {k: v[:5] for k, v in list(collisions.items())[:10]},
        "advice": advice,
    }


def crosscheck(
    root: Path, catalog_path: Path, index_dir: Path | None = None, *, include_all: bool = False
) -> dict[str, Any]:
    """Raw-file finding counts vs the catalogue, matched on ingest path."""
    conn = sqlite3.connect(str(catalog_path))
    conn.row_factory = sqlite3.Row
    stored: dict[str, sqlite3.Row] = {}
    for row in conn.execute(
        "SELECT r.report_id, r.ingest_status, r.ingest_path, "
        "(SELECT COUNT(*) FROM findings f WHERE f.report_id = r.report_id) AS n, "
        "(SELECT COUNT(*) FROM chunks c WHERE c.report_id = r.report_id) AS chunks "
        "FROM reports r"
    ):
        if row["ingest_path"]:
            stored[row["ingest_path"]] = row

    index_counts: Counter = Counter()
    ids_file = Path(index_dir or Path(catalog_path).parent / "index") / "ids.json"
    if ids_file.exists():
        import json

        index_counts.update(json.loads(ids_file.read_text(encoding="utf-8")).get("report_ids", []))

    problems: list[str] = []
    matched = quarantined = 0
    for path in discover_reports(Path(root), include_all=include_all):
        try:
            data = load_json(path)
        except Exception as exc:
            problems.append(f"UNREADABLE   {path.name}: {exc}")
            continue
        reports = expand_reports(data)
        if not reports:
            problems.append(f"NOT_A_REPORT {path.name}: top-level JSON holds no report or findings list")
            continue
        for suffix, report in reports:
            label = path.name + suffix
            raw_n = len(find_findings_array(report)[1])
            row = stored.get(str(path.resolve()) + suffix)
            if row is None:
                problems.append(f"NOT_INGESTED {label} (raw findings={raw_n})")
                continue
            if row["ingest_status"] == "quarantined":
                quarantined += 1
                continue
            if row["n"] != raw_n:
                problems.append(f"COUNT_DIFF   {row['report_id']}: raw={raw_n} stored={row['n']}")
                continue
            if ids_file.exists() and index_counts.get(row["report_id"], 0) != row["chunks"]:
                problems.append(
                    f"VECTORS      {row['report_id']}: {row['chunks']} chunks but "
                    f"{index_counts.get(row['report_id'], 0)} vectors (re-run ingest to repair)"
                )
                continue
            matched += 1
    conn.close()
    return {
        "matched": matched,
        "quarantined_not_compared": quarantined,
        "problems": problems,
        "ok": not problems,
    }
