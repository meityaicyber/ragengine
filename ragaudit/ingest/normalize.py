"""Canonical field mapping.

Vendors spell the same concept differently: the supplied corpus uses
``detailed_observations`` where ``report_structure.json`` specifies
``vulnerabilities``, and ``observation_description`` where the schema says
``detailed_observation``. Across thousands of reports this drift is the norm.

Normalising at ingest means every downstream component -- SQL store, chunker,
retriever, chatbot -- sees one shape. The alias tables below are the hand-built
seed; ``schema_extractor``'s clustering output supersedes them at scale by
supplying the mapping empirically (D04).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from ragaudit.domain.models import (
    FIGURE_TOKEN_RE,
    Finding,
    FindingStatus,
    Report,
    ReportMeta,
    Severity,
)

# Canonical field -> the spellings seen in the wild, in priority order.
#: Edit these tables when ``python run.py survey <folder>`` shows your JSON uses
#: a spelling not listed here. Order is priority: the first key present wins.
#: The validator imports these same tables, so one edit covers both.
FINDINGS_ARRAY_ALIASES = (
    "detailed_observations",
    "vulnerabilities",
    "findings",
    "observations",
    "detailed_findings",
    "vulnerability_details",
)

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "finding_id": ("finding_id", "id", "observation_id", "vulnerability_id", "sr_no", "s_no"),
    "title": (
        "title", "vulnerability_title", "finding_title", "observation_title",
        "vulnerability_name", "name", "issue_title",
    ),
    "severity": ("severity", "risk_rating", "risk_level", "severity_level", "risk_severity", "criticality", "risk"),
    "status": (
        "status", "finding_status", "remediation_status", "revalidation_status", "current_status",
    ),
    "cvss_score": ("cvss_score", "cvss", "cvss_base_score"),
    "cvss_vector": ("cvss_vector", "cvss_string"),
    "cwe_ids": ("cwe", "cwe_ids", "cwe_id"),
    "cve_ids": ("cve", "cve_ids", "cve_id"),
    "affected_assets": (
        "affected_asset", "affected_assets", "asset", "assets", "affected_url",
        "affected_urls", "vulnerable_url", "affected_hosts",
    ),
    "observation": (
        "observation_description",
        "detailed_observation",
        "description",
        "vulnerability_description",
        "observation",
        "details",
    ),
    "impact": ("impact", "business_impact", "risk_impact"),
    "recommendation": (
        "recommendation", "recommendations", "remediation", "mitigation", "recommended_solution",
    ),
    "preconditions": ("preconditions", "prerequisites", "precondition"),
    "root_cause": ("likely_root_cause", "root_cause", "root_cause_analysis", "cause"),
    "references": ("reference", "references", "ref", "links"),
    "severity_rationale": ("severity_rationale", "rationale", "severity_justification"),
    "regulatory_references": ("regulatory_references", "regulatory_reference", "regulations",
                              "regulatory_mapping", "compliance_references"),
    "impact_level": ("impact_level", "impact_rating"),
    "likelihood_level": ("likelihood_level", "likelihood", "probability", "likelihood_rating"),
    "observation_date": ("observation_date", "date_observed", "tested_on", "finding_date"),
    "epss": ("reported_epss_score", "epss_score", "epss"),
    "proof": ("proof_of_concept_steps", "proof_of_concept", "poc", "reproduction_steps", "steps"),
    "repeat": ("new_or_repeat", "repeat", "is_repeat"),
}

# Narrative fields, in the order a reader would want them. These are the only
# fields that get embedded; everything else is answered by SQL (D02).
NARRATIVE_FIELDS = ("observation", "impact", "recommendation", "preconditions", "root_cause",
                    "severity_rationale", "regulatory_references", "references")


def _first(row: dict[str, Any], canonical: str) -> Any:
    for alias in FIELD_ALIASES.get(canonical, ()):
        if alias in row and row[alias] not in (None, "", [], {}):
            return row[alias]
    return None


def _as_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                out.extend(str(v).strip() for v in item.values() if str(v).strip())
        return out
    return [part.strip() for part in str(value).replace("\n", ",").split(",") if part.strip()]


def _as_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, list):
        return "\n".join(_as_text(v) for v in value if _as_text(v))
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_as_text(v)}" for k, v in value.items() if _as_text(v))
    # Extracted PDF text carries hard line wraps mid-sentence; collapse them so
    # chunks read as prose and embeddings are not polluted by layout artefacts.
    return " ".join(str(value).split())


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_CWE_TOKEN_RE = re.compile(r"CWE[-\s:]*(\d+)", re.I)
_CVE_TOKEN_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.I)


def _as_float(value: Any) -> float | None:
    """CVSS score from "8.1", 8.1, "8.1 (High)" or "CVSS: 8.1". None if absent."""
    if value in (None, "", [], {}):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    match = _NUMBER_RE.search(str(value))
    if not match:
        return None
    number = float(match.group(0))
    return number if 0.0 <= number <= 10.0 else None


def _identifiers(value: Any, token_re: re.Pattern, prefix: str) -> list[str]:
    """Canonical ids out of messy strings.

    Real extractions hold "CWE-284: Improper Access Control" and even
    "CWE-639 / CWE-862 Branch Dispatcher - BLR-WEST". Storing those verbatim
    split one CWE into many spellings, so top-CWE counts and cross-report
    repeat detection disagreed with the report. Every CWE-n / CVE-y-n token is
    extracted and stored bare; values with no token are kept as given.
    """
    items = _as_list(value)
    out: list[str] = []
    for item in items:
        found = token_re.findall(item)
        if found:
            for tok in found:
                ident = f"{prefix}-{tok}" if prefix == "CWE" else tok.upper()
                if ident not in out:
                    out.append(ident)
        elif item not in out:
            out.append(item)
    return out


def _looks_like_findings(rows: list[Any]) -> bool:
    """A list whose items are mostly dicts carrying a title or severity field."""
    dicts = [r for r in rows if isinstance(r, dict)]
    if not dicts or len(dicts) < len(rows) * 0.8:
        return False
    marker_keys = set(FIELD_ALIASES["title"]) | set(FIELD_ALIASES["severity"])
    hits = sum(1 for r in dicts if marker_keys & set(r.keys()))
    return hits >= max(1, len(dicts) * 0.5)


def find_findings_array(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Locate the findings list. Returns (key path, rows); ("", []) if none.

    1. A known key at the top level (FINDINGS_ARRAY_ALIASES), as before.
    2. Otherwise, a known key nested inside a top-level object, e.g.
       {"report": {"vulnerabilities": [...]}}.
    3. Otherwise, any list (top level or one level down) whose items look like
       findings (most carry a title or severity field). Extractors vary, and
       a report silently quarantined for a key-name difference is the most
       likely way a demo corpus loses reports.
    """
    for key in FINDINGS_ARRAY_ALIASES:
        value = data.get(key)
        if isinstance(value, list):
            return key, [v for v in value if isinstance(v, dict)]
    for outer, inner in data.items():
        if isinstance(inner, dict):
            for key in FINDINGS_ARRAY_ALIASES:
                value = inner.get(key)
                if isinstance(value, list):
                    return f"{outer}.{key}", [v for v in value if isinstance(v, dict)]
    # Heuristic: never pick the executive-summary table, which also has titles.
    skip = {"summary_table", "summary_findings", "executive_summary"}
    for outer, value in data.items():
        if outer in skip:
            continue
        if isinstance(value, list) and _looks_like_findings(value):
            return outer, [v for v in value if isinstance(v, dict)]
    for outer, inner in data.items():
        if outer in skip or not isinstance(inner, dict):
            continue
        for key, value in inner.items():
            if key not in skip and isinstance(value, list) and _looks_like_findings(value):
                return f"{outer}.{key}", [v for v in value if isinstance(v, dict)]
    return "", []


_KEY_SEP_RE = re.compile(r"[\s\-./()]+")


def _norm_key(key: Any) -> str:
    """"Finding ID" -> "finding_id", "Risk-Rating" -> "risk_rating", "Sr. No." -> "sr_no"."""
    text = _KEY_SEP_RE.sub("_", str(key).strip().lower())
    text = re.sub(r"_+", "_", text).rstrip("_")
    return text or str(key)


def normalise_keys(value: Any) -> Any:
    """Lower-case and underscore every object key, recursively. Values untouched.

    Extractors emit "Finding ID", "Severity", "Detailed Observations" as
    readily as snake_case; matching aliases on normalised keys makes all of
    them work without listing every spelling. On a clash the first
    non-empty value wins.
    """
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            nk = _norm_key(k)
            nv = normalise_keys(v)
            if nk not in out or out[nk] in (None, "", [], {}):
                out[nk] = nv
        return out
    if isinstance(value, list):
        return [normalise_keys(v) for v in value]
    return value


def expand_reports(data: Any) -> list[tuple[str, dict[str, Any]]]:
    """Turn one parsed JSON file into the report(s) it holds.

    Returns (suffix, report_dict) pairs. ``suffix`` is "" for a single report,
    or "#1", "#2"... when one file holds a list of reports. Handles:

    * an object                      -> one report (the normal case)
    * a list of report objects       -> one report each
    * a bare list of finding objects -> one report wrapping them
    * anything else                  -> [] (not a report)

    Keys are normalised first (see ``normalise_keys``).
    """
    data = normalise_keys(data)
    if isinstance(data, dict):
        key, _ = find_findings_array(data)
        if "." in key:
            # {"report": {"report_meta": ..., "findings": [...]}}: unwrap so the
            # wrapper's metadata and sections are read like a top-level report.
            outer = key.split(".", 1)[0]
            inner = data.get(outer)
            if isinstance(inner, dict):
                merged = {k: v for k, v in data.items() if k != outer}
                merged.update(inner)
                return [("", merged)]
        return [("", data)]
    if isinstance(data, list) and data and all(isinstance(x, dict) for x in data):
        if any(find_findings_array(x)[0] for x in data):
            return [(f"#{i + 1}", x) for i, x in enumerate(data)]
        if _looks_like_findings(data):
            return [("", {"findings": data})]
    return []


def normalize_finding(row: dict[str, Any], report_id: str, position: int) -> Finding:
    raw_id = _first(row, "finding_id")
    finding_id = str(raw_id).strip() if raw_id not in (None, "") else f"F-{position:03d}"

    proof = _first(row, "proof")
    figure_tokens = FIGURE_TOKEN_RE.findall(_as_text(proof) if proof else "")
    figure_tokens = [f"[IMAGE_PAGE_{p}_FIG_{i}]" for p, i in figure_tokens]

    repeat_raw = str(_first(row, "repeat") or "").strip().lower()
    is_repeat = None
    if repeat_raw in ("repeat", "repeated", "yes", "true"):
        is_repeat = True
    elif repeat_raw in ("new", "no", "false"):
        is_repeat = False

    return Finding(
        finding_id=finding_id,
        report_id=report_id,
        title=_as_text(_first(row, "title")),
        severity=Severity.parse(_first(row, "severity")),
        status=FindingStatus.parse(_first(row, "status")),
        cvss_score=_as_float(_first(row, "cvss_score")),
        cvss_vector=_as_text(_first(row, "cvss_vector")),
        cwe_ids=_identifiers(_first(row, "cwe_ids"), _CWE_TOKEN_RE, "CWE"),
        cve_ids=_identifiers(_first(row, "cve_ids"), _CVE_TOKEN_RE, "CVE"),
        affected_assets=_as_list(_first(row, "affected_assets")),
        observation=_as_text(_first(row, "observation")),
        impact=_as_text(_first(row, "impact")),
        recommendation=_as_text(_first(row, "recommendation")),
        preconditions=_as_text(_first(row, "preconditions")),
        root_cause=_as_text(_first(row, "root_cause")),
        references=_as_text(_first(row, "references")),
        severity_rationale=_as_text(_first(row, "severity_rationale")),
        regulatory_references=_as_text(_first(row, "regulatory_references")),
        impact_level=_as_text(_first(row, "impact_level")),
        likelihood_level=_as_text(_first(row, "likelihood_level")),
        observation_date=_as_text(_first(row, "observation_date")),
        epss=_as_text(_first(row, "epss")),
        figure_tokens=figure_tokens,
        is_repeat=is_repeat,
        raw=row,
    )


def _nested(data: dict[str, Any], *path: str) -> Any:
    node: Any = data
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def normalize_meta(data: dict[str, Any], report_id: str, source_path: Path | None = None) -> ReportMeta:
    meta = data.get("report_meta") if isinstance(data.get("report_meta"), dict) else {}
    control = data.get("document_control") if isinstance(data.get("document_control"), dict) else {}
    top_meta = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
    extractor = data.get("_meta") if isinstance(data.get("_meta"), dict) else {}

    def pick(*candidates: Any) -> str:
        for candidate in candidates:
            # Template v2 sign-offs are objects: {"name": ..., "auditing_team_sr_no": ...}.
            if isinstance(candidate, dict) and "name" in candidate:
                candidate = candidate.get("name")
            text = _as_text(candidate)
            if text:
                return text
        return ""

    standards = _as_list(
        _nested(data, "methodology", "standards_referred")
        or top_meta.get("standards_referenced")
        or []
    )

    sha = ""
    if source_path and source_path.exists():
        digest = hashlib.sha256()
        with source_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        sha = digest.hexdigest()

    return ReportMeta(
        report_id=report_id,
        title=pick(meta.get("report_title"), control.get("document_title"), top_meta.get("document_title"),
                   meta.get("title"), data.get("title"), data.get("report_title"),
                   # Last resort: the file name, so lists and the UI never show a blank.
                   source_path.stem.replace("_", " ") if source_path else ""),
        report_type=pick(meta.get("report_type"), top_meta.get("report_type")),
        audit_type=pick(meta.get("audit_type"), top_meta.get("audit_type")),
        organisation=pick(top_meta.get("organization_audited")),
        auditor=pick(top_meta.get("auditing_organization"), control.get("prepared_by")),
        classification=pick(meta.get("classification"), top_meta.get("document_classification")),
        release_date=pick(meta.get("report_release_date"), control.get("release_date")),
        audit_period_from=pick(_nested(data, "report_meta", "audit_period", "from")),
        audit_period_to=pick(_nested(data, "report_meta", "audit_period", "to")),
        standards=standards,
        source_file=pick(extractor.get("source_file"), str(source_path) if source_path else ""),
        source_sha256=sha,
        extractor_version=pick(extractor.get("pipeline_version")),
        ingest_path=str(source_path.resolve()) if source_path else "",
    )


def _disambiguate_ids(findings: list[Finding]) -> None:
    """Make finding ids unique within a report.

    Extracted reports sometimes repeat an id (two rows both "OBS-03"). The
    catalogue keys findings on (report_id, finding_id), so a repeat used to
    raise an integrity error and fail the whole report. The first occurrence
    keeps its id; later ones become ``OBS-03#2``, ``OBS-03#3`` so they stay
    queryable and visibly distinct.
    """
    used: set[str] = set()
    counts: dict[str, int] = {}
    for finding in findings:
        base = finding.finding_id
        if base not in used:
            used.add(base)
            counts[base] = 1
            continue
        n = counts.get(base, 1)
        candidate = base
        while candidate in used:
            n += 1
            candidate = f"{base}#{n}"
        counts[base] = n
        used.add(candidate)
        finding.finding_id = candidate


def normalize_report(data: dict[str, Any], report_id: str, source_path: Path | None = None) -> Report:
    """Build a canonical Report. Chunks and figures are attached later."""
    _, rows = find_findings_array(data)
    findings = [normalize_finding(row, report_id, i + 1) for i, row in enumerate(rows)]
    _disambiguate_ids(findings)
    return Report(meta=normalize_meta(data, report_id, source_path), findings=findings, raw=data)
