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
FINDINGS_ARRAY_ALIASES = ("detailed_observations", "vulnerabilities", "findings", "observations")

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "finding_id": ("finding_id", "id", "observation_id", "sr_no"),
    "title": ("title", "vulnerability_title", "finding_title", "name", "issue_title"),
    "severity": ("severity", "risk_rating", "risk_level", "criticality"),
    "status": ("status", "finding_status", "remediation_status", "revalidation_status"),
    "cvss_score": ("cvss_score", "cvss", "cvss_base_score"),
    "cvss_vector": ("cvss_vector", "cvss_string"),
    "cwe_ids": ("cwe", "cwe_ids", "cwe_id"),
    "cve_ids": ("cve", "cve_ids", "cve_id"),
    "affected_assets": ("affected_asset", "affected_assets", "asset", "assets", "affected_url"),
    "observation": (
        "observation_description",
        "detailed_observation",
        "description",
        "observation",
        "details",
    ),
    "impact": ("impact", "business_impact", "risk_impact"),
    "recommendation": ("recommendation", "recommendations", "remediation", "mitigation"),
    "preconditions": ("preconditions", "prerequisites", "precondition"),
    "proof": ("proof_of_concept_steps", "proof_of_concept", "poc", "reproduction_steps", "steps"),
    "repeat": ("new_or_repeat", "repeat", "is_repeat"),
}

# Narrative fields, in the order a reader would want them. These are the only
# fields that get embedded; everything else is answered by SQL (D02).
NARRATIVE_FIELDS = ("observation", "impact", "recommendation", "preconditions")


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


def _as_float(value: Any) -> float | None:
    if value in (None, "", [], {}):
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def find_findings_array(data: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for key in FINDINGS_ARRAY_ALIASES:
        value = data.get(key)
        if isinstance(value, list):
            return key, [v for v in value if isinstance(v, dict)]
    return "", []


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
        cwe_ids=_as_list(_first(row, "cwe_ids")),
        cve_ids=_as_list(_first(row, "cve_ids")),
        affected_assets=_as_list(_first(row, "affected_assets")),
        observation=_as_text(_first(row, "observation")),
        impact=_as_text(_first(row, "impact")),
        recommendation=_as_text(_first(row, "recommendation")),
        preconditions=_as_text(_first(row, "preconditions")),
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
        title=pick(meta.get("report_title"), control.get("document_title"), top_meta.get("document_title")),
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
    )


def normalize_report(data: dict[str, Any], report_id: str, source_path: Path | None = None) -> Report:
    """Build a canonical Report. Chunks and figures are attached later."""
    _, rows = find_findings_array(data)
    findings = [normalize_finding(row, report_id, i + 1) for i, row in enumerate(rows)]
    return Report(meta=normalize_meta(data, report_id, source_path), findings=findings, raw=data)
