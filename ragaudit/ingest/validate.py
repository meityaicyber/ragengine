"""Extraction validation gate.

Why this exists (D04): PDF-to-JSON extraction fails silently and at scale. In
the supplied corpus, every row of ``executive_summary.summary_table`` in
``2-report.json`` carries a URL in its ``title`` field -- a column
misalignment affecting 7 of 7 rows that no count-based check detects, because
the counts are all internally consistent (7 findings, 7 observations,
severity_count summing to 7, matching finding-id sets).

A report that fails a BLOCKING check does not enter the index. Across
thousands of reports nobody reviews extraction by eye, so the alternative to
this gate is answering questions from corrupt evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

URL_RE = re.compile(r"https?://", re.IGNORECASE)
FIGURE_TOKEN_RE = re.compile(r"\[IMAGE_PAGE_\d+_FIG_\d+\]")
CWE_RE = re.compile(r"CWE-\d+", re.IGNORECASE)
CVE_RE = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)


class Level(str, Enum):
    BLOCKING = "blocking"
    WARNING = "warning"


@dataclass
class Issue:
    check: str
    level: Level
    message: str
    locator: str = ""

    def __str__(self) -> str:
        where = f" at {self.locator}" if self.locator else ""
        return f"[{self.level.value}] {self.check}{where}: {self.message}"


@dataclass
class ValidationResult:
    report_id: str
    issues: list[Issue] = field(default_factory=list)

    @property
    def blocking(self) -> list[Issue]:
        return [i for i in self.issues if i.level is Level.BLOCKING]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.level is Level.WARNING]

    @property
    def ok(self) -> bool:
        return not self.blocking

    def add(self, check: str, level: Level, message: str, locator: str = "") -> None:
        self.issues.append(Issue(check, level, message, locator))


def _findings_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Locate the findings array under either known spelling.

    ``2-report.json`` uses ``detailed_observations``; ``report_structure.json``
    specifies ``vulnerabilities``. Field drift across vendors is the norm, which
    is what the canonical mapping in normalize.py exists to absorb.
    """
    for key in ("detailed_observations", "vulnerabilities", "findings"):
        value = data.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    return []


def _summary_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    summary = data.get("executive_summary")
    if not isinstance(summary, dict):
        return []
    for key in ("summary_table", "summary_findings"):
        value = summary.get(key)
        if isinstance(value, list):
            return [v for v in value if isinstance(v, dict)]
    return []


def _finding_id(row: dict[str, Any]) -> str:
    for key in ("finding_id", "id", "sr_no", "observation_id"):
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _title(row: dict[str, Any]) -> str:
    for key in ("title", "vulnerability_title", "finding_title", "name"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def validate_report(data: dict[str, Any], report_id: str) -> ValidationResult:
    """Run every structural check over one extracted report."""
    result = ValidationResult(report_id=report_id)
    findings = _findings_list(data)
    summary_rows = _summary_rows(data)
    summary = data.get("executive_summary") if isinstance(data.get("executive_summary"), dict) else {}

    _check_findings_present(result, data, findings)
    _check_counts_agree(result, summary, findings)
    _check_id_sets_match(result, summary_rows, findings)
    _check_title_fields(result, summary_rows, findings)
    _check_identifier_fields(result, findings)
    _check_severity_variation(result, findings)
    _check_titles_are_phrases(result, findings)
    _check_narrative_present(result, findings)
    _check_figures(result, data)
    _check_metadata(result, data)
    return result


def _check_findings_present(
    result: ValidationResult, data: dict[str, Any], findings: list[dict[str, Any]]
) -> None:
    if findings:
        return
    # An empty findings array is legitimate (a clean audit) but only if the
    # report says so. An absent key is an extraction failure.
    has_key = any(k in data for k in ("detailed_observations", "vulnerabilities", "findings"))
    if not has_key:
        result.add(
            "findings_array_present",
            Level.BLOCKING,
            "no findings array under any known key (detailed_observations / vulnerabilities / findings)",
        )
    else:
        result.add("findings_array_empty", Level.WARNING, "findings array is present but empty")


def _check_counts_agree(
    result: ValidationResult, summary: dict[str, Any], findings: list[dict[str, Any]]
) -> None:
    n = len(findings)
    declared = summary.get("total_findings", summary.get("total_vulnerabilities"))
    if isinstance(declared, int) and declared != n:
        result.add(
            "total_matches_array",
            Level.BLOCKING,
            f"executive summary declares {declared} findings but the array holds {n}",
            "executive_summary.total_findings",
        )

    counts = summary.get("severity_count")
    if isinstance(counts, dict):
        numeric = [v for v in counts.values() if isinstance(v, int)]
        if numeric and sum(numeric) != n:
            result.add(
                "severity_counts_sum",
                Level.BLOCKING,
                f"severity_count sums to {sum(numeric)} but the array holds {n} findings",
                "executive_summary.severity_count",
            )


def _check_id_sets_match(
    result: ValidationResult, summary_rows: list[dict[str, Any]], findings: list[dict[str, Any]]
) -> None:
    if not summary_rows or not findings:
        return
    summary_ids = {_finding_id(r) for r in summary_rows} - {""}
    detail_ids = {_finding_id(r) for r in findings} - {""}
    if not summary_ids or not detail_ids:
        return
    only_summary = sorted(summary_ids - detail_ids)
    only_detail = sorted(detail_ids - summary_ids)
    if only_summary or only_detail:
        result.add(
            "summary_detail_ids_match",
            Level.BLOCKING,
            f"finding ids disagree; summary-only={only_summary or '[]'}, detail-only={only_detail or '[]'}",
            "executive_summary.summary_table",
        )


def _check_title_fields(
    result: ValidationResult, summary_rows: list[dict[str, Any]], findings: list[dict[str, Any]]
) -> None:
    """Catch column misalignment: a URL or a figure token sitting in a title.

    This is the check that fires on the supplied sample. A title is a short
    human phrase; a URL in that position means the extractor shifted columns.
    """
    for label, rows in (("executive_summary.summary_table", summary_rows), ("findings", findings)):
        if not rows:
            continue
        urlish = [i for i, row in enumerate(rows) if URL_RE.search(_title(row) or "")]
        if not urlish:
            continue
        ratio = len(urlish) / len(rows)
        # A systematic shift affects most rows; one stray row is a data problem
        # in that row, not a structural extraction failure.
        level = Level.BLOCKING if ratio >= 0.5 else Level.WARNING
        result.add(
            "title_not_a_url",
            level,
            f"{len(urlish)} of {len(rows)} rows have a URL in the title field "
            f"({ratio:.0%}); likely column misalignment in extraction",
            label,
        )

    for row in findings:
        title = _title(row)
        if title and FIGURE_TOKEN_RE.search(title):
            result.add(
                "title_not_a_figure_token",
                Level.BLOCKING,
                f"finding {_finding_id(row) or '?'} has a figure placeholder in its title",
                "findings[].title",
            )


def _check_identifier_fields(result: ValidationResult, findings: list[dict[str, Any]]) -> None:
    """CWE/CVE fields must hold identifiers of the right shape, when populated."""
    for row in findings:
        fid = _finding_id(row) or "?"
        for key, pattern, label in (
            ("cwe", CWE_RE, "CWE-nnn"),
            ("cwe_ids", CWE_RE, "CWE-nnn"),
            ("cve_ids", CVE_RE, "CVE-yyyy-nnnn"),
        ):
            value = row.get(key)
            if not value:
                continue
            items = value if isinstance(value, list) else [value]
            bad = [str(v) for v in items if isinstance(v, str) and v.strip() and not pattern.search(v)]
            if bad:
                result.add(
                    "identifier_shape",
                    Level.WARNING,
                    f"finding {fid}: {key} holds {bad[:3]} which do not look like {label}",
                    f"findings[].{key}",
                )


#: Methodology phase names that appear as section headings. When these turn up
#: as finding titles the extractor has swallowed the methodology table into the
#: findings array -- observed in ``4-report.json``, rows OBS-01 to OBS-04.
PHASE_WORDS = {
    "initiation",
    "discovery",
    "validation",
    "reporting",
    "reconnaissance",
    "scoping",
    "planning",
    "execution",
    "closure",
    "retesting",
    "revalidation",
}


def _check_severity_variation(result: ValidationResult, findings: list[dict[str, Any]]) -> None:
    """A single severity across many findings is a constant-fill artefact.

    Real audits produce a spread. ``4-report.json`` reports High for all 15
    findings, which would make any severity-filtered query meaningless.
    """
    if len(findings) < 5:
        return
    severities = {str(_first_value(row, ("severity", "risk_rating", "criticality"))).strip().lower() for row in findings}
    severities.discard("")
    severities.discard("none")
    if len(severities) == 1:
        result.add(
            "severity_variation",
            Level.WARNING,
            f"all {len(findings)} findings share severity '{next(iter(severities))}'; "
            "likely a constant-fill extraction artefact, severity filters will be unreliable",
            "findings[].severity",
        )


def _check_titles_are_phrases(result: ValidationResult, findings: list[dict[str, Any]]) -> None:
    """Finding titles should be short noun phrases, not prose or phase names."""
    if not findings:
        return
    phase_hits = [
        _finding_id(row) or "?"
        for row in findings
        if _title(row).strip().lower().rstrip(".") in PHASE_WORDS
    ]
    if phase_hits:
        result.add(
            "title_not_a_methodology_phase",
            Level.BLOCKING,
            f"findings {phase_hits[:5]} are titled with methodology phase names; "
            "the methodology table was likely parsed into the findings array",
            "findings[].title",
        )

    sentence_like = [
        _finding_id(row) or "?"
        for row in findings
        if len(_title(row)) > 120 or (_title(row).endswith(".") and len(_title(row).split()) > 12)
    ]
    if len(sentence_like) >= max(2, len(findings) // 4):
        result.add(
            "title_is_a_phrase",
            Level.WARNING,
            f"{len(sentence_like)} of {len(findings)} titles read as sentences rather than labels; "
            "check for cell-boundary loss in extraction",
            "findings[].title",
        )


def _first_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if row.get(key) not in (None, "", [], {}):
            return row[key]
    return ""


def _check_narrative_present(result: ValidationResult, findings: list[dict[str, Any]]) -> None:
    """The narrative fields are what retrieval actually searches.

    A findings array whose prose is empty indexes to nothing, so the chatbot
    would silently have no evidence to answer from.
    """
    narrative_keys = (
        "observation_description",
        "detailed_observation",
        "description",
        "impact",
        "recommendation",
    )
    empty = 0
    for row in findings:
        if not any(str(row.get(k, "")).strip() for k in narrative_keys):
            empty += 1
    if findings and empty == len(findings):
        result.add(
            "narrative_fields_present",
            Level.BLOCKING,
            f"all {len(findings)} findings have empty narrative fields; nothing to retrieve over",
            "findings[]",
        )
    elif empty:
        result.add(
            "narrative_fields_present",
            Level.WARNING,
            f"{empty} of {len(findings)} findings have no narrative text",
            "findings[]",
        )


def _check_figures(result: ValidationResult, data: dict[str, Any]) -> None:
    """Figure placeholders are reported, never silently dropped.

    The previous pipeline deleted these tokens during parsing. A proof-of-concept
    step list that is nothing but placeholders means the evidence for that
    finding is images the system does not hold (D07).
    """
    tokens = FIGURE_TOKEN_RE.findall(_as_text(data))
    if not tokens:
        return
    result.add(
        "figures_referenced",
        Level.WARNING,
        f"{len(tokens)} figure placeholders referenced; image assets must be staged separately",
        "proof_of_concept_steps",
    )


def _check_metadata(result: ValidationResult, data: dict[str, Any]) -> None:
    meta = data.get("report_meta") if isinstance(data.get("report_meta"), dict) else {}
    control = data.get("document_control") if isinstance(data.get("document_control"), dict) else {}
    title = str(meta.get("report_title") or control.get("document_title") or "").strip()
    if not title:
        result.add("report_title_present", Level.WARNING, "no report title found", "report_meta.report_title")
    classification = str(meta.get("classification") or "").strip()
    if not classification:
        result.add(
            "classification_present",
            Level.WARNING,
            "no classification marking; defaulting to the most restrictive handling",
            "report_meta.classification",
        )


def _as_text(value: Any) -> str:
    import json

    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return str(value)
