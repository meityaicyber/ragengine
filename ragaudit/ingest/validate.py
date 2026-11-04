"""Extraction validation gate.

Policy (revised after testing on the real 2-report/4-report extractions):
BLOCK a report only when the data answers are built from -- the detailed
findings -- is structurally corrupt (missing, URL/figure/phase-name titles,
no narrative text). When only the executive summary disagrees with the
details (totals, severity counts, id sets, a misaligned summary table), WARN:
the summary is never used to answer, so quarantining a report whose details
are clean throws away good evidence. 2-report.json is exactly that case: all
seven detailed findings are clean, only its summary table is misaligned.

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

# One source of truth for field spellings. The validator previously kept its
# own shorter alias lists, so a report using a spelling the normaliser accepts
# (e.g. ``observations``) was quarantined as having "no findings array".
from ragaudit.ingest.normalize import FIELD_ALIASES, FINDINGS_ARRAY_ALIASES, find_findings_array

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
    """Locate the findings array, exactly as the normaliser does.

    ``2-report.json`` uses ``detailed_observations``; ``report_structure.json``
    specifies ``vulnerabilities``. Field drift across vendors is the norm, which
    is what the canonical mapping in normalize.py exists to absorb.
    """
    return find_findings_array(data)[1]


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
    for key in FIELD_ALIASES["finding_id"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _title(row: dict[str, Any]) -> str:
    for key in FIELD_ALIASES["title"]:
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
    _check_duplicate_ids(result, findings)
    _check_severity_variation(result, findings)
    _check_titles_are_phrases(result, findings)
    _check_narrative_present(result, findings)
    _check_figures(result, data)
    _check_metadata(result, data)
    _check_closure_evidence(result, data, findings)
    _check_distinct_reviewer(result, data)
    return result


def _person(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("name", "")
    return str(value or "").strip().lower()


def _check_closure_evidence(result: ValidationResult, data: dict[str, Any], findings: list[dict[str, Any]]) -> None:
    """Standard Template §9/§11: a Closed or Partially Remediated finding needs
    a retest entry (Retest & Closure log, or a proof-of-concept entry marked as a
    retest); without one the closure is treated as unverified. Warning only."""
    from ragaudit.domain.models import FindingStatus

    appendices = data.get("appendices") if isinstance(data.get("appendices"), dict) else {}
    logged = {str(r.get("finding_id", "")).strip() for r in appendices.get("retest_closure") or []
              if isinstance(r, dict)}
    unverified = []
    for row in findings:
        status = FindingStatus.parse(_first_value(row, FIELD_ALIASES["status"]))
        if status not in (FindingStatus.CLOSED, FindingStatus.PARTIALLY_REMEDIATED):
            continue
        fid = _finding_id(row)
        poc = _first_value(row, FIELD_ALIASES["proof"])
        retest_poc = isinstance(poc, list) and any(
            isinstance(e, dict) and str(e.get("is_retest", "")).strip().lower() in ("true", "yes", "1")
            for e in poc)
        if fid not in logged and not retest_poc:
            unverified.append(fid or "?")
    if unverified:
        result.add(
            "closure_verified",
            Level.WARNING,
            f"{len(unverified)} finding(s) marked Closed/Partially Remediated have no retest evidence "
            f"(no Retest & Closure entry or retest proof-of-concept): {unverified[:8]}; "
            "per the standard template these closures are unverified",
            "appendices.retest_closure",
        )


def _check_distinct_reviewer(result: ValidationResult, data: dict[str, Any]) -> None:
    """Standard Template §2: the reviewer must be a different person from the preparer."""
    control = data.get("document_control") if isinstance(data.get("document_control"), dict) else {}
    prepared, reviewed = _person(control.get("prepared_by")), _person(control.get("reviewed_by"))
    if prepared and not reviewed:
        result.add("reviewer_present", Level.WARNING, "no reviewer named in document control",
                   "document_control.reviewed_by")
    elif prepared and reviewed and prepared == reviewed:
        result.add("reviewer_distinct", Level.WARNING,
                   "the report was reviewed by the same person who prepared it",
                   "document_control.reviewed_by")


def _check_findings_present(
    result: ValidationResult, data: dict[str, Any], findings: list[dict[str, Any]]
) -> None:
    if findings:
        return
    # An empty findings array is legitimate (a clean audit) but only if the
    # report says so. An absent key is an extraction failure.
    has_key = bool(find_findings_array(data)[0]) or any(k in data for k in FINDINGS_ARRAY_ALIASES)
    if not has_key:
        top = ", ".join(sorted(str(k) for k in data.keys())[:15])
        result.add(
            "findings_array_present",
            Level.BLOCKING,
            "no findings array under any known key ("
            + " / ".join(FINDINGS_ARRAY_ALIASES)
            + f"). Top-level keys present: {top}. If the findings live under another "
            "key, add it to FINDINGS_ARRAY_ALIASES in ragaudit/ingest/normalize.py",
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
            Level.WARNING,
            f"executive summary declares {declared} findings but the detailed list holds {n}; "
            "counts are taken from the detailed list",
            "executive_summary.total_findings",
        )

    counts = summary.get("severity_count")
    if isinstance(counts, dict):
        numeric = [v for v in counts.values() if isinstance(v, int)]
        if numeric and sum(numeric) != n:
            result.add(
                "severity_counts_sum",
                Level.WARNING,
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
            Level.WARNING,
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
        # The summary table is never used to answer, so a misaligned summary
        # is a warning; misaligned DETAIL titles are what gets blocked.
        level = Level.BLOCKING if ratio >= 0.5 and label == "findings" else Level.WARNING
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
            *((k, CWE_RE, "CWE-nnn") for k in FIELD_ALIASES["cwe_ids"]),
            *((k, CVE_RE, "CVE-yyyy-nnnn") for k in FIELD_ALIASES["cve_ids"]),
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


def _check_duplicate_ids(result: ValidationResult, findings: list[dict[str, Any]]) -> None:
    """Repeated finding ids inside one report.

    Previously this crashed the whole report at insert time (primary-key
    violation). Normalisation now disambiguates repeats as ``ID#2``; this
    warning makes the rename visible.
    """
    seen: dict[str, int] = {}
    for row in findings:
        fid = _finding_id(row)
        if fid:
            seen[fid] = seen.get(fid, 0) + 1
    dupes = sorted(k for k, n in seen.items() if n > 1)
    if dupes:
        result.add(
            "finding_ids_unique",
            Level.WARNING,
            f"finding ids repeat within the report: {dupes[:8]}; repeats are stored as ID#2, ID#3 ...",
            "findings[].finding_id",
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
    severities = {str(_first_value(row, FIELD_ALIASES["severity"])).strip().lower() for row in findings}
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
        FIELD_ALIASES["observation"] + FIELD_ALIASES["impact"] + FIELD_ALIASES["recommendation"]
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
