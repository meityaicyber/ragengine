"""Data-quality tools: surface the extraction gate inside the chat.

The validation gate already quarantines or flags reports at ingest. These make
that visible to a user — "which reports failed validation and why", "which
findings are missing CWEs" — turning an internal safeguard into an answerable
question.
"""

from __future__ import annotations

import json
from typing import Any

from ragaudit.agent.coerce import scoped_report_ids, resolve_report_ids
from ragaudit.agent.registry import ToolContext, ToolGroup, ToolRegistry, ToolResult


def _report_quality(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Validation status per report, from the stored validation_json."""
    rows = []
    conn = ctx.catalog._conn  # read-only access to the stored validation record
    for r in conn.execute("SELECT report_id, ingest_status, validation_json FROM reports ORDER BY report_id"):
        validation = json.loads(r["validation_json"]) if r["validation_json"] else {}
        rows.append({
            "report_id": r["report_id"],
            "status": r["ingest_status"],
            "ok": validation.get("ok", None),
            "blocking": validation.get("blocking", []),
            "warnings": validation.get("warnings", []),
        })
    quarantined = [row["report_id"] for row in rows if row["status"] == "quarantined"]
    return ToolResult(content={
        "reports": rows,
        "quarantined": quarantined,
        "any_blocking": any(row["blocking"] for row in rows),
    })


def _find_data_gaps(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Findings missing expected fields, corpus-wide or scoped."""
    report_ids = scoped_report_ids(ctx, args)
    findings = ctx.catalog.findings(report_ids=report_ids)

    missing_cwe = [{"report_id": f.report_id, "finding_id": f.finding_id}
                   for f in findings if not f.cwe_ids]
    missing_cvss = [{"report_id": f.report_id, "finding_id": f.finding_id}
                    for f in findings if f.cvss_score is None]
    missing_reco = [{"report_id": f.report_id, "finding_id": f.finding_id}
                    for f in findings if not f.recommendation]
    unknown_sev = [{"report_id": f.report_id, "finding_id": f.finding_id}
                   for f in findings if f.severity.value == "Unknown"]

    return ToolResult(content={
        "total_findings": len(findings),
        "missing_cwe": {"count": len(missing_cwe), "findings": missing_cwe[:25]},
        "missing_cvss": {"count": len(missing_cvss), "findings": missing_cvss[:25]},
        "missing_recommendation": {"count": len(missing_reco), "findings": missing_reco[:25]},
        "unknown_severity": {"count": len(unknown_sev), "findings": unknown_sev[:25]},
    })


def register(registry: ToolRegistry) -> None:
    registry.add(
        "report_quality",
        "Validation status of each report: which are quarantined and why, and what blocking "
        "or warning issues extraction found. Use for 'which reports failed validation'.",
        {"type": "object", "properties": {}},
        _report_quality,
        group=ToolGroup.QUALITY,
    )
    registry.add(
        "find_data_gaps",
        "Findings missing expected fields (CWE, CVSS, recommendation) or with unknown severity. "
        "Use for 'which findings are missing CWEs' or data-completeness questions.",
        {"type": "object", "properties": {
            "report_id": {"type": "string"},
            "report_ids": {"type": "array", "items": {"type": "string"}},
        }},
        _find_data_gaps,
        group=ToolGroup.QUALITY,
    )
