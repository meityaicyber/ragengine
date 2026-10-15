"""Structured query tools: exact answers from fields, no model involved.

These are the tools that keep numbers exact. A count comes from SQL, never
from the model reading a sample of rows. Every list-shaped result carries the
true ``total`` alongside the (bounded) ``returned`` rows, because a truncated
result without a true total invites "25 findings" when the answer is 213 — the
worst error class an audit assistant can produce.

None of these registers evidence. ``observation`` is surfaced only as a short,
clearly-labelled ``observation_excerpt`` that the system prompt forbids
quoting: it has no chunk id and no offsets, so a quote from it would be
unverifiable. Full, quotable finding prose comes from ``get_finding_text`` in
the search group.
"""

from __future__ import annotations

from typing import Any

from ragaudit.agent.coerce import (
    as_str,
    bounded_limit,
    coerce_severity,
    coerce_status,
    scoped_report_ids,
)
from ragaudit.agent.registry import ToolContext, ToolError, ToolGroup, ToolRegistry, ToolResult

_EXCERPT = 160


def _finding_brief(f) -> dict[str, Any]:
    excerpt = f.observation[:_EXCERPT].strip()
    if len(f.observation) > _EXCERPT:
        excerpt += "..."
    return {
        "report_id": f.report_id,
        "finding_id": f.finding_id,
        "title": f.title,
        "severity": f.severity.value,
        "status": f.status.value,
        "cwe_ids": f.cwe_ids,
        "cve_ids": f.cve_ids,
        "observation_excerpt": excerpt,  # NOT quotable; for the model's orientation only
    }


def _filters(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    severity = coerce_severity(args.get("severity"))
    status = coerce_status(args.get("status"))
    return {
        "report_ids": scoped_report_ids(ctx, args),
        "severities": [severity] if severity else None,
        "statuses": [status] if status else None,
        "cwe": as_str(args.get("cwe")) or None,
        "cve": as_str(args.get("cve")) or None,
    }


def _query_findings(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    filters = _filters(ctx, args)
    limit = bounded_limit(args.get("limit"), default=25, maximum=100)
    total = ctx.catalog.count_findings(**filters)
    rows = ctx.catalog.findings(**filters, limit=limit)
    return ToolResult(content={
        "findings": [_finding_brief(f) for f in rows],
        "returned": len(rows),
        "total": total,
        "truncated": total > len(rows),
    })


def _count_findings(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    filters = _filters(ctx, args)
    return ToolResult(content={"count": ctx.catalog.count_findings(**filters), "filters": _describe(filters)})


def _aggregate_findings(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """One- or two-dimensional breakdown. The result is already renderable."""
    by = as_str(args.get("by") or "severity").lower()
    second = as_str(args.get("then_by")).lower() or None
    report_ids = scoped_report_ids(ctx, args)

    valid = {"severity", "status", "report"}
    if by not in valid:
        raise ToolError(f"'by' must be one of {sorted(valid)}")
    if second and second not in valid:
        raise ToolError(f"'then_by' must be one of {sorted(valid)}")

    findings = ctx.catalog.findings(report_ids=report_ids)
    key = {"severity": lambda f: f.severity.value, "status": lambda f: f.status.value,
           "report": lambda f: f.report_id}

    if second is None:
        counts: dict[str, int] = {}
        for f in findings:
            counts[key[by](f)] = counts.get(key[by](f), 0) + 1
        return ToolResult(
            content={"dimension": by, "counts": counts, "total": len(findings)},
            blocks=[{"kind": "chart", "chart": "bar", "title": f"Findings by {by}",
                     "x": list(counts.keys()), "series": [{"name": by, "values": list(counts.values())}]}],
        )

    cross: dict[str, dict[str, int]] = {}
    second_keys: list[str] = []
    for f in findings:
        a, b = key[by](f), key[second](f)
        cross.setdefault(a, {})
        cross[a][b] = cross[a].get(b, 0) + 1
        if b not in second_keys:
            second_keys.append(b)
    x = list(cross.keys())
    series = [{"name": sk, "values": [cross[a].get(sk, 0) for a in x]} for sk in second_keys]
    return ToolResult(
        content={"rows": by, "columns": second, "cross_tab": cross, "total": len(findings)},
        blocks=[{"kind": "chart", "chart": "stacked_bar", "title": f"{by} by {second}",
                 "x": x, "series": series}],
    )


def _distinct_values(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    field = as_str(args.get("field") or "severity").lower()
    report_ids = scoped_report_ids(ctx, args)
    findings = ctx.catalog.findings(report_ids=report_ids)

    extract = {
        "severity": lambda f: [f.severity.value],
        "status": lambda f: [f.status.value],
        "cwe": lambda f: f.cwe_ids,
        "cve": lambda f: f.cve_ids,
        "asset": lambda f: f.affected_assets,
    }
    if field not in extract:
        raise ToolError(f"'field' must be one of {sorted(extract)}")

    counts: dict[str, int] = {}
    for f in findings:
        for v in extract[field](f):
            if v:
                counts[v] = counts.get(v, 0) + 1
    return ToolResult(content={"field": field, "values": dict(sorted(counts.items(), key=lambda kv: -kv[1]))})


def _top_values(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    result = _distinct_values(ctx, args)
    n = bounded_limit(args.get("n"), default=5, maximum=25)
    top = list(result.content["values"].items())[:n]
    return ToolResult(content={"field": result.content["field"], "top": [{"value": v, "count": c} for v, c in top]})


def _get_finding(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    fid = as_str(args.get("finding_id"))
    if not fid:
        raise ToolError("finding_id is required")
    report_ids = scoped_report_ids(ctx, args)
    matches = [f for f in ctx.catalog.findings(report_ids=report_ids) if f.finding_id == fid]
    if not matches:
        raise ToolError(f"no finding '{fid}'" + (f" in {report_ids}" if report_ids else " in any report"))
    if len(matches) > 1:
        reps = ", ".join(f.report_id for f in matches)
        raise ToolError(f"finding '{fid}' exists in several reports ({reps}); pass report_id to disambiguate")
    f = matches[0]
    brief = _finding_brief(f)
    brief.update({
        "impact_excerpt": (f.impact[:_EXCERPT] + "...") if len(f.impact) > _EXCERPT else f.impact,
        "recommendation_excerpt": (f.recommendation[:_EXCERPT] + "...") if len(f.recommendation) > _EXCERPT else f.recommendation,
        "cvss_score": f.cvss_score,
        "affected_assets": f.affected_assets,
        "has_figures": bool(f.figure_tokens),
        "note": "Excerpts are for orientation and are NOT quotable. "
                "Use get_finding_text to retrieve quotable, citable passages.",
    })
    return ToolResult(content={"finding": brief})


def _describe(filters: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in filters.items() if v}


_REPORT_SCOPE = {
    "report_id": {"type": "string", "description": "a report id or name to scope to"},
    "report_ids": {"type": "array", "items": {"type": "string"}, "description": "several reports to scope to"},
}
_FILTERS = {
    **_REPORT_SCOPE,
    "severity": {"type": "string", "enum": ["Critical", "High", "Medium", "Low", "Informational"]},
    "status": {"type": "string", "enum": ["Open", "Closed", "In Progress", "Risk Accepted"]},
    "cwe": {"type": "string", "description": "CWE id, e.g. CWE-284"},
    "cve": {"type": "string", "description": "CVE id"},
}


def register(registry: ToolRegistry) -> None:
    registry.add(
        "query_findings",
        "List findings matching filters. Returns a bounded page plus the true total count. "
        "For exact counts use count_findings instead.",
        {"type": "object", "properties": {**_FILTERS, "limit": {"type": "integer"}}},
        _query_findings,
        group=ToolGroup.QUERY,
    )
    registry.add(
        "count_findings",
        "Exact count of findings matching filters. Use this for any 'how many' question; "
        "never count from a query_findings page.",
        {"type": "object", "properties": _FILTERS},
        _count_findings,
        group=ToolGroup.QUERY,
    )
    registry.add(
        "aggregate_findings",
        "Break findings down by severity, status or report. Pass then_by for a two-dimensional "
        "cross-tab (e.g. by=severity, then_by=status). Auto-renders a chart.",
        {"type": "object", "properties": {
            "by": {"type": "string", "enum": ["severity", "status", "report"]},
            "then_by": {"type": "string", "enum": ["severity", "status", "report"]},
            **_REPORT_SCOPE,
        }},
        _aggregate_findings,
        group=ToolGroup.QUERY,
    )
    registry.add(
        "distinct_values",
        "The distinct values that actually occur for a field (severity, status, cwe, cve, asset), "
        "with how often each appears.",
        {"type": "object", "properties": {
            "field": {"type": "string", "enum": ["severity", "status", "cwe", "cve", "asset"]},
            **_REPORT_SCOPE,
        }, "required": ["field"]},
        _distinct_values,
        group=ToolGroup.QUERY,
    )
    registry.add(
        "top_values",
        "The most common values for a field, e.g. the most cited CWE or most affected asset.",
        {"type": "object", "properties": {
            "field": {"type": "string", "enum": ["severity", "status", "cwe", "cve", "asset"]},
            "n": {"type": "integer"}, **_REPORT_SCOPE,
        }, "required": ["field"]},
        _top_values,
        group=ToolGroup.QUERY,
    )
    registry.add(
        "get_finding",
        "Full detail for one finding by id: severity, status, CWE/CVE, assets, and short excerpts. "
        "Excerpts are NOT quotable; use get_finding_text for citable passages.",
        {"type": "object", "properties": {
            "finding_id": {"type": "string"}, "report_id": {"type": "string"},
        }, "required": ["finding_id"]},
        _get_finding,
        group=ToolGroup.QUERY,
    )
