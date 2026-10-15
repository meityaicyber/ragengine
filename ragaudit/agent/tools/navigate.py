"""Navigation tools: how the agent orients before it digs.

These answer "what is here and what can I ask of it". Crucially, none of them
return chunk-addressable prose, so none registers evidence — a report title or
a field name is not a quotable source passage.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from ragaudit.agent.coerce import resolve_report_id
from ragaudit.agent.registry import ToolContext, ToolGroup, ToolRegistry, ToolResult


def _list_reports(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    reports = []
    for rid in ctx.catalog.report_ids():
        meta = ctx.catalog.get_report_meta(rid)
        reports.append({
            "report_id": rid,
            "title": meta.title if meta else "",
            "classification": meta.classification if meta else "",
            "report_type": meta.report_type if meta else "",
        })
    return ToolResult(content={"reports": reports, "count": len(reports)})


def _find_report(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Resolve a report named in words to its exact id. Closes the scope gap."""
    rid = resolve_report_id(args.get("query", ""), ctx.catalog)
    meta = ctx.catalog.get_report_meta(rid)
    return ToolResult(content={
        "report_id": rid,
        "title": meta.title if meta else "",
        "classification": meta.classification if meta else "",
    })


def _get_report_outline(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    rid = resolve_report_id(args.get("report_id", ""), ctx.catalog)
    meta = ctx.catalog.get_report_meta(rid)
    findings = ctx.catalog.findings(report_ids=[rid])
    sections: dict[str, int] = {}
    for row in ctx.catalog.all_chunks(report_ids=[rid]):
        sections[row["section"]] = sections.get(row["section"], 0) + 1
    return ToolResult(content={
        "report_id": rid,
        "title": meta.title if meta else "",
        "finding_count": len(findings),
        "finding_ids": [f.finding_id for f in findings],
        "severity_breakdown": ctx.catalog.severity_breakdown([rid]),
        "status_breakdown": ctx.catalog.status_breakdown([rid]),
        "sections": sections,
    })


def _corpus_stats(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    return ToolResult(content=ctx.catalog.stats())


def _describe_schema(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """What fields exist and which are actually populated.

    Without this the agent cannot know what it is able to answer, so it
    invents. Population is reported per field so the agent can decline a
    temporal or per-organisation question the corpus cannot support, rather
    than approximate (both sample reports have an empty ``organisation``).
    """
    findings = ctx.catalog.findings()
    total = len(findings) or 1

    def populated(predicate) -> dict[str, Any]:
        n = sum(1 for f in findings if predicate(f))
        return {"populated": n, "of": len(findings), "fully": n == len(findings)}

    reports = [ctx.catalog.get_report_meta(r) for r in ctx.catalog.report_ids()]
    reports = [asdict(m) for m in reports if m]

    def report_field(key) -> dict[str, Any]:
        n = sum(1 for m in reports if str(m.get(key, "")).strip())
        return {"populated": n, "of": len(reports)}

    return ToolResult(content={
        "finding_fields": {
            "severity": populated(lambda f: f.severity.value != "Unknown"),
            "status": populated(lambda f: f.status.value != "Unknown"),
            "cvss_score": populated(lambda f: f.cvss_score is not None),
            "cwe_ids": populated(lambda f: bool(f.cwe_ids)),
            "cve_ids": populated(lambda f: bool(f.cve_ids)),
            "observation": populated(lambda f: bool(f.observation)),
            "recommendation": populated(lambda f: bool(f.recommendation)),
            "affected_assets": populated(lambda f: bool(f.affected_assets)),
        },
        "report_fields": {
            "organisation": report_field("organisation"),
            "release_date": report_field("release_date"),
            "auditor": report_field("auditor"),
            "standards": report_field("standards"),
        },
        "note": "A field with populated < of cannot answer questions for the unpopulated rows. "
                "Do not infer an unpopulated field from context.",
    })


def register(registry: ToolRegistry) -> None:
    registry.add(
        "list_reports",
        "List every indexed audit report with its id, title and classification. "
        "Call this first when the user names a report in words.",
        {"type": "object", "properties": {}},
        _list_reports,
        group=ToolGroup.NAVIGATE,
    )
    registry.add(
        "find_report",
        "Resolve a report named in words (a title fragment, or a partial id) to its "
        "exact report_id. Use the returned report_id to scope other tools.",
        {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "report name, title fragment, or id"}},
            "required": ["query"],
        },
        _find_report,
        group=ToolGroup.NAVIGATE,
    )
    registry.add(
        "get_report_outline",
        "Overview of one report: finding ids, severity and status breakdowns, and sections. "
        "Use before diving in, to see what a report contains.",
        {
            "type": "object",
            "properties": {"report_id": {"type": "string"}},
            "required": ["report_id"],
        },
        _get_report_outline,
        group=ToolGroup.NAVIGATE,
    )
    registry.add(
        "corpus_stats",
        "Counts across the whole corpus: reports, findings, chunks, figures.",
        {"type": "object", "properties": {}},
        _corpus_stats,
        group=ToolGroup.NAVIGATE,
    )
    registry.add(
        "describe_schema",
        "What fields exist and which are populated. Call this when unsure whether the "
        "data can answer a question, especially temporal or per-organisation questions.",
        {"type": "object", "properties": {}},
        _describe_schema,
        group=ToolGroup.NAVIGATE,
    )
