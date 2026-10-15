"""Analysis tools: comparison, repeat detection, and safe arithmetic.

These are what the user asked for by name — compare reports, find findings
that repeat across audits, compute percentages. Unlocked via use_toolset so a
small model is not shown them until it needs them.
"""

from __future__ import annotations

from typing import Any

from ragaudit.agent.compute import ComputeError, compute
from ragaudit.agent.coerce import scoped_report_ids, as_str, resolve_report_ids
from ragaudit.agent.registry import ToolContext, ToolError, ToolGroup, ToolRegistry, ToolResult


def _compare_reports(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Side-by-side of two or more reports across structured dimensions."""
    report_ids = scoped_report_ids(ctx, args)
    if not report_ids or len(report_ids) < 2:
        all_ids = ctx.catalog.report_ids()
        if len(all_ids) < 2:
            raise ToolError("need at least two reports to compare")
        report_ids = all_ids

    columns = []
    for rid in report_ids:
        meta = ctx.catalog.get_report_meta(rid)
        findings = ctx.catalog.findings(report_ids=[rid])
        columns.append({
            "report_id": rid,
            "title": meta.title if meta else "",
            "total_findings": len(findings),
            "severity": ctx.catalog.severity_breakdown([rid]),
            "status": ctx.catalog.status_breakdown([rid]),
            "open": sum(1 for f in findings if f.status.value == "Open"),
        })

    x = [c["report_id"] for c in columns]
    return ToolResult(
        content={"reports": columns},
        blocks=[{
            "kind": "chart", "chart": "bar", "title": "Findings per report",
            "x": x, "series": [{"name": "findings", "values": [c["total_findings"] for c in columns]}],
        }, {
            "kind": "comparison", "title": "Report comparison",
            "columns": x,
            "rows": [
                {"label": "Total findings", "values": [c["total_findings"] for c in columns]},
                {"label": "Open", "values": [c["open"] for c in columns]},
            ],
        }],
    )


def _find_repeat_findings(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Findings recurring across reports, matched by CWE, CVE, or title.

    The is_repeat field and cross-report CWE/CVE overlap exist in the schema
    and were previously unused.
    """
    report_ids = scoped_report_ids(ctx, args)
    findings = ctx.catalog.findings(report_ids=report_ids)

    by_key: dict[tuple[str, str], list[dict[str, str]]] = {}
    for f in findings:
        keys: set[tuple[str, str]] = set()
        for cwe in f.cwe_ids:
            keys.add(("cwe", cwe.split(":")[0].strip().upper()))
        for cve in f.cve_ids:
            keys.add(("cve", cve.strip().upper()))
        if f.title:
            keys.add(("title", f.title.strip().lower()))
        for key in keys:
            by_key.setdefault(key, []).append({"report_id": f.report_id, "finding_id": f.finding_id})

    repeats = []
    for (kind, value), occurrences in by_key.items():
        reports = {o["report_id"] for o in occurrences}
        if len(reports) > 1:  # the same key in more than one report
            repeats.append({
                "match_on": kind, "value": value,
                "reports": sorted(reports), "occurrences": occurrences,
            })
    repeats.sort(key=lambda r: -len(r["reports"]))
    return ToolResult(content={"repeat_findings": repeats, "count": len(repeats)})


def _compute(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Deterministic arithmetic. Values come from step references or literals.

    ``from_steps`` resolves each entry against a recorded tool result so the
    model never retypes a number it might corrupt; ``values`` allows literals
    for things not in a prior result. The provenance of every input is kept.
    """
    operation = as_str(args.get("operation"))
    values: list[float] = []
    provenance: list[str] = []

    for ref in args.get("from_steps") or []:
        step = ref.get("step") if isinstance(ref, dict) else None
        key = ref.get("key") if isinstance(ref, dict) else None
        if step is None or key is None:
            raise ToolError("each from_steps entry needs {step, key}")
        content = ctx.step_results.get(int(step))
        if content is None:
            raise ToolError(f"no result recorded for step {step}")
        if key not in content:
            raise ToolError(f"step {step} result has no key '{key}'; keys: {list(content)[:8]}")
        values.append(content[key])
        provenance.append(f"step {step}.{key}={content[key]}")

    for literal in args.get("values") or []:
        values.append(literal)
        provenance.append(f"literal={literal}")

    try:
        result = compute(operation, values, provenance=provenance)
    except ComputeError as exc:
        raise ToolError(str(exc)) from exc

    return ToolResult(
        content=result.to_dict(),
        blocks=[{"kind": "metric", "label": f"{operation}", "value": result.result,
                 "detail": result.expression}],
    )


def register(registry: ToolRegistry) -> None:
    registry.add(
        "compare_reports",
        "Compare two or more reports side by side across finding counts, severity and status. "
        "Auto-renders a chart and a comparison table.",
        {"type": "object", "properties": {
            "report_ids": {"type": "array", "items": {"type": "string"},
                           "description": "reports to compare; omit for all"},
        }},
        _compare_reports,
        group=ToolGroup.ANALYSE,
    )
    registry.add(
        "find_repeat_findings",
        "Find findings that recur across reports, matched by CWE, CVE or title. "
        "Use for 'which findings repeat' or cross-audit trend questions.",
        {"type": "object", "properties": {
            "report_ids": {"type": "array", "items": {"type": "string"}},
        }},
        _find_repeat_findings,
        group=ToolGroup.ANALYSE,
    )
    registry.add(
        "compute",
        "Exact arithmetic (percentage, ratio, difference, sum, mean) over numbers from earlier "
        "tool results. Use this for any calculation; never do the arithmetic yourself. "
        "Reference values by step: from_steps=[{step, key}].",
        {"type": "object", "properties": {
            "operation": {"type": "string", "enum": ["percentage", "ratio", "difference", "sum", "mean"]},
            "from_steps": {"type": "array", "items": {"type": "object", "properties": {
                "step": {"type": "integer"}, "key": {"type": "string"}}}},
            "values": {"type": "array", "items": {"type": "number"}},
        }, "required": ["operation"]},
        _compute,
        group=ToolGroup.ANALYSE,
    )
