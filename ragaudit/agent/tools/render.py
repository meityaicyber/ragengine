"""Render tools: presentation by reference, never model-typed data.

The model chooses *how* to present; the server owns *what* the data is. A
render tool takes ``from_step`` and column names, and the server pulls the rows
from that step's recorded result. The model never retypes a number, so a chart
cannot carry a value the model invented — the failure mode a data-carrying
show_chart would open.

Auto-blocks already cover the common case (aggregate_findings emits its own
chart), so these are for re-presenting an existing result a different way.
"""

from __future__ import annotations

from typing import Any

from ragaudit.agent.coerce import as_str
from ragaudit.agent.registry import ToolContext, ToolError, ToolGroup, ToolRegistry, ToolResult


def _step_content(ctx: ToolContext, args: dict[str, Any]) -> dict[str, Any]:
    step = args.get("from_step")
    if step is None:
        raise ToolError("from_step is required: reference the step whose data to render")
    content = ctx.step_results.get(int(step)) if str(step).lstrip("-").isdigit() else None
    if content is None:
        raise ToolError(f"no result recorded for step {step}")
    return content


def _show_chart(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    content = _step_content(ctx, args)
    chart = as_str(args.get("chart") or "bar")
    counts = content.get("counts") or content.get("values")
    if not isinstance(counts, dict) or not counts:
        raise ToolError("that step has no {label: number} data to chart; "
                        "use aggregate_findings or distinct_values first")
    x = [str(k) for k in counts.keys()]
    values = list(counts.values())
    block = {"kind": "chart", "chart": chart, "title": as_str(args.get("title")) or "Chart",
             "x": x, "series": [{"name": "count", "values": values}]}
    return ToolResult(content={"rendered": "chart", "points": len(x)}, blocks=[block])


def _show_table(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    content = _step_content(ctx, args)
    rows_data = content.get("findings") or content.get("reports") or content.get("repeat_findings")
    if not isinstance(rows_data, list) or not rows_data:
        raise ToolError("that step has no row list to tabulate")
    columns = [as_str(c) for c in (args.get("columns") or [])]
    if not columns:
        columns = [k for k in rows_data[0].keys() if not str(k).endswith("_excerpt")][:6]
    rows = [[row.get(c, "") for c in columns] for row in rows_data[:100]]
    block = {"kind": "table", "title": as_str(args.get("title")) or "Findings",
             "columns": columns, "rows": rows}
    return ToolResult(content={"rendered": "table", "rows": len(rows)}, blocks=[block])


def _show_note(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    text = as_str(args.get("text"))
    if not text:
        raise ToolError("note text is required")
    tone = as_str(args.get("tone") or "info")
    return ToolResult(content={"rendered": "note"},
                      blocks=[{"kind": "note", "tone": tone, "text": text}])


_FROM_STEP = {"from_step": {"type": "integer", "description": "the step whose recorded result to render"}}


def register(registry: ToolRegistry) -> None:
    registry.add(
        "show_chart",
        "Render a chart from an earlier step's data (e.g. an aggregate). You choose the chart "
        "type; the data comes from the step, so you never retype numbers.",
        {"type": "object", "properties": {
            **_FROM_STEP, "chart": {"type": "string", "enum": ["bar", "stacked_bar", "line", "donut"]},
            "title": {"type": "string"},
        }, "required": ["from_step"]},
        _show_chart,
        group=ToolGroup.RENDER,
    )
    registry.add(
        "show_table",
        "Render a table from an earlier step's row list (findings, reports). Pick columns by name; "
        "omit to use sensible defaults.",
        {"type": "object", "properties": {
            **_FROM_STEP, "columns": {"type": "array", "items": {"type": "string"}},
            "title": {"type": "string"},
        }, "required": ["from_step"]},
        _show_table,
        group=ToolGroup.RENDER,
    )
    registry.add(
        "show_note",
        "Add a short caveat or callout to the answer (e.g. a data limitation the user should know).",
        {"type": "object", "properties": {
            "text": {"type": "string"}, "tone": {"type": "string", "enum": ["info", "warning", "caution"]},
        }, "required": ["text"]},
        _show_note,
        group=ToolGroup.RENDER,
    )
