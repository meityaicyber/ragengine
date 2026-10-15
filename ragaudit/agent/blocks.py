"""Render blocks and their validator.

A block is a small JSON object the UI draws: a table, a chart, a metric tile,
a comparison, a note, a figure set. Tools emit them; the loop collects them;
the API ships them; the UI renders them.

Everything here is validated before it reaches the API, for two reasons that
are not optional:

- **No non-finite numbers.** ``json.dumps`` emits bare ``NaN``/``Infinity`` by
  default, which is invalid JSON and throws in the browser's parse. A chart
  value that came out non-finite is rejected at the source.
- **Every string is report-derived and report text is untrusted.** An
  unescaped table cell is stored XSS delivered by an ingested PDF. The
  validator does not escape (that is the UI's job, per-field), but it caps
  sizes and rejects structurally bad blocks so a malformed or hallucinated
  block never reaches the renderer.
"""

from __future__ import annotations

import math
from typing import Any

MAX_SERIES = 12
MAX_POINTS = 60
MAX_ROWS = 100
MAX_COLS = 12
MAX_LABEL = 200

CHART_KINDS = {"bar", "stacked_bar", "line", "donut"}
BLOCK_KINDS = {"table", "chart", "metric", "comparison", "note", "figures"}


class BlockError(ValueError):
    """A block that cannot be rendered safely as supplied."""


def _finite(value: Any, where: str) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError):
        raise BlockError(f"{where}: {value!r} is not a number") from None
    if not math.isfinite(n):
        raise BlockError(f"{where}: non-finite value ({value!r})")
    return n


def _label(value: Any) -> str:
    text = "" if value is None else str(value)
    return text[:MAX_LABEL]


def validate_block(block: dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned, render-safe copy, or raise BlockError.

    Numbers are coerced to finite floats; labels are stringified and capped;
    sizes are bounded. The UI still escapes every string on render — this is
    the structural gate, not the HTML-escaping one.
    """
    if not isinstance(block, dict):
        raise BlockError("block must be an object")
    kind = block.get("kind")
    if kind not in BLOCK_KINDS:
        raise BlockError(f"unknown block kind {kind!r}; allowed: {sorted(BLOCK_KINDS)}")

    if kind == "chart":
        return _validate_chart(block)
    if kind == "table":
        return _validate_table(block)
    if kind == "metric":
        return _validate_metric(block)
    if kind == "comparison":
        return _validate_comparison(block)
    if kind == "figures":
        return _validate_figures(block)
    return _validate_note(block)


def _validate_chart(block: dict[str, Any]) -> dict[str, Any]:
    chart = block.get("chart")
    if chart not in CHART_KINDS:
        raise BlockError(f"unknown chart type {chart!r}; allowed: {sorted(CHART_KINDS)}")
    x = [_label(v) for v in (block.get("x") or [])][:MAX_POINTS]
    if not x:
        raise BlockError("chart has no x values")
    series_in = block.get("series") or []
    if not series_in:
        raise BlockError("chart has no series")
    if len(series_in) > MAX_SERIES:
        raise BlockError(f"too many series ({len(series_in)} > {MAX_SERIES})")

    series = []
    for i, s in enumerate(series_in):
        values = [_finite(v, f"series[{i}]") for v in (s.get("values") or [])][:MAX_POINTS]
        if len(values) != len(x):
            raise BlockError(f"series[{i}] has {len(values)} values for {len(x)} x points")
        series.append({"name": _label(s.get("name", f"series {i + 1}")), "values": values})

    return {"kind": "chart", "chart": chart, "title": _label(block.get("title", "")),
            "x": x, "series": series}


def _validate_table(block: dict[str, Any]) -> dict[str, Any]:
    columns = [_label(c) for c in (block.get("columns") or [])][:MAX_COLS]
    if not columns:
        raise BlockError("table has no columns")
    rows_in = (block.get("rows") or [])[:MAX_ROWS]
    rows = [[_label(cell) for cell in row[:MAX_COLS]] for row in rows_in]
    return {"kind": "table", "title": _label(block.get("title", "")),
            "columns": columns, "rows": rows}


def _validate_metric(block: dict[str, Any]) -> dict[str, Any]:
    value = block.get("value")
    # A metric value may be a number or a short string ("PASS").
    if isinstance(value, (int, float)):
        value = _finite(value, "metric.value")
    else:
        value = _label(value)
    return {"kind": "metric", "label": _label(block.get("label", "")),
            "value": value, "detail": _label(block.get("detail", ""))}


def _validate_comparison(block: dict[str, Any]) -> dict[str, Any]:
    columns = [_label(c) for c in (block.get("columns") or [])][:MAX_COLS]
    if not columns:
        raise BlockError("comparison has no columns")
    rows = []
    for row in (block.get("rows") or [])[:MAX_ROWS]:
        values = row.get("values") or []
        clean = [(_finite(v, "comparison") if isinstance(v, (int, float)) else _label(v))
                 for v in values[:MAX_COLS]]
        rows.append({"label": _label(row.get("label", "")), "values": clean})
    return {"kind": "comparison", "title": _label(block.get("title", "")),
            "columns": columns, "rows": rows}


def _validate_figures(block: dict[str, Any]) -> dict[str, Any]:
    figures = []
    for fig in (block.get("figures") or [])[:MAX_ROWS]:
        figures.append({
            "token": _label(fig.get("token", "")),
            "report_id": _label(fig.get("report_id", "")),
            "available": bool(fig.get("available")),
            "url": _label(fig.get("url")) if fig.get("url") else None,
        })
    return {"kind": "figures", "title": _label(block.get("title", "")), "figures": figures}


def _validate_note(block: dict[str, Any]) -> dict[str, Any]:
    return {"kind": "note", "tone": _label(block.get("tone", "info")),
            "text": _label(block.get("text", ""))[:1000]}


def validate_blocks(blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate a list, dropping bad blocks and reporting why.

    One malformed block must not sink the others, so failures become flags
    rather than an exception — the answer is still useful without a chart.
    """
    clean: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, block in enumerate(blocks or []):
        try:
            clean.append(validate_block(block))
        except BlockError as exc:
            errors.append(f"block[{i}]: {exc}")
    return clean, errors
