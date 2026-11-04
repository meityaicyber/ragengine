"""Small units: pooling fix, compute, blocks, router."""

from __future__ import annotations

import pytest

from ragaudit.agent.blocks import validate_blocks
from ragaudit.agent.compute import ComputeError, compute
from ragaudit.retrieve.router import Route, route_query


def test_last_token_pool_left_and_right_padding():
    torch = pytest.importorskip("torch")
    from ragaudit.retrieve.embedder import _last_token_pool

    hidden = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    # Left padded: real tokens are at the end -> last position for both rows.
    left = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    assert _last_token_pool(hidden, left, torch).flatten().tolist() == [3.0, 7.0]
    # Right padded: row 0 ends at index 1.
    right = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]])
    assert _last_token_pool(hidden, right, torch).flatten().tolist() == [1.0, 7.0]


def test_compute_whitelist():
    assert compute("percentage", [1, 4]).result == 25.0
    with pytest.raises(ComputeError):
        compute("percentage", [1, 0])
    with pytest.raises(ComputeError):
        compute("__import__", [1])


def test_blocks_drop_non_finite():
    clean, errors = validate_blocks([
        {"kind": "chart", "chart": "bar", "x": ["a"], "series": [{"name": "n", "values": [float("nan")]}]},
        {"kind": "note", "text": "ok"},
    ])
    assert len(clean) == 1 and len(errors) == 1


@pytest.mark.parametrize("q,route", [
    ("hi", Route.CHITCHAT),
    ("How many high-severity findings are open?", Route.STRUCTURED),
    ("Why were the high findings accepted?", Route.HYBRID),
    ("What access control weaknesses were identified?", Route.NARRATIVE),
])
def test_router(q, route):
    assert route_query(q).route is route


def test_show_chart_renders_cross_tab_and_top_values():
    from ragaudit.agent.registry import ToolContext
    from ragaudit.agent.tools.render import _show_chart

    ctx = ToolContext(catalog=None, retriever=None, settings=None)
    ctx.step_results[1] = {"rows": "severity", "columns": "status",
                           "cross_tab": {"High": {"Open": 1, "Closed": 2}, "Low": {"Open": 3}}}
    ctx.step_results[2] = {"field": "cwe", "top": [{"value": "CWE-284", "count": 3},
                                                   {"value": "CWE-89", "count": 1}]}
    stacked = _show_chart(ctx, {"from_step": 1}).blocks[0]
    assert stacked["chart"] == "stacked_bar" and stacked["x"] == ["High", "Low"]
    assert {s["name"]: s["values"] for s in stacked["series"]} == {"Open": [1, 3], "Closed": [2, 0]}
    bar = _show_chart(ctx, {"from_step": 2}).blocks[0]
    assert bar["x"] == ["CWE-284", "CWE-89"] and bar["series"][0]["values"] == [3, 1]
