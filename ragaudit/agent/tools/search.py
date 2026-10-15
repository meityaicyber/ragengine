"""Narrative retrieval tools: the only chunk-addressable evidence path.

Every tool here registers what it returns in the evidence ledger, and
registers *exactly the text it serialises to the model* — not the fuller chunk
behind it. That is the rule that keeps citation verification honest: a quote
is checkable only against what the model was shown.

The structured tools deliberately do not do this. Their prose excerpts have no
chunk id and no offsets, so a quote from them is unlocatable. If quotable
finding prose is needed, it comes through ``get_finding_text`` here, where it
arrives as chunk-addressed spans.
"""

from __future__ import annotations

from typing import Any

from ragaudit.agent.coerce import scoped_report_ids, as_str, bounded_limit, resolve_report_ids
from ragaudit.agent.registry import ToolContext, ToolError, ToolGroup, ToolRegistry, ToolResult

#: How much of each chunk is shown to the model, and registered as citable.
#: The ledger stores exactly this slice, so a quote from beyond it cannot
#: verify — which is correct, the model never saw beyond it.
_SHOWN = 600


def _register_and_show(ctx: ToolContext, chunk, retriever: str) -> dict[str, Any]:
    shown = chunk.quotable_text[:_SHOWN]
    if ctx.ledger is not None:
        ctx.ledger.register(chunk, shown_text=shown, step=ctx.current_step, retriever=retriever)
    return {
        "chunk_id": chunk.chunk_id,
        "report_id": chunk.report_id,
        "finding_id": chunk.finding_id,
        "field": chunk.field_path,
        "text": shown,
        "truncated": len(chunk.quotable_text) > _SHOWN,
    }


def _search_reports(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    query = as_str(args.get("query"))
    if not query:
        raise ToolError("query is required")
    report_ids = scoped_report_ids(ctx, args)
    top_k = bounded_limit(args.get("top_k"), default=8, maximum=20)

    hits = ctx.retriever.retrieve(query, report_ids=report_ids, top_k=top_k)
    passages = [_register_and_show(ctx, h.chunk, h.retriever) for h in hits]
    return ToolResult(content={
        "query": query,
        "scope": report_ids or "all reports",
        "passages": passages,
        "count": len(passages),
    })


def _keyword_search(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Exact lexical lookup, for identifiers where semantic drift hurts.

    'CVE-2024-1234' or a precise error string should match literally, not by
    nearest embedding. Backed by the catalogue's FTS index.
    """
    query = as_str(args.get("query"))
    if not query:
        raise ToolError("query is required")
    report_ids = scoped_report_ids(ctx, args)
    limit = bounded_limit(args.get("limit"), default=10, maximum=25)

    hits = ctx.catalog.search_lexical(query, report_ids=report_ids, limit=limit)
    chunk_rows = ctx.catalog.get_chunks([cid for cid, _ in hits])
    passages = []
    for cid, score in hits:
        row = chunk_rows.get(cid)
        if row is None:
            continue
        chunk = _row_to_chunk(row)
        entry = _register_and_show(ctx, chunk, "lexical")
        entry["score"] = round(score, 4)
        passages.append(entry)
    return ToolResult(content={"query": query, "passages": passages, "count": len(passages)})


def _get_finding_text(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    """Full, quotable prose for one finding, as chunk-addressed spans.

    This is how narrative finding text becomes citable. It selects the chunks
    belonging to the finding and registers each, so quotes verify against real
    chunk ids rather than the unquotable excerpts the query tools return.
    """
    fid = as_str(args.get("finding_id"))
    if not fid:
        raise ToolError("finding_id is required")
    report_ids = scoped_report_ids(ctx, args)

    rows = ctx.catalog.all_chunks(report_ids=report_ids)
    matching = [r for r in rows if r["finding_id"] == fid]
    if not matching:
        raise ToolError(f"no indexed text for finding '{fid}'")

    passages = [_register_and_show(ctx, _row_to_chunk(r), "finding_text") for r in matching]
    return ToolResult(content={"finding_id": fid, "passages": passages, "count": len(passages)})


def _row_to_chunk(row):
    import json

    from ragaudit.domain.models import Chunk

    return Chunk(
        chunk_id=row["chunk_id"],
        report_id=row["report_id"],
        section=row["section"],
        field_path=row["field_path"],
        text=row["text"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        token_count=row["token_count"],
        finding_id=row["finding_id"],
        figure_tokens=json.loads(row["figures_json"]) if row["figures_json"] else [],
        context_prefix=row["context_prefix"],
    )


_SCOPE = {
    "report_id": {"type": "string"},
    "report_ids": {"type": "array", "items": {"type": "string"}},
}


def register(registry: ToolRegistry) -> None:
    from ragaudit.agent.registry import ToolCost

    registry.add(
        "search_reports",
        "Semantic search over report narrative (observations, impact, recommendations, scope). "
        "Returns quotable passages with chunk ids. Use for open-ended questions.",
        {"type": "object", "properties": {
            "query": {"type": "string"}, "top_k": {"type": "integer"}, **_SCOPE,
        }, "required": ["query"]},
        _search_reports,
        group=ToolGroup.SEARCH,
        cost=ToolCost.SEARCH,
    )
    registry.add(
        "keyword_search",
        "Exact keyword/identifier search (CVE, CWE, a precise phrase). Use when a semantic "
        "search would drift off an exact term.",
        {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}, **_SCOPE,
        }, "required": ["query"]},
        _keyword_search,
        group=ToolGroup.SEARCH,
    )
    registry.add(
        "get_finding_text",
        "Full quotable text of one finding as citable passages. Use this, not get_finding, "
        "when you need to quote a finding's observation or recommendation.",
        {"type": "object", "properties": {
            "finding_id": {"type": "string"}, "report_id": {"type": "string"},
        }, "required": ["finding_id"]},
        _get_finding_text,
        group=ToolGroup.SEARCH,
    )
