"""Argument coercion and entity resolution for tools.

The model's arguments are suggestions, not commands. Captured qwen3:1.7b
output returns ``severity: "high"`` and ``report_id: "4"`` — lowercase, and a
bare number where the catalogue holds ``4-REPORT``. Coercing in the tool layer
means the model is never relied on for the step that matters most, and a
resolution it cannot make returns a *correctable* error naming the valid
options rather than a guess.
"""

from __future__ import annotations

from typing import Any

from ragaudit.agent.registry import ToolError
from ragaudit.domain.models import FindingStatus, Severity


def as_str(value: Any) -> str:
    return "" if value is None else str(value).strip()


def coerce_severity(value: Any) -> str | None:
    """Return a canonical severity value, or raise if it is unrecognisable."""
    if value in (None, ""):
        return None
    parsed = Severity.parse(value)
    if parsed is Severity.UNKNOWN and as_str(value).lower() != "unknown":
        valid = ", ".join(s.value for s in Severity if s is not Severity.UNKNOWN)
        raise ToolError(f"unknown severity '{value}'. Use one of: {valid}")
    return parsed.value


def coerce_status(value: Any) -> str | None:
    if value in (None, ""):
        return None
    parsed = FindingStatus.parse(value)
    if parsed is FindingStatus.UNKNOWN and as_str(value).lower() != "unknown":
        valid = ", ".join(s.value for s in FindingStatus if s is not FindingStatus.UNKNOWN)
        raise ToolError(f"unknown status '{value}'. Use one of: {valid}")
    return parsed.value


def coerce_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, list):
        return [as_str(v) for v in value if as_str(v)]
    return [as_str(value)]


#: Words the model uses to mean "no scope / every report". Treated as absence
#: of a filter, not as a report to resolve — the agent reaches for "all" or
#: "all reports" constantly and must not error on it.
_ALL_SENTINELS = {"all", "all reports", "every", "everything", "any", "*", "-", "none"}


def resolve_report_id(value: Any, catalog: Any) -> str:
    """Resolve a report reference to an exact id, or raise correctably.

    Accepts an exact id, a case-insensitive id, a bare suffix ("4" ->
    "4-REPORT"), or a unique title substring ("Tatva"). Ambiguity and absence
    both raise, naming the candidates so the model can choose.
    """
    wanted = as_str(value)
    if not wanted:
        raise ToolError("report id is required but was empty")

    ids = catalog.report_ids()
    if not ids:
        raise ToolError("no reports are indexed")

    # Exact, then case-insensitive.
    if wanted in ids:
        return wanted
    lowered = {rid.lower(): rid for rid in ids}
    if wanted.lower() in lowered:
        return lowered[wanted.lower()]

    # Suffix/prefix on the id, e.g. "4" -> "4-REPORT".
    id_matches = [
        rid for rid in ids
        if rid.lower().startswith(wanted.lower()) or rid.lower().split("-")[0] == wanted.lower()
    ]
    if len(id_matches) == 1:
        return id_matches[0]

    # Title substring.
    title_matches = []
    for rid in ids:
        meta = catalog.get_report_meta(rid)
        if meta and wanted.lower() in (meta.title or "").lower():
            title_matches.append(rid)
    if len(title_matches) == 1:
        return title_matches[0]

    candidates = id_matches or title_matches or ids
    labelled = []
    for rid in candidates[:8]:
        meta = catalog.get_report_meta(rid)
        labelled.append(f"{rid} ({meta.title})" if meta and meta.title else rid)
    if len(candidates) > 1:
        raise ToolError(f"'{value}' matches several reports; be specific: {'; '.join(labelled)}")
    raise ToolError(f"no report matches '{value}'. Known reports: {'; '.join(labelled)}")


def resolve_report_ids(value: Any, catalog: Any) -> list[str] | None:
    """Resolve a list of report references. None/empty/'all' means 'all reports'."""
    items = coerce_list(value)
    # A single "all"/"every"/etc means no scope, as does an empty list.
    items = [i for i in items if i.strip().lower() not in _ALL_SENTINELS]
    if not items:
        return None
    return [resolve_report_id(item, catalog) for item in items]


def scoped_report_ids(ctx: Any, args: dict[str, Any]) -> list[str] | None:
    """Report scope for a tool call, honouring the context's default.

    The model's explicit report_ids win; absent them, the context default
    applies (the UI selector, or a singular named report). This is what stops
    a "Tatva report" question from pulling 4-REPORT evidence when the small
    model forgets to scope its own calls.

    Note the one exception: an explicit 'all' from the model means the user
    widened on purpose, so the default is not re-imposed.
    """
    raw = args.get("report_ids") or args.get("report_id")
    if raw is not None and as_str(raw).lower() not in _ALL_SENTINELS:
        explicit = resolve_report_ids(raw, ctx.catalog)
        if explicit is not None:
            return explicit
        # raw was a list of sentinels -> explicit widening
        return None
    # No scope from the model: apply the context default (may be None = all).
    return getattr(ctx, "default_scope", None)


def bounded_limit(value: Any, *, default: int, maximum: int) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(n, maximum))
