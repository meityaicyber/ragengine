"""Tool registry: the contract every agent tool implements.

A tool is a typed, JSON-schema-described callable the model can invoke. The
registry owns three concerns the loop should not have to think about:

- **Schema.** Each tool advertises an Ollama-format function schema. The model
  sees only the schemas of the groups currently exposed (progressive
  disclosure), so a small model is not drowned in 29 options at once.
- **Dispatch.** A tool call is matched by name, its arguments validated and
  coerced, the handler run, and the result shaped into a JSON-safe dict. A
  tool that raises returns a correctable error to the model, never an
  exception to the loop — the model can fix a bad argument, the loop cannot
  catch its way out of a crashed step mid-plan.
- **Classification.** Each tool declares a group (for disclosure), a cost
  class (for per-tool caps), and whether it mutates (for the confirmation
  gate).

Tools never serialise their own output beyond returning a plain dict; the loop
owns truncation and the evidence ledger, because only it knows the remaining
context budget and what was actually shown.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolGroup(str, Enum):
    NAVIGATE = "navigate"
    QUERY = "query"
    SEARCH = "search"
    ANALYSE = "analyse"
    QUALITY = "quality"
    RENDER = "render"
    EXPORT = "export"
    META = "meta"


class ToolCost(str, Enum):
    """How expensive a call is, for per-run caps.

    ``search`` is the only genuinely heavy one — embed + lexical + RRF + a CPU
    cross-encoder rerank. Everything else is a SQL query or an in-memory op.
    """

    CHEAP = "cheap"
    SEARCH = "search"


class ToolError(Exception):
    """A correctable failure. The message is returned to the model verbatim.

    Raise this for anything the model could fix by calling again differently —
    an unknown report id, a bad enum, a missing argument. Infrastructure
    failures (a SQLite error, an embedder crash) are NOT this: those are left
    to propagate so the dispatcher can mark them non-correctable and the loop
    can short-circuit to an honest abstention.
    """


@dataclass
class ToolContext:
    """Everything a tool handler may touch.

    Passed to every handler so tools stay free functions rather than methods
    bound to a service. ``ledger`` and ``step_results`` are how a tool records
    shown evidence and how render tools reach prior results by reference.
    """

    catalog: Any
    retriever: Any
    settings: Any
    ledger: Any = None
    step_results: dict[int, Any] = field(default_factory=dict)
    current_step: int = 0
    resolved: dict[str, Any] = field(default_factory=dict)
    # Default report scope applied when a tool call names no report. Enforces
    # the UI scope selector and a singular named report ("the Tatva report")
    # across every tool, so evidence never silently leaks from another report.
    default_scope: list[str] | None = None


@dataclass
class ToolResult:
    """What a dispatched tool returns.

    ``content`` is the JSON-safe payload the model sees (after the loop's
    truncation). ``blocks`` are render blocks a tool emitted directly (an
    aggregate auto-renders). ``error`` set means the call failed correctably.
    """

    content: dict[str, Any]
    blocks: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]           # JSON schema for the arguments object
    handler: Callable[[ToolContext, dict[str, Any]], ToolResult]
    group: ToolGroup
    cost: ToolCost = ToolCost.CHEAP
    mutating: bool = False

    def schema(self) -> dict[str, Any]:
        """Ollama function-tool schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def add(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[[ToolContext, dict[str, Any]], ToolResult],
        *,
        group: ToolGroup,
        cost: ToolCost = ToolCost.CHEAP,
        mutating: bool = False,
    ) -> None:
        self.register(Tool(name, description, parameters, handler, group, cost, mutating))

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self, groups: set[ToolGroup] | None = None) -> list[dict[str, Any]]:
        """Schemas for the exposed groups, or all tools when ``groups`` is None."""
        tools = self._tools.values()
        if groups is not None:
            tools = [t for t in tools if t.group in groups]
        return [t.schema() for t in sorted(tools, key=lambda t: t.name)]

    def group_of(self, name: str) -> ToolGroup | None:
        tool = self._tools.get(name)
        return tool.group if tool else None

    def dispatch(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        """Run a tool by name. Correctable failures become error results.

        A ``ToolError`` (bad argument, unknown id) is returned to the model so
        it can retry. Any other exception propagates — the dispatcher wraps it
        as a non-correctable infrastructure error that the loop treats as a
        stop condition rather than something to retry forever.
        """
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                content={},
                error=f"unknown tool '{name}'. Available: {', '.join(self.names())}",
            )
        try:
            return tool.handler(ctx, arguments or {})
        except ToolError as exc:
            return ToolResult(content={}, error=str(exc))
