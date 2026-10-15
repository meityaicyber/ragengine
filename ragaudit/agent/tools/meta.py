"""Meta tools: the agent explaining itself.

describe_capabilities answers "what can you do" from the live registry rather
than a hardcoded string, so it never drifts from the actual tool set.
"""

from __future__ import annotations

from typing import Any

from ragaudit.agent.registry import ToolContext, ToolGroup, ToolRegistry, ToolResult

# Set by build_registry so this tool can introspect its own registry.
_REGISTRY_REF: dict[str, Any] = {}


def _describe_capabilities(ctx: ToolContext, args: dict[str, Any]) -> ToolResult:
    registry = _REGISTRY_REF.get("registry")
    if registry is None:
        return ToolResult(content={"capabilities": []})
    groups: dict[str, list[dict[str, str]]] = {}
    for name in registry.names():
        tool = registry.get(name)
        groups.setdefault(tool.group.value, []).append({
            "name": name,
            "description": tool.description.split(".")[0] + ".",
        })
    return ToolResult(content={
        "groups": groups,
        "note": "Counts and lookups are exact (from report fields). Open-ended questions are "
                "answered from report text with citations. Unsupported data is declined, not guessed.",
    })


def register(registry: ToolRegistry) -> None:
    _REGISTRY_REF["registry"] = registry
    registry.add(
        "describe_capabilities",
        "List what this assistant can do, derived from its live tool set. Use when the user asks "
        "what you can do or what data is available.",
        {"type": "object", "properties": {}},
        _describe_capabilities,
        group=ToolGroup.META,
    )
