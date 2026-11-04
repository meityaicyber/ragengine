"""The agent loop: plan -> call -> observe -> finalize.

Runs the tool-calling loop against the model, accumulates an evidence ledger
and render blocks, and hands the result back for the existing citation-
verification path to finalise. The loop never builds the final Answer itself —
that stays in ChatService so ``_verify_citations`` and ``_build_answer`` are
reused unchanged.

Termination is layered, because ``max_steps`` alone is the weakest guard. In
priority order the loop stops on: a wall-clock deadline, a token budget, a
repeated identical call, no new evidence across several steps, a per-tool cap,
too many tool errors, or the step cap. All of them are necessary — with
temperature 0 and a fixed seed, a stuck model repeats a call byte-identically
forever, so duplicate suppression is not optional.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ragaudit.agent.evidence import EvidenceLedger
from ragaudit.agent.registry import ToolContext, ToolCost, ToolGroup, ToolRegistry
from ragaudit.generate.llm import Completion, LLMError, LLMUnavailable, OllamaClient, ToolCall
from ragaudit.generate.prompts import AGENT_SYSTEM_PROMPT, estimate_tokens

logger = logging.getLogger("ragaudit.agent.loop")

#: Groups the model always sees. The rest are unlocked via use_toolset, so a
#: small model is not asked to choose among 29 tools at once.
BASE_GROUPS = {ToolGroup.NAVIGATE, ToolGroup.QUERY, ToolGroup.SEARCH, ToolGroup.RENDER}


class StopReason(str, Enum):
    FINISHED = "finished"                 # model stopped calling tools
    MAX_STEPS = "agent_max_steps"
    DEADLINE = "agent_deadline"
    TOKEN_BUDGET = "agent_token_budget"
    NO_PROGRESS = "agent_no_progress"
    TOOL_ERRORS = "agent_tool_errors"
    LLM_UNAVAILABLE = "llm_unavailable"
    INFRA_ERROR = "agent_infrastructure_error"


@dataclass
class AgentConfig:
    max_steps: int = 10
    deadline_s: float = 90.0
    step_timeout_s: float = 45.0
    max_tool_errors: int = 4
    max_searches: int = 4
    no_progress_steps: int = 3
    step_max_tokens: int = 384
    max_exposed_tools: int = 24           # a larger model can be given all of them
    prompt_token_ceiling: int = 12000     # compact/stop before the server truncates


@dataclass
class StepRecord:
    step: int
    tool: str
    arguments: dict[str, Any]
    summary: str
    duration_s: float
    error: str | None = None


@dataclass
class AgentRun:
    """Everything the loop produced, for ChatService to finalise."""

    stop_reason: StopReason
    ledger: EvidenceLedger
    steps: list[StepRecord] = field(default_factory=list)
    blocks: list[dict[str, Any]] = field(default_factory=list)
    #: The tool that produced each entry of ``blocks`` (same order), so the
    #: finalize path can tell a chart the model asked for (show_chart) from
    #: one a query tool emitted automatically.
    block_sources: list[str] = field(default_factory=list)
    structured_summary: str = ""
    step_results: dict[int, Any] = field(default_factory=dict)
    exposed_groups: set[ToolGroup] = field(default_factory=set)
    infra_error: str = ""
    #: What the model said when it stopped calling tools. It is asked to say
    #: SHOW when its last result, shown as-is, answers the question, and WRITE
    #: when the answer has to be written (explained, described, combined).
    final_note: str = ""

    @property
    def model_chose_show(self) -> bool:
        """The model judged the tool result itself to be the answer."""
        note = self.final_note.upper()
        return "SHOW" in note and "WRITE" not in note

    @property
    def only_structured_tool(self) -> str | None:
        """Name of the single structured tool, if that is all that ran AND the
        model chose to show its result as the answer.

        Enables the deterministic shortcut: the reply is rendered from
        templates with no second model call — exact number, no finalize
        latency. Whether the result alone answers the question is the model's
        call (SHOW), not inferred from how many tools ran: "tell me in detail
        about the medium findings" may need only one query, but it needs a
        written answer, not the query's list.
        """
        if not self.model_chose_show:
            return None
        tool_steps = [s for s in self.steps if s.error is None]
        if len(tool_steps) != 1:
            return None
        name = tool_steps[0].tool
        return name if name in _SHORTCUTTABLE else None


#: Tools whose result can be rendered deterministically without a finalize call.
_SHORTCUTTABLE = {
    "count_findings", "aggregate_findings", "query_findings", "distinct_values", "top_values",
    # Corpus-level questions ("how many reports", "what fields exist", "which
    # reports failed validation"). Previously these tools ran but their result
    # reached neither a template nor the finalize prompt, so the answer was
    # always an abstention.
    "list_reports", "corpus_stats", "describe_schema", "get_report_outline",
    "report_quality", "find_data_gaps", "compare_reports", "find_repeat_findings",
    "describe_capabilities", "find_report",
}


class AgentLoop:
    def __init__(
        self,
        llm: OllamaClient,
        registry: ToolRegistry,
        config: AgentConfig | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.config = config or AgentConfig()

    def run(
        self,
        question: str,
        ctx: ToolContext,
        *,
        history: list[dict[str, Any]] | None = None,
        trace_id: str = "",
    ) -> AgentRun:
        ledger = ctx.ledger or EvidenceLedger()
        ctx.ledger = ledger
        run = AgentRun(stop_reason=StopReason.FINISHED, ledger=ledger, exposed_groups=set(BASE_GROUPS))

        messages: list[dict[str, Any]] = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": self._user_message(question, ctx)})

        deadline = time.monotonic() + self.config.deadline_s
        seen_signatures: set[str] = set()
        tool_errors = 0
        search_calls = 0
        steps_without_progress = 0

        for step in range(1, self.config.max_steps + 1):
            if time.monotonic() >= deadline:
                run.stop_reason = StopReason.DEADLINE
                break

            if estimate_tokens(_serialise(messages)) > self.config.prompt_token_ceiling:
                run.stop_reason = StopReason.TOKEN_BUDGET
                break

            remaining = max(1.0, deadline - time.monotonic())
            try:
                completion = self.llm.chat(
                    messages,
                    tools=self._exposed_schemas(run.exposed_groups),
                    max_tokens=self.config.step_max_tokens,
                    timeout_s=min(self.config.step_timeout_s, remaining),
                )
            except LLMUnavailable as exc:
                logger.warning("agent llm unavailable: %s", exc, extra={"trace_id": trace_id})
                run.stop_reason = StopReason.LLM_UNAVAILABLE
                break
            except LLMError as exc:
                logger.error("agent llm error: %s", exc, extra={"trace_id": trace_id})
                run.stop_reason = StopReason.INFRA_ERROR
                run.infra_error = str(exc)
                break

            if not completion.tool_calls:
                run.stop_reason = StopReason.FINISHED
                run.final_note = (completion.text or "").strip()
                break

            messages.append(_assistant_message(completion))
            progressed = False

            for call in completion.tool_calls:
                ctx.current_step = step

                if call.name == "use_toolset":
                    note = self._unlock(call, run)
                    messages.append(_tool_message(call, note))
                    continue

                signature = call.signature()
                if signature in seen_signatures:
                    # A stuck model repeats a call identically; do not re-run it.
                    messages.append(_tool_message(
                        call, {"note": "duplicate call; you already have this result. "
                                        "Use it or finish."}))
                    continue
                seen_signatures.add(signature)

                if self.registry.group_of(call.name) is None:
                    messages.append(_tool_message(call, {"error": f"unknown tool '{call.name}'"}))
                    tool_errors += 1
                    continue

                tool = self.registry.get(call.name)
                if tool.cost is ToolCost.SEARCH and search_calls >= self.config.max_searches:
                    messages.append(_tool_message(
                        call, {"error": "search limit reached for this question; "
                                        "answer from what you have."}))
                    continue

                record, result, infra = self._dispatch(step, call, ctx)
                run.steps.append(record)

                if infra is not None:
                    run.stop_reason = StopReason.INFRA_ERROR
                    run.infra_error = infra
                    return self._finish(run, ledger)

                if tool.cost is ToolCost.SEARCH:
                    search_calls += 1

                if result.error is not None:
                    tool_errors += 1
                else:
                    run.step_results[step] = result.content
                    ctx.step_results[step] = result.content
                    run.blocks.extend(result.blocks)
                    run.block_sources.extend([call.name] * len(result.blocks))
                    if _is_structured(call.name):
                        run.structured_summary = _summarise_structured(
                            run.structured_summary, call, result.content)
                    if _advanced(result, ledger, run):
                        progressed = True

                messages.append(_tool_message(call, _tool_payload(result)))

            if tool_errors >= self.config.max_tool_errors:
                run.stop_reason = StopReason.TOOL_ERRORS
                break

            steps_without_progress = 0 if progressed else steps_without_progress + 1
            if steps_without_progress >= self.config.no_progress_steps:
                run.stop_reason = StopReason.NO_PROGRESS
                break
        else:
            run.stop_reason = StopReason.MAX_STEPS

        return self._finish(run, ledger)

    # -- internals -------------------------------------------------------

    def _dispatch(self, step: int, call: ToolCall, ctx: ToolContext):
        started = time.monotonic()
        try:
            result = self.registry.dispatch(call.name, call.arguments, ctx)
        except Exception as exc:  # non-correctable: infrastructure, not a bad argument
            logger.exception("tool %s raised", call.name)
            record = StepRecord(step, call.name, call.arguments, "infrastructure error",
                                time.monotonic() - started, error=str(exc))
            return record, None, f"{call.name}: {exc}"

        duration = time.monotonic() - started
        summary = result.error if result.error else _result_summary(call.name, result.content)
        record = StepRecord(step, call.name, call.arguments, summary, duration, error=result.error)
        return record, result, None

    def _exposed_schemas(self, groups: set[ToolGroup]) -> list[dict[str, Any]]:
        schemas = self.registry.schemas(groups)
        # use_toolset is always available so locked groups can be unlocked.
        schemas.append(_USE_TOOLSET_SCHEMA)
        return schemas[: self.config.max_exposed_tools]

    def _unlock(self, call: ToolCall, run: AgentRun) -> dict[str, Any]:
        wanted = str(call.arguments.get("toolset", "")).strip().lower()
        try:
            group = ToolGroup(wanted)
        except ValueError:
            return {"error": f"unknown toolset '{wanted}'. "
                             f"Options: analyse, quality, meta, export."}
        run.exposed_groups.add(group)
        return {"note": f"{group.value} tools are now available."}

    def _finish(self, run: AgentRun, ledger: EvidenceLedger) -> AgentRun:
        run.ledger = ledger
        # Validate every block before it can reach the API: drop non-finite
        # numbers and structurally bad blocks (one bad block must not sink the
        # rest). The UI still escapes each string on render.
        from ragaudit.agent.blocks import BlockError, validate_block

        sources = run.block_sources + ["?"] * (len(run.blocks) - len(run.block_sources))
        clean, kept_sources, block_errors = [], [], []
        for i, (block, source) in enumerate(zip(run.blocks, sources)):
            try:
                clean.append(validate_block(block))
                kept_sources.append(source)
            except BlockError as exc:
                block_errors.append(f"block[{i}]: {exc}")
        run.blocks, run.block_sources = clean, kept_sources
        if block_errors:
            logger.warning("dropped %d malformed block(s): %s", len(block_errors), block_errors)
        return run

    def _user_message(self, question: str, ctx: ToolContext) -> str:
        resolved = ctx.resolved or {}
        hints = [f"{k}={v}" for k, v in resolved.items()]
        if getattr(ctx, "default_scope", None):
            if getattr(ctx, "scope_is_hard", False):
                hints.append(f"report scope={ctx.default_scope} (selected by the user; "
                             "every call is limited to it)")
            else:
                hints.append(f"default report scope={ctx.default_scope} (applies when a call names "
                             "no report; pass report_id 'all' or explicit ids if the question "
                             "needs other reports)")
        if hints:
            return f"{question}\n\n(resolved context: {'; '.join(hints)})"
        return question


# --------------------------------------------------------------------------
# message / result helpers
# --------------------------------------------------------------------------

_USE_TOOLSET_SCHEMA = {
    "type": "function",
    "function": {
        "name": "use_toolset",
        "description": "Unlock an additional group of tools: analyse (compare, repeats, compute), "
                       "quality (validation, data gaps), meta (capabilities), or export.",
        "parameters": {
            "type": "object",
            "properties": {"toolset": {"type": "string",
                                       "enum": ["analyse", "quality", "meta", "export"]}},
            "required": ["toolset"],
        },
    },
}

#: Tools whose results are NOT exact facts for the finalize prompt: search
#: tools deliver quotable passages through the evidence ledger instead, and
#: render tools only re-present earlier results.
_NOT_FACTS = {"search_reports", "keyword_search", "get_finding_text",
              "show_chart", "show_table", "show_note"}


def _is_structured(name: str) -> bool:
    """Every other tool returns exact data computed from the catalogue.

    This used to be a whitelist of eight query tools, so results from
    list_reports, describe_schema, report_quality and the other navigation /
    quality / analysis tools were silently dropped before the final answer.
    """
    return name not in _NOT_FACTS


def _advanced(result, ledger: EvidenceLedger, run: AgentRun) -> bool:
    """Did this step add anything new — evidence, a block, or a structured fact?"""
    before = getattr(run, "_ledger_mark", 0)
    now = len(ledger)
    run._ledger_mark = now  # type: ignore[attr-defined]
    return now > before or bool(result.blocks) or bool(result.content)


def _serialise(messages: list[dict[str, Any]]) -> str:
    import json
    return json.dumps(messages, default=str)


def _assistant_message(completion: Completion) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": completion.text,
        "tool_calls": [
            {"function": {"name": c.name, "arguments": c.arguments}} for c in completion.tool_calls
        ],
    }


def _tool_message(call: ToolCall, payload: dict[str, Any]) -> dict[str, Any]:
    import json
    # Newer Ollama reads ``tool_name``; older builds and proxies use ``name``.
    return {"role": "tool", "tool_name": call.name, "name": call.name,
            "content": json.dumps(payload, default=str)}


def _tool_payload(result) -> dict[str, Any]:
    if result.error is not None:
        return {"error": result.error}
    return result.content


def _result_summary(name: str, content: dict[str, Any]) -> str:
    if "counts" in content or "cross_tab" in content:
        return f"{name}: {content.get('total', '?')} findings grouped"
    if "count" in content:
        return f"{name}: count={content['count']}"
    if "total" in content and "returned" in content:
        return f"{name}: {content['returned']} of {content['total']}"
    if "passages" in content:
        return f"{name}: {content.get('count', len(content['passages']))} passages"
    if "report_id" in content:
        return f"{name}: {content['report_id']}"
    return name


#: Budget for the exact-results block handed to the finalize call.
_SUMMARY_LINE_CHARS = 2500
_SUMMARY_TOTAL_CHARS = 8000


def _compact(value: Any, depth: int = 0) -> Any:
    """Shrink a tool result for the prompt: drop non-quotable excerpts, cap lists."""
    if isinstance(value, dict):
        return {k: _compact(v, depth + 1) for k, v in value.items()
                if not str(k).endswith("_excerpt") and k != "note"}
    if isinstance(value, list):
        items = [_compact(v, depth + 1) for v in value[:40]]
        if len(value) > 40:
            items.append(f"... {len(value) - 40} more")
        return items
    return value


def _summarise_structured(existing: str, call: ToolCall, content: dict[str, Any]) -> str:
    import json
    body = json.dumps(_compact(content), default=str, ensure_ascii=False)
    if len(body) > _SUMMARY_LINE_CHARS:
        body = body[:_SUMMARY_LINE_CHARS] + " ...(truncated)"
    line = f"{call.name}({json.dumps(call.arguments, sort_keys=True)}) -> {body}"
    combined = f"{existing}\n{line}".strip()
    return combined[-_SUMMARY_TOTAL_CHARS:] if len(combined) > _SUMMARY_TOTAL_CHARS else combined
