"""The chatbot.

Orchestrates: resolve the question against conversation history, route it,
answer it from the structured store or from retrieved narrative, verify every
citation against the evidence actually supplied, and attach any figures bound
to the findings involved.

Citation verification is the part that matters (D06). A model that cites a
chunk_id it was never shown, or quotes text that does not appear in the
evidence, has fabricated its grounding. Both are checked here, and a failed
check downgrades the answer rather than being logged and ignored.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import time
import uuid
from collections.abc import Sequence

from ragaudit.chat.session import Session, SessionStore, Turn
from ragaudit.domain.models import Answer, Citation, FigureRef, RetrievedChunk
from ragaudit.generate.llm import LLMError, LLMUnavailable, OllamaClient
from ragaudit.generate.prompts import (
    AGENT_ANSWER_SCHEMA,
    AGENT_FINALIZE_SYSTEM_PROMPT,
    ANSWER_SCHEMA,
    REWRITE_SYSTEM,
    SYSTEM_PROMPT,
    build_agent_finalize_prompt,
    build_chat_prompt,
    build_rewrite_prompt,
    estimate_tokens,
    fit_evidence_to_budget,
)
from ragaudit.retrieve.hybrid import HybridRetriever
from ragaudit.retrieve.router import Route, route_query
from ragaudit.retrieve.structured import StructuredAnswerer
from ragaudit.store.catalog import Catalog

logger = logging.getLogger("ragaudit.chat")

ORDINALS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1,
}
ORDINAL_RE = re.compile(
    r"\b(" + "|".join(ORDINALS) + r")\s+(?:one|finding|observation|issue|item)\b", re.I
)
FOLLOWUP_RE = re.compile(
    r"\b(it|its|that|those|them|they|this|the (?:same|above|previous))\b", re.I
)

#: A quote counts as grounded at or above this similarity to the best-matching
#: window of evidence text. Set high: the model is asked to copy, so anything
#: much below an exact match means it reworded, and a reworded quote presented
#: as a quotation is a provenance failure.
QUOTE_MATCH_THRESHOLD = 0.82


class ChatService:
    def __init__(
        self,
        catalog: Catalog,
        retriever: HybridRetriever,
        llm: OllamaClient,
        *,
        sessions: SessionStore | None = None,
        selector=None,
        max_history_turns: int = 8,
        rewrite_queries: bool = True,
        icl_k: int = 3,
        generation_k: int = 8,
        agent=None,
        settings=None,
    ) -> None:
        self.catalog = catalog
        self.retriever = retriever
        self.llm = llm
        # "is not None", not "or": SessionStore defines __len__, so an EMPTY
        # shared store is falsy, and "or" silently gave every request a fresh
        # private store -- the web console never remembered a conversation.
        self.sessions = sessions if sessions is not None else SessionStore()
        self.structured = StructuredAnswerer(catalog)
        self.selector = selector
        self.max_history_turns = max_history_turns
        self.rewrite_queries = rewrite_queries
        self.icl_k = icl_k
        self.generation_k = generation_k
        # When an AgentLoop is injected, every non-chitchat question goes
        # through it. Without one, the old router path runs unchanged — the
        # kill switch and the way the test suite stays green.
        self.agent = agent
        self.settings = settings

    # -- public ----------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        session_id: str | None = None,
        report_ids: Sequence[str] | None = None,
    ) -> tuple[Answer, Session]:
        started = time.monotonic()
        trace_id = uuid.uuid4().hex[:12]
        session = self.sessions.get_or_create(
            session_id, list(report_ids) if report_ids else None
        )
        scope = list(report_ids) if report_ids is not None else session.report_scope

        search_query = self._resolve_query(question, session)
        plan = route_query(search_query)

        logger.info(
            "question routed",
            extra={
                "trace_id": trace_id,
                "route": plan.route.value,
                "reasons": plan.reasons,
                "rewritten": search_query != question,
            },
        )

        if plan.route is Route.CHITCHAT:
            # Greetings never reach the model, agent or not.
            answer = self._chitchat_answer(plan, scope)
        elif self.agent is None and _CORPUS_META_RE.fullmatch(_normalise_question(search_query)):
            # Without an agent, "how many reports are indexed?" is a catalogue
            # lookup. With one, the model decides (list_reports answers it
            # exactly through the deterministic template).
            answer = self._catalog_answer()
        elif self.agent is not None:
            answer = self._agent_answer(question, search_query, session, scope, trace_id)
            if _gathered_nothing(answer):
                # The model answered without using any tool (or every call
                # failed), so nothing it said can be grounded. Rather than
                # abstain, run the deterministic single-pass path, which
                # always consults the data.
                fallback = self._single_pass(question, search_query, plan, session, scope, trace_id)
                fallback.steps = answer.steps
                fallback.review_flags.append("agent_fallback:single_pass")
                answer = fallback
        else:
            answer = self._single_pass(question, search_query, plan, session, scope, trace_id)

        answer.trace_id = trace_id
        answer.latency_s = time.monotonic() - started
        self.sessions.append(session, Turn(question=question, answer=answer))
        return answer, session

    def _single_pass(self, question, search_query, plan, session, scope, trace_id) -> Answer:
        """The pre-agent path: SQL for field predicates, retrieval otherwise."""
        if plan.route is Route.STRUCTURED:
            return self.structured.answer(search_query, plan, report_ids=scope)
        answer = self._narrative_answer(question, search_query, plan, session, scope, trace_id)
        if plan.route is Route.HYBRID:
            answer = self._merge_structured(answer, search_query, plan, scope)
        answer.route = plan.route.value
        return answer

    def _catalog_answer(self) -> Answer:
        from ragaudit.agent.blocks import validate_blocks
        from ragaudit.agent.registry import ToolContext
        from ragaudit.agent.tools.navigate import _list_reports

        result = _list_reports(ToolContext(catalog=self.catalog, retriever=None, settings=self.settings), {})
        answer = Answer(
            text=_render_structured_text("list_reports", {}, result.content) or "",
            route="structured",
            review_flags=["catalog_lookup"],
        )
        answer.blocks, _ = validate_blocks(result.blocks)
        return answer

    # -- agent -----------------------------------------------------------

    def _agent_answer(self, question, search_query, session, scope, trace_id) -> Answer:
        """Run the agent loop, then finalise through the existing verifier.

        The loop gathers evidence into a ledger; the answer is produced either
        by a deterministic shortcut (one structured tool answered outright) or
        by one schema-constrained finalize call whose verification reuses
        ``_build_answer`` unchanged.

        ``search_query`` is the follow-up rewritten as a standalone question
        ("list them" -> "list the findings that cite CWE-284"). The agent works
        from it; previously it got only the literal follow-up, and a model that
        read "list them" as "list the reports" answered the wrong question.
        """
        from ragaudit.agent.evidence import EvidenceLedger
        from ragaudit.agent.loop import StopReason
        from ragaudit.agent.registry import ToolContext

        if search_query and search_query.strip() != question.strip():
            question = f'{search_query}\n(The user literally asked: "{question}")'
        resolved = _pre_resolve(question, self.catalog)
        ctx = ToolContext(
            catalog=self.catalog,
            retriever=self.retriever,
            settings=self.settings,
            ledger=EvidenceLedger(),
            resolved=resolved,
            default_scope=_default_scope(scope, resolved, question),
            scope_is_hard=bool(scope),
        )
        history = _agent_history(session, self.max_history_turns)
        run = self.agent.run(question, ctx, history=history, trace_id=trace_id)

        # Infrastructure failure or a dead backend: abstain, but keep whatever
        # exact structured results and blocks were gathered — they are true.
        if run.stop_reason in (StopReason.LLM_UNAVAILABLE, StopReason.INFRA_ERROR):
            return self._agent_failed(run, run.stop_reason)

        shortcut = self._shortcut_answer(run, question, ctx)
        if shortcut is not None:
            return shortcut

        return self._finalize_answer(question, run, ctx)

    def _shortcut_answer(self, run, question, ctx) -> Answer | None:
        """Deterministic reply when one structured tool answered outright.

        No finalize model call: the number is from SQL and the sentence is
        templated, exactly as the old structured route produced. Returns None
        when the run does not qualify, so the finalize path takes over.
        """
        tool = run.only_structured_tool
        if tool is None:
            return None
        step = next(s for s in run.steps if s.error is None)
        content = run.step_results.get(step.step, {})

        text = _render_structured_text(tool, step.arguments, content)
        if text is None:
            return None

        answer = Answer(text=text, route="structured", abstained=False)
        answer.blocks = list(run.blocks)
        answer.steps = _step_dicts(run.steps)
        answer.review_flags.append(f"agent_shortcut:{tool}")
        return answer

    def _finalize_answer(self, question, run, ctx) -> Answer:
        from ragaudit.agent.loop import StopReason

        topped_up = _top_up_finding_text(ctx, run)
        retrieved = run.ledger.as_retrieved()
        coverage = _coverage_note(run)
        if topped_up:
            coverage += f"; full text of the {topped_up} queried finding(s) included as passages"
        visuals = [_describe_block(b) for b in run.blocks]
        # Keep the finalize prompt inside the model's context window. Ollama
        # truncates an over-long prompt silently (from the front), losing the
        # system prompt; trimming the latest-gathered passages is the safe side.
        budget = getattr(self.llm, "prompt_budget_tokens", 8192)
        fixed = estimate_tokens(AGENT_FINALIZE_SYSTEM_PROMPT + question + (run.structured_summary or "")) + 300
        retrieved, dropped = fit_evidence_to_budget(retrieved, budget, reserve=fixed)
        if dropped:
            coverage += f"; {dropped} lower-priority passage(s) left out to fit the context window"

        # Budget-limited runs still try to answer from what was gathered,
        # labelled partial, rather than discarding the work.
        partial = run.stop_reason in (
            StopReason.MAX_STEPS, StopReason.DEADLINE, StopReason.TOKEN_BUDGET,
            StopReason.NO_PROGRESS, StopReason.TOOL_ERRORS,
        )

        prompt = build_agent_finalize_prompt(
            question, retrieved, structured_summary=run.structured_summary, coverage_note=coverage,
            visuals=visuals,
        )
        try:
            completion = self.llm.chat(
                [{"role": "system", "content": AGENT_FINALIZE_SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                format_schema=AGENT_ANSWER_SCHEMA,
            )
        except (LLMError, LLMUnavailable) as exc:
            logger.error("agent finalize failed: %s", exc)
            return self._agent_failed(run, StopReason.LLM_UNAVAILABLE)

        answer = self._build_answer(completion, retrieved, _AgentPlan())
        decisions = _parse_json(completion.text) or {}
        answer.blocks = _chosen_blocks(run, decisions.get("visuals"))
        answer.steps = _step_dicts(run.steps)
        answer.route = _effective_route(run)

        # Agent-path hardening: a free-formed narrative claim with zero verified
        # chunk citations is not grounded. The single-pass path only flags this;
        # after many tool steps it must force an abstention.
        #
        # Exact results excuse the missing citation only when they come from a
        # tool that computes data, and only for an answer that sticks to them.
        # Looking a report up (find_report, list_reports) is orientation, not
        # support: before, any such call let an uncited claim through. Whether
        # the answer goes beyond counts and lists into what the reports SAY is
        # the model's own declaration (describes_report_text), judged from the
        # question's meaning rather than matched on its wording. Missing or
        # malformed, it counts as true: the strict side.
        # The declaration only matters when report text was actually shown: with
        # no passages at all, the answer can only restate the exact results.
        describes_text = len(run.ledger) > 0 and decisions.get("describes_report_text", True) is not False
        if not answer.abstained and not answer.citations and (
            not _data_tools_used(run) or describes_text
        ):
            answer.abstained = True
            answer.abstention_reason = "no_verified_citation"
            answer.text = (
                "I gathered evidence but could not ground an answer in it. "
                + _coverage_sentence(run)
            )
            listing = _exact_listing(run)
            if listing:
                answer.text += "\n\nExact results from the report data:\n" + listing
            answer.review_flags.append("agent_forced_abstention")

        if partial:
            answer.review_flags.append(f"{run.stop_reason.value}:partial")
        return answer

    def _agent_failed(self, run, stop_reason) -> Answer:
        """Honest failure that keeps the exact results already gathered."""
        text = (
            "The language model backend is unavailable, so I could not compose a full answer."
            if stop_reason.value == "llm_unavailable"
            else "I hit an internal error while working on that."
        )
        if run.structured_summary or run.blocks:
            text += " " + _coverage_sentence(run) + " (These exact results are still valid.)"
        answer = Answer(
            text=text,
            abstained=True,
            abstention_reason=stop_reason.value,
            route=_effective_route(run),
        )
        answer.blocks = list(run.blocks)
        answer.steps = _step_dicts(run.steps)
        answer.review_flags.append(stop_reason.value)
        return answer

    # -- non-questions ---------------------------------------------------

    def _chitchat_answer(self, plan, scope: Sequence[str] | None) -> Answer:
        """Answer a greeting or a capability question without retrieving.

        Deliberately has no citations and no evidence: there is nothing in the
        reports that answers "hey", and presenting retrieved passages here
        would imply the reply was grounded in them.
        """
        kind = plan.chitchat_kind

        if kind in {"acknowledgement", "farewell"}:
            text = "Happy to help. Ask me anything else about the reports."
        else:
            text = self._orientation(scope)

        return Answer(
            text=text,
            route=Route.CHITCHAT.value,
            abstained=False,
            review_flags=[f"not_a_report_question:{kind}"],
        )

    def _orientation(self, scope: Sequence[str] | None) -> str:
        """Describe what is indexed and what can be asked of it."""
        try:
            # Only searchable reports: quarantined ones are never cited.
            report_ids = list(scope) if scope else self.catalog.report_ids(status="indexed")
            stats = self.catalog.stats()
        except Exception:  # orientation must never be the thing that breaks
            report_ids, stats = [], {}

        lines = [
            "I answer questions about the cybersecurity audit reports indexed here, "
            "and cite the passage behind every claim."
        ]

        if report_ids:
            titles = []
            for report_id in report_ids[:5]:
                meta = self.catalog.get_report_meta(report_id)
                titles.append(f"  - {report_id}: {meta.title}" if meta and meta.title else f"  - {report_id}")
            more = f"\n  ... and {len(report_ids) - 5} more" if len(report_ids) > 5 else ""
            quarantined = stats.get("reports_quarantined", 0)
            q_note = (f"\n  ({quarantined} more report(s) quarantined by the quality gate; "
                      "ask 'which reports failed validation' for reasons)") if quarantined and not scope else ""
            lines.append(
                f"\nIndexed ({stats.get('findings', 0)} findings across "
                f"{len(report_ids)} report(s)):\n" + "\n".join(titles) + more + q_note
            )

        lines.append(
            "\nThings you can ask:\n"
            "  - How many high-severity findings are still open?\n"
            "  - Are all findings closed?\n"
            "  - Which findings cite CWE-284?\n"
            "  - What access control weaknesses were identified?\n"
            "  - What remediation was recommended for the token API issue?\n"
            "  - Show me the proof of concept for a finding"
        )
        lines.append(
            "\nCounts and lookups are answered exactly from the report fields. "
            "Open-ended questions are answered from the report text, with citations. "
            "If the reports do not cover something, I will say so rather than guess."
        )
        return "\n".join(lines)

    # -- query resolution ------------------------------------------------

    def _resolve_query(self, question: str, session: Session) -> str:
        """Turn a follow-up into a standalone search query."""
        if not session.turns or not self.rewrite_queries:
            return question

        # With an agent, every follow-up goes through the model's rewrite: it
        # judges from the conversation whether the question depends on it
        # ("list them", "what about the second one?", "and the other report?")
        # and returns it unchanged when it does not. Without an agent, the old
        # cheap paths remain: ordinal lookup, and a rewrite only on a pronoun.
        if self.agent is None:
            if match := ORDINAL_RE.search(question):
                ids = session.last_finding_ids()
                position = ORDINALS[match.group(1).lower()]
                if ids and -len(ids) <= position < len(ids):
                    return f"{question} (referring to finding {ids[position]})"
            if not FOLLOWUP_RE.search(question):
                return question

        try:
            completion = self.llm.generate(
                build_rewrite_prompt(question, session.history_pairs(self.max_history_turns)),
                system=REWRITE_SYSTEM,
                max_tokens=128,
            )
            rewritten = completion.text.strip().strip('"')
            # A rewrite that collapses or explodes the question is a model
            # failure; fall back rather than search on nonsense.
            if rewritten and 3 <= len(rewritten) <= 400:
                return rewritten
        except (LLMError, LLMUnavailable) as exc:
            logger.warning("query rewrite failed, using original question: %s", exc)
        return question

    # -- narrative path --------------------------------------------------

    def _narrative_answer(
        self,
        question: str,
        search_query: str,
        plan,
        session: Session,
        scope: Sequence[str] | None,
        trace_id: str,
    ) -> Answer:
        retrieved = self.retriever.retrieve(search_query, report_ids=scope)
        if not retrieved:
            return Answer(
                text=(
                    "I could not find anything in the indexed reports that addresses that. "
                    "This may mean the reports do not cover it, or that the relevant report "
                    "is not in the current scope."
                ),
                abstained=True,
                abstention_reason="no_evidence_retrieved",
                review_flags=["no_evidence_retrieved"],
                route=plan.route.value,
            )

        demos = self.selector.select(search_query, self.icl_k) if self.selector else []
        history = session.history_pairs(self.max_history_turns)

        # Retrieval ran wide for recall; generation sees only the reranked head.
        # The full list stays on the answer for evaluation, so retrieval quality
        # is still measured at top_k even though the model reads fewer passages.
        for_generation = retrieved[: self.generation_k]

        # Fit the evidence to the model's context before sending. Without
        # this the server truncates the tail of the prompt -- which is where
        # the response-format instruction lives -- and returns malformed
        # output with no error anywhere (D03).
        budget = getattr(self.llm, "prompt_budget_tokens", 8192)
        fixed_cost = estimate_tokens(
            SYSTEM_PROMPT + question + "".join(d.render(0) for d in demos)
            + "".join(t for _, t in history)
        )
        for_generation, dropped = fit_evidence_to_budget(for_generation, budget, reserve=fixed_cost)
        if not for_generation:
            logger.error(
                "prompt budget exhausted before any evidence fit",
                extra={"trace_id": trace_id, "budget": budget, "fixed_cost": fixed_cost},
            )
            return Answer(
                text=(
                    "The evidence for that question does not fit in the model's context window. "
                    "Narrow the question, or configure a model with a larger context."
                ),
                abstained=True,
                abstention_reason="prompt_budget_exhausted",
                review_flags=["prompt_budget_exhausted"],
                route=plan.route.value,
            )

        prompt = build_chat_prompt(question, for_generation, demos, history)

        try:
            completion = self.llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                format_schema=ANSWER_SCHEMA,
            )
        except LLMUnavailable as exc:
            logger.error("llm unavailable: %s", exc, extra={"trace_id": trace_id})
            return Answer(
                text="The language model backend is unavailable, so I cannot answer right now.",
                abstained=True,
                abstention_reason="llm_unavailable",
                review_flags=["llm_unavailable"],
                retrieved=retrieved,
                route=plan.route.value,
            )
        except LLMError as exc:
            logger.error("llm error: %s", exc, extra={"trace_id": trace_id})
            return Answer(
                text="I hit an error generating an answer from the retrieved evidence.",
                abstained=True,
                abstention_reason="llm_error",
                review_flags=["llm_error"],
                retrieved=retrieved,
                route=plan.route.value,
            )

        # Citations are verified against what the model was actually shown.
        # Validating against the wider retrieved list would accept a citation
        # to a passage the model never saw, which is the fabrication this
        # check exists to catch.
        answer = self._build_answer(completion, for_generation, plan)
        # The full retrieved list is reported for evaluation, so retrieval
        # recall is still measured at top_k.
        answer.retrieved = retrieved
        if dropped:
            answer.review_flags.append(f"evidence_trimmed_to_context:{dropped}")
        return answer

    def _build_answer(self, completion, retrieved: list[RetrievedChunk], plan) -> Answer:
        parsed = _parse_json(completion.text)
        flags: list[str] = []

        if parsed is None:
            # Distinguish "ran out of room" from "produced nonsense". The first
            # is a configuration problem with an obvious fix; reporting both as
            # a parse failure hides that.
            truncated = getattr(completion, "truncated", False)
            reason = "truncated_generation" if truncated else "unparseable_model_output"
            text = (
                "The answer was cut off before it finished. Raise generation.max_tokens, "
                "or ask a narrower question."
                if truncated
                else "I could not produce a well-formed answer from the retrieved evidence."
            )
            logger.warning(
                "model output could not be parsed",
                extra={"reason": reason, "completion_tokens": completion.completion_tokens},
            )
            return Answer(
                text=text,
                abstained=True,
                abstention_reason=reason,
                review_flags=[reason],
                retrieved=retrieved,
                model=completion.model,
                route=plan.route.value,
            )

        answered = bool(parsed.get("answered", True))
        text = str(parsed.get("answer", "")).strip()
        if not text:
            answered = False
            text = "The evidence retrieved does not answer that question."
            flags.append("empty_answer_text")

        citations, citation_flags = self._verify_citations(parsed.get("citations", []), retrieved)
        flags.extend(citation_flags)

        # An answer asserting something with no verified citation behind it is
        # not grounded. Downgrade rather than present it as evidenced.
        if answered and not citations:
            flags.append("no_verified_citation")

        if _contains_injection(retrieved):
            flags.append("prompt_injection_like_text_in_evidence")

        figures = self._figures_for(retrieved, citations)

        return Answer(
            text=text,
            citations=citations,
            figures=figures,
            retrieved=retrieved,
            abstained=not answered,
            abstention_reason=str(parsed.get("missing_information", "")) if not answered else "",
            review_flags=flags,
            model=completion.model,
            route=plan.route.value,
        )

    # -- verification ----------------------------------------------------

    def _verify_citations(
        self, raw_citations, retrieved: list[RetrievedChunk]
    ) -> tuple[list[Citation], list[str]]:
        """Keep only citations that point at supplied evidence and quote it.

        Two independent failures are caught: a chunk_id the model was never
        shown (fabricated reference), and a quote that does not appear in the
        chunk it names (fabricated wording).
        """
        by_id = {item.chunk.chunk_id: item.chunk for item in retrieved}
        verified: list[Citation] = []
        flags: list[str] = []

        for entry in raw_citations or []:
            if not isinstance(entry, dict):
                continue
            chunk_id = str(entry.get("chunk_id", "")).strip()
            quote = str(entry.get("quote", "")).strip()
            if not chunk_id:
                continue

            chunk = by_id.get(chunk_id)
            if chunk is None:
                flags.append(f"citation_not_in_evidence:{chunk_id}")
                continue
            if quote and not _quote_supported(quote, chunk.quotable_text):
                flags.append(f"quote_not_in_chunk:{chunk_id}")
                continue

            verified.append(
                Citation(
                    chunk_id=chunk_id,
                    report_id=chunk.report_id,
                    field_path=chunk.field_path,
                    quote=quote,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    finding_id=chunk.finding_id,
                )
            )
        return verified, flags

    def _figures_for(
        self, retrieved: list[RetrievedChunk], citations: list[Citation]
    ) -> list[FigureRef]:
        """Attach figures bound to the findings actually cited.

        This is a join on ``finding_id``, not a visual search. The extractor
        already binds each image to its finding, so retrieving the finding and
        following the link is exact and costs nothing (D07).
        """
        # Only findings the answer actually cites. The old fallback attached
        # the top-3 retrieved findings' figures to answers with no citation,
        # so an abstention ("the reports do not say") showed unrelated PoC
        # screenshots.
        cited_findings = {c.finding_id for c in citations if c.finding_id}
        if not cited_findings:
            return []
        report_ids = {item.chunk.report_id for item in retrieved}

        figures: list[FigureRef] = []
        for report_id in report_ids:
            for finding_id in sorted(f for f in cited_findings if f):
                for row in self.catalog.figures_for_finding(report_id, finding_id):
                    figures.append(
                        FigureRef(
                            token=row["token"],
                            page=row["page"],
                            index=row["idx"],
                            report_id=row["report_id"],
                            asset_path=row["asset_path"],
                        )
                    )
        return figures

    # -- hybrid merge ----------------------------------------------------

    def _merge_structured(self, answer: Answer, question: str, plan, scope) -> Answer:
        """Prepend exact field results to a narrative answer.

        Where a question has both a field predicate and a judgement component,
        the counts come from SQL and the interpretation from the model, so the
        numbers are never the model's to invent.

        Only runs when there is an actual predicate to evaluate. A figure
        request routes hybrid without one, and merging there would prepend an
        unfiltered dump of every finding to an otherwise focused answer.
        """
        has_predicate = bool(
            plan.severities or plan.statuses or plan.cwe or plan.cve or plan.wants_count
        )
        if not has_predicate:
            return answer

        try:
            structured = self.structured.answer(question, plan, report_ids=scope)
        except Exception as exc:
            logger.warning("structured half of hybrid answer failed: %s", exc)
            return answer
        answer.text = f"{structured.text}\n\n{answer.text}"
        answer.citations = structured.citations + answer.citations
        answer.review_flags.extend(structured.review_flags)
        return answer


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

_INJECTION_PATTERNS = (
    r"ignore (?:all )?(?:previous|prior|above) instructions",
    r"disregard (?:the )?(?:system|previous)",
    r"you are now",
    r"system prompt",
    r"reveal (?:the )?(?:prompt|instructions)",
    r"<\s*/?\s*(?:system|assistant|tool)\s*>",
)


def _contains_injection(retrieved: Sequence[RetrievedChunk]) -> bool:
    blob = "\n".join(item.chunk.quotable_text for item in retrieved)
    return any(re.search(p, blob, re.I) for p in _INJECTION_PATTERNS)


def _parse_json(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


#: Characters that separate fields in a rendered chunk header. A model that
#: copies "[OBS-02] Discovery | Severity: High" back as "Discovery, Severity:
#: High" has quoted faithfully; only the punctuation differs. Comparing at
#: character level treats that as a rewrite, so separators are flattened
#: before comparison.
_SEPARATORS = str.maketrans({c: " " for c in "|,;:[]()\"'`*_-–—/\\"})


def _normalise_ws(text: str) -> str:
    return " ".join(text.lower().split())


def _tokens(text: str) -> list[str]:
    # Sentence punctuation is stripped from token EDGES only, not treated as a
    # separator: a model quoting a partial sentence ("...token endpoint") must
    # match a source ending "...token endpoint.", while an internal dot in
    # "CVE-2024-1234" or "3.5" stays intact.
    raw = _normalise_ws(text.translate(_SEPARATORS)).split()
    return [t.strip(".?!") for t in raw if t.strip(".?!")]


def _quote_supported(quote: str, source: str) -> bool:
    """Is ``quote`` actually present in ``source``?

    Compared as word sequences with separators flattened, so a faithful copy
    that differs only in punctuation passes while a quote lifted from a
    *different* chunk still fails. That distinction is the whole point: the
    common model failure here is not rewording, it is attributing one chunk's
    text to another chunk's id.
    """
    q_tokens = _tokens(quote)
    s_tokens = _tokens(source)
    if not q_tokens or not s_tokens:
        return False

    # Contiguous run of the same words, punctuation ignored.
    if _is_sublist(q_tokens, s_tokens):
        return True

    # Otherwise require most of the quote's words to align in order against
    # the source. Operating on tokens rather than characters means a changed
    # separator costs nothing while a substituted clause still fails.
    matcher = difflib.SequenceMatcher(None, q_tokens, s_tokens, autojunk=False)
    return matcher.ratio() * (len(q_tokens) + len(s_tokens)) / (2 * len(q_tokens)) >= (
        QUOTE_MATCH_THRESHOLD
    )


def _is_sublist(needle: list[str], haystack: list[str]) -> bool:
    if len(needle) > len(haystack):
        return False
    first = needle[0]
    for start in range(len(haystack) - len(needle) + 1):
        if haystack[start] == first and haystack[start : start + len(needle)] == needle:
            return True
    return False


# --------------------------------------------------------------------------
# agent helpers
# --------------------------------------------------------------------------

import re as _re  # noqa: E402

_OBS_RE = _re.compile(r"\bOBS[-\s]?(\d+)\b", _re.I)
_CWE_RE = _re.compile(r"\bCWE[-\s]?(\d+)\b", _re.I)
_CVE_RE = _re.compile(r"\b(CVE-\d{4}-\d{4,})\b", _re.I)
_IDLIKE_RE = _re.compile(r"\b[A-Za-z]{1,8}[-_]?\d{1,4}(?:#\d+)?\b")


class _AgentPlan:
    """Minimal plan stand-in so _build_answer's plan.route.value works.

    The agent computes the effective route after verification, so the value
    here is a placeholder that _finalize_answer overwrites.
    """

    class _R:
        value = "agent"

    route = _R()


def _pre_resolve(question: str, catalog) -> dict:
    """Deterministically resolve entities named in the question.

    Report names, finding ids and CWE/CVE are pulled out and resolved before
    the model sees the question, so the agent needs fewer turns and is never
    relied on for the resolution step that matters most.
    """
    resolved: dict = {}
    findings = {m.group(0).upper().replace(" ", "-") for m in _OBS_RE.finditer(question)}
    # Also match any id-shaped token against the finding ids actually stored,
    # so "VAPT-03", "V-12" or "F-007" resolve, not just OBS-nn.
    try:
        tokens = {t.upper() for t in _IDLIKE_RE.findall(question)}
        if tokens:
            known = {fid.upper(): fid for fid in catalog.distinct_finding_ids()}
            findings |= {known[t] for t in tokens if t in known}
    except Exception:
        pass
    if findings:
        resolved["finding_ids"] = sorted(findings)
    cwes = {f"CWE-{m.group(1)}" for m in _CWE_RE.finditer(question)}
    if cwes:
        resolved["cwe"] = sorted(cwes)
    cves = {m.group(1).upper() for m in _CVE_RE.finditer(question)}
    if cves:
        resolved["cve"] = sorted(cves)

    try:
        q_words = set(_re.findall(r"[a-z0-9]+", question.lower()))
        reports = []
        for rid, title in catalog.report_titles().items():
            title = title or ""
            # A report is "named" by a DISTINCTIVE title word. Common audit
            # vocabulary ("report", "security", "api") matches every title and
            # would name them all, defeating the scoping — so it is excluded.
            distinctive = [
                w.lower() for w in title.split()
                if len(w) >= 4 and w.lower() not in _TITLE_STOPWORDS
            ]
            if any(w in q_words for w in distinctive):
                reports.append(rid)
        if reports:
            resolved["report_ids"] = sorted(set(reports))
    except Exception:  # resolution is best-effort; never block the answer
        pass
    return resolved


#: Audit-domain words common to most report titles; matching on them would
#: "name" every report and defeat scoping.
#: Questions about the catalogue itself rather than report content. Matched
#: against the WHOLE (normalised) question, so "which reports have critical
#: findings" does not match and goes to the agent as usual.
_CORPUS_META_RE = _re.compile(
    r"(?:how many|number of|count of|count the|count)\s+(?:audit\s+|security\s+|vapt\s+|cert-?in\s+)?reports"
    r"(?:\s+(?:are|have been|were|do you have|are there|is there))?"
    r"(?:\s+(?:indexed|available|loaded|ingested|stored|there|uploaded|in total|in the (?:system|index|database|catalogue|catalog)))*"
    r"(?:\s+(?:do you have|in total))?"
    r"|(?:list|show|show me|which|what|what are)\s+(?:all\s+)?(?:the\s+)?(?:audit\s+|available\s+|indexed\s+|loaded\s+)?reports"
    r"(?:\s+(?:are|do you have|have you got))?(?:\s+(?:indexed|available|loaded|ingested|there|in the system))?"
    r"(?:\s+(?:do you have))?",
)


def _normalise_question(text: str) -> str:
    return " ".join(_re.sub(r"[?.!]+$", "", text.strip().lower()).split())


def _gathered_nothing(answer: Answer) -> bool:
    """True when an agent answer rests on no data at all.

    No successful tool call (the model answered from its own head, or every
    call errored) means no exact result and no evidence reached the answer.
    """
    if not answer.abstained:
        return False
    return not any(step.get("error") is None for step in answer.steps)


_TITLE_STOPWORDS = {
    "report", "reports", "security", "assessment", "audit", "testing", "test",
    "vapt", "api", "apis", "application", "web", "network", "mobile", "review",
    "penetration", "pentest", "cyber", "cybersecurity", "final", "draft", "the",
}


def _default_scope(ui_scope, resolved: dict, question: str) -> list[str] | None:
    """The report scope tools default to when a call names none.

    The UI selector is a hard choice and always wins. Otherwise a singular
    named report scopes the run, as a default only: the model is told the
    default and widens it (report_id 'all' or explicit ids) when the question
    needs other reports, such as a comparison. That judgement used to be a
    keyword list ("compare", "versus", ...); it is the model's now. Returns
    None (all reports) when nothing constrains it.
    """
    if ui_scope:
        return list(ui_scope)
    named = resolved.get("report_ids")
    if named and len(named) == 1:
        return named
    return None


def _agent_history(session, max_turns: int) -> list[dict]:
    """Recent turns as chat messages for the agent's context.

    Each answer carries the tool calls behind it, so a follow-up ("list them",
    "which of those are open?") can reuse the exact filters instead of
    guessing them back out of a templated sentence.
    """
    out: list[dict] = []
    for turn in session.turns[-max_turns:]:
        out.append({"role": "user", "content": turn.question})
        text = turn.answer.text
        text = text if len(text) <= 300 else text[:300] + "..."
        calls = [f"{s['tool']} {json.dumps(s['arguments'], sort_keys=True)}"
                 for s in (turn.answer.steps or []) if s.get("error") is None][:4]
        if calls:
            text += "\n[tools used: " + "; ".join(calls) + "]"
        out.append({"role": "assistant", "content": text})
    return out


def _step_dicts(steps) -> list[dict]:
    return [
        {"step": s.step, "tool": s.tool, "arguments": s.arguments,
         "summary": s.summary, "duration_s": round(s.duration_s, 3),
         "error": s.error}
        for s in steps
    ]


def _effective_route(run) -> str:
    """Report the lane the agent actually used, so eval's expected_route still
    means something and 'agent' is a value rather than a flattening."""
    tools = {s.tool for s in run.steps if s.error is None}
    searched = bool(tools & {"search_reports", "keyword_search", "get_finding_text"})
    structured = bool(tools & {"count_findings", "aggregate_findings", "query_findings",
                               "distinct_values", "top_values", "get_finding"})
    n_search = sum(1 for s in run.steps if s.tool in
                   {"search_reports", "keyword_search", "get_finding_text"} and s.error is None)
    if n_search > 1 or len(tools) > 3:
        return "agent"
    if searched and structured:
        return "hybrid"
    if searched:
        return "narrative"
    if structured:
        return "structured"
    return "agent"


def _coverage_note(run) -> str:
    bits = [f"{len(run.ledger)} evidence passage(s) gathered over {len(run.steps)} tool call(s)"]
    if run.stop_reason.value != "finished":
        bits.append(f"the run stopped early ({run.stop_reason.value})")
    return "; ".join(bits)


def _coverage_sentence(run) -> str:
    return (
        f"I examined {len(run.steps)} tool result(s) and gathered "
        f"{len(run.ledger)} evidence passage(s)."
    )


#: Tools that only orient the model (which report is which, what tools exist).
#: Their results are exact, but they support no claim about report content.
_ORIENTATION_TOOLS = {"find_report", "describe_capabilities"}

#: Explicit render tools: a block from these was asked for by the model.
_RENDER_TOOLS = {"show_chart", "show_table", "show_note"}

#: At most this many queried findings get their full text attached.
_TOP_UP_MAX_FINDINGS = 8


def _data_tools_used(run) -> set[str]:
    """Successful tools whose result is computed data, not orientation or search."""
    from ragaudit.agent.loop import _is_structured

    return {s.tool for s in run.steps
            if s.error is None and _is_structured(s.tool) and s.tool not in _ORIENTATION_TOOLS}


def _describe_block(block: dict) -> str:
    """One line naming a visual, for the finalize call to choose from."""
    kind = block.get("chart") or block.get("kind", "block")
    title = block.get("title") or block.get("label") or ""
    return f"{kind}: {title}".strip(": ")


def _chosen_blocks(run, choice) -> list[dict]:
    """Blocks to show beside a finalized answer, as the model chose them.

    Query tools emit charts and tables automatically; whether one helps answer
    this question is the finalize model's call (its ``visuals`` field), made
    from the question's meaning. Anything the model rendered on purpose with a
    show_* tool is always kept. If the model gave no usable choice, every block
    is kept: showing an extra chart is a smaller failure than hiding the one
    that was asked for.
    """
    sources = run.block_sources + ["?"] * (len(run.blocks) - len(run.block_sources))
    if not isinstance(choice, list):
        return list(run.blocks)
    picked = {i for i in choice if isinstance(i, int) and not isinstance(i, bool)}
    return [b for i, (b, src) in enumerate(zip(run.blocks, sources))
            if src in _RENDER_TOOLS or i in picked]


def _exact_listing(run) -> str:
    """The run's exact results, rendered by the deterministic templates.

    Used when an answer is refused for lack of a citation: the counts and
    lists the tools computed are still true, so the user keeps them.
    """
    texts: list[str] = []
    for s in run.steps:
        if s.error is None and s.tool in _data_tools_used(run):
            content = run.step_results.get(s.step) or {}
            text = _render_structured_text(s.tool, s.arguments, content)
            if text and text not in texts:
                texts.append(text)
    return "\n".join(texts)


def _top_up_finding_text(ctx, run) -> int:
    """Make the findings the model queried citable in the final answer.

    Query tools return labels and short, deliberately unquotable excerpts; the
    quotable text arrives only through get_finding_text, which a model asked
    to describe findings often never calls. It then finalizes with no passages
    and abstains, though the findings were found.

    So the findings the model itself chose to look up (query_findings,
    count_findings, get_finding) have their full text registered in the ledger.
    No reading of the question: the model's tool choice is the signal. That
    text is rendered into the finalize prompt, so a citation to it is a
    citation to text the model was shown, which is the ledger's rule. Bounded.
    Returns how many findings were added.
    """
    if ctx is None:
        return 0
    from ragaudit.agent.tools.search import _FULL, _register_and_show, _row_to_chunk

    wanted: list[tuple[str, str]] = []
    for s in run.steps:
        if s.error is not None:
            continue
        content = run.step_results.get(s.step) or {}
        if s.tool in ("query_findings", "count_findings"):
            wanted += [(f["report_id"], f["finding_id"]) for f in content.get("findings") or []]
        elif s.tool == "get_finding" and content.get("finding"):
            wanted.append((content["finding"]["report_id"], content["finding"]["finding_id"]))

    seen_in_ledger = {(c.report_id, c.finding_id) for c in run.ledger.chunks()}
    added = 0
    ctx.current_step = len(run.steps) + 1
    for rid, fid in dict.fromkeys(wanted):
        if added >= _TOP_UP_MAX_FINDINGS:
            break
        if (rid, fid) in seen_in_ledger:
            continue
        rows = ctx.catalog.chunks_for_finding(fid, [rid])
        for row in rows:
            _register_and_show(ctx, _row_to_chunk(row), "finding_text", limit=_FULL)
        added += 1 if rows else 0
    return added


def _render_structured_text(tool: str, args: dict, content: dict) -> str | None:
    """Deterministic one-liner for the structured shortcut. None = no shortcut."""
    if tool == "count_findings":
        filt = content.get("filters") or {}
        where = _describe_filters(filt)
        n = content.get("count", 0)
        if where:
            head = f"{n} finding{'s' if n != 1 else ''} match{'es' if n == 1 else ''} {where}"
        else:
            head = f"There {'is' if n == 1 else 'are'} {n} finding{'s' if n != 1 else ''} in scope"
        listed = content.get("findings") or []
        if not listed:
            return head + "."
        lines = [f"- [{r['report_id']} {r['finding_id']}] {r['title'] or '(untitled)'}: "
                 f"{r['severity']}, {r['status']}" for r in listed]
        return head + ":\n" + "\n".join(lines)
    if tool == "aggregate_findings":
        counts = content.get("counts") or content.get("cross_tab")
        dim = content.get("dimension") or f"{content.get('rows')} by {content.get('columns')}"
        if isinstance(counts, dict) and content.get("counts"):
            parts = ", ".join(f"{k}: {v}" for k, v in content["counts"].items())
            return f"Findings by {dim}: {parts}. (See the chart.)"
        cross = content.get("cross_tab") or {}
        if isinstance(cross, dict) and cross:
            rows = "; ".join(
                f"{row}: " + ", ".join(f"{col} {n}" for col, n in cols.items())
                for row, cols in cross.items()
            )
            return (f"Findings by {content.get('rows')} and {content.get('columns')} "
                    f"({content.get('total', '?')} total). {rows}. (See the chart.)")
        return f"Cross-tab of {dim} computed; see the chart."
    if tool in ("distinct_values", "top_values"):
        field = content.get("field", "value")
        vals = content.get("values") or {v["value"]: v["count"] for v in content.get("top", [])}
        if not vals:
            return f"No {field} values are present in scope."
        parts = ", ".join(f"{k} ({v})" for k, v in list(vals.items())[:10])
        label = {"cwe": "CWE", "cve": "CVE"}.get(field, field.title())
        return f"{label} values in scope (most frequent first): {parts}."
    if tool == "query_findings":
        rows = content.get("findings") or []
        total = content.get("total", len(rows))
        where = _describe_filters(content.get("filters") or {})
        scope = f" matching {where}" if where else ""
        if not rows:
            return f"No findings{scope}."
        lines = [f"- [{r['report_id']} {r['finding_id']}] {r['title'] or '(untitled)'}: "
                 f"{r['severity']}, {r['status']}" for r in rows[:15]]
        more = f"\n... and {total - min(len(rows), 15)} more (see the table)." if total > 15 else ""
        return f"{total} finding{'s' if total != 1 else ''}{scope}:\n" + "\n".join(lines) + more
    if tool == "list_reports":
        reports = content.get("reports") or []
        if not reports:
            return "No reports have been ingested yet."
        q = content.get("quarantined", 0)
        head = (f"{content.get('count', len(reports))} report(s) are in the catalogue: "
                f"{content.get('indexed', 0)} indexed and searchable"
                + (f", {q} quarantined by the quality gate (not searchable)" if q else "") + ".")
        lines = [f"- {r['report_id']}: {r['title'] or '(no title)'} [{r['status']}, {r['findings']} findings]"
                 for r in reports[:20]]
        more = f"\n... and {len(reports) - 20} more (see the table)." if len(reports) > 20 else ""
        return head + "\n" + "\n".join(lines) + more
    if tool == "corpus_stats":
        return (f"{content.get('reports', 0)} report(s) in the catalogue "
                f"({content.get('reports_indexed', 0)} indexed, "
                f"{content.get('reports_quarantined', 0)} quarantined), "
                f"{content.get('findings', 0)} findings, {content.get('chunks', 0)} searchable passages, "
                f"{content.get('figures', 0)} referenced figures "
                f"({content.get('figures_available', 0)} with images).")
    if tool == "describe_schema":
        fields = content.get("finding_fields") or {}
        report_fields = content.get("report_fields") or {}
        lines = ["Each finding can carry these fields (populated count / total findings):"]
        for name, info in fields.items():
            lines.append(f"- {name.replace('_', ' ')}: {info.get('populated', 0)} / {info.get('of', 0)}")
        if report_fields:
            lines.append("Report-level fields (populated / total reports):")
            for name, info in report_fields.items():
                lines.append(f"- {name.replace('_', ' ')}: {info.get('populated', 0)} / {info.get('of', 0)}")
        lines.append("Each finding also has a title and id, and its observation, impact and "
                     "recommendation text is searchable with citations. Fields with low counts "
                     "cannot answer questions for the reports that lack them.")
        return "\n".join(lines)
    if tool == "get_report_outline":
        sev = ", ".join(f"{k} {v}" for k, v in (content.get("severity_breakdown") or {}).items())
        st = ", ".join(f"{k} {v}" for k, v in (content.get("status_breakdown") or {}).items())
        sections = ", ".join((content.get("sections") or {}).keys())
        return (f"{content.get('report_id')}: {content.get('title') or '(no title)'}. "
                f"{content.get('finding_count', 0)} findings"
                + (f"; by severity: {sev}" if sev else "") + (f"; by status: {st}" if st else "")
                + (f". Sections indexed: {sections}." if sections else "."))
    if tool == "find_report":
        return f"That is report {content.get('report_id')}: {content.get('title') or '(no title)'}."
    if tool == "report_quality":
        rows = content.get("reports") or []
        bad = [r for r in rows if r.get("status") == "quarantined"]
        # Standard-template compliance warnings worth surfacing even for
        # indexed reports (unverified closures, self-reviewed reports).
        notable = ("closure_verified", "reviewer_distinct", "reviewer_present", "total_matches_array",
                   "severity_counts_sum", "summary_detail_ids_match")
        flagged = []
        for r in rows:
            if r.get("status") == "quarantined":
                continue
            hits = [w.split(": ", 1)[-1] for w in r.get("warnings", []) if any(f"] {n}" in w for n in notable)]
            if hits:
                flagged.append(f"- {r['report_id']} (indexed, with warnings): " + "; ".join(hits[:3]))
        tail = ("\nIndexed reports with template-compliance warnings:\n" + "\n".join(flagged)) if flagged else ""
        if not bad:
            return f"All {len(rows)} report(s) passed the extraction quality gate." + tail
        def reasons(r):
            seen: list[str] = []
            for issue in r.get("blocking", []):
                text = issue.split(": ", 1)[-1]
                if text not in seen:
                    seen.append(text)
            return "; ".join(seen[:2])

        lines = [f"- {r['report_id']}: {reasons(r)}" for r in bad[:15]]
        return (f"{len(bad)} of {len(rows)} report(s) were quarantined by the quality gate "
                "(their content is not searchable):\n" + "\n".join(lines) + tail)
    if tool == "find_data_gaps":
        total = content.get("total_findings", 0)
        parts = []
        for key, label in (("missing_cwe", "no CWE"), ("missing_cvss", "no CVSS score"),
                           ("missing_recommendation", "no recommendation"),
                           ("unknown_severity", "unknown severity")):
            parts.append(f"{(content.get(key) or {}).get('count', 0)} with {label}")
        return f"Of {total} findings: " + ", ".join(parts) + "."
    if tool == "compare_reports":
        cols = content.get("reports") or []
        lines = [f"- {c['report_id']}: {c['total_findings']} findings, {c['open']} open; "
                 + ", ".join(f"{k} {v}" for k, v in (c.get("severity") or {}).items()) for c in cols]
        return "Report comparison (see the chart and table):\n" + "\n".join(lines)
    if tool == "find_repeat_findings":
        repeats = content.get("repeat_findings") or []
        if not repeats:
            return "No finding recurs across reports (matched by CWE, CVE or title)."
        # The same findings often match on several keys (title AND CWE);
        # merge those so one recurring issue is reported once.
        groups: dict[tuple, dict] = {}
        for r in repeats:
            key = tuple(sorted(f"{o['report_id']}:{o['finding_id']}" for o in r["occurrences"]))
            g = groups.setdefault(key, {"reports": r["reports"], "occurrences": r["occurrences"], "on": []})
            g["on"].append(f"{r['match_on']} '{r['value']}'" if r["match_on"] == "title"
                           else r["value"])
        lines = []
        for g in list(groups.values())[:15]:
            occ = ", ".join(f"{o['report_id']} {o['finding_id']}" for o in g["occurrences"][:6])
            lines.append(f"- matched on {' and '.join(g['on'])}: {occ}")
        n = len(groups)
        return (f"{n} issue{'s' if n != 1 else ''} recur{'' if n != 1 else 's'} across reports:\n"
                + "\n".join(lines))
    if tool == "describe_capabilities":
        groups = content.get("groups") or {}
        lines = ["I can:"]
        for group, tools in groups.items():
            lines.append(f"- {group}: " + "; ".join(t["description"].rstrip(".") for t in tools[:4]))
        return "\n".join(lines)
    return None


def _describe_filters(filters: dict) -> str:
    bits = []
    if filters.get("severities"):
        bits.append("severity " + "/".join(filters["severities"]))
    if filters.get("statuses"):
        bits.append("status " + "/".join(filters["statuses"]))
    if filters.get("cwe"):
        cwe = str(filters["cwe"]).upper()
        bits.append(cwe if cwe.startswith("CWE") else f"CWE-{cwe}")
    if filters.get("cve"):
        bits.append(str(filters["cve"]).upper())
    if filters.get("report_ids"):
        bits.append("in " + ", ".join(filters["report_ids"]))
    return " and ".join(bits)
