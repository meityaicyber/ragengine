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
        self.sessions = sessions or SessionStore()
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
        elif self.agent is not None:
            answer = self._agent_answer(question, session, scope, trace_id)
        elif plan.route is Route.STRUCTURED:
            answer = self.structured.answer(search_query, plan, report_ids=scope)
        else:
            answer = self._narrative_answer(question, search_query, plan, session, scope, trace_id)
            if plan.route is Route.HYBRID:
                answer = self._merge_structured(answer, search_query, plan, scope)
            answer.route = plan.route.value

        answer.trace_id = trace_id
        answer.latency_s = time.monotonic() - started
        self.sessions.append(session, Turn(question=question, answer=answer))
        return answer, session

    # -- agent -----------------------------------------------------------

    def _agent_answer(self, question, session, scope, trace_id) -> Answer:
        """Run the agent loop, then finalise through the existing verifier.

        The loop gathers evidence into a ledger; the answer is produced either
        by a deterministic shortcut (one structured tool answered outright) or
        by one schema-constrained finalize call whose verification reuses
        ``_build_answer`` unchanged.
        """
        from ragaudit.agent.evidence import EvidenceLedger
        from ragaudit.agent.loop import StopReason
        from ragaudit.agent.registry import ToolContext

        resolved = _pre_resolve(question, self.catalog)
        ctx = ToolContext(
            catalog=self.catalog,
            retriever=self.retriever,
            settings=self.settings,
            ledger=EvidenceLedger(),
            resolved=resolved,
            default_scope=_default_scope(scope, resolved, question),
        )
        history = _agent_history(session, self.max_history_turns)
        run = self.agent.run(question, ctx, history=history, trace_id=trace_id)

        # Infrastructure failure or a dead backend: abstain, but keep whatever
        # exact structured results and blocks were gathered — they are true.
        if run.stop_reason in (StopReason.LLM_UNAVAILABLE, StopReason.INFRA_ERROR):
            return self._agent_failed(run, run.stop_reason)

        shortcut = self._shortcut_answer(run, question, scope)
        if shortcut is not None:
            return shortcut

        return self._finalize_answer(question, run, scope)

    def _shortcut_answer(self, run, question, scope) -> Answer | None:
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

    def _finalize_answer(self, question, run, scope) -> Answer:
        from ragaudit.agent.loop import StopReason

        retrieved = run.ledger.as_retrieved()
        coverage = _coverage_note(run)

        # Budget-limited runs still try to answer from what was gathered,
        # labelled partial, rather than discarding the work.
        partial = run.stop_reason in (
            StopReason.MAX_STEPS, StopReason.DEADLINE, StopReason.TOKEN_BUDGET,
            StopReason.NO_PROGRESS, StopReason.TOOL_ERRORS,
        )

        prompt = build_agent_finalize_prompt(
            question, retrieved, structured_summary=run.structured_summary, coverage_note=coverage
        )
        try:
            completion = self.llm.chat(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                format_schema=ANSWER_SCHEMA,
            )
        except (LLMError, LLMUnavailable) as exc:
            logger.error("agent finalize failed: %s", exc)
            return self._agent_failed(run, StopReason.LLM_UNAVAILABLE)

        answer = self._build_answer(completion, retrieved, _AgentPlan())
        answer.blocks = list(run.blocks)
        answer.steps = _step_dicts(run.steps)
        answer.route = _effective_route(run)

        # Agent-path hardening: a free-formed narrative claim with zero verified
        # chunk citations is not grounded. The single-pass path only flags this;
        # after many tool steps it must force an abstention.
        if not answer.abstained and not answer.citations and not run.structured_summary:
            answer.abstained = True
            answer.abstention_reason = "no_verified_citation"
            answer.text = (
                "I gathered evidence but could not ground an answer in it. "
                + _coverage_sentence(run)
            )
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
            report_ids = list(scope) if scope else self.catalog.report_ids()
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
            lines.append(
                f"\nIndexed ({stats.get('findings', 0)} findings across "
                f"{len(report_ids)} report(s)):\n" + "\n".join(titles) + more
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
        if not session.turns:
            return question

        # Ordinal references resolve from session state, with no model call.
        # Deterministic and free, where a rewrite would be neither.
        if match := ORDINAL_RE.search(question):
            ids = session.last_finding_ids()
            position = ORDINALS[match.group(1).lower()]
            if ids and -len(ids) <= position < len(ids):
                return f"{question} (referring to finding {ids[position]})"

        if not self.rewrite_queries or not FOLLOWUP_RE.search(question):
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
        cited_findings = {c.finding_id for c in citations if c.finding_id}
        if not cited_findings:
            cited_findings = {
                item.chunk.finding_id for item in retrieved[:3] if item.chunk.finding_id
            }
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
        for rid in catalog.report_ids():
            meta = catalog.get_report_meta(rid)
            title = (meta.title if meta else "") or ""
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
_TITLE_STOPWORDS = {
    "report", "reports", "security", "assessment", "audit", "testing", "test",
    "vapt", "api", "apis", "application", "web", "network", "mobile", "review",
    "penetration", "pentest", "cyber", "cybersecurity", "final", "draft", "the",
}


_COMPARISON_RE = _re.compile(
    r"\b(compare|versus|vs\.?|difference between|across (?:all |both )?reports|"
    r"both reports|each report|every report|which report)\b", _re.I)


def _default_scope(ui_scope, resolved: dict, question: str) -> list[str] | None:
    """The report scope tools default to when a call names none.

    The UI selector is a hard choice and always wins. Otherwise a singular
    named report scopes the run — unless the question is a comparison, where
    naming one report must not exclude the others. Returns None (all reports)
    when nothing constrains it.
    """
    if ui_scope:
        return list(ui_scope)
    if _COMPARISON_RE.search(question):
        return None
    named = resolved.get("report_ids")
    if named and len(named) == 1:
        return named
    return None


def _agent_history(session, max_turns: int) -> list[dict]:
    """Recent turns as chat messages for the agent's context."""
    out: list[dict] = []
    for turn in session.turns[-max_turns:]:
        out.append({"role": "user", "content": turn.question})
        text = turn.answer.text
        out.append({"role": "assistant", "content": text if len(text) <= 300 else text[:300] + "..."})
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
    reports = {s.arguments.get("report_id") for s in run.steps if s.arguments.get("report_id")}
    bits = [f"{len(run.ledger)} evidence passage(s) gathered over {len(run.steps)} tool call(s)"]
    if run.stop_reason.value != "finished":
        bits.append(f"the run stopped early ({run.stop_reason.value})")
    return "; ".join(bits)


def _coverage_sentence(run) -> str:
    return (
        f"I examined {len(run.steps)} tool result(s) and gathered "
        f"{len(run.ledger)} evidence passage(s)."
    )


def _render_structured_text(tool: str, args: dict, content: dict) -> str | None:
    """Deterministic one-liner for the structured shortcut. None = no shortcut."""
    if tool == "count_findings":
        filt = content.get("filters") or {}
        where = _describe_filters(filt)
        n = content.get("count", 0)
        return f"{n} finding{'s' if n != 1 else ''} match{'' if n == 1 else ''} {where}." if where \
            else f"There are {n} finding{'s' if n != 1 else ''} in scope."
    if tool == "aggregate_findings":
        counts = content.get("counts") or content.get("cross_tab")
        dim = content.get("dimension") or f"{content.get('rows')} by {content.get('columns')}"
        if isinstance(counts, dict) and content.get("counts"):
            parts = ", ".join(f"{k}: {v}" for k, v in content["counts"].items())
            return f"Findings by {dim} — {parts}. (See the chart.)"
        return f"Cross-tab of {dim} computed; see the chart."
    if tool in ("distinct_values", "top_values"):
        field = content.get("field", "value")
        vals = content.get("values") or {v["value"]: v["count"] for v in content.get("top", [])}
        if not vals:
            return f"No {field} values are present in scope."
        parts = ", ".join(f"{k} ({v})" for k, v in list(vals.items())[:10])
        return f"{field.title()} values in scope: {parts}."
    # query_findings is a list; let the finalize call phrase it rather than
    # dumping rows deterministically.
    return None


def _describe_filters(filters: dict) -> str:
    bits = []
    if filters.get("severities"):
        bits.append("severity " + "/".join(filters["severities"]))
    if filters.get("statuses"):
        bits.append("status " + "/".join(filters["statuses"]))
    if filters.get("cwe"):
        bits.append(f"CWE {filters['cwe']}")
    if filters.get("cve"):
        bits.append(f"CVE {filters['cve']}")
    if filters.get("report_ids"):
        bits.append("in " + ", ".join(filters["report_ids"]))
    return " and ".join(bits)
