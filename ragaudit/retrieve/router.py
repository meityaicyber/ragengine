"""Query routing: structured, narrative, or both.

The premise (D02): in the supplied benchmark every one of the eight
``real_reports`` questions is a predicate over extracted fields. Routing those
to SQL makes them exact and free; routing them to a language model makes them
slow, costly and occasionally wrong.

The router is deliberately a deterministic rule set, not a model. An audit
system must be able to explain why a question took the path it did, and a
regex that can be read in the source is a better explanation than a classifier
logit. Rules that fire are reported on the answer, so a misroute is visible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from ragaudit.domain.models import FindingStatus, Severity


class Route(str, Enum):
    STRUCTURED = "structured"
    NARRATIVE = "narrative"
    HYBRID = "hybrid"
    #: Not a question about the reports. Greetings, acknowledgements, and
    #: "what can you do" are answered directly and never reach retrieval.
    CHITCHAT = "chitchat"


@dataclass
class QueryPlan:
    route: Route
    reasons: list[str] = field(default_factory=list)
    #: For CHITCHAT: which kind, so the service can answer appropriately.
    chitchat_kind: str = ""
    severities: list[str] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    cwe: str | None = None
    cve: str | None = None
    wants_count: bool = False
    wants_list: bool = False
    universal: bool = False          # "all", "every" -- needs complete data
    existential: bool = False        # "any", "at least one"
    negated: bool = False            # "no findings", "none"
    asks_for_figures: bool = False


COUNT_RE = re.compile(r"\b(how many|number of|count of|total (?:number|count))\b", re.I)
LIST_RE = re.compile(r"\b(list|which|what are|show me|enumerate|name the)\b", re.I)
UNIVERSAL_RE = re.compile(r"\b(all|every|each|always)\b", re.I)
EXISTENTIAL_RE = re.compile(r"\b(any|at least one|some|does the report (?:contain|include|have))\b", re.I)
NEGATION_RE = re.compile(r"\b(no|none|zero|without|absent|not)\b", re.I)
CWE_RE = re.compile(r"\bCWE[-\s]?(\d+)\b", re.I)
CVE_RE = re.compile(r"\b(CVE-\d{4}-\d{4,})\b", re.I)
FIGURE_RE = re.compile(
    r"\b(screenshot|screenshots|figure|figures|image|images|diagram|proof of concept|poc|evidence image)\b",
    re.I,
)

#: Narrative intent. These ask for judgement or explanation, which no field
#: holds -- they must go to retrieval even when they mention a field name.
NARRATIVE_RE = re.compile(
    r"\b(why|how (?:was|did|does|were)|explain|describe|summar(?:ise|ize|y)|"
    r"adequate|sufficient|appropriate|rationale|reasoning|recommend(?:ation)?s? (?:address|cover)|"
    r"what (?:is|was) the (?:impact|risk|root cause)|interpret|assess whether)\b",
    re.I,
)

#: Field-presence intent, the RQ1-RQ8 shape.
PRESENCE_RE = re.compile(
    r"\b(contain|include|provide|specify|state|document|record|present|populated|mark(?:ed)?)\b", re.I
)

#: Nouns that name something the extractor actually stores. Presence intent
#: alone is not enough to route to SQL: "do all vendor contracts include a
#: breach clause?" reads as a presence question but no field holds it, so it
#: belongs in retrieval. The structured path requires both the intent and a
#: noun the catalogue can answer for.
FIELD_NOUN_RE = re.compile(
    r"\b(finding|findings|vulnerabilit(?:y|ies)|observation|observations|severity|severities|"
    r"cwe|cve|cvss|status|remediation|recommendation|recommendations|"
    r"audit timeline|timeline|affected asset|assets|classification|"
    r"audit team|tools used|scope section)\b",
    re.I,
)

#: A predicate the catalogue cannot evaluate on its own. "Remediated within
#: the organisation's SLA" needs both the finding statuses (structured) and
#: the SLA window (narrative, if the report states it at all). Answering from
#: either half alone would overstate what the evidence supports.
POLICY_QUALIFIER_RE = re.compile(
    # "within <up to four qualifying words> SLA/window/deadline" -- the span is
    # bounded so this cannot swallow an unrelated clause later in the sentence.
    r"\bwithin\b(?:\s+[\w'’-]+){0,4}\s+"
    r"(?:sla|slas|timeframes?|time frames?|windows?|periods?|deadlines?|cycles?)\b"
    r"|\b(?:per|against|in line with|in accordance with) (?:the |its |their )?"
    r"(?:polic(?:y|ies)|standard|baseline|requirements?)\b"
    r"|\bmeet or exceed\b"
    r"|\brequired (?:annual |periodic )?(?:review )?cycle\b",
    re.I,
)

STATUS_TERMS = {
    "open": FindingStatus.OPEN,
    "unresolved": FindingStatus.OPEN,
    "outstanding": FindingStatus.OPEN,
    "closed": FindingStatus.CLOSED,
    "resolved": FindingStatus.CLOSED,
    "remediated": FindingStatus.CLOSED,
    "fixed": FindingStatus.CLOSED,
    "in progress": FindingStatus.IN_PROGRESS,
    "risk accepted": FindingStatus.RISK_ACCEPTED,
}


#: Inputs that are not questions about the reports. Matched against the WHOLE
#: input, so "hey, how many findings are open?" still routes normally -- only
#: a bare greeting is intercepted.
_GREETING = re.compile(
    r"^(hi|hey+|hello+|yo|hiya|howdy|sup|greetings|good\s+(morning|afternoon|evening|day))"
    r"[\s!.,?]*(there|everyone|team)?[\s!.,?]*$",
    re.I,
)
_ACK = re.compile(
    r"^(thanks?|thank\s+you|thx|ty|ok(ay)?|cool|nice|great|got\s+it|understood|"
    r"perfect|sure|alright|fine|yep|yes|no|nope|k)[\s!.,]*$",
    re.I,
)
_FAREWELL = re.compile(r"^(bye|goodbye|cya|see\s+you|later|good\s*night)[\s!.,]*$", re.I)
_CAPABILITY = re.compile(
    r"^(help|/?help|what\s+can\s+you\s+do|what\s+do\s+you\s+do|who\s+are\s+you|"
    r"what\s+is\s+this|what'?s\s+this|how\s+do(es)?\s+(this|you)\s+work|"
    r"what\s+(reports?|data)\s+do\s+you\s+have|commands?)[\s!.,?]*$",
    re.I,
)

#: A question needs at least one word that could refer to report content.
#: Without this, a stray token routes to retrieval and the model is handed
#: evidence it must say something about.
_CONTENT_WORD = re.compile(r"[A-Za-z]{3,}")


def _chitchat_kind(q: str) -> str:
    """Classify a non-question input, or return '' if it is a real question."""
    if _CAPABILITY.match(q):
        return "capability"
    if _GREETING.match(q):
        return "greeting"
    if _ACK.match(q):
        return "acknowledgement"
    if _FAREWELL.match(q):
        return "farewell"
    # Nothing substantive to search for: no word long enough to name anything
    # in a report.
    if not _CONTENT_WORD.search(q):
        return "empty"
    return ""


def route_query(question: str) -> QueryPlan:
    """Classify one question into a retrieval plan."""
    q = question.strip()

    # Checked first, and against the whole input. A greeting handed to
    # retrieval returns arbitrary chunks, and a model told to answer from
    # evidence will produce a confident summary of whatever it was given --
    # an answer to a question nobody asked.
    kind = _chitchat_kind(q)
    if kind:
        return QueryPlan(
            route=Route.CHITCHAT,
            reasons=[f"input is {kind}, not a question about the reports"],
            chitchat_kind=kind,
        )

    plan = QueryPlan(route=Route.NARRATIVE)

    for term, severity in (
        ("critical", Severity.CRITICAL),
        ("high", Severity.HIGH),
        ("medium", Severity.MEDIUM),
        ("low", Severity.LOW),
        ("informational", Severity.INFORMATIONAL),
    ):
        if re.search(rf"\b{term}[- ]?(?:severity|risk|finding|vulnerabilit)", q, re.I) or re.search(
            rf"\bseverity\s+{term}\b", q, re.I
        ):
            plan.severities.append(severity.value)

    for term, status in STATUS_TERMS.items():
        if re.search(rf"\b{re.escape(term)}\b", q, re.I):
            if status.value not in plan.statuses:
                plan.statuses.append(status.value)

    if match := CWE_RE.search(q):
        plan.cwe = f"CWE-{match.group(1)}"
    if match := CVE_RE.search(q):
        plan.cve = match.group(1).upper()

    plan.wants_count = bool(COUNT_RE.search(q))
    plan.wants_list = bool(LIST_RE.search(q))
    plan.universal = bool(UNIVERSAL_RE.search(q))
    plan.existential = bool(EXISTENTIAL_RE.search(q))
    plan.negated = bool(NEGATION_RE.search(q))
    plan.asks_for_figures = bool(FIGURE_RE.search(q))

    narrative_intent = bool(NARRATIVE_RE.search(q))
    names_field = bool(FIELD_NOUN_RE.search(q))
    presence_intent = bool(PRESENCE_RE.search(q)) and names_field
    has_field_predicate = bool(
        plan.severities or plan.statuses or plan.cwe or plan.cve or plan.wants_count
    )

    # Precedence matters. A question can mention a severity and still be asking
    # for judgement ("why were the high findings accepted?"), so narrative
    # intent is checked first and wins outright.
    if narrative_intent:
        plan.route = Route.HYBRID if has_field_predicate else Route.NARRATIVE
        plan.reasons.append("narrative intent detected")
        if has_field_predicate:
            plan.reasons.append("field predicate also present; retrieval scoped by structured filter")
        return plan

    if POLICY_QUALIFIER_RE.search(q):
        plan.route = Route.HYBRID if has_field_predicate else Route.NARRATIVE
        plan.reasons.append(
            "policy qualifier present; the threshold itself must come from the report text"
        )
        return plan

    if plan.wants_count or (plan.universal and has_field_predicate):
        # A count or a universal claim is only sound over the complete set,
        # which retrieval cannot promise. These must be answered by SQL.
        plan.route = Route.STRUCTURED
        plan.reasons.append("aggregate or universal claim requires complete data, not top-k")
        return plan

    if has_field_predicate:
        plan.route = Route.STRUCTURED
        plan.reasons.append("question is a predicate over extracted fields")
        return plan

    if presence_intent and not narrative_intent:
        plan.route = Route.STRUCTURED
        plan.reasons.append("field-presence question")
        return plan

    if plan.asks_for_figures:
        plan.route = Route.HYBRID
        plan.reasons.append("figure request; locate the finding, then join its attached figures")
        return plan

    plan.reasons.append("no field predicate recognised; searching narrative text")
    return plan
