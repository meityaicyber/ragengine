"""Core domain types.

Provenance rule (D04): a chunk's ``start_char``/``end_char`` are *carried* from
the slice that produced the text, never recovered afterwards by searching for
the text in the document. Offsets found by search mislocate whenever a phrase
repeats, which silently breaks the citation guarantee this product rests on.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# --------------------------------------------------------------------------
# Controlled vocabularies
# --------------------------------------------------------------------------


class Severity(str, Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    INFORMATIONAL = "Informational"
    UNKNOWN = "Unknown"

    @classmethod
    def parse(cls, raw: Any) -> Severity:
        text = str(raw or "").strip().lower()
        for member in cls:
            if member.value.lower() == text:
                return member
        aliases = {
            "info": cls.INFORMATIONAL,
            "informational": cls.INFORMATIONAL,
            "crit": cls.CRITICAL,
            "med": cls.MEDIUM,
            "moderate": cls.MEDIUM,
            "severe": cls.HIGH,
        }
        return aliases.get(text, cls.UNKNOWN)


class FindingStatus(str, Enum):
    OPEN = "Open"
    CLOSED = "Closed"
    IN_PROGRESS = "In Progress"
    RISK_ACCEPTED = "Risk Accepted"
    UNKNOWN = "Unknown"

    @classmethod
    def parse(cls, raw: Any) -> FindingStatus:
        text = str(raw or "").strip().lower()
        for member in cls:
            if member.value.lower() == text:
                return member
        aliases = {
            "opened": cls.OPEN,
            "fixed": cls.CLOSED,
            "resolved": cls.CLOSED,
            "remediated": cls.CLOSED,
            "closed - verified": cls.CLOSED,
            "wip": cls.IN_PROGRESS,
            "in-progress": cls.IN_PROGRESS,
            "ongoing": cls.IN_PROGRESS,
            "accepted": cls.RISK_ACCEPTED,
        }
        return aliases.get(text, cls.UNKNOWN)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

FIGURE_TOKEN_RE = re.compile(r"\[IMAGE_PAGE_(\d+)_FIG_(\d+)\]")


@dataclass(frozen=True)
class FigureRef:
    """A figure referenced by an extractor placeholder token.

    ``asset_path`` is None when the extractor emitted the token but no image
    file was staged. The chatbot reports such figures as referenced-but-missing
    rather than pretending they do not exist (D07).
    """

    token: str
    page: int
    index: int
    report_id: str
    asset_path: str | None = None

    @property
    def available(self) -> bool:
        return self.asset_path is not None

    @classmethod
    def from_token(cls, token: str, report_id: str, asset_path: str | None = None) -> FigureRef | None:
        match = FIGURE_TOKEN_RE.fullmatch(token.strip())
        if not match:
            return None
        return cls(
            token=token.strip(),
            page=int(match.group(1)),
            index=int(match.group(2)),
            report_id=report_id,
            asset_path=asset_path,
        )


# --------------------------------------------------------------------------
# Chunks
# --------------------------------------------------------------------------


def stable_chunk_id(report_id: str, field_path: str, start: int, end: int) -> str:
    """Deterministic chunk id.

    Keyed on the *field path* and character span rather than a running counter,
    so re-ingesting an unchanged report yields identical ids and any stored
    evaluation label stays valid.
    """
    key = f"{report_id}|{field_path}|{start}|{end}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


@dataclass
class Chunk:
    chunk_id: str
    report_id: str
    section: str
    field_path: str
    text: str
    start_char: int
    end_char: int
    token_count: int
    finding_id: str | None = None
    figure_tokens: list[str] = field(default_factory=list)
    # Contextual Retrieval prefix, stored separately so citations quote only
    # text that actually appears in the source report (D04).
    context_prefix: str = ""

    @property
    def embedding_text(self) -> str:
        """What goes to the embedder: context prefix plus the chunk body."""
        return f"{self.context_prefix}\n\n{self.text}".strip() if self.context_prefix else self.text

    @property
    def quotable_text(self) -> str:
        """What a citation may quote: the source span only."""
        return self.text


# --------------------------------------------------------------------------
# Findings and reports
# --------------------------------------------------------------------------


@dataclass
class Finding:
    finding_id: str
    report_id: str
    title: str
    severity: Severity
    status: FindingStatus
    cvss_score: float | None = None
    cvss_vector: str = ""
    cwe_ids: list[str] = field(default_factory=list)
    cve_ids: list[str] = field(default_factory=list)
    affected_assets: list[str] = field(default_factory=list)
    observation: str = ""
    impact: str = ""
    recommendation: str = ""
    preconditions: str = ""
    figure_tokens: list[str] = field(default_factory=list)
    is_repeat: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportMeta:
    report_id: str
    title: str = ""
    report_type: str = ""
    audit_type: str = ""
    organisation: str = ""
    auditor: str = ""
    classification: str = ""
    release_date: str = ""
    audit_period_from: str = ""
    audit_period_to: str = ""
    standards: list[str] = field(default_factory=list)
    source_file: str = ""
    source_sha256: str = ""
    extractor_version: str = ""
    ingested_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Report:
    meta: ReportMeta
    findings: list[Finding] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    figures: list[FigureRef] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def report_id(self) -> str:
        return self.meta.report_id


# --------------------------------------------------------------------------
# Answers
# --------------------------------------------------------------------------


@dataclass
class Citation:
    chunk_id: str
    report_id: str
    field_path: str
    quote: str
    start_char: int
    end_char: int
    finding_id: str | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    rank: int
    retriever: str


@dataclass
class Answer:
    """A chatbot answer.

    ``text`` is free-form prose -- the product is a question-answering
    assistant, not a yes/no classifier (D06). Compliance verdicts are a
    separate, narrower surface.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    figures: list[FigureRef] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    abstained: bool = False
    abstention_reason: str = ""
    review_flags: list[str] = field(default_factory=list)
    route: str = "narrative"
    latency_s: float = 0.0
    model: str = ""
    trace_id: str = ""
    # Agent additions: render blocks (tables, charts, metrics) the UI draws,
    # and the tool-call trace that makes a multi-step answer auditable. Empty
    # on the single-pass path.
    blocks: list[dict] = field(default_factory=list)
    steps: list[dict] = field(default_factory=list)
