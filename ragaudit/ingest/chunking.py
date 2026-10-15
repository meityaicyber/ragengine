"""Field-aware chunking for structured audit reports.

Design (D04). The previous pipeline flattened structured JSON into text lines
and ran a generic text splitter over the result. That discards the structure
the extractor worked to recover, then asks an embedding model to recover it
statistically.

Here a finding is the natural retrieval unit: one finding becomes one chunk
when it fits, and splits on field boundaries when it does not, so a
recommendation is never severed from the observation that triggered it. Each
chunk carries its originating ``field_path``, which gives the chatbot a precise
citation target without needing character offsets into a reconstructed
document.

Offsets are into the *canonical field text* and are carried from the slice that
produced them. They are never recovered with ``str.find``.
"""

from __future__ import annotations

import re
from typing import Iterable

from ragaudit.domain.models import Chunk, Finding, Report, stable_chunk_id

# Token counting. tiktoken when staged, else a calibrated word ratio. The
# fallback matters only for budget decisions, never for correctness.
try:  # pragma: no cover - depends on local staging
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))

except Exception:  # pragma: no cover

    def count_tokens(text: str) -> int:
        return max(1, int(len(text.split()) * 1.3))


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

FIELD_LABELS = {
    "observation": "Observation",
    "impact": "Impact",
    "recommendation": "Recommendation",
    "preconditions": "Preconditions",
}


class FieldAwareChunker:
    """Chunk a normalised report along its own field structure."""

    def __init__(
        self,
        max_tokens: int = 420,
        overlap_tokens: int = 60,
        min_tokens: int = 40,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens

    # -- public ----------------------------------------------------------

    def chunk_report(self, report: Report) -> list[Chunk]:
        chunks: list[Chunk] = []
        for finding in report.findings:
            chunks.extend(self.chunk_finding(finding))
        chunks.extend(self._chunk_report_sections(report))
        return chunks

    def chunk_finding(self, finding: Finding) -> list[Chunk]:
        """One chunk per finding when it fits; otherwise split per field."""
        body = self._finding_body(finding)
        if not body.strip():
            return []

        header = self._finding_header(finding)
        whole = f"{header}\n{body}"

        if count_tokens(whole) <= self.max_tokens:
            return [
                self._make(
                    report_id=finding.report_id,
                    section="Findings",
                    field_path=f"findings[{finding.finding_id}]",
                    text=whole,
                    start=0,
                    end=len(whole),
                    finding_id=finding.finding_id,
                    figure_tokens=finding.figure_tokens,
                )
            ]

        out: list[Chunk] = []
        for name in ("observation", "impact", "recommendation", "preconditions"):
            value = getattr(finding, name, "")
            if not value.strip():
                continue
            label = FIELD_LABELS[name]
            field_path = f"findings[{finding.finding_id}].{name}"
            # Repeat the header on every piece so a chunk retrieved alone still
            # identifies its finding, severity and status.
            prefix = f"{header}\n{label}: "
            for start, end, piece in self._split_text(value):
                text = f"{prefix}{piece}"
                out.append(
                    self._make(
                        report_id=finding.report_id,
                        section="Findings",
                        field_path=field_path,
                        text=text,
                        start=start,
                        end=end,
                        finding_id=finding.finding_id,
                        # Figures belong to the proof, which rides with the
                        # observation; attaching them to every piece would
                        # over-report evidence.
                        figure_tokens=finding.figure_tokens if name == "observation" else [],
                    )
                )
        return out

    # -- internals -------------------------------------------------------

    def _finding_header(self, finding: Finding) -> str:
        bits = [f"[{finding.finding_id}] {finding.title}".strip()]
        bits.append(f"Severity: {finding.severity.value}")
        bits.append(f"Status: {finding.status.value}")
        if finding.cwe_ids:
            bits.append("CWE: " + ", ".join(finding.cwe_ids))
        if finding.cve_ids:
            bits.append("CVE: " + ", ".join(finding.cve_ids))
        if finding.affected_assets:
            assets = ", ".join(finding.affected_assets[:3])
            if len(finding.affected_assets) > 3:
                assets += f" (+{len(finding.affected_assets) - 3} more)"
            bits.append(f"Affected: {assets}")
        return " | ".join(bits)

    def _finding_body(self, finding: Finding) -> str:
        parts = []
        for name, label in FIELD_LABELS.items():
            value = getattr(finding, name, "")
            if value.strip():
                parts.append(f"{label}: {value.strip()}")
        return "\n".join(parts)

    def _split_text(self, text: str) -> Iterable[tuple[int, int, str]]:
        """Split long field text on sentence boundaries, carrying offsets.

        Yields ``(start_char, end_char, piece)`` where the offsets index into
        ``text``. Offsets come from the scan, not from a later search.
        """
        text = text.strip()
        if not text:
            return

        spans: list[tuple[int, int]] = []
        cursor = 0
        for match in SENTENCE_BOUNDARY.finditer(text):
            spans.append((cursor, match.start()))
            cursor = match.end()
        spans.append((cursor, len(text)))
        spans = [(s, e) for s, e in spans if text[s:e].strip()]

        if not spans:
            return

        buf_start = spans[0][0]
        buf_end = spans[0][1]
        buf_tokens = count_tokens(text[buf_start:buf_end])

        for start, end in spans[1:]:
            piece_tokens = count_tokens(text[start:end])
            if buf_tokens + piece_tokens > self.max_tokens:
                yield buf_start, buf_end, text[buf_start:buf_end].strip()
                # Overlap by stepping the window back a bounded number of
                # characters, measured in the same unit as the offsets.
                overlap_chars = min(self.overlap_tokens * 4, buf_end - buf_start)
                buf_start = max(buf_start, buf_end - overlap_chars)
                buf_tokens = count_tokens(text[buf_start:buf_end])
            buf_end = end
            buf_tokens += piece_tokens

        tail = text[buf_start:buf_end].strip()
        if tail:
            yield buf_start, buf_end, tail

    def _chunk_report_sections(self, report: Report) -> list[Chunk]:
        """Narrative sections outside the findings array.

        Scope, methodology and conclusions answer questions findings cannot
        ("was the testing authenticated?"), so they are indexed too.
        """
        data = report.raw
        out: list[Chunk] = []
        candidates: list[tuple[str, str, str]] = []

        for section, path in (
            ("Introduction", "introduction"),
            ("Scope", "engagement_scope"),
            ("Methodology", "methodology"),
            ("Tools", "tools_used"),
            ("Timeline", "audit_timeline"),
            ("Conclusion", "conclusion"),
            ("Appendices", "appendices"),
        ):
            value = data.get(path)
            text = _render(value)
            if text.strip():
                candidates.append((section, path, text))

        for section, path, text in candidates:
            if count_tokens(text) <= self.max_tokens:
                body = f"{section}: {text}"
                out.append(
                    self._make(
                        report_id=report.report_id,
                        section=section,
                        field_path=path,
                        text=body,
                        start=0,
                        end=len(body),
                    )
                )
                continue
            for start, end, piece in self._split_text(text):
                body = f"{section}: {piece}"
                out.append(
                    self._make(
                        report_id=report.report_id,
                        section=section,
                        field_path=path,
                        text=body,
                        start=start,
                        end=end,
                    )
                )
        return out

    def _make(
        self,
        *,
        report_id: str,
        section: str,
        field_path: str,
        text: str,
        start: int,
        end: int,
        finding_id: str | None = None,
        figure_tokens: list[str] | None = None,
    ) -> Chunk:
        return Chunk(
            chunk_id=stable_chunk_id(report_id, field_path, start, end),
            report_id=report_id,
            section=section,
            field_path=field_path,
            text=text,
            start_char=start,
            end_char=end,
            token_count=count_tokens(text),
            finding_id=finding_id,
            figure_tokens=list(figure_tokens or []),
        )


def _render(value, depth: int = 0) -> str:
    """Flatten a nested section into readable prose, dropping figure tokens."""
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        text = " ".join(value.split())
        return "" if text.startswith("[IMAGE_PAGE_") else text
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_render(v, depth + 1) for v in value]
        return "; ".join(p for p in parts if p)
    if isinstance(value, dict):
        parts = []
        for key, child in value.items():
            rendered = _render(child, depth + 1)
            if rendered:
                label = str(key).replace("_", " ")
                parts.append(f"{label}: {rendered}")
        return ". ".join(parts)
    return str(value)
