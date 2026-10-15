"""The evidence ledger.

This is the integrity core of the agent. In the single-pass system, citations
are verified against ``for_generation`` — the exact list serialised into the
prompt. An agent gathers evidence across many tool calls, so that one list no
longer exists. The ledger reconstructs the same guarantee: a citation is
verifiable only against text the model was *actually shown*.

Three rules, each easy to get wrong and silent when wrong:

1. Register the text that was serialised into the tool result, never the
   retriever's fuller return value. If a search returns 2000 characters but
   only 400 reached the model, a quote from the unshown tail must not verify.

2. A chunk is citable only through a tool that returns chunk-addressable text
   (``search_reports``, ``get_finding_text``). A tool that returns a field
   value or a row label does not make that text citable — it has no chunk id
   and no offsets, so a quote from it is unlocatable by construction.

3. The same chunk seen in two searches keeps the longest shown span, so a
   later, shorter view never shrinks what an earlier, fuller view made
   citable.

The failure that looks like success: if the ledger is ever seeded from the
corpus, from ``all_chunks``, or from everything the retriever returned rather
than everything the model saw, every real chunk id passes verification and
fabrication detection silently becomes a no-op. Nothing errors; the metric
just reads as an improvement. Guard this with the adversarial tests, not with
inspection.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ragaudit.domain.models import Chunk, RetrievedChunk


@dataclass
class LedgerEntry:
    chunk: Chunk
    shown_text: str
    first_step: int
    retriever: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


class EvidenceLedger:
    """Accumulates the chunks a model was shown over a run.

    Keyed on chunk id. Registering the same chunk again keeps whichever view
    showed more text, so the citable span only ever grows.
    """

    def __init__(self) -> None:
        self._entries: dict[str, LedgerEntry] = {}

    def register(self, chunk: Chunk, shown_text: str, *, step: int, retriever: str = "") -> None:
        """Record that ``shown_text`` for ``chunk`` reached the model at ``step``.

        ``shown_text`` must be exactly what was serialised into the tool result
        — the possibly-truncated view, not ``chunk.quotable_text``. The
        verification check runs against this string.
        """
        shown = (shown_text or "").strip()
        if not shown:
            return
        existing = self._entries.get(chunk.chunk_id)
        if existing is None:
            self._entries[chunk.chunk_id] = LedgerEntry(chunk, shown, step, retriever)
        elif len(shown) > len(existing.shown_text):
            # Keep the fuller view; preserve the earliest step it was seen.
            self._entries[chunk.chunk_id] = LedgerEntry(
                chunk, shown, existing.first_step, retriever or existing.retriever
            )

    def get(self, chunk_id: str) -> LedgerEntry | None:
        return self._entries.get(chunk_id)

    def shown_text(self, chunk_id: str) -> str | None:
        """The exact text a citation to ``chunk_id`` may be checked against."""
        entry = self._entries.get(chunk_id)
        return entry.shown_text if entry else None

    def chunks(self) -> list[Chunk]:
        return [e.chunk for e in self._entries.values()]

    def __contains__(self, chunk_id: str) -> bool:
        return chunk_id in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def as_retrieved(self) -> list[RetrievedChunk]:
        """The ledger as a RetrievedChunk list for the existing finalize path.

        Ordered by the step each chunk first appeared, which is the closest
        thing to a rank a union-over-searches has. This is handed to
        ``_build_answer`` in place of ``for_generation``; verification then
        runs through the unchanged ``_verify_citations``.

        Note: this is deliberately NOT what feeds retrieval metrics. MRR and
        nDCG are rank-sensitive and a union has no global ranking, so the eval
        keeps the first search's ranked result separately (see the plan's
        evaluation-compatibility note).
        """
        ordered = sorted(self._entries.values(), key=lambda e: (e.first_step, e.chunk_id))
        return [
            RetrievedChunk(chunk=e.chunk, score=0.0, rank=i, retriever=e.retriever or "ledger")
            for i, e in enumerate(ordered)
        ]
