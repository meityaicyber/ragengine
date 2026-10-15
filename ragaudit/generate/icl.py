"""In-context learning: demonstration banks and selection strategies.

The brief names In-Context Learning as a required capability, so this is a
product feature rather than an ablation axis (D06).

The prior evaluation found retrieval few-shot (62.5%) scoring *below*
zero-shot (75%) and concluded ICL hurt. That conclusion is unsafe: the
demonstrations being compared taught the model to emit empty citation lists
and ellipsis-truncated quotes, which the scorer then penalised. The
demonstrations here cite real ids and quote in full, so the comparison
measures ICL rather than a defect in the renderer.

Selection strategies are kept so the comparison can be re-run properly.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from ragaudit.generate.prompts import Demonstration


class DemonstrationSelector(Protocol):
    name: str

    def select(self, query: str, k: int) -> list[Demonstration]: ...


#: Built-in bank. These are synthetic by design: a demonstration must never be
#: a real passage from an indexed report, or the model may quote it as if it
#: were evidence for the question actually being asked.
DEFAULT_BANK: list[Demonstration] = [
    Demonstration(
        question="What access-control weaknesses did the assessment identify?",
        evidence=[
            (
                "d1a2b3c4",
                "[OBS-04] Missing authorisation on admin endpoint | Severity: High | Status: Open "
                "Observation: The /admin/users endpoint returned full user records to a "
                "session authenticated as a standard, non-administrative user.",
            ),
            (
                "d5e6f7a8",
                "[OBS-09] Verbose error messages | Severity: Low | Status: Closed "
                "Observation: Stack traces disclosing framework versions were returned on malformed input.",
            ),
        ],
        evidence_review=(
            "OBS-04 is an access-control failure and bears directly on the question. "
            "OBS-09 is information disclosure, not access control, so it is excluded."
        ),
        answer=(
            "One access-control weakness was identified. OBS-04, rated High and still Open, "
            "found that the /admin/users endpoint returned full user records to a session "
            "authenticated as a standard, non-administrative user, meaning authorisation was "
            "not enforced at that endpoint."
        ),
        citations=[
            (
                "d1a2b3c4",
                "The /admin/users endpoint returned full user records to a session "
                "authenticated as a standard, non-administrative user.",
            )
        ],
    ),
    Demonstration(
        question="Does the report say whether the database server was patched?",
        evidence=[
            (
                "b9c8d7e6",
                "[OBS-02] Outdated TLS configuration | Severity: Medium | Status: Open "
                "Observation: The web tier negotiated TLS 1.0 on two endpoints.",
            ),
        ],
        evidence_review=(
            "The only passage available concerns TLS configuration on the web tier. "
            "Nothing addresses database server patching."
        ),
        answer=(
            "The evidence available does not address database server patching. The passages "
            "retrieved cover TLS configuration on the web tier only. The report may discuss "
            "patching elsewhere, but it is not in the evidence provided here."
        ),
        citations=[],
        answered=False,
        missing_information=(
            "A passage covering the database server's patch level, or a scope statement "
            "saying whether the database tier was in scope."
        ),
    ),
    Demonstration(
        question="How severe was the token-management issue and what was recommended?",
        evidence=[
            (
                "f1e2d3c4",
                "[OBS-01] Broken access control on token API | Severity: High | Status: Open | "
                "CWE: CWE-284 Observation: PATCH token endpoints allowed suspend and activate "
                "operations without authentication. Recommendation: Enforce authentication and "
                "per-object authorisation on all token state transitions before release.",
            ),
        ],
        evidence_review="OBS-01 covers both the severity and the recommendation asked about.",
        answer=(
            "OBS-01 is rated High and remains Open. It found that PATCH token endpoints allowed "
            "suspend and activate operations without authentication, classified as CWE-284. The "
            "report recommends enforcing authentication and per-object authorisation on all token "
            "state transitions before release."
        ),
        citations=[
            (
                "f1e2d3c4",
                "Enforce authentication and per-object authorisation on all token state "
                "transitions before release.",
            )
        ],
    ),
]


def load_bank(path: str | Path | None = None) -> list[Demonstration]:
    """Load a demonstration bank, falling back to the built-in set."""
    if path is None:
        return list(DEFAULT_BANK)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    bank: list[Demonstration] = []
    for item in raw:
        bank.append(
            Demonstration(
                question=item["question"],
                evidence=[(e["chunk_id"], e["text"]) for e in item.get("evidence", [])],
                evidence_review=item.get("evidence_review", ""),
                answer=item["answer"],
                citations=[(c["chunk_id"], c["quote"]) for c in item.get("citations", [])],
                answered=item.get("answered", True),
                missing_information=item.get("missing_information", ""),
            )
        )
    return bank


class ZeroShot:
    name = "zero_shot"

    def select(self, query: str, k: int = 0) -> list[Demonstration]:
        return []


class FixedFewShot:
    """The same demonstrations every time.

    Chosen to span the behaviours the model must learn: a grounded answer, an
    honest abstention, and a multi-field answer. Coverage of the abstention
    case matters most -- it is the behaviour a model is least likely to
    produce unprompted.
    """

    name = "fixed_few_shot"

    def __init__(self, bank: Sequence[Demonstration] | None = None, k: int = 3) -> None:
        self.bank = list(bank or DEFAULT_BANK)
        self.k = k

    def select(self, query: str, k: int | None = None) -> list[Demonstration]:
        limit = k or self.k
        answered = [d for d in self.bank if d.answered]
        abstained = [d for d in self.bank if not d.answered]
        # Always keep at least one abstention example in the window.
        chosen = answered[: max(1, limit - 1)] + abstained[:1]
        return chosen[:limit]


class RetrievalFewShot:
    """Demonstrations nearest the query, with abstention coverage guaranteed.

    Pure nearest-neighbour selection can return an all-answered window, which
    teaches the model that an answer always exists. One abstention slot is
    reserved -- that is the diversity guarantee the previous implementation
    documented but did not implement.
    """

    name = "retrieval_few_shot"

    def __init__(self, bank: Sequence[Demonstration], embedder, k: int = 3) -> None:
        self.bank = list(bank)
        self.embedder = embedder
        self.k = k
        self._vectors: np.ndarray | None = None

    def _ensure_vectors(self) -> None:
        if self._vectors is None:
            texts = [d.question for d in self.bank]
            self._vectors = self.embedder.embed_passages(texts) if texts else np.zeros((0, 1))

    def select(self, query: str, k: int | None = None) -> list[Demonstration]:
        limit = k or self.k
        if not self.bank:
            return []
        self._ensure_vectors()
        query_vector = self.embedder.embed_queries([query])[0]
        scores = np.asarray(self._vectors) @ query_vector
        order = [int(i) for i in np.argsort(-scores)]

        chosen: list[Demonstration] = []
        has_abstention = False
        for index in order:
            demo = self.bank[index]
            if len(chosen) >= limit:
                break
            chosen.append(demo)
            has_abstention = has_abstention or not demo.answered

        if not has_abstention:
            for index in order:
                if not self.bank[index].answered:
                    if chosen:
                        chosen[-1] = self.bank[index]
                    else:
                        chosen.append(self.bank[index])
                    break
        return chosen


STRATEGIES = {
    "zero_shot": ZeroShot,
    "fixed_few_shot": FixedFewShot,
    "retrieval_few_shot": RetrievalFewShot,
}
