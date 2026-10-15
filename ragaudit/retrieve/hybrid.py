"""Hybrid retrieval: dense + lexical, fused by Reciprocal Rank Fusion.

Why RRF over score blending (D05): dense cosine similarity and BM25 live on
different, query-dependent scales, so any weighted sum of the two raw scores is
arbitrary. RRF consumes only the rank each retriever assigns, which is the
property both agree on.

For the same reason, downstream confidence gating uses *rank*, not score. The
previous implementation compared a single ``min_retrieval_score`` against
whichever of four incomparable scales happened to be active -- cross-encoder
logits (unbounded, often negative), RRF scores (bounded near 0.03), cosine
similarity, and a synthetic ``1.0/(rank+1)``. That threshold could not be
meaningful for more than one of them at a time.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from ragaudit.domain.models import Chunk, RetrievedChunk
from ragaudit.retrieve.embedder import CrossEncoderReranker, Embedder
from ragaudit.store.catalog import Catalog
from ragaudit.store.vectors import VectorIndex

logger = logging.getLogger("ragaudit.retrieve.hybrid")


@dataclass
class RetrievalConfig:
    top_k: int = 20
    candidates_k: int = 60
    rrf_k: int = 60
    use_reranker: bool = True


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[str, float]]:
    """Fuse ranked id lists. Returns (id, score) best first."""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


class HybridRetriever:
    """Dense + lexical retrieval over one corpus, with optional reranking."""

    def __init__(
        self,
        catalog: Catalog,
        index: VectorIndex,
        embedder: Embedder,
        reranker: CrossEncoderReranker | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.catalog = catalog
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        query: str,
        *,
        report_ids: Sequence[str] | None = None,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve chunks for one query.

        ``report_ids`` is a hard scope boundary, applied inside both retrievers.
        An empty scope returns nothing; it never widens to the whole corpus.
        """
        top_k = top_k or self.config.top_k
        if report_ids is not None and len(report_ids) == 0:
            return []

        dense_hits = self._dense(query, report_ids)
        lexical_hits = self.catalog.search_lexical(
            query, report_ids=report_ids, limit=self.config.candidates_k
        )

        fused = reciprocal_rank_fusion(
            [[cid for cid, _ in dense_hits], [cid for cid, _ in lexical_hits]],
            k=self.config.rrf_k,
        )
        if not fused:
            return []

        candidate_ids = [cid for cid, _ in fused[: self.config.candidates_k]]
        rows = self.catalog.get_chunks(candidate_ids)
        chunks = [_row_to_chunk(rows[cid]) for cid in candidate_ids if cid in rows]
        if not chunks:
            return []

        if self.reranker and self.config.use_reranker:
            scores = self.reranker.score(query, [c.embedding_text for c in chunks])
            ranked = sorted(zip(chunks, scores), key=lambda p: p[1], reverse=True)[:top_k]
            retriever_name = "hybrid+rerank"
        else:
            score_by_id = dict(fused)
            ranked = [(c, score_by_id.get(c.chunk_id, 0.0)) for c in chunks][:top_k]
            retriever_name = "hybrid"

        return [
            RetrievedChunk(chunk=chunk, score=float(score), rank=i, retriever=retriever_name)
            for i, (chunk, score) in enumerate(ranked)
        ]

    def _dense(self, query: str, report_ids: Sequence[str] | None) -> list[tuple[str, float]]:
        vector = self.embedder.embed_queries([query])
        return self.index.search(vector, self.config.candidates_k, allowed_report_ids=report_ids)


def _row_to_chunk(row) -> Chunk:
    import json

    return Chunk(
        chunk_id=row["chunk_id"],
        report_id=row["report_id"],
        section=row["section"],
        field_path=row["field_path"],
        text=row["text"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        token_count=row["token_count"],
        finding_id=row["finding_id"],
        figure_tokens=json.loads(row["figures_json"]) if row["figures_json"] else [],
        context_prefix=row["context_prefix"],
    )
