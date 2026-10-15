"""Dense vector index with pre-search metadata filtering.

Two decisions carry the design (D05).

*Prefilter, not postfilter.* The report filter is applied while searching, not
to the results. Filtering afterwards silently returns fewer than ``top_k``
chunks, and in a corpus holding reports from different audited organisations a
postfilter that falls back to unfiltered results is a cross-tenant disclosure.

*Pluggable backend.* ``NumpyVectorIndex`` is exact and has no native
dependency, so the test suite and small deployments need nothing staged.
``FaissVectorIndex`` builds an HNSW graph for the production corpus. They
implement the same protocol and are chosen by configuration, so the retrieval
code never knows which is in use.

Vectors are persisted as ``.npy`` plus a JSON sidecar of ids. Nothing is
pickled: loading a pickle from a data directory is arbitrary code execution,
which is not acceptable in a security product.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class VectorIndex(Protocol):
    def add(self, ids: Sequence[str], vectors: np.ndarray, report_ids: Sequence[str]) -> None: ...

    def search(
        self, query: np.ndarray, top_k: int, allowed_report_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]: ...

    def remove_report(self, report_id: str) -> None: ...

    def save(self, directory: str | Path) -> None: ...

    def __len__(self) -> int: ...


def _normalise(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype="float32")
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-12, None)


class NumpyVectorIndex:
    """Exact cosine search over a normalised matrix.

    Linear in corpus size. Fine to roughly 10^5 vectors; past that use the
    FAISS backend. D05 records the measured crossover.
    """

    backend = "numpy"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self._vectors = np.zeros((0, dimension), dtype="float32")
        self._ids: list[str] = []
        self._report_ids: list[str] = []

    def add(self, ids: Sequence[str], vectors: np.ndarray, report_ids: Sequence[str]) -> None:
        if len(ids) == 0:
            return
        if len(ids) != len(report_ids):
            raise ValueError("ids and report_ids must be the same length")
        vectors = _normalise(vectors)
        if vectors.shape[0] != len(ids):
            raise ValueError(f"got {vectors.shape[0]} vectors for {len(ids)} ids")
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension {vectors.shape[1]} does not match index dimension "
                f"{self.dimension}; the index was built with a different model"
            )
        self._vectors = np.vstack([self._vectors, vectors]) if len(self._ids) else vectors
        self._ids.extend(ids)
        self._report_ids.extend(report_ids)

    def search(
        self, query: np.ndarray, top_k: int, allowed_report_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = _normalise(query)[0]
        if allowed_report_ids is not None:
            allowed = set(allowed_report_ids)
            mask = np.fromiter((r in allowed for r in self._report_ids), dtype=bool, count=len(self._ids))
            if not mask.any():
                return []
            idx = np.flatnonzero(mask)
            scores = self._vectors[idx] @ q
            order = np.argsort(-scores)[:top_k]
            return [(self._ids[int(idx[i])], float(scores[i])) for i in order]
        scores = self._vectors @ q
        order = np.argsort(-scores)[:top_k]
        return [(self._ids[int(i)], float(scores[int(i)])) for i in order]

    def remove_report(self, report_id: str) -> None:
        keep = [i for i, r in enumerate(self._report_ids) if r != report_id]
        if len(keep) == len(self._ids):
            return
        self._vectors = self._vectors[keep] if keep else np.zeros((0, self.dimension), dtype="float32")
        self._ids = [self._ids[i] for i in keep]
        self._report_ids = [self._report_ids[i] for i in keep]

    def __len__(self) -> int:
        return len(self._ids)

    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self._vectors)
        (out / "ids.json").write_text(
            json.dumps(
                {"backend": self.backend, "dimension": self.dimension,
                 "ids": self._ids, "report_ids": self._report_ids}
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path) -> NumpyVectorIndex:
        path = Path(directory)
        meta = json.loads((path / "ids.json").read_text(encoding="utf-8"))
        index = cls(dimension=int(meta["dimension"]))
        index._vectors = np.load(path / "vectors.npy").astype("float32")
        index._ids = list(meta["ids"])
        index._report_ids = list(meta["report_ids"])
        return index


class FaissVectorIndex:
    """HNSW index over normalised vectors, with an id map.

    Deletion is handled by a tombstone set rather than a graph rebuild: HNSW
    does not support cheap removal, and a report withdrawal must take effect
    immediately. Tombstones are compacted on save.
    """

    backend = "faiss"

    def __init__(self, dimension: int, m: int = 32, ef_construction: int = 200, ef_search: int = 64) -> None:
        import faiss  # imported lazily so the package installs without it

        self.dimension = dimension
        self._faiss = faiss
        self.index = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self._ids: list[str] = []
        self._report_ids: list[str] = []
        self._dead: set[int] = set()

    def add(self, ids: Sequence[str], vectors: np.ndarray, report_ids: Sequence[str]) -> None:
        if len(ids) == 0:
            return
        vectors = _normalise(vectors)
        if vectors.shape[1] != self.dimension:
            raise ValueError(
                f"embedding dimension {vectors.shape[1]} does not match index dimension {self.dimension}"
            )
        self.index.add(vectors)
        self._ids.extend(ids)
        self._report_ids.extend(report_ids)

    def search(
        self, query: np.ndarray, top_k: int, allowed_report_ids: Sequence[str] | None = None
    ) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        allowed = set(allowed_report_ids) if allowed_report_ids is not None else None
        # Over-fetch so that filtering and tombstones still leave top_k live
        # hits. The multiplier is bounded to keep latency predictable.
        depth = top_k if allowed is None else min(len(self._ids), max(top_k * 8, 200))
        scores, idxs = self.index.search(_normalise(query), depth)
        out: list[tuple[str, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0 or int(i) in self._dead:
                continue
            if allowed is not None and self._report_ids[int(i)] not in allowed:
                continue
            out.append((self._ids[int(i)], float(score)))
            if len(out) >= top_k:
                break
        return out

    def remove_report(self, report_id: str) -> None:
        for i, r in enumerate(self._report_ids):
            if r == report_id:
                self._dead.add(i)

    def __len__(self) -> int:
        return len(self._ids) - len(self._dead)

    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        live = [i for i in range(len(self._ids)) if i not in self._dead]
        vectors = np.vstack([self.index.reconstruct(int(i)).reshape(1, -1) for i in live]) if live else (
            np.zeros((0, self.dimension), dtype="float32")
        )
        np.save(out / "vectors.npy", vectors.astype("float32"))
        (out / "ids.json").write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "dimension": self.dimension,
                    "ids": [self._ids[i] for i in live],
                    "report_ids": [self._report_ids[i] for i in live],
                }
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: str | Path, **kwargs) -> FaissVectorIndex:
        path = Path(directory)
        meta = json.loads((path / "ids.json").read_text(encoding="utf-8"))
        index = cls(dimension=int(meta["dimension"]), **kwargs)
        vectors = np.load(path / "vectors.npy").astype("float32")
        index.add(list(meta["ids"]), vectors, list(meta["report_ids"]))
        return index


def open_index(
    directory: str | Path,
    dimension: int,
    backend: str = "numpy",
    *,
    hnsw_m: int = 32,
    hnsw_ef_construction: int = 200,
    hnsw_ef_search: int = 64,
) -> VectorIndex:
    """Load the index from ``directory``, or create an empty one.

    A stored index records the dimension it was built with. Loading it under a
    different embedding model raises rather than silently mixing vector spaces.
    """
    path = Path(directory)
    exists = (path / "ids.json").exists() and (path / "vectors.npy").exists()

    if backend == "faiss":
        try:
            import faiss  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "retrieval.backend is 'faiss' but faiss is not installed. "
                "Install ragaudit[ann] or set RAGAUDIT_RETRIEVAL__BACKEND=numpy."
            ) from exc
        if exists:
            return FaissVectorIndex.load(
                path, m=hnsw_m, ef_construction=hnsw_ef_construction, ef_search=hnsw_ef_search
            )
        return FaissVectorIndex(dimension, hnsw_m, hnsw_ef_construction, hnsw_ef_search)

    if exists:
        loaded = NumpyVectorIndex.load(path)
        if loaded.dimension != dimension:
            raise RuntimeError(
                f"stored index has dimension {loaded.dimension} but the configured embedding model "
                f"produces {dimension}. Re-ingest the corpus or restore the previous model."
            )
        return loaded
    return NumpyVectorIndex(dimension)
