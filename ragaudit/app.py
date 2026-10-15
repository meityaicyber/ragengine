"""Application assembly.

One place that turns settings into a working system, so the API, the CLI and
the evaluation harness all construct the same objects the same way.
"""

from __future__ import annotations

import logging

from ragaudit.chat.service import ChatService
from ragaudit.chat.session import SessionStore
from ragaudit.config import Settings, load_settings
from ragaudit.generate.icl import FixedFewShot, RetrievalFewShot, ZeroShot, load_bank
from ragaudit.generate.llm import OllamaClient
from ragaudit.ingest.chunking import FieldAwareChunker
from ragaudit.ingest.pipeline import IngestPipeline
from ragaudit.observability import configure_logging
from ragaudit.retrieve.embedder import CrossEncoderReranker, Embedder
from ragaudit.retrieve.hybrid import HybridRetriever, RetrievalConfig
from ragaudit.store.catalog import Catalog
from ragaudit.store.vectors import open_index

logger = logging.getLogger("ragaudit.app")


class Application:
    """Lazily constructed components, wired from one Settings object."""

    def __init__(self, settings: Settings | None = None, *, configure_logs: bool = True) -> None:
        self.settings = settings or load_settings()
        self.settings.ensure_dirs()
        if configure_logs:
            configure_logging(self.settings.log_level, self.settings.log_json)
        self._catalog: Catalog | None = None
        self._embedder = None
        self._reranker = None
        self._index = None
        self._llm: OllamaClient | None = None
        self._sessions = SessionStore(ttl_s=self.settings.chat.session_ttl_s)

    # -- components ------------------------------------------------------

    @property
    def catalog(self) -> Catalog:
        if self._catalog is None:
            self._catalog = Catalog(self.settings.catalog_path)
        return self._catalog

    @property
    def embedder(self):
        if self._embedder is None:
            cfg = self.settings.embedding
            self._embedder = Embedder(
                cfg.model_path,
                device=cfg.device,
                batch_size=cfg.batch_size,
                max_sequence_tokens=cfg.max_sequence_tokens,
                query_instruction=cfg.query_instruction,
            )
        return self._embedder

    @property
    def reranker(self):
        cfg = self.settings.reranker
        if not cfg.enabled:
            return None
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(
                cfg.model_path, device=cfg.device, batch_size=cfg.batch_size
            )
        return self._reranker

    @property
    def index(self):
        if self._index is None:
            cfg = self.settings.retrieval
            self._index = open_index(
                self.settings.index_dir,
                dimension=self.settings.embedding.dimension,
                backend=cfg.backend,
                hnsw_m=cfg.hnsw_m,
                hnsw_ef_construction=cfg.hnsw_ef_construction,
                hnsw_ef_search=cfg.hnsw_ef_search,
            )
        return self._index

    @property
    def llm(self) -> OllamaClient:
        if self._llm is None:
            cfg = self.settings.generation
            self._llm = OllamaClient(
                base_url=cfg.base_url,
                model_name=cfg.model_name,
                temperature=cfg.temperature,
                seed=cfg.seed,
                max_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
                think=cfg.think,
                context_tokens=cfg.context_tokens,
                timeout_s=cfg.timeout_s,
                connect_timeout_s=cfg.connect_timeout_s,
                max_retries=cfg.max_retries,
                backoff_base_s=cfg.backoff_base_s,
                circuit_breaker_threshold=cfg.circuit_breaker_threshold,
                circuit_breaker_reset_s=cfg.circuit_breaker_reset_s,
            )
        return self._llm

    # -- services --------------------------------------------------------

    def build_retriever(self) -> HybridRetriever:
        cfg = self.settings.retrieval
        return HybridRetriever(
            self.catalog,
            self.index,
            self.embedder,
            reranker=self.reranker,
            config=RetrievalConfig(
                top_k=cfg.top_k,
                candidates_k=cfg.candidates_k,
                rrf_k=cfg.rrf_k,
                use_reranker=self.settings.reranker.enabled,
            ),
        )

    def build_selector(self, strategy: str = "fixed_few_shot", k: int = 3):
        bank = load_bank()
        if strategy == "zero_shot":
            return ZeroShot()
        if strategy == "retrieval_few_shot":
            return RetrievalFewShot(bank, self.embedder, k=k)
        return FixedFewShot(bank, k=k)

    def build_agent(self):
        """The agent loop, or None when disabled (single-pass fallback)."""
        if not self.settings.agent.enabled:
            return None
        from ragaudit.agent.loop import AgentConfig, AgentLoop
        from ragaudit.agent.tools import build_registry

        cfg = self.settings.agent
        return AgentLoop(
            self.llm,
            build_registry(),
            AgentConfig(
                max_steps=cfg.max_steps,
                deadline_s=cfg.deadline_s,
                step_timeout_s=cfg.step_timeout_s,
                max_tool_errors=cfg.max_tool_errors,
                max_searches=cfg.max_searches,
                no_progress_steps=cfg.no_progress_steps,
                step_max_tokens=cfg.step_max_tokens,
                max_exposed_tools=cfg.max_exposed_tools,
                prompt_token_ceiling=cfg.prompt_token_ceiling,
            ),
        )

    def build_chat(self, icl_strategy: str = "fixed_few_shot", icl_k: int = 3) -> ChatService:
        return ChatService(
            self.catalog,
            self.build_retriever(),
            self.llm,
            sessions=self._sessions,
            selector=self.build_selector(icl_strategy, icl_k),
            max_history_turns=self.settings.chat.max_history_turns,
            rewrite_queries=self.settings.chat.rewrite_queries,
            icl_k=icl_k,
            generation_k=self.settings.retrieval.generation_k,
            agent=self.build_agent(),
            settings=self.settings,
        )

    def build_ingest(self) -> IngestPipeline:
        cfg = self.settings.ingest
        return IngestPipeline(
            self.catalog,
            self.index,
            self.embedder,
            chunker=FieldAwareChunker(
                max_tokens=cfg.chunk_max_tokens,
                overlap_tokens=cfg.chunk_overlap_tokens,
                min_tokens=cfg.chunk_min_tokens,
            ),
            figure_dir=self.settings.figure_dir,
            fail_on_validation_error=cfg.fail_on_validation_error,
        )

    # -- health ----------------------------------------------------------

    def readiness(self) -> dict:
        """What is actually ready, component by component.

        Reported honestly: a missing model or an empty index is a 'not ready'
        signal, not something to discover on the first user question.
        """
        checks: dict[str, object] = {}
        try:
            checks["catalog"] = self.catalog.stats()
        except Exception as exc:
            checks["catalog"] = f"error: {exc}"
        try:
            checks["index_vectors"] = len(self.index)
        except Exception as exc:
            checks["index_vectors"] = f"error: {exc}"
        try:
            checks["embedding_model"] = self.embedder.fingerprint
        except Exception as exc:
            checks["embedding_model"] = f"unavailable: {exc}"
        checks["llm_reachable"] = self.llm.health()
        ready = (
            isinstance(checks.get("index_vectors"), int)
            and checks["index_vectors"] > 0
            and checks["llm_reachable"] is True
            and not str(checks.get("embedding_model", "")).startswith("unavailable")
        )
        return {"ready": ready, "checks": checks}

    def close(self) -> None:
        if self._catalog is not None:
            self._catalog.close()
        if self._llm is not None:
            self._llm.close()
