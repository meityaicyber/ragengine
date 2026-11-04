"""Runtime configuration.

Every value is overridable by environment variable (prefix ``RAGAUDIT_``) or by
an explicit settings object passed at construction. Nothing in this package
reads a hardcoded host, path or model name at call time -- D03 records why.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingSettings(BaseModel):
    # Hugging Face repo that `python run.py models` downloads into model_path.
    repo_id: str = "Qwen/Qwen3-Embedding-0.6B"
    model_path: str = "models/Qwen3-Embedding-0.6B"
    # How a sentence vector is pooled from token states. "auto" picks from the
    # model: last-token for Qwen3/decoder embedders, CLS for BGE, mean otherwise.
    pooling: Literal["auto", "last", "cls", "mean"] = "auto"
    # Query/passage formatting. "auto" applies each family's documented format
    # (Qwen3 "Instruct: ...\nQuery: ...", BGE query prefix, E5 "query:"/"passage:").
    # Custom: a string containing {query} (and optionally {instruction}) /
    # containing {text}.
    query_template: str = "auto"
    passage_template: str = "auto"
    device: str = "cpu"
    batch_size: int = 32
    max_sequence_tokens: int = 1024
    # Qwen3-Embedding is instruction-aware; the retrieval instruction is part of
    # the query, not the passage. Changing it invalidates an existing index.
    query_instruction: str = (
        "Given a question about a cybersecurity audit report, "
        "retrieve the audit evidence that answers it"
    )
    # Fallback only: the real dimension is probed from the loaded model.
    dimension: int = 1024


class RerankerSettings(BaseModel):
    enabled: bool = True
    repo_id: str = "Qwen/Qwen3-Reranker-0.6B"
    model_path: str = "models/Qwen3-Reranker-0.6B"
    device: str = "cpu"
    batch_size: int = 16


class RetrievalSettings(BaseModel):
    # Anthropic's contextual-retrieval study found 20 retrieved chunks
    # outperformed 5 and 10; we rerank a wider candidate pool down to this.
    top_k: int = 20
    candidates_k: int = 60
    # Chunks actually placed in the generation prompt. Retrieval stays wide for
    # recall, but handing a small model twenty passages makes it try to cover
    # all of them and run out of output budget mid-answer. The reranked head
    # carries the signal; the tail mostly adds length.
    generation_k: int = 8
    # How many fused candidates the cross-encoder reranks. Reranking is the
    # slowest retrieval step on CPU (one 0.6B forward pass per passage); the
    # rest of the candidates keep their fused order behind the reranked head.
    rerank_k: int = 24
    rrf_k: int = 60
    # Rank-based, not score-based: fused and cross-encoder scores are not on a
    # common scale, so an absolute cutoff is meaningless. See D05.
    min_rank_confidence: int = 3
    backend: Literal["numpy", "faiss"] = "numpy"
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64


class GenerationSettings(BaseModel):
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen3:8b"
    temperature: float = 0.0
    seed: int = 42
    max_tokens: int = 2048
    # Ollama defaults num_ctx to 4096 and silently truncates anything longer,
    # which drops the tail of the prompt -- including the response-format
    # instruction -- and yields malformed output with no error. Always set it
    # explicitly, and keep the prompt budget below it. See D03.
    context_tokens: int = 16384
    top_p: float = 1.0
    # Qwen3-style "thinking". None = auto: send think=false to model families
    # that support the switch (qwen3, deepseek-r1, gpt-oss, ...) and omit it for
    # others, which may reject the option.
    think: bool | None = None
    # How long Ollama keeps the model in memory after a request. Its default
    # (5m) means the first question after any pause pays a full cold load,
    # which on CPU can exceed the agent's per-step timeout.
    keep_alive: str = "60m"
    timeout_s: float = 120.0
    connect_timeout_s: float = 5.0
    max_retries: int = 3
    backoff_base_s: float = 1.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_s: float = 30.0


class ChatSettings(BaseModel):
    max_history_turns: int = 8
    rewrite_queries: bool = True
    session_ttl_s: int = 3600


class AgentSettings(BaseModel):
    # Agent-first: every non-chitchat question goes through the tool loop.
    # The kill switch (false) restores the single-pass router path, which is
    # how the pre-agent test suite stays meaningful and how to A/B.
    enabled: bool = True
    max_steps: int = 10
    deadline_s: float = 90.0
    step_timeout_s: float = 45.0
    max_tool_errors: int = 4
    max_searches: int = 4
    no_progress_steps: int = 3
    step_max_tokens: int = 384
    max_exposed_tools: int = 24
    prompt_token_ceiling: int = 12000


class IngestSettings(BaseModel):
    chunk_max_tokens: int = 420
    chunk_overlap_tokens: int = 60
    chunk_min_tokens: int = 40
    fail_on_validation_error: bool = True
    # Save the vector index to disk after this many newly indexed reports, so
    # an interrupted ingest loses at most this many reports' embeddings (and
    # the next run re-embeds them automatically). The removed ``workers`` and
    # ``contextualize`` options were never wired to anything.
    checkpoint_every: int = 10


class ApiSettings(BaseModel):
    # Optional shared secret. When set, every endpoint except /healthz needs
    # header ``X-API-Key: <value>``; the web console asks for it once.
    # Empty (default) means no auth, so keep the server on 127.0.0.1.
    api_key: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAGAUDIT_",
        env_nested_delimiter="__",
        extra="ignore",
        # Settings can also live in a .env file in the working directory, so
        # run.py and the server pick up the same values without exporting
        # variables by hand.
        env_file=".env",
        env_file_encoding="utf-8",
    )

    data_dir: Path = Path("var")
    catalog_path: Path = Path("var/catalog.sqlite3")
    index_dir: Path = Path("var/index")
    figure_dir: Path = Path("var/figures")
    log_level: str = "INFO"
    log_json: bool = True

    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    api: ApiSettings = Field(default_factory=ApiSettings)

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.index_dir, self.figure_dir):
            path.mkdir(parents=True, exist_ok=True)
        self.catalog_path.parent.mkdir(parents=True, exist_ok=True)


def load_settings(**overrides) -> Settings:
    return Settings(**overrides)
