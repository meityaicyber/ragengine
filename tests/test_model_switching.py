"""Switching models from .env: config loading, embedder families, index guard, think flag."""

from __future__ import annotations

import json

import pytest
from conftest import FakeEmbedder, make_app

from ragaudit.config import Settings
from ragaudit.generate.llm import OllamaClient
from ragaudit.retrieve.embedder import Embedder
from ragaudit.store.vectors import open_index


def test_env_file_switches_every_model(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "RAGAUDIT_GENERATION__MODEL_NAME=llama3.1:8b\n"
        "RAGAUDIT_EMBEDDING__REPO_ID=BAAI/bge-large-en-v1.5\n"
        "RAGAUDIT_EMBEDDING__MODEL_PATH=models/bge-large-en-v1.5\n"
        "RAGAUDIT_RERANKER__ENABLED=false\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    s = Settings()
    assert s.generation.model_name == "llama3.1:8b"
    assert s.embedding.repo_id == "BAAI/bge-large-en-v1.5"
    assert s.embedding.model_path == "models/bge-large-en-v1.5"
    assert s.reranker.enabled is False


def _model_dir(tmp_path, name, model_type="bert"):
    d = tmp_path / name
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


@pytest.mark.parametrize("name,model_type,pooling,query,passage", [
    ("Qwen3-Embedding-0.6B", "qwen3", "last", "Instruct: I\nQuery: q", "t"),
    ("bge-large-en-v1.5", "bert", "cls", "Represent this sentence for searching relevant passages: q", "t"),
    ("bge-m3", "xlm-roberta", "cls", "q", "t"),
    ("multilingual-e5-large", "xlm-roberta", "mean", "query: q", "passage: t"),
    ("all-MiniLM-L6-v2", "bert", "mean", "q", "t"),
])
def test_embedder_family_defaults(tmp_path, name, model_type, pooling, query, passage):
    e = Embedder(_model_dir(tmp_path, name, model_type), query_instruction="I")
    assert e.pooling == pooling
    assert e.query_template.format(instruction="I", query="q") == query
    assert e.passage_template.format(text="t") == passage


def test_changing_embedding_model_is_refused_until_reset(tmp_path, corpus):
    app = make_app(tmp_path)
    app.build_ingest().ingest_paths(sorted(corpus.glob("*.json")))
    app.close()
    index_dir = tmp_path / "var" / "index"
    # Same model -> opens fine.
    assert len(open_index(index_dir, 1024, fingerprint=FakeEmbedder.fingerprint)) > 0
    # Different model, same dimension -> refused with the fix in the message.
    with pytest.raises(RuntimeError, match="reset --yes"):
        open_index(index_dir, 1024, fingerprint="bge-large-en-v1.5:1024:cls:abcd1234")


def test_think_flag_only_sent_to_thinking_models():
    sent = []

    class Session:
        def post(self, url, json=None, timeout=None):
            sent.append(json)

            class R:
                status_code = 200

                @staticmethod
                def json():
                    return {"message": {"content": "ok"}, "eval_count": 1}
            return R()

    for model, expect in (("qwen3:8b", False), ("llama3.1:8b", None), ("mistral-nemo", None)):
        OllamaClient(model_name=model, session=Session()).chat([{"role": "user", "content": "hi"}])
        assert sent[-1].get("think", None) is expect, model
    # Explicit setting always wins.
    OllamaClient(model_name="llama3.1:8b", think=False, session=Session()).chat([{"role": "user", "content": "hi"}])
    assert sent[-1]["think"] is False
