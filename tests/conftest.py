"""Test fixtures: the real system with the two heavy pieces swapped for fakes.

``FakeEmbedder`` is a deterministic hashed bag-of-words embedder (1024-d, same
interface as the Qwen3 embedder), so dense retrieval behaves sensibly without
weights. ``FakeLLM`` speaks the OllamaClient interface and plays a small,
well-behaved tool-calling agent: it picks a tool from the question, then
finalises with a citation copied from the evidence it was shown. A ``liar``
mode cites chunk ids and quotes it never saw, to prove verification catches it.

Everything else (catalogue, vector index, ingest, router, agent loop, ledger,
citation verification, API) is the production code.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from ragaudit.app import Application
from ragaudit.config import load_settings
from ragaudit.generate.llm import Completion, ToolCall

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


class FakeEmbedder:
    dimension = 1024
    fingerprint = "fake-hash-embedder:1024"

    def __init__(self, fail_on_call: int | None = None, exc: type[BaseException] = RuntimeError):
        self.calls = 0
        self.fail_on_call = fail_on_call
        self.exc = exc

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimension, dtype="float32")
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dimension] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_passages(self, texts):
        self.calls += 1
        if self.fail_on_call is not None and self.calls == self.fail_on_call:
            raise self.exc("simulated embedder failure")
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dimension), "float32")

    def embed_queries(self, queries):
        return np.stack([self._vec(q) for q in queries])


_CHUNK_RE = re.compile(r"\[chunk_id=([0-9a-f]+) \|[^\]]*\]\n(.+)")


class FakeLLM:
    model_name = "fake-llm"
    prompt_budget_tokens = 12000

    def __init__(self, *, liar: bool = False, lazy: bool = False):
        self.liar = liar
        self.lazy = lazy  # never calls a tool, like a model that answers from its head
        self.calls: list[dict] = []

    # -- OllamaClient interface ---------------------------------------
    def health(self) -> bool:
        return True

    def close(self) -> None:
        pass

    def generate(self, prompt, *, system=None, format_schema=None, max_tokens=None, timeout_s=None):
        # Used for follow-up rewriting: echo the follow-up line back.
        m = re.search(r"Follow-up question: (.+)", prompt)
        return Completion(text=m.group(1) if m else prompt[:200], latency_s=0.0, model=self.model_name)

    def chat(self, messages, *, format_schema=None, tools=None, max_tokens=None, timeout_s=None):
        self.calls.append({"tools": bool(tools), "schema": bool(format_schema)})
        if format_schema:
            return self._finalize(messages[-1]["content"])
        return self._agent_step(messages)

    # -- behaviour ----------------------------------------------------
    def _agent_step(self, messages):
        if messages[-1]["role"] == "tool":
            # Finish. A narrative search needs a written answer; any other single
            # result is shown as it is (the model's SHOW / WRITE decision).
            first_call = next((m for m in messages if m.get("role") == "assistant" and m.get("tool_calls")), None)
            searched = first_call and "search" in json.dumps(first_call.get("tool_calls"))
            return Completion(text="WRITE" if searched else "SHOW", latency_s=0.0, model=self.model_name)
        if self.lazy:
            return Completion(text="I think there are about 12.", latency_s=0.0, model=self.model_name)
        question = messages[-1]["content"].split("\n\n(resolved context")[0]
        q = question.lower()
        if "types of information" in q or "what fields" in q:
            call = ToolCall("describe_schema", {})
        elif "failed validation" in q or "quarantined" in q:
            call = ToolCall("report_quality", {})
        elif "reports" in q and "finding" not in q and ("how many" in q or "list" in q):
            call = ToolCall("list_reports", {})
        elif "repeat" in q:
            call = ToolCall("find_repeat_findings", {})
        elif "compare" in q:
            call = ToolCall("compare_reports", {})
        elif "which findings cite" in q:
            call = ToolCall("query_findings", {"cwe": re.search(r"CWE-\d+", question).group(0)})
        elif "most common cwe" in q:
            call = ToolCall("top_values", {"field": "cwe", "n": 8})
        elif "corpus stats" in q:
            call = ToolCall("corpus_stats", {})
        elif "how many" in q:
            args = {}
            for sev in ("critical", "high", "medium", "low"):
                if sev in q:
                    args["severity"] = sev  # lowercase on purpose: coercion must fix it
            if "open" in q:
                args["status"] = "open"
            call = ToolCall("count_findings", args)
        elif "break down" in q or "by severity" in q:
            args = {"by": "severity"}
            if "and status" in q:
                args["then_by"] = "status"
            call = ToolCall("aggregate_findings", args)
        else:
            call = ToolCall("search_reports", {"query": question})
        return Completion(text="", latency_s=0.0, model=self.model_name, tool_calls=[call])

    def _finalize(self, prompt: str):
        evidence = _CHUNK_RE.findall(prompt)
        if self.liar:
            payload = {"evidence_review": "x", "answer": "The server was patched in 2019.",
                       "citations": [{"chunk_id": "deadbeefdeadbeef", "quote": "patched in 2019"}],
                       "answered": True}
        elif not evidence:
            payload = {"evidence_review": "none", "answer": "The evidence does not say.",
                       "citations": [], "answered": False, "missing_information": "no passages"}
        else:
            cid, first_line = evidence[0]
            quote = " ".join(first_line.split()[:8])
            payload = {"evidence_review": f"{cid} is relevant.",
                       "answer": f"According to the report: {quote}",
                       "citations": [{"chunk_id": cid, "quote": quote}], "answered": True}
        return Completion(text=json.dumps(payload), latency_s=0.0, model=self.model_name)


def make_app(tmp_path: Path, *, embedder=None, llm=None, **overrides) -> Application:
    settings = load_settings(
        data_dir=tmp_path / "var",
        catalog_path=tmp_path / "var" / "catalog.sqlite3",
        index_dir=tmp_path / "var" / "index",
        figure_dir=tmp_path / "var" / "figures",
        reranker={"enabled": False},
        **overrides,
    )
    app = Application(settings, configure_logs=False)
    app._embedder = embedder or FakeEmbedder()
    app._llm = llm or FakeLLM()
    return app


@pytest.fixture
def corpus(tmp_path) -> Path:
    """A private copy of the sample corpus."""
    dest = tmp_path / "corpus"
    shutil.copytree(SAMPLE, dest)
    return dest


@pytest.fixture
def app(tmp_path, corpus):
    application = make_app(tmp_path)
    application.build_ingest().ingest_paths(sorted(corpus.glob("*.json")))
    yield application
    application.close()


def write_report(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def mini_report(findings, **meta) -> dict:
    return {"report_meta": {"report_title": "Mini report", "classification": "Internal", **meta},
            "detailed_observations": findings}


def finding(fid, title="Some issue", severity="High", status="Open", cwe="CWE-79") -> dict:
    return {"finding_id": fid, "title": title, "severity": severity, "status": status, "cwe": cwe,
            "observation_description": f"Observation text for {title}.",
            "recommendation": "Fix it properly."}
