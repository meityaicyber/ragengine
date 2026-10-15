"""Local LLM client.

Hardening over the previous implementation (D03):

* base URL comes from configuration, never a module constant;
* one pooled ``requests.Session`` rather than a fresh connection per call;
* separate connect and read timeouts;
* retries only on transient failures -- a 400 from a malformed request will
  never succeed on attempt three, and retrying it just triples the latency of
  a guaranteed failure;
* exponential backoff with jitter, so concurrent workers do not retry in step;
* a circuit breaker, so a dead backend fails fast instead of making every
  request wait out its full retry budget.

Schema note. Constrained decoding removes malformed JSON but costs correctness,
worst on small models, partly because the schema forces answer tokens before
reasoning tokens. Where a schema is used here, reasoning fields are declared
*before* the conclusion they support.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any

import requests

logger = logging.getLogger("ragaudit.generate.llm")


class LLMError(RuntimeError):
    pass


class LLMUnavailable(LLMError):
    """The backend is unreachable or the circuit is open."""


@dataclass
class ToolCall:
    """One tool invocation requested by the model.

    ``arguments`` is already a dict: Ollama returns parsed arguments, unlike
    the OpenAI wire format where they arrive as a JSON string. ``call_id`` is
    often absent, so nothing may depend on it being present.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str = ""

    def signature(self) -> str:
        """Stable key for duplicate-call detection.

        With ``temperature=0`` and a fixed seed, a stuck model repeats a call
        byte-identically forever, so the loop needs to recognise a repeat.
        """
        return f"{self.name}({json.dumps(self.arguments, sort_keys=True, default=str)})"


@dataclass
class Completion:
    text: str
    latency_s: float
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    attempts: int = 1
    tool_calls: list[ToolCall] = field(default_factory=list)
    #: True when generation stopped because it hit the token cap rather
    #: than because the model finished. A truncated structured response is
    #: unparseable, and 'ran out of room' is a far more actionable
    #: diagnosis than 'malformed output'.
    truncated: bool = False


class _CircuitBreaker:
    def __init__(self, threshold: int, reset_after_s: float) -> None:
        self.threshold = threshold
        self.reset_after_s = reset_after_s
        self._failures = 0
        self._opened_at: float | None = None

    @property
    def is_open(self) -> bool:
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at >= self.reset_after_s:
            # Half-open: allow one probe through.
            self._opened_at = None
            self._failures = self.threshold - 1
            return False
        return True

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self.threshold and self._opened_at is None:
            self._opened_at = time.monotonic()
            logger.warning("llm circuit opened after %d consecutive failures", self._failures)


# Status codes worth retrying. Everything else is a client error that will
# fail identically on every attempt.
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _parse_tool_calls(message: dict[str, Any]) -> list[ToolCall]:
    """Extract tool calls from an Ollama chat message.

    Tolerant by design. A malformed entry is skipped rather than raising:
    the loop can recover by telling the model its call was unreadable, but it
    cannot recover from an exception thrown inside the client. Ollama returns
    ``arguments`` already parsed, but older builds and some proxies send a
    JSON string, so both are accepted.
    """
    raw = message.get("tool_calls")
    if not isinstance(raw, list):
        return []

    calls: list[ToolCall] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function")
        if not isinstance(function, dict):
            continue
        name = str(function.get("name") or "").strip()
        if not name:
            continue

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                logger.warning("tool call %s had unparseable argument string", name)
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}

        calls.append(
            ToolCall(name=name, arguments=arguments, call_id=str(entry.get("id") or ""))
        )
    return calls


class OllamaClient:
    """Chat/generate client for a local Ollama server."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "qwen3:8b",
        *,
        temperature: float = 0.0,
        seed: int = 42,
        max_tokens: int = 2048,
        top_p: float = 1.0,
        think: bool = False,
        context_tokens: int = 16384,
        timeout_s: float = 120.0,
        connect_timeout_s: float = 5.0,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_reset_s: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.think = think
        self.context_tokens = context_tokens
        self.timeout = (connect_timeout_s, timeout_s)
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self._session = session or requests.Session()
        self._breaker = _CircuitBreaker(circuit_breaker_threshold, circuit_breaker_reset_s)

    # -- public ----------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        format_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> Completion:
        """Chat completion, optionally with tools or a response schema.

        ``tools`` and ``format_schema`` are mutually exclusive. Ollama's
        handling of both together is unreliable, and the agent design does not
        need it: the tool loop runs unconstrained and a separate, tool-free
        call produces the schema-constrained answer.
        """
        if tools and format_schema:
            raise ValueError(
                "tools and format_schema cannot be combined; run the tool loop "
                "unconstrained and use a separate schema-constrained call to finalise"
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": self.think,
            "options": self._options(max_tokens),
        }
        if format_schema:
            payload["format"] = format_schema
        if tools:
            payload["tools"] = tools

        data, latency, attempts = self._post("/api/chat", payload, timeout_s=timeout_s)
        limit = max_tokens or self.max_tokens
        produced = int(data.get("eval_count") or 0)
        message = data.get("message") or {}
        return Completion(
            text=(message.get("content") or "").strip(),
            latency_s=latency,
            model=self.model_name,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=produced,
            attempts=attempts,
            truncated=produced >= limit,
            tool_calls=_parse_tool_calls(message),
        )

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        format_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "system": system or "",
            "stream": False,
            "think": self.think,
            "options": self._options(max_tokens),
        }
        if format_schema:
            payload["format"] = format_schema
        data, latency, attempts = self._post("/api/generate", payload, timeout_s=timeout_s)
        limit = max_tokens or self.max_tokens
        produced = int(data.get("eval_count") or 0)
        return Completion(
            text=data.get("response", "").strip(),
            latency_s=latency,
            model=self.model_name,
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=produced,
            attempts=attempts,
            truncated=produced >= limit,
        )

    def health(self) -> bool:
        try:
            response = self._session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def close(self) -> None:
        self._session.close()

    # -- internals -------------------------------------------------------

    def _options(self, max_tokens: int | None) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "seed": self.seed,
            "num_predict": max_tokens or self.max_tokens,
            "top_p": self.top_p,
            # Explicit, because the server default silently truncates.
            "num_ctx": self.context_tokens,
        }

    @property
    def prompt_budget_tokens(self) -> int:
        """Tokens available for the prompt, leaving room for the response."""
        return max(512, self.context_tokens - self.max_tokens - 256)

    def _post(
        self, path: str, payload: dict[str, Any], *, timeout_s: float | None = None
    ) -> tuple[dict[str, Any], float, int]:
        if self._breaker.is_open:
            raise LLMUnavailable(
                f"LLM circuit is open for {self.base_url}; backend has been failing. "
                "Not sending further requests until it resets."
            )

        url = f"{self.base_url}{path}"
        last_error: Exception | None = None
        started = time.monotonic()
        # A per-call override lets the agent loop derive its read timeout from
        # the time left in the run budget, instead of every step being free to
        # consume the full construction-time timeout.
        effective_timeout = (
            (self.timeout[0], timeout_s) if timeout_s is not None else self.timeout
        )

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._session.post(url, json=payload, timeout=effective_timeout)
            except requests.RequestException as exc:
                last_error = exc
                self._breaker.record_failure()
                if attempt == self.max_retries:
                    break
                self._sleep(attempt)
                continue

            if response.status_code == 200:
                self._breaker.record_success()
                try:
                    return response.json(), time.monotonic() - started, attempt
                except ValueError as exc:
                    raise LLMError(f"LLM returned non-JSON body: {response.text[:200]}") from exc

            if response.status_code not in RETRYABLE_STATUS:
                # Deterministic client error: fail now with the server's reason.
                self._breaker.record_success()  # the server is up, the request was wrong
                raise LLMError(
                    f"LLM rejected the request ({response.status_code}): {response.text[:300]}"
                )

            last_error = LLMError(f"HTTP {response.status_code}: {response.text[:200]}")
            self._breaker.record_failure()
            if attempt == self.max_retries:
                break
            self._sleep(attempt)

        raise LLMUnavailable(
            f"LLM call to {url} failed after {self.max_retries} attempt(s). "
            f"Is the server running and is model '{self.model_name}' pulled? "
            f"Last error: {last_error}"
        )

    def _sleep(self, attempt: int) -> None:
        # Full jitter: concurrent workers must not retry in lockstep.
        ceiling = self.backoff_base_s * (2 ** (attempt - 1))
        time.sleep(random.uniform(0, ceiling))
