"""Structured logging and per-stage timing.

The previous pipeline wrote one plaintext ``run.log`` and timed only the LLM
call, so retrieval and rerank latency were never measured at all (D10). Here
every stage is timed and every record is machine-readable, with a trace id
threaded through a request so its stages can be reassembled.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")

_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename", "module",
    "exc_info", "exc_text", "stack_info", "lineno", "funcName", "created", "msecs",
    "relativeCreated", "thread", "threadName", "processName", "process", "taskName",
    "message", "asctime",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace = trace_id_var.get() or getattr(record, "trace_id", "")
        if trace:
            payload["trace_id"] = trace
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", json_output: bool = True) -> None:
    root = logging.getLogger("ragaudit")
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonFormatter()
        if json_output
        else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(handler)
    root.propagate = False


def new_trace_id() -> str:
    trace = uuid.uuid4().hex[:12]
    trace_id_var.set(trace)
    return trace


@dataclass
class StageTimer:
    """Collects per-stage wall time for one request.

    Retrieval and rerank cost is as operationally important as generation
    cost, and was previously invisible.
    """

    stages: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.monotonic()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.monotonic() - started)

    @property
    def total_s(self) -> float:
        return sum(self.stages.values())

    def to_dict(self) -> dict[str, float]:
        out = {f"{k}_s": round(v, 4) for k, v in self.stages.items()}
        out["total_s"] = round(self.total_s, 4)
        return out
