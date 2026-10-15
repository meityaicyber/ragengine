"""Safe arithmetic over recorded step results.

Language models are unreliable at arithmetic, and an audit percentage must be
right. So the model never does the sum: it names an operation and the values
(by step reference or literally), and this module computes it deterministically
over a whitelisted set of operations — no ``eval``, no attribute access, no
arbitrary expression.

Every computed number carries the inputs it came from, so the finalize answer
can show provenance and a figure with none never reaches the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OPERATIONS = {"percentage", "ratio", "difference", "sum", "mean"}


@dataclass
class Computation:
    operation: str
    inputs: dict[str, Any]
    result: float
    expression: str
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "result": self.result,
            "expression": self.expression,
            "inputs": self.inputs,
            "provenance": self.provenance,
        }


class ComputeError(Exception):
    """A computation that cannot be performed as asked (bad op, missing value)."""


def _as_number(value: Any, label: str) -> float:
    if isinstance(value, bool):  # bool is an int subclass; reject it explicitly
        raise ComputeError(f"{label} must be a number, not a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        raise ComputeError(f"{label}={value!r} is not a number") from None


def compute(operation: str, values: list[float], *, provenance: list[str] | None = None) -> Computation:
    """Perform one whitelisted operation over numeric values.

    ``percentage`` and ``ratio`` take [part, whole]; ``difference`` [a, b];
    ``sum`` and ``mean`` take any count. Division by zero and empty inputs
    raise, rather than producing NaN/Inf that would break JSON and the UI.
    """
    op = (operation or "").strip().lower()
    if op not in OPERATIONS:
        raise ComputeError(f"unknown operation '{operation}'. Use one of: {sorted(OPERATIONS)}")

    nums = [_as_number(v, f"value[{i}]") for i, v in enumerate(values)]
    prov = list(provenance or [])

    if op in ("percentage", "ratio"):
        if len(nums) != 2:
            raise ComputeError(f"{op} needs exactly [part, whole]")
        part, whole = nums
        if whole == 0:
            raise ComputeError(f"{op} undefined: whole is zero")
        value = (part / whole) * (100.0 if op == "percentage" else 1.0)
        symbol = "%" if op == "percentage" else ""
        return Computation(op, {"part": part, "whole": whole}, round(value, 2),
                           f"{part} / {whole}{(' * 100' if op == 'percentage' else '')} = {round(value, 2)}{symbol}", prov)

    if op == "difference":
        if len(nums) != 2:
            raise ComputeError("difference needs exactly [a, b]")
        value = nums[0] - nums[1]
        return Computation(op, {"a": nums[0], "b": nums[1]}, value, f"{nums[0]} - {nums[1]} = {value}", prov)

    if not nums:
        raise ComputeError(f"{op} needs at least one value")
    if op == "sum":
        value = sum(nums)
        return Computation(op, {"values": nums}, value, f"sum({nums}) = {value}", prov)
    # mean
    value = round(sum(nums) / len(nums), 4)
    return Computation(op, {"values": nums}, value, f"mean({nums}) = {value}", prov)
