"""Evaluation metrics with uncertainty.

Why the statistics are not optional (D09). The previous evaluation reported
81.25% accuracy on 16 question/report pairs, to two decimal places, and drew a
per-variable decision from every 6-to-12 point gap in an ablation table. On
n=16 a single flipped answer moves the figure 6.25 points, so most of those
gaps were inside the noise floor.

Every accuracy here comes with a bootstrap interval, and paired comparisons
use McNemar's exact test, which is the right test when two systems are scored
on the *same* items.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# Retrieval metrics
# --------------------------------------------------------------------------


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for r in top if r in set(relevant))
    return hits / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float | None:
    gold = set(relevant)
    if not gold:
        return None  # undefined, not zero -- excluded from the mean
    return sum(1 for r in retrieved[:k] if r in gold) / len(gold)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    gold = set(relevant)
    return 1.0 if any(r in gold for r in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    gold = set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if item in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    gold = set(relevant)
    if not gold:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank, item in enumerate(retrieved[:k], start=1) if item in gold)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    return dcg / ideal if ideal else 0.0


# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------


@dataclass
class Interval:
    point: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.1%} [{self.low:.1%}, {self.high:.1%}] (n={self.n})"

    @property
    def width(self) -> float:
        return self.high - self.low


def bootstrap_mean(
    values: Sequence[float], *, iterations: int = 10_000, alpha: float = 0.05, seed: int = 42
) -> Interval:
    """Percentile bootstrap interval for a mean."""
    data = [float(v) for v in values]
    if not data:
        return Interval(0.0, 0.0, 0.0, 0)
    if len(data) == 1:
        return Interval(data[0], data[0], data[0], 1)

    rng = random.Random(seed)
    n = len(data)
    means = []
    for _ in range(iterations):
        means.append(sum(data[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * iterations)]
    hi = means[min(iterations - 1, int((1 - alpha / 2) * iterations))]
    return Interval(sum(data) / n, lo, hi, n)


@dataclass
class McNemarResult:
    """Paired comparison of two systems on the same items.

    ``b`` counts items system A got right and B got wrong; ``c`` the reverse.
    Items both systems agree on carry no information about which is better,
    which is exactly why an unpaired test wastes the pairing.
    """

    b: int
    c: int
    p_value: float
    n_pairs: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def verdict(self, name_a: str = "A", name_b: str = "B") -> str:
        if self.b + self.c == 0:
            return f"{name_a} and {name_b} agree on every item; no evidence of a difference."
        better = name_a if self.b > self.c else name_b
        if not self.significant:
            return (
                f"No significant difference (p={self.p_value:.3f}, discordant pairs "
                f"{self.b}/{self.c} of {self.n_pairs}). The observed gap favours {better} "
                f"but is within what chance produces at this sample size."
            )
        return (
            f"{better} is better (p={self.p_value:.3f}, discordant pairs {self.b}/{self.c} "
            f"of {self.n_pairs})."
        )


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> McNemarResult:
    """Exact binomial McNemar test."""
    if len(correct_a) != len(correct_b):
        raise ValueError("paired test requires equal-length sequences")
    b = sum(1 for a, bb in zip(correct_a, correct_b) if a and not bb)
    c = sum(1 for a, bb in zip(correct_a, correct_b) if bb and not a)
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 1.0, len(correct_a))

    # Two-sided exact test: P(X <= min(b,c)) under Binomial(n, 0.5), doubled.
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return McNemarResult(b, c, min(1.0, 2 * tail), len(correct_a))


def minimum_detectable_difference(n: int, baseline: float = 0.8) -> float:
    """Roughly the smallest accuracy gap this sample size could resolve.

    Reported alongside results so a reader can see immediately whether an
    observed gap is even measurable at the current n.
    """
    if n <= 0:
        return 1.0
    se = math.sqrt(2 * baseline * (1 - baseline) / n)
    return 1.96 * se


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

#: Pipeline failures are their own outcome, never folded into a content class.
#: The previous harness mapped ParseError and Error into "Insufficient
#: Evidence", which made a crashed run look like principled abstention.
FAILURE_CLASS = "PipelineFailure"


@dataclass
class ClassificationReport:
    classes: list[str]
    matrix: list[list[int]]
    accuracy: Interval
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    macro_f1: float = 0.0
    n: int = 0
    failures: int = 0

    def to_dict(self) -> dict:
        return {
            "classes": self.classes,
            "confusion_matrix": self.matrix,
            "accuracy": {
                "point": self.accuracy.point,
                "ci_low": self.accuracy.low,
                "ci_high": self.accuracy.high,
            },
            "macro_f1": self.macro_f1,
            "per_class": self.per_class,
            "n": self.n,
            "pipeline_failures": self.failures,
            "minimum_detectable_difference": minimum_detectable_difference(self.n),
        }


def classification_report(
    y_true: Sequence[str], y_pred: Sequence[str], classes: Sequence[str] | None = None
) -> ClassificationReport:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")

    labels = list(classes) if classes else sorted(set(y_true) | set(y_pred))
    if FAILURE_CLASS not in labels and FAILURE_CLASS in set(y_pred):
        labels.append(FAILURE_CLASS)

    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)
    matrix = [[0] * size for _ in range(size)]
    correct: list[float] = []

    for truth, pred in zip(y_true, y_pred):
        if truth not in index:
            continue
        # An unrecognised prediction is a pipeline failure in its own right.
        pred_label = pred if pred in index else FAILURE_CLASS
        if pred_label not in index:
            labels.append(FAILURE_CLASS)
            index[FAILURE_CLASS] = len(labels) - 1
            for row in matrix:
                row.append(0)
            matrix.append([0] * len(labels))
            size = len(labels)
        matrix[index[truth]][index[pred_label]] += 1
        correct.append(1.0 if pred_label == truth else 0.0)

    per_class: dict[str, dict[str, float]] = {}
    f1s: list[float] = []
    for label in labels:
        i = index[label]
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(size)) - tp
        fn = sum(matrix[i]) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        support = sum(matrix[i])
        per_class[label] = {
            "precision": precision, "recall": recall, "f1": f1, "support": support
        }
        # A class with no gold examples contributes nothing meaningful to a
        # macro average; including it drags the figure toward zero and makes
        # macro-F1 unreadable. The previous harness averaged over a class the
        # gold set never contained.
        if support > 0 and label != FAILURE_CLASS:
            f1s.append(f1)

    return ClassificationReport(
        classes=labels,
        matrix=matrix,
        accuracy=bootstrap_mean(correct),
        per_class=per_class,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        n=len(correct),
        failures=sum(matrix[r][index[FAILURE_CLASS]] for r in range(size))
        if FAILURE_CLASS in index
        else 0,
    )


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------


@dataclass
class GroundingReport:
    answered: int
    with_citation: int
    fabricated_citations: int
    fabricated_quotes: int
    abstentions: int

    @property
    def citation_rate(self) -> float:
        return self.with_citation / self.answered if self.answered else 0.0

    @property
    def fabrication_rate(self) -> float:
        total = self.answered or 1
        return (self.fabricated_citations + self.fabricated_quotes) / total

    def to_dict(self) -> dict:
        return {
            "answered": self.answered,
            "with_verified_citation": self.with_citation,
            "citation_rate": self.citation_rate,
            "fabricated_citations": self.fabricated_citations,
            "fabricated_quotes": self.fabricated_quotes,
            "fabrication_rate": self.fabrication_rate,
            "abstentions": self.abstentions,
        }


def grounding_report(answers: Sequence) -> GroundingReport:
    """Measure grounding from verification outcomes, not word overlap.

    The previous harness reported a 'grounding rate' computed from 70%
    bag-of-words overlap between the model's quote and the retrieved text,
    which largely measures vocabulary reuse. These counts come from the
    citation checks the chat service actually performs.
    """
    answered = sum(1 for a in answers if not a.abstained)
    with_citation = sum(1 for a in answers if not a.abstained and a.citations)
    fabricated_citations = sum(
        1 for a in answers if any(f.startswith("citation_not_in_evidence") for f in a.review_flags)
    )
    fabricated_quotes = sum(
        1 for a in answers if any(f.startswith("quote_not_in_chunk") for f in a.review_flags)
    )
    return GroundingReport(
        answered=answered,
        with_citation=with_citation,
        fabricated_citations=fabricated_citations,
        fabricated_quotes=fabricated_quotes,
        abstentions=sum(1 for a in answers if a.abstained),
    )
