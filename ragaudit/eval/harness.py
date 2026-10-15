"""Evaluation harness.

Runs a question set through the live chat service and writes per-question
results plus aggregate metrics. Three differences from the previous harness
(D09):

* it does not rebuild the index -- evaluation reads the same index serving
  traffic, so what is measured is what runs;
* every accuracy carries a bootstrap interval and the minimum difference the
  sample size can resolve;
* grounding is counted from citation verification, not word overlap.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ragaudit.eval.metrics import (
    bootstrap_mean,
    classification_report,
    grounding_report,
    hit_rate_at_k,
    minimum_detectable_difference,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

logger = logging.getLogger("ragaudit.eval")


def load_questions(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = raw.get("questions", [])
    out = []
    for i, item in enumerate(raw, start=1):
        out.append(
            {
                "question_id": str(item.get("question_id") or item.get("id") or f"Q{i}"),
                "text": item["text"],
                "report_id": item.get("report_id"),
                "gold_answer": item.get("gold_answer") or item.get("answer"),
                "gold_chunk_ids": item.get("gold_chunk_ids", []),
                "expected_route": item.get("expected_route"),
            }
        )
    return out


def run_evaluation(
    application,
    questions_path: Path,
    out_dir: Path,
    *,
    icl_strategy: str = "fixed_few_shot",
    report_ids: Sequence[str] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    if not questions:
        raise ValueError(f"no questions found in {questions_path}")

    run_id = f"{int(time.time())}_{icl_strategy}"
    out = Path(out_dir) / run_id
    out.mkdir(parents=True, exist_ok=True)

    chat = application.build_chat(icl_strategy=icl_strategy)
    k = top_k or application.settings.retrieval.top_k

    rows: list[dict[str, Any]] = []
    answers = []
    retrieval_rows: list[dict[str, float]] = []
    route_correct: list[float] = []
    latencies: list[float] = []

    for question in questions:
        scope = [question["report_id"]] if question.get("report_id") else (
            list(report_ids) if report_ids else None
        )
        answer, _ = chat.ask(question["text"], report_ids=scope)
        answers.append(answer)
        latencies.append(answer.latency_s)

        retrieved_ids = [r.chunk.chunk_id for r in answer.retrieved]
        gold_chunks = question.get("gold_chunk_ids") or []
        if gold_chunks:
            recall = recall_at_k(retrieved_ids, gold_chunks, k)
            retrieval_rows.append(
                {
                    "question_id": question["question_id"],
                    "precision_at_k": precision_at_k(retrieved_ids, gold_chunks, k),
                    "recall_at_k": recall if recall is not None else float("nan"),
                    "hit_rate_at_k": hit_rate_at_k(retrieved_ids, gold_chunks, k),
                    "mrr": reciprocal_rank(retrieved_ids, gold_chunks),
                    "ndcg_at_k": ndcg_at_k(retrieved_ids, gold_chunks, k),
                }
            )

        if question.get("expected_route"):
            route_correct.append(1.0 if answer.route == question["expected_route"] else 0.0)

        rows.append(
            {
                "question_id": question["question_id"],
                "question": question["text"],
                "report_id": question.get("report_id") or "",
                "route": answer.route,
                "answer": answer.text,
                "abstained": answer.abstained,
                "abstention_reason": answer.abstention_reason,
                "n_citations": len(answer.citations),
                "cited_chunk_ids": [c.chunk_id for c in answer.citations],
                "retrieved_chunk_ids": retrieved_ids,
                "gold_answer": question.get("gold_answer") or "",
                "gold_chunk_ids": gold_chunks,
                "review_flags": answer.review_flags,
                "figures": [f.token for f in answer.figures],
                "latency_s": round(answer.latency_s, 3),
                "model": answer.model,
            }
        )
        logger.info(
            "evaluated question",
            extra={
                "question_id": question["question_id"],
                "route": answer.route,
                "abstained": answer.abstained,
                "citations": len(answer.citations),
            },
        )

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "n_questions": len(questions),
        "icl_strategy": icl_strategy,
        "top_k": k,
        "grounding": grounding_report(answers).to_dict(),
        "latency": {
            "mean_s": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "max_s": round(max(latencies), 3) if latencies else 0.0,
        },
        "routes": _count(r["route"] for r in rows),
        "abstention_rate": round(sum(1 for r in rows if r["abstained"]) / len(rows), 4),
    }

    if retrieval_rows:
        metrics["retrieval"] = _aggregate_retrieval(retrieval_rows)
    else:
        metrics["retrieval"] = {
            "note": "no gold chunk ids supplied; retrieval metrics not computed"
        }

    labelled = [r for r in rows if r["gold_answer"]]
    if labelled:
        report = classification_report(
            [r["gold_answer"] for r in labelled],
            [_verdict(r) for r in labelled],
        )
        metrics["answer_quality"] = report.to_dict()

    if route_correct:
        interval = bootstrap_mean(route_correct)
        metrics["routing_accuracy"] = {
            "point": interval.point,
            "ci_low": interval.low,
            "ci_high": interval.high,
            "n": interval.n,
        }

    metrics["interpretation"] = _interpretation(len(questions), metrics)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "questions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(out / "questions.csv", rows)

    logger.info("evaluation complete", extra={"run_id": run_id, "out": str(out)})
    return {"run_id": run_id, "out_dir": str(out), "metrics": metrics}


def _verdict(row: dict[str, Any]) -> str:
    """Map a free-text answer onto the gold label vocabulary.

    Only used when a question set carries Yes/No gold labels. This is a
    reporting convenience for comparison against the legacy benchmark; the
    product itself does not classify.
    """
    if row["abstained"]:
        return "Insufficient Evidence"
    text = row["answer"].strip().lower()
    if text.startswith(("yes", "correct,")):
        return "Yes"
    if text.startswith("no"):
        return "No"
    if "does not" in text[:120] or "no findings" in text[:120]:
        return "No"
    return "Yes" if row["n_citations"] else "Insufficient Evidence"


def _aggregate_retrieval(rows: list[dict[str, float]]) -> dict[str, Any]:
    import math

    def interval(key: str) -> dict[str, float]:
        values = [r[key] for r in rows if not math.isnan(r[key])]
        if not values:
            return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
        got = bootstrap_mean(values)
        return {"point": got.point, "ci_low": got.low, "ci_high": got.high, "n": got.n}

    return {metric: interval(metric) for metric in
            ("precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k")}


def _interpretation(n: int, metrics: dict[str, Any]) -> list[str]:
    """Plain-language caveats, written into the artefact itself.

    The previous summary reported figures to two decimal places on n=16 and
    drew per-variable decisions from gaps inside the noise floor. Stating the
    resolution limit next to the numbers makes that harder to repeat.
    """
    notes = []
    mdd = minimum_detectable_difference(n)
    notes.append(
        f"With n={n}, the smallest accuracy difference this sample can resolve is about "
        f"{mdd:.0%}. Gaps smaller than that are not evidence of a real difference."
    )
    if n < 100:
        notes.append(
            "Below 100 labelled questions, treat every figure as directional. "
            "Use McNemar's test on paired runs rather than comparing point estimates."
        )
    grounding = metrics.get("grounding", {})
    if grounding.get("fabricated_citations") or grounding.get("fabricated_quotes"):
        notes.append(
            f"{grounding.get('fabricated_citations', 0)} fabricated citation(s) and "
            f"{grounding.get('fabricated_quotes', 0)} fabricated quote(s) were caught and "
            "removed by verification. These are model failures, not retrieval failures."
        )
    if isinstance(metrics.get("retrieval"), dict) and "note" in metrics["retrieval"]:
        notes.append(
            "Retrieval metrics are absent because the question set has no gold chunk ids. "
            "Answer-level results remain valid; rank quality is unmeasured."
        )
    return notes


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values:
        out[value] = out.get(value, 0) + 1
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    flat = []
    for row in rows:
        item = dict(row)
        for key, value in item.items():
            if isinstance(value, (list, dict)):
                item[key] = json.dumps(value)
        flat.append(item)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0].keys()))
        writer.writeheader()
        writer.writerows(flat)
