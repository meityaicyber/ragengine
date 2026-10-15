"""Demo setup and scripted walkthrough.

``ragaudit demo setup`` prepares a working system from a clean checkout:
checks dependencies, ingests the sample corpus, and reports what is missing
rather than failing at the first gap.

``ragaudit demo run`` executes a scripted sequence of questions that exercises
every path -- structured, narrative, hybrid, figures, follow-up and honest
abstention -- so a walkthrough does not depend on remembering what to type.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass
class Step:
    question: str
    shows: str
    expect_route: str | None = None


#: Ordered to build an argument: exactness first, then language understanding,
#: then the honesty property, then conversation.
SCRIPT: list[Step] = [
    Step(
        "How many high-severity findings are still open?",
        "Counted by SQL over every row, not sampled from retrieved chunks. "
        "A count from top-k cannot be trusted because retrieval never promises "
        "it saw everything.",
        "structured",
    ),
    Step(
        "Are all findings closed?",
        "Answered by searching for a counterexample. One finding that is not "
        "closed settles it, and the complete scan is what makes a 'yes' credible.",
        "structured",
    ),
    Step(
        "Which findings cite CWE-284?",
        "Identifier lookup against an indexed column, not a similarity search.",
        "structured",
    ),
    Step(
        "What access control weaknesses were identified?",
        "Open-ended: the answer is prose with a citation, which the previous "
        "Yes/No/Insufficient schema could not express at all.",
        "narrative",
    ),
    Step(
        "What remediation was recommended for the token API issue?",
        "Retrieves the recommendation field and quotes it. The quote is checked "
        "against the chunk it names before it is shown.",
        "narrative",
    ),
    Step(
        "Was the database server patched?",
        "Nothing in the corpus addresses this. The assistant says so instead of "
        "assembling a plausible answer.",
        "narrative",
    ),
    Step(
        "Show me the proof of concept for the access control finding",
        "Figures are joined from the cited finding by id. No visual search is "
        "involved; the extractor already binds each image to its finding.",
        "hybrid",
    ),
    Step(
        "Tell me more about it",
        "A follow-up with no standalone meaning. Conversation state resolves "
        "what 'it' refers to before retrieval runs.",
        None,
    ),
]


def check_environment(settings) -> dict:
    """Report what is ready and what is missing, without raising."""
    report: dict[str, object] = {}

    embed = Path(settings.embedding.model_path)
    report["embedding_weights"] = str(embed) if embed.exists() else f"MISSING: {embed}"

    rerank = Path(settings.reranker.model_path)
    report["reranker_weights"] = (
        str(rerank) if rerank.exists() else f"MISSING: {rerank} (set reranker.enabled=false to skip)"
    )

    base = settings.generation.base_url.rstrip("/")
    try:
        response = requests.get(f"{base}/api/tags", timeout=4)
        models = [m.get("name", "") for m in response.json().get("models", [])]
        report["llm_server"] = base
        report["llm_models"] = models or "none pulled"
        wanted = settings.generation.model_name
        report["llm_model_ready"] = any(m.split(":")[0] == wanted.split(":")[0] for m in models)
    except requests.RequestException as exc:
        report["llm_server"] = f"UNREACHABLE at {base}: {exc}"
        report["llm_models"] = []
        report["llm_model_ready"] = False

    return report


def run_script(chat, steps: list[Step] | None = None, *, pause: float = 0.0) -> list[dict]:
    """Run the walkthrough against a live ChatService."""
    steps = steps or SCRIPT
    session_id = None
    results: list[dict] = []

    for i, step in enumerate(steps, start=1):
        print(f"\n{'=' * 74}")
        print(f"[{i}/{len(steps)}]  {step.question}")
        print(f"{'-' * 74}")
        print(f"what this shows: {step.shows}")
        print()

        started = time.monotonic()
        answer, session = chat.ask(step.question, session_id=session_id)
        session_id = session.session_id
        elapsed = time.monotonic() - started

        route_note = ""
        if step.expect_route and answer.route != step.expect_route:
            route_note = f"  (expected {step.expect_route})"
        print(f"route: {answer.route}{route_note}   latency: {elapsed:.2f}s")
        print()
        print(answer.text)

        if answer.citations:
            print("\nevidence:")
            for citation in answer.citations[:4]:
                print(f"  - {citation.report_id} {citation.field_path}")
                if citation.quote:
                    print(f'      "{citation.quote[:150]}"')
        if answer.figures:
            staged = sum(1 for f in answer.figures if f.available)
            print(f"\nfigures: {len(answer.figures)} referenced, {staged} staged")
        if answer.review_flags:
            print(f"\nflags: {', '.join(answer.review_flags[:4])}")

        results.append(
            {
                "question": step.question,
                "route": answer.route,
                "route_as_expected": step.expect_route is None or answer.route == step.expect_route,
                "abstained": answer.abstained,
                "citations": len(answer.citations),
                "figures": len(answer.figures),
                "latency_s": round(elapsed, 2),
                "flags": answer.review_flags,
            }
        )
        if pause:
            time.sleep(pause)

    print(f"\n{'=' * 74}")
    routed = sum(1 for r in results if r["route_as_expected"])
    print(
        f"{len(results)} questions · {routed}/{len(results)} routed as expected · "
        f"{sum(1 for r in results if r['abstained'])} honest abstention(s) · "
        f"mean {sum(r['latency_s'] for r in results) / len(results):.2f}s"
    )
    return results


def write_transcript(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
