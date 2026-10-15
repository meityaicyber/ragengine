"""Command-line interface.

Ingest, validate, query, chat, serve and evaluate. ``validate`` runs the gate
without writing anything, so extraction quality across a corpus can be
surveyed before committing to an ingest run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragaudit.app import Application
from ragaudit.config import load_settings
from ragaudit.ingest.pipeline import discover_reports
from ragaudit.ingest.validate import validate_report
from ragaudit.retrieve.router import route_query


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> int:
    """Dry-run the extraction gate over a corpus. Writes nothing."""
    paths = discover_reports(Path(args.path)) if Path(args.path).is_dir() else [Path(args.path)]
    if not paths:
        print(f"no report JSON found under {args.path}", file=sys.stderr)
        return 1

    blocked = 0
    summary = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            blocked += 1
            summary.append({"file": str(path), "ok": False, "error": f"invalid JSON: {exc}"})
            continue
        result = validate_report(data, path.stem)
        blocked += 0 if result.ok else 1
        summary.append(
            {
                "file": str(path),
                "ok": result.ok,
                "blocking": [str(i) for i in result.blocking],
                "warnings": [str(i) for i in result.warnings],
            }
        )

    if args.json:
        _print({"checked": len(paths), "would_quarantine": blocked, "reports": summary})
    else:
        for item in summary:
            mark = "PASS" if item["ok"] else "QUARANTINE"
            print(f"[{mark}] {item['file']}")
            for issue in item.get("blocking", []):
                print(f"    {issue}")
            for issue in item.get("warnings", []):
                print(f"    {issue}")
            if item.get("error"):
                print(f"    {item['error']}")
        print(f"\n{len(paths)} checked, {blocked} would be quarantined.")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    app = Application(load_settings())
    try:
        pipeline = app.build_ingest()
        paths = discover_reports(Path(args.path)) if Path(args.path).is_dir() else [Path(args.path)]
        report = pipeline.ingest_paths(paths, force=args.force)
        pipeline.save_index(app.settings.index_dir)

        _print(
            {
                "summary": report.summary(),
                "catalog": app.catalog.stats(),
                "outcomes": [
                    {
                        "report_id": o.report_id,
                        "status": o.status,
                        "findings": o.findings,
                        "chunks": o.chunks,
                        "figures": o.figures,
                        "figures_missing": o.figures_missing,
                        "blocking": o.blocking,
                        "warnings": o.warnings if args.verbose else len(o.warnings),
                        "error": o.error,
                    }
                    for o in report.outcomes
                ],
            }
        )
        return 0 if report.count("failed") == 0 else 1
    finally:
        app.close()


def cmd_route(args: argparse.Namespace) -> int:
    """Show how a question would be routed, without answering it."""
    plan = route_query(args.question)
    _print(
        {
            "question": args.question,
            "route": plan.route.value,
            "reasons": plan.reasons,
            "severities": plan.severities,
            "statuses": plan.statuses,
            "cwe": plan.cwe,
            "cve": plan.cve,
            "wants_count": plan.wants_count,
            "universal": plan.universal,
        }
    )
    return 0


def cmd_findings(args: argparse.Namespace) -> int:
    app = Application(load_settings())
    try:
        rows = app.catalog.findings(
            report_ids=[args.report] if args.report else None,
            severities=[args.severity] if args.severity else None,
            statuses=[args.status] if args.status else None,
            cwe=args.cwe,
            limit=args.limit,
        )
        _print(
            {
                "count": len(rows),
                "findings": [
                    {
                        "report_id": f.report_id,
                        "finding_id": f.finding_id,
                        "title": f.title,
                        "severity": f.severity.value,
                        "status": f.status.value,
                        "cwe_ids": f.cwe_ids,
                    }
                    for f in rows
                ],
            }
        )
        return 0
    finally:
        app.close()


def cmd_ask(args: argparse.Namespace) -> int:
    app = Application(load_settings())
    try:
        chat = app.build_chat(icl_strategy=args.icl, icl_k=args.icl_k)
        answer, session = chat.ask(args.question, report_ids=args.report)
        if args.json:
            _print(
                {
                    "answer": answer.text,
                    "route": answer.route,
                    "abstained": answer.abstained,
                    "citations": [
                        {"chunk_id": c.chunk_id, "report_id": c.report_id,
                         "field_path": c.field_path, "quote": c.quote}
                        for c in answer.citations
                    ],
                    "figures": [{"token": f.token, "available": f.available} for f in answer.figures],
                    "review_flags": answer.review_flags,
                    "session_id": session.session_id,
                    "latency_s": round(answer.latency_s, 3),
                }
            )
        else:
            print(answer.text)
            if answer.citations:
                print("\nSources:")
                for c in answer.citations:
                    print(f"  - {c.report_id} {c.field_path}")
                    if c.quote:
                        print(f'    "{c.quote[:160]}"')
            if answer.figures:
                print("\nFigures:")
                for f in answer.figures:
                    state = f.asset_path if f.available else "referenced, image not staged"
                    print(f"  - {f.token} (page {f.page}): {state}")
            if answer.review_flags:
                print("\nFlags: " + ", ".join(answer.review_flags))
        return 0
    finally:
        app.close()


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive multi-turn session."""
    app = Application(load_settings())
    try:
        chat = app.build_chat(icl_strategy=args.icl)
        session_id = None
        print("Audit assistant. Ctrl-D or 'exit' to quit.\n")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"exit", "quit"}:
                return 0
            if not question:
                continue
            answer, session = chat.ask(question, session_id=session_id, report_ids=args.report)
            session_id = session.session_id
            print(f"\n{answer.text}\n")
            for c in answer.citations:
                print(f"  [{c.report_id} {c.field_path}]")
            if answer.review_flags:
                print(f"  flags: {', '.join(answer.review_flags)}")
            print()
    finally:
        app.close()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "ragaudit.api.server:app", host=args.host, port=args.port, log_level=args.log_level
    )
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    app = Application(load_settings())
    try:
        _print(app.readiness())
        return 0
    finally:
        app.close()


def cmd_demo(args: argparse.Namespace) -> int:
    """Prepare or run the demo walkthrough."""
    from ragaudit.demo import check_environment, run_script, write_transcript

    settings = load_settings()

    if args.action == "check":
        _print(check_environment(settings))
        return 0

    if args.action == "setup":
        report = check_environment(settings)
        _print(report)
        missing = [k for k, v in report.items() if isinstance(v, str) and v.startswith(("MISSING", "UNREACHABLE"))]
        if missing:
            print(f"\nNot ready: {', '.join(missing)}", file=sys.stderr)
            print("Ingest will still run if the embedding weights are present.", file=sys.stderr)

        app = Application(settings)
        try:
            pipeline = app.build_ingest()
            paths = discover_reports(Path(args.corpus))
            if not paths:
                print(f"no reports found under {args.corpus}", file=sys.stderr)
                return 1
            result = pipeline.ingest_paths(paths, force=args.force)
            pipeline.save_index(settings.index_dir)
            _print({"ingest": result.summary(), "catalog": app.catalog.stats()})
            return 0
        finally:
            app.close()

    app = Application(settings)
    try:
        chat = app.build_chat(icl_strategy=args.icl)
        results = run_script(chat, pause=args.pause)
        if args.transcript:
            write_transcript(results, Path(args.transcript))
            print(f"\ntranscript written to {args.transcript}")
        return 0
    finally:
        app.close()


def cmd_evaluate(args: argparse.Namespace) -> int:
    from ragaudit.eval.harness import run_evaluation

    app = Application(load_settings())
    try:
        result = run_evaluation(
            app,
            questions_path=Path(args.questions),
            out_dir=Path(args.out),
            icl_strategy=args.icl,
            report_ids=args.report,
        )
        _print(result)
        return 0
    finally:
        app.close()


# --------------------------------------------------------------------------
# parser
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragaudit", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="check extraction quality; writes nothing")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("ingest", help="parse, validate, chunk, embed and index reports")
    p.add_argument("path")
    p.add_argument("--force", action="store_true", help="re-ingest even if the source is unchanged")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("route", help="show how a question would be routed")
    p.add_argument("question")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("findings", help="query the structured store directly")
    p.add_argument("--report")
    p.add_argument("--severity")
    p.add_argument("--status")
    p.add_argument("--cwe")
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_findings)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--report", action="append", help="restrict to this report id (repeatable)")
    p.add_argument("--icl", default="fixed_few_shot",
                   choices=["zero_shot", "fixed_few_shot", "retrieval_few_shot"])
    p.add_argument("--icl-k", type=int, default=3, dest="icl_k")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive multi-turn session")
    p.add_argument("--report", action="append")
    p.add_argument("--icl", default="fixed_few_shot",
                   choices=["zero_shot", "fixed_few_shot", "retrieval_few_shot"])
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="run the HTTP API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="readiness of every component")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("demo", help="prepare or run the scripted walkthrough")
    p.add_argument("action", choices=["check", "setup", "run"],
                   help="check: report readiness. setup: check then ingest. run: scripted questions")
    p.add_argument("--corpus", default="sample_data", help="reports to ingest during setup")
    p.add_argument("--force", action="store_true")
    p.add_argument("--pause", type=float, default=0.0, help="seconds between questions when presenting")
    p.add_argument("--transcript", help="write the run to this JSON file")
    p.add_argument("--icl", default="fixed_few_shot",
                   choices=["zero_shot", "fixed_few_shot", "retrieval_few_shot"])
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("evaluate", help="run the evaluation harness")
    p.add_argument("--questions", required=True)
    p.add_argument("--out", default="var/eval")
    p.add_argument("--report", action="append")
    p.add_argument("--icl", default="fixed_few_shot",
                   choices=["zero_shot", "fixed_few_shot", "retrieval_few_shot"])
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
