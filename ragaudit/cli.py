"""Command-line interface.

    python -m ragaudit.cli <command> ...

Inspect:   survey, validate, status, route
Build:     ingest, reset
Use:       ask, chat, findings, serve
Verify:    crosscheck, evaluate, demo

``validate`` and ``survey`` write nothing, so extraction quality across a
corpus can be surveyed before committing to an ingest run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

from ragaudit.config import load_settings
from ragaudit.ingest.normalize import expand_reports
from ragaudit.ingest.pipeline import discover_reports, load_json
from ragaudit.ingest.validate import validate_report
from ragaudit.retrieve.router import route_query


def _print(payload: object) -> None:
    print(json.dumps(payload, indent=2, default=str))


def _err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _discover(path: str, include_all: bool) -> list[Path]:
    root = Path(path)
    if not root.exists():
        _err(f"path not found: {root}")
        return []
    skipped: list[Path] = []
    paths = discover_reports(root, include_all=include_all, skipped=skipped)
    if skipped:
        _err(f"note: skipped {len(skipped)} file(s) by name (use --include-all to keep them):")
        for p in skipped[:10]:
            _err(f"  - {p}")
    return paths


def _application():
    # Imported lazily so survey/validate/route work without torch installed.
    from ragaudit.app import Application

    return Application(load_settings())


# --------------------------------------------------------------------------
# inspect
# --------------------------------------------------------------------------


def cmd_survey(args: argparse.Namespace) -> int:
    """Describe the shape of a JSON corpus. Writes nothing."""
    from ragaudit.corpus_tools import survey

    result = survey(Path(args.path), include_all=args.include_all, sample=args.sample)
    if args.json:
        _print(result)
        return 0
    print(f"\nSurvey of {result['folder']}: {result['files']} file(s), "
          f"{result['findings_total']} finding rows\n")
    for key in ("top_level_types", "findings_array_key_used", "canonical_field_coverage",
                "unmapped_finding_keys (ignored by ingest)", "severity_values", "status_values"):
        print(f"{key}:")
        value = result[key]
        if not value:
            print("  (none)")
        for k, v in value.items():
            print(f"  {k}: {v}")
        print()
    if result["unreadable"]:
        print("unreadable files:")
        for line in result["unreadable"][:10]:
            print(f"  {line}")
        print()
    if result["report_id_collisions"]:
        print("report ids shared by several files:")
        for k, v in result["report_id_collisions"].items():
            print(f"  {k}: {v}")
        print()
    print("advice:")
    for line in result["advice"]:
        print(f"  * {line}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Dry-run the extraction gate over a corpus. Writes nothing."""
    paths = _discover(args.path, args.include_all)
    if not paths:
        _err(f"no report JSON found under {args.path}")
        return 1

    blocked = 0
    summary = []
    check_counts: Counter = Counter()
    for path in paths:
        try:
            data = load_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            blocked += 1
            summary.append({"file": str(path), "ok": False, "error": f"invalid JSON: {exc}"})
            check_counts["invalid_json"] += 1
            continue
        reports = expand_reports(data)
        if not reports:
            blocked += 1
            summary.append({"file": str(path), "ok": False,
                            "error": "top-level JSON holds no report object, list of reports, "
                                     "or list of findings"})
            check_counts["not_a_report"] += 1
            continue
        for suffix, report in reports:
            result = validate_report(report, path.stem + suffix)
            blocked += 0 if result.ok else 1
            for issue in result.issues:
                check_counts[f"{issue.level.value}:{issue.check}"] += 1
            summary.append(
                {
                    "file": str(path) + suffix,
                    "ok": result.ok,
                    "blocking": [str(i) for i in result.blocking],
                    "warnings": [str(i) for i in result.warnings],
                }
            )

    if args.json:
        _print({"checked": len(summary), "files": len(paths), "would_quarantine": blocked,
                "check_counts": dict(check_counts.most_common()), "reports": summary})
    else:
        for item in summary:
            if args.only_failing and item["ok"]:
                continue
            mark = "PASS" if item["ok"] else "QUARANTINE"
            print(f"[{mark}] {item['file']}")
            for issue in item.get("blocking", []):
                print(f"    {issue}")
            if not args.quiet:
                for issue in item.get("warnings", []):
                    print(f"    {issue}")
            if item.get("error"):
                print(f"    {item['error']}")
        print("\nMost frequent checks:")
        for name, n in check_counts.most_common(12):
            print(f"  {n:5d}  {name}")
        print(f"\n{len(summary)} report(s) in {len(paths)} file(s) checked, {blocked} would be "
              f"quarantined ({(blocked / max(1, len(summary))):.0%}).")
    return 0


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


def cmd_status(args: argparse.Namespace) -> int:
    app = _application()
    try:
        _print(app.readiness())
        return 0
    finally:
        app.close()


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    paths = _discover(args.path, args.include_all)
    if not paths:
        _err(f"no report JSON found under {args.path}")
        return 1

    app = _application()
    try:
        pipeline = app.build_ingest()
        if args.no_quarantine:
            pipeline.fail_on_validation_error = False
            _err("--no-quarantine: reports failing blocking checks are indexed anyway "
                 "(their counts may be wrong; the issues stay recorded).")
        _err(f"ingesting {len(paths)} file(s) into {app.settings.catalog_path} "
             f"(index: {app.settings.index_dir}). Ctrl-C is safe: progress is saved.\n")

        def progress(n, total, outcome, elapsed):
            eta = (elapsed / n) * (total - n)
            extra = f"{outcome.findings} findings, {outcome.chunks} chunks" if outcome.status == "indexed" else (
                outcome.blocking[0][:110] if outcome.blocking else outcome.error[:110])
            _err(f"[{n}/{total}] {outcome.status:<11} {outcome.report_id:<28} {extra}"
                 f"   (elapsed {elapsed:,.0f}s, eta {eta:,.0f}s)")
            if outcome.note:
                _err(f"           note: {outcome.note}")

        report = pipeline.ingest_paths(paths, force=args.force, progress=progress)

        result = {
            "summary": report.summary(),
            "interrupted": report.interrupted,
            "catalog": app.catalog.stats(),
            "index_vectors": len(app.index),
        }
        if args.verbose:
            result["outcomes"] = [
                {
                    "report_id": o.report_id, "path": o.path, "status": o.status,
                    "findings": o.findings, "chunks": o.chunks,
                    "figures": o.figures, "figures_missing": o.figures_missing,
                    "blocking": o.blocking, "warnings": o.warnings, "error": o.error, "note": o.note,
                }
                for o in report.outcomes
            ]
        else:
            result["quarantined"] = {o.report_id: o.blocking[:2] for o in report.outcomes
                                     if o.status == "quarantined"}
            result["failed"] = {o.report_id: o.error for o in report.outcomes if o.status == "failed"}
        print()
        _print(result)
        if report.interrupted:
            _err("\ninterrupted: re-run the same command to continue where it stopped.")
        q = report.count("quarantined")
        if q:
            _err(f"\n{q} report(s) quarantined by the quality gate (reasons above). To index them "
                 f"anyway: python run.py ingest {args.path} --no-quarantine")
        return 0 if report.count("failed") == 0 and not report.interrupted else 1
    finally:
        app.close()


def cmd_reset(args: argparse.Namespace) -> int:
    """Delete the catalogue and vector index (start a fresh corpus)."""
    settings = load_settings()
    targets = [settings.catalog_path, Path(f"{settings.catalog_path}-wal"),
               Path(f"{settings.catalog_path}-shm"), settings.index_dir]
    if not args.yes:
        _err("This deletes: " + ", ".join(str(t) for t in targets))
        _err("Re-run with --yes to confirm.")
        return 1
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    print("reset done; ingest again to rebuild.")
    return 0


# --------------------------------------------------------------------------
# use
# --------------------------------------------------------------------------


def cmd_findings(args: argparse.Namespace) -> int:
    app = _application()
    try:
        kwargs = dict(
            report_ids=[args.report] if args.report else None,
            severities=[args.severity] if args.severity else None,
            statuses=[args.status] if args.status else None,
            cwe=args.cwe,
        )
        rows = app.catalog.findings(**kwargs, limit=args.limit)
        _print(
            {
                "count": app.catalog.count_findings(**kwargs),
                "returned": len(rows),
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


def _num(value) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)


def _print_answer(answer, *, trace: bool) -> None:
    print(answer.text)
    if answer.citations:
        print("\nSources:")
        for c in answer.citations:
            print(f"  - {c.report_id} {c.field_path}")
            if c.quote:
                print(f'    "{c.quote[:160]}"')
    for block in answer.blocks:
        if block.get("kind") == "chart":
            pairs = ", ".join(
                f"{x}={'/'.join(_num(s['values'][i]) for s in block['series'])}"
                for i, x in enumerate(block["x"])
            )
            print(f"\n[chart] {block.get('title', '')}: {pairs}")
        elif block.get("kind") == "metric":
            print(f"\n[metric] {block.get('label')}: {_num(block.get('value'))} ({block.get('detail')})")
        elif block.get("kind") in ("table", "comparison"):
            print(f"\n[{block['kind']}] {block.get('title', '')} ({len(block.get('rows', []))} rows; "
                  "see the web console for the full table)")
        elif block.get("kind") == "note":
            print(f"\n[note] {block.get('text')}")
    if answer.figures:
        print("\nFigures:")
        for f in answer.figures:
            state = f.asset_path if f.available else "referenced, image not staged"
            print(f"  - {f.token} (page {f.page}): {state}")
    if trace and answer.steps:
        print("\nTool trace:")
        for s in answer.steps:
            err = f"  ERROR: {s['error']}" if s.get("error") else ""
            print(f"  {s['step']}. {s['tool']}({json.dumps(s['arguments'])}) -> {s['summary']}"
                  f" [{s['duration_s']}s]{err}")
    print(f"\n(route: {answer.route}; {answer.latency_s:.1f}s"
          + (f"; flags: {', '.join(answer.review_flags)}" if answer.review_flags else "") + ")")


def cmd_ask(args: argparse.Namespace) -> int:
    app = _application()
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
                    "blocks": answer.blocks,
                    "steps": answer.steps,
                    "figures": [{"token": f.token, "available": f.available} for f in answer.figures],
                    "review_flags": answer.review_flags,
                    "session_id": session.session_id,
                    "latency_s": round(answer.latency_s, 3),
                }
            )
        else:
            _print_answer(answer, trace=args.trace)
        return 0
    finally:
        app.close()


def cmd_chat(args: argparse.Namespace) -> int:
    """Interactive multi-turn session."""
    app = _application()
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
            print()
            _print_answer(answer, trace=args.trace)
            print()
    finally:
        app.close()


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    host = args.host
    if host not in ("127.0.0.1", "localhost") and not load_settings().api.api_key:
        _err("warning: serving on a non-local address with no API key. Anyone who can reach "
             "this port can query the reports. Set RAGAUDIT_API__API_KEY in .env.")
    _err(f"web console: http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{args.port}")
    uvicorn.run(
        "ragaudit.api.server:app", host=host, port=args.port, log_level=args.log_level
    )
    return 0


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def cmd_crosscheck(args: argparse.Namespace) -> int:
    from ragaudit.corpus_tools import crosscheck

    settings = load_settings()
    result = crosscheck(Path(args.path), settings.catalog_path, settings.index_dir,
                        include_all=args.include_all)
    for line in result["problems"]:
        print(line)
    print(f"\nmatched={result['matched']}  problems={len(result['problems'])}  "
          f"quarantined(not compared)={result['quarantined_not_compared']}")
    print("OK: stored counts match the raw files." if result["ok"]
          else "Mismatches found; see lines above.")
    return 0 if result["ok"] else 1


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
            _err(f"\nNot ready: {', '.join(missing)}")
            _err("Ingest will still run if the embedding weights are present.")

        from ragaudit.app import Application

        app = Application(settings)
        try:
            pipeline = app.build_ingest()
            paths = discover_reports(Path(args.corpus))
            if not paths:
                _err(f"no reports found under {args.corpus}")
                return 1
            result = pipeline.ingest_paths(paths, force=args.force)
            _print({"ingest": result.summary(), "catalog": app.catalog.stats()})
            return 0
        finally:
            app.close()

    from ragaudit.app import Application

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

    app = _application()
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

_ICL = ["zero_shot", "fixed_few_shot", "retrieval_few_shot"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragaudit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("survey", help="describe a JSON corpus's shape and field coverage; writes nothing")
    p.add_argument("path")
    p.add_argument("--sample", type=int, default=0, help="only look at the first N files")
    p.add_argument("--include-all", action="store_true", dest="include_all")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_survey)

    p = sub.add_parser("validate", help="check extraction quality; writes nothing")
    p.add_argument("path")
    p.add_argument("--json", action="store_true")
    p.add_argument("--quiet", action="store_true", help="hide warnings, show only blocking issues")
    p.add_argument("--only-failing", action="store_true", dest="only_failing")
    p.add_argument("--include-all", action="store_true", dest="include_all")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("ingest", help="parse, validate, chunk, embed and index reports")
    p.add_argument("path")
    p.add_argument("--force", action="store_true", help="re-ingest even if the source is unchanged")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--include-all", action="store_true", dest="include_all",
                   help="do not skip files named like eval/benchmark artefacts")
    p.add_argument("--no-quarantine", action="store_true", dest="no_quarantine",
                   help="index reports even if they fail blocking validation checks")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("reset", help="delete the catalogue and vector index")
    p.add_argument("--yes", action="store_true")
    p.set_defaults(func=cmd_reset)

    p = sub.add_parser("route", help="show how a question would be routed")
    p.add_argument("question")
    p.set_defaults(func=cmd_route)

    p = sub.add_parser("findings", help="query the structured store directly (no model)")
    p.add_argument("--report")
    p.add_argument("--severity")
    p.add_argument("--status")
    p.add_argument("--cwe")
    p.add_argument("--limit", type=int, default=100)
    p.set_defaults(func=cmd_findings)

    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question")
    p.add_argument("--report", action="append", help="restrict to this report id (repeatable)")
    p.add_argument("--icl", default="fixed_few_shot", choices=_ICL)
    p.add_argument("--icl-k", type=int, default=3, dest="icl_k")
    p.add_argument("--trace", action="store_true", help="print the agent's tool calls")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("chat", help="interactive multi-turn session")
    p.add_argument("--report", action="append")
    p.add_argument("--icl", default="fixed_few_shot", choices=_ICL)
    p.add_argument("--trace", action="store_true")
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("serve", help="run the HTTP API and web console")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8077)
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("status", help="readiness of every component")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("crosscheck", help="compare raw JSON finding counts with what was stored")
    p.add_argument("path")
    p.add_argument("--include-all", action="store_true", dest="include_all")
    p.set_defaults(func=cmd_crosscheck)

    p = sub.add_parser("demo", help="prepare or run the scripted walkthrough")
    p.add_argument("action", choices=["check", "setup", "run"],
                   help="check: report readiness. setup: check then ingest. run: scripted questions")
    p.add_argument("--corpus", default="sample_data", help="reports to ingest during setup")
    p.add_argument("--force", action="store_true")
    p.add_argument("--pause", type=float, default=0.0, help="seconds between questions when presenting")
    p.add_argument("--transcript", help="write the run to this JSON file")
    p.add_argument("--icl", default="fixed_few_shot", choices=_ICL)
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("evaluate", help="run the evaluation harness")
    p.add_argument("--questions", required=True)
    p.add_argument("--out", default="var/eval")
    p.add_argument("--report", action="append")
    p.add_argument("--icl", default="fixed_few_shot", choices=_ICL)
    p.set_defaults(func=cmd_evaluate)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except RuntimeError as exc:
        # Configuration problems (missing weights, an index built with another
        # embedding model, an unreachable backend) carry their own fix in the
        # message; print that, not a traceback.
        _err(f"\nerror: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
