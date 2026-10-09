"""Command-line entry point: ``bioeval <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.resources import files
from pathlib import Path

import pandas as pd

from bioeval import analysis, storage
from bioeval.adapters import AdapterConfigError, make_adapter
from bioeval.dataset import load_dataset
from bioeval.judge import LLMJudge
from bioeval.report import markdown_table, write_report
from bioeval.runner import run_evaluation


def default_dataset() -> Path:
    return Path(str(files("bioeval") / "data" / "biology_v1.jsonl"))


def _print_frame(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df.to_string(index=False) if not df.empty else "(empty)")


def cmd_validate(args) -> int:
    report = load_dataset(args.dataset)
    if args.json:
        print(json.dumps({"summary": report.summary(),
                          "issues": report.issues_frame().to_dict(orient="records")}, indent=2))
        return 0 if report.ok else 1
    for k, v in report.summary().items():
        print(f"{k:>18}: {v}")
    if report.questions:
        print("\nCoverage (subject x difficulty):")
        print(report.coverage_frame().to_string())
    if report.issues:
        print(f"\n{len(report.issues)} issue(s):")
        _print_frame(report.issues_frame())
    print("\nOK: dataset is valid." if report.ok else "\nFAILED: fix the errors above.")
    return 0 if report.ok else 1


def cmd_run(args) -> int:
    dataset = load_dataset(args.dataset)
    if not dataset.ok:
        print(f"{args.dataset} has {len(dataset.errors)} error(s); run `bioeval validate`.",
              file=sys.stderr)
        return 1
    if args.resume and len(args.model) != 1:
        print("--resume takes exactly one --model", file=sys.stderr)
        return 2
    judge = None
    if args.judge:
        judge = LLMJudge(make_adapter(args.judge, dataset.questions))
    status = 0
    for spec in args.model:
        kwargs = {"seed": args.seed} if spec.startswith("mock") else {}
        try:
            adapter = make_adapter(spec, dataset.questions, **kwargs)
            summary = run_evaluation(args.db, dataset, adapter, judge=judge,
                                     workers=args.workers, limit=args.limit,
                                     resume_run_id=args.resume)
        except AdapterConfigError as exc:
            print(f"[{spec}] stopped: {exc}", file=sys.stderr)
            status = 1
            continue
        print(f"{summary.adapter}: run {summary.run_id}  answered={summary.attempted} "
              f"failed={summary.failed} mean_score={summary.mean_score}")
        status = status or (1 if summary.failed else 0)
    return status


def _run_ids(args) -> list[str] | None:
    return [r.strip() for r in args.runs.split(",")] if args.runs else None


def cmd_report(args) -> int:
    board = write_report(args.db, args.out, _run_ids(args), args.markdown)
    _print_frame(board)
    print(f"\nreport written to {args.out}")
    return 0


def cmd_leaderboard(args) -> int:
    runs = storage.load_runs(args.db)
    df = storage.load_results(args.db, _run_ids(args) or analysis.latest_runs(runs))
    board = analysis.leaderboard(df)
    if args.markdown:
        print(markdown_table(board))
    else:
        _print_frame(board)
        comp = analysis.comparisons(df)
        if not comp.empty:
            print("\nPaired bootstrap (top model vs others):")
            _print_frame(comp)
    return 0


def cmd_demo(args) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    db = out / "results.db"
    if db.exists():
        db.unlink()
    dataset = load_dataset(args.dataset)
    if not dataset.ok:
        print("the bundled dataset failed validation", file=sys.stderr)
        return 1
    run_ids = []
    for profile in ("strong", "medium", "weak"):
        adapter = make_adapter(f"mock:{profile}", dataset.questions, seed=args.seed)
        run_ids.append(run_evaluation(db, dataset, adapter, workers=1).run_id)
    board = write_report(db, out / "report.html", run_ids, out / "leaderboard.md")
    print()
    _print_frame(board)
    print(f"\nreport: {out / 'report.html'}\ndatabase: {db}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bioeval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    v = sub.add_parser("validate", help="audit a question set before grading against it")
    v.add_argument("dataset", nargs="?", default=default_dataset())
    v.add_argument("--json", action="store_true", help="machine-readable output")
    v.set_defaults(func=cmd_validate)

    r = sub.add_parser("run", help="answer and grade every question with one or more models")
    r.add_argument("--model", action="append", required=True,
                   help="mock:strong|medium|weak, openai:<model>, anthropic:<model>[@effort]")
    r.add_argument("--dataset", default=default_dataset())
    r.add_argument("--db", default="results.db")
    r.add_argument("--judge", help="optional LLM judge, same spec format as --model")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--limit", type=int, help="only the first N questions")
    r.add_argument("--resume", metavar="RUN_ID", help="continue an interrupted run")
    r.add_argument("--seed", type=int, default=0, help="seed for mock models")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="write the HTML report")
    rep.add_argument("--db", default="results.db")
    rep.add_argument("--out", default="report.html")
    rep.add_argument("--markdown", help="also write the leaderboard as Markdown here")
    rep.add_argument("--runs", help="comma-separated run ids (default: latest per model)")
    rep.set_defaults(func=cmd_report)

    lb = sub.add_parser("leaderboard", help="print the leaderboard")
    lb.add_argument("--db", default="results.db")
    lb.add_argument("--runs")
    lb.add_argument("--markdown", action="store_true")
    lb.set_defaults(func=cmd_leaderboard)

    d = sub.add_parser("demo", help="offline end-to-end demo with three mock models")
    d.add_argument("--out-dir", default="demo_output")
    d.add_argument("--dataset", default=default_dataset())
    d.add_argument("--seed", type=int, default=0)
    d.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
