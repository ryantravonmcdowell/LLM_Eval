"""Render a self-contained HTML report (and a Markdown leaderboard) from the database."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bioeval import analysis, storage
from bioeval.scoring.taxonomy import ErrorType

CSS = """
:root { --fg:#1d232b; --muted:#5b6673; --bg:#ffffff; --panel:#f5f7fa; --line:#dde3ea;
        --accent:#2f6f9f; --bad:#b3412e; --good:#2e7d4f; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e6ebf0; --muted:#9aa6b2; --bg:#14181d; --panel:#1c2229; --line:#2c343d;
          --accent:#7fb4dd; --bad:#e38b78; --good:#7fcf9d; } }
* { box-sizing: border-box; }
body { margin:0; padding:32px 16px 64px; background:var(--bg); color:var(--fg);
       font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 40px 0 8px; padding-top: 8px; border-top: 1px solid var(--line); }
p.lead, .muted { color: var(--muted); }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; margin: 8px 0 4px; font-variant-numeric: tabular-nums; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: right; }
th { background: var(--panel); font-weight: 600; }
td:first-child, th:first-child { text-align: left; }
code, pre { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 13px; }
pre { white-space: pre-wrap; background: var(--panel); padding: 10px; border-radius: 6px; }
details { background: var(--panel); border-radius: 8px; padding: 8px 12px; margin: 8px 0; }
summary { cursor: pointer; font-weight: 600; }
.tag { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px;
       border:1px solid var(--line); margin-right:4px; }
.tag.factual_error { color: var(--bad); border-color: var(--bad); }
dl { display:grid; grid-template-columns: max-content 1fr; gap: 2px 14px; }
dt { color: var(--muted); }
"""


def _table(df: pd.DataFrame, index: bool = False) -> str:
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'
    return f'<div class="scroll">{df.to_html(index=index, border=0, na_rep="-", escape=True)}</div>'


def _answer_card(row) -> str:
    details = json.loads(row.details_json) if row.details_json else {}
    tags = "".join(f'<span class="tag {html.escape(e)}">{html.escape(e)}</span>'
                   for e in row.errors) or '<span class="tag">none</span>'
    facts = ", ".join(details.get("missing_facts", []) + details.get("missing_reasoning", []))
    extra = [("Missing", facts or "-"),
             ("Misconceptions", ", ".join(details.get("misconceptions", [])) or "-"),
             ("Instruction issues", "; ".join(details.get("violations", [])) or "-")]
    dl = "".join(f"<dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd>" for k, v in extra)
    return (
        f"<details><summary>{html.escape(row.question_id)} &middot; {html.escape(row.adapter)}"
        f" &middot; score {row.score:.2f}</summary><p>{tags}</p><dl>{dl}</dl>"
        f"<pre>{html.escape(row.response or '')}</pre></details>"
    )


def build_report(db_path: str | Path, run_ids: list[str] | None = None, seed: int = 0) -> str:
    runs = storage.load_runs(db_path)
    run_ids = run_ids or analysis.latest_runs(runs)
    df = storage.load_results(db_path, run_ids)
    if df.empty:
        raise ValueError(f"no results in {db_path} for runs {run_ids}")
    shown = runs[runs["run_id"].isin(run_ids)]

    board = analysis.leaderboard(df, seed=seed)
    taxonomy = pd.DataFrame([{"error type": e.value, "meaning": e.description} for e in ErrorType])
    sections = [
        ("Leaderboard",
         "<p>Mean rubric score per model, with a 95% bootstrap confidence interval over "
         "questions. <em>rating</em> is a Bradley-Terry strength from head-to-head comparisons "
         "on the same questions, on an Elo-like scale.</p>" + _table(board)),
        ("Is the gap real?",
         "<p>Paired bootstrap of the top model against each other model: the difference in "
         "mean score with its 95% interval. An interval that excludes zero is a real gap at "
         "this sample size; one that straddles zero is not.</p>"
         + _table(analysis.comparisons(df, seed=seed))),
        ("Score by subject", _table(analysis.score_by(df, "subject"), index=True)),
        ("Score by difficulty", _table(analysis.score_by(df, "difficulty"), index=True)),
        ("Error taxonomy",
         "<p>Number of answers carrying each error label. One answer can carry several.</p>"
         + _table(analysis.error_counts(df), index=True)
         + "<details><summary>What each label means</summary>" + _table(taxonomy) + "</details>"),
    ]
    breakdown = storage.error_breakdown(db_path, run_ids)
    if not breakdown.empty:
        parts = []
        for adapter, g in breakdown.groupby("adapter"):
            pivot = g.pivot_table(index="subject", columns="error_type", values="n",
                                  aggfunc="sum", fill_value=0)
            parts.append(f"<details><summary>{html.escape(adapter)}</summary>"
                         f"{_table(pivot, index=True)}</details>")
        sections.append(("Errors by subject", "<p>Where each model fails, by subject area "
                         "(computed with SQL over the <code>result_errors</code> table).</p>"
                         + "".join(parts)))
    agreement = analysis.judge_agreement(df)
    if not agreement.empty:
        sections.append(("Rubric vs LLM judge", "<p>Rank correlation between the pattern "
                         "rubric and the LLM judge on the same answers.</p>" + _table(agreement)))
    worst = analysis.worst_answers(df)
    sections.append(("Lowest-scoring answers",
                     "".join(_answer_card(r) for r in worst.itertuples(index=False))))

    sha = shown["dataset_sha256"].iloc[0][:12] if not shown.empty else "?"
    meta = (f"{len(df['question_id'].unique())} questions &middot; {len(run_ids)} runs &middot; "
            f"dataset <code>{html.escape(sha)}</code> &middot; generated "
            f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
    body = "".join(f"<h2>{html.escape(t)}</h2>{c}" for t, c in sections)
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Biology LLM Evaluation</title><style>{CSS}</style></head><body><main>"
        f'<h1>Biology LLM Evaluation</h1><p class="lead">{meta}</p>{body}</main></body></html>'
    )


def markdown_table(df: pd.DataFrame) -> str:
    """A GitHub-flavoured Markdown table, without the optional ``tabulate`` dependency."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        cells = ["-" if pd.isna(v) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(db_path: str | Path, out_html: str | Path, run_ids: list[str] | None = None,
                 markdown_path: str | Path | None = None) -> pd.DataFrame:
    Path(out_html).write_text(build_report(db_path, run_ids), encoding="utf-8")
    runs = storage.load_runs(db_path)
    df = storage.load_results(db_path, run_ids or analysis.latest_runs(runs))
    board = analysis.leaderboard(df)
    if markdown_path:
        Path(markdown_path).write_text(markdown_table(board) + "\n", encoding="utf-8")
    return board
