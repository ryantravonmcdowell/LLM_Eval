"""Turn stored results into the tables a report needs.

Everything here takes the DataFrame from :func:`bioeval.storage.load_results`,
so the same numbers back the HTML report, the CLI leaderboard and the notebook.
"""

from __future__ import annotations

import pandas as pd

from bioeval.ranking import rank_models
from bioeval.scoring.taxonomy import ErrorType
from bioeval.stats import bootstrap_mean, paired_bootstrap, spearman


def latest_runs(runs: pd.DataFrame) -> list[str]:
    """The most recent finished run for each adapter."""
    usable = runs[runs["status"].isin(["complete", "partial"])]
    if usable.empty:
        return []
    latest = usable.sort_values("started_at").groupby("adapter").tail(1)
    return latest["run_id"].tolist()


def _has(errors: list[str], kind: ErrorType) -> bool:
    return kind.value in errors


def leaderboard(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rows = []
    for adapter, g in df.groupby("adapter"):
        graded = g[g["score"].notna()]
        ci = bootstrap_mean(graded["score"], seed=seed)
        refused = g["errors"].apply(_has, kind=ErrorType.REFUSAL_OR_EMPTY)
        failed = g["errors"].apply(_has, kind=ErrorType.ADAPTER_ERROR)
        rows.append({
            "model": adapter,
            "answered": len(graded),
            "mean_score": round(ci.estimate, 3),
            "ci_low": round(ci.low, 3),
            "ci_high": round(ci.high, 3),
            "fact_coverage": round(graded["coverage"].mean(), 3),
            "misconception_rate": round((graded["n_misconceptions"] > 0).mean(), 3),
            "instruction_pass": round(graded["instruction_ok"].mean(), 3),
            "refusal_rate": round(refused.mean(), 3),
            "api_failures": int(failed.sum()),
            "median_latency_ms": round(graded["latency_ms"].median(), 1),
        })
    board = pd.DataFrame(rows)
    if board.empty:
        return board
    scores = df.loc[df["score"].notna(), ["adapter", "question_id", "score"]]
    ratings = rank_models(scores.rename(columns={"adapter": "model"}))
    if not ratings.empty:
        board = board.merge(ratings[["model", "rating", "win_rate"]], on="model", how="left")
    return board.sort_values("mean_score", ascending=False, ignore_index=True)


def comparisons(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Paired bootstrap of the best model against every other model."""
    wide = df.pivot_table(index="question_id", columns="adapter", values="score")
    if wide.shape[1] < 2:
        return pd.DataFrame()
    order = wide.mean().sort_values(ascending=False).index.tolist()
    best = order[0]
    rows = []
    for other in order[1:]:
        pair = wide[[best, other]].dropna()
        c = paired_bootstrap(pair[best], pair[other], seed=seed)
        rows.append({
            "comparison": f"{best} vs {other}",
            "questions": c.n,
            "mean_diff": round(c.mean_diff, 3),
            "ci_low": round(c.low, 3),
            "ci_high": round(c.high, 3),
            "p_value": c.p_value_text,
            "significant": "yes" if c.significant else "no",
        })
    return pd.DataFrame(rows)


def score_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    table = df.pivot_table(index=column, columns="adapter", values="score", aggfunc="mean")
    return table.round(3)


def error_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Rows = model, columns = error type, values = number of answers with that error."""
    exploded = df[["adapter", "errors"]].explode("errors", ignore_index=True).dropna()
    if exploded.empty:
        return pd.DataFrame()
    table = pd.crosstab(exploded["adapter"], exploded["errors"])
    order = [e.value for e in ErrorType if e.value in table.columns]
    return table[order]


def judge_agreement(df: pd.DataFrame) -> pd.DataFrame:
    judged = df[df["judge_score"].notna() & df["score"].notna()]
    if len(judged) < 3:
        return pd.DataFrame()
    rows = []
    groups = [("all models", judged)] + list(judged.groupby("adapter"))
    for name, g in groups:
        rows.append({
            "model": name,
            "judged": len(g),
            "spearman_rho": round(spearman(g["score"], g["judge_score"]), 3),
            "mean_abs_diff": round((g["score"] - g["judge_score"]).abs().mean(), 3),
        })
    return pd.DataFrame(rows)


def worst_answers(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    graded = df[df["score"].notna()].sort_values(["score", "question_id"]).head(n)
    return graded[["adapter", "question_id", "subject", "score", "errors", "response",
                   "details_json"]]
