"""Pairwise ranking of models from per-question scores.

Mean score answers "how good is each model"; pairwise ranking answers "which
model gives the better answer when both see the same question", which is how
human preference data is usually collected. Strengths are fitted with the
Bradley-Terry model and reported on an Elo-like scale.
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import pandas as pd


def pairwise_outcomes(scores: pd.DataFrame, tie_margin: float = 0.05) -> pd.DataFrame:
    """One row per (question, model pair): who won, or a tie.

    ``scores`` needs columns ``model``, ``question_id`` and ``score``. Scores
    within ``tie_margin`` of each other count as a tie.
    """
    wide = scores.pivot_table(index="question_id", columns="model", values="score")
    rows = []
    for a, b in combinations(sorted(wide.columns), 2):
        pair = wide[[a, b]].dropna()
        for qid, (sa, sb) in pair.iterrows():
            if abs(sa - sb) <= tie_margin:
                winner = None
            else:
                winner = a if sa > sb else b
            rows.append({"question_id": qid, "model_a": a, "model_b": b, "winner": winner})
    return pd.DataFrame(rows, columns=["question_id", "model_a", "model_b", "winner"])


def bradley_terry(outcomes: pd.DataFrame, prior: float = 0.5, max_iter: int = 1000,
                  tol: float = 1e-10) -> pd.Series:
    """Fit Bradley-Terry strengths with the MM algorithm (Hunter, 2004).

    A tie counts as half a win for each side. ``prior`` adds that many virtual
    wins in each direction for every pair that met, so a model that won every
    comparison still gets a finite strength.
    """
    models = sorted(set(outcomes["model_a"]) | set(outcomes["model_b"]))
    if not models:
        return pd.Series(dtype=float)
    idx = {m: i for i, m in enumerate(models)}
    k = len(models)
    wins = np.zeros((k, k))
    for row in outcomes.itertuples(index=False):
        i, j = idx[row.model_a], idx[row.model_b]
        # pandas may store a tie as NaN rather than None, so test with isna.
        if pd.isna(row.winner):
            wins[i, j] += 0.5
            wins[j, i] += 0.5
        elif row.winner == row.model_a:
            wins[i, j] += 1
        else:
            wins[j, i] += 1
    met = (wins + wins.T) > 0
    wins = wins + prior * met
    games = wins + wins.T

    p = np.ones(k)
    for _ in range(max_iter):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        new = np.where(denom > 0, wins.sum(axis=1) / np.where(denom > 0, denom, 1), p)
        new /= np.exp(np.log(new).mean())  # fix the scale: geometric mean = 1
        if np.max(np.abs(new - p)) < tol:
            p = new
            break
        p = new
    return pd.Series(p, index=models, name="strength")


def rank_models(scores: pd.DataFrame, tie_margin: float = 0.05) -> pd.DataFrame:
    outcomes = pairwise_outcomes(scores, tie_margin)
    if outcomes.empty:
        return pd.DataFrame(columns=["model", "rating", "wins", "losses", "ties", "win_rate"])
    strength = bradley_terry(outcomes)
    rows = []
    for m in strength.index:
        involved = outcomes[(outcomes.model_a == m) | (outcomes.model_b == m)]
        wins = int((involved.winner == m).sum())
        ties = int(involved.winner.isna().sum())
        losses = len(involved) - wins - ties
        rows.append({
            "model": m,
            "rating": round(1000 + 400 * math.log10(strength[m]), 1),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": round((wins + 0.5 * ties) / len(involved), 3) if len(involved) else None,
        })
    return pd.DataFrame(rows).sort_values("rating", ascending=False, ignore_index=True)
