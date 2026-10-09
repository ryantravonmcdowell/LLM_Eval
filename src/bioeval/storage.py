"""SQLite results store.

Every run, response and score is written here, so any report can be rebuilt
from the database alone and a crashed run can resume where it stopped. Error
labels are stored one row per label in ``result_errors`` so breakdowns are a
plain ``GROUP BY`` rather than JSON parsing.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from bioeval.schema import Question

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    dataset_sha256   TEXT NOT NULL,
    question_id      TEXT NOT NULL,
    subject          TEXT NOT NULL,
    difficulty       TEXT NOT NULL,
    prompt           TEXT NOT NULL,
    n_key_facts      INTEGER NOT NULL,
    n_misconceptions INTEGER NOT NULL,
    PRIMARY KEY (dataset_sha256, question_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    adapter        TEXT NOT NULL,
    dataset_path   TEXT,
    dataset_sha256 TEXT NOT NULL,
    started_at     TEXT NOT NULL,
    finished_at    TEXT,
    status         TEXT NOT NULL DEFAULT 'running',
    config_json    TEXT
);

CREATE TABLE IF NOT EXISTS results (
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    question_id      TEXT NOT NULL,
    response         TEXT,
    model            TEXT,
    latency_ms       REAL,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    stop_reason      TEXT,
    score            REAL,
    coverage         REAL,
    instruction_ok   INTEGER,
    n_misconceptions INTEGER,
    details_json     TEXT,
    adapter_error    TEXT,
    judge_score      REAL,
    judge_rationale  TEXT,
    created_at       TEXT NOT NULL,
    PRIMARY KEY (run_id, question_id)
);

CREATE TABLE IF NOT EXISTS result_errors (
    run_id      TEXT NOT NULL,
    question_id TEXT NOT NULL,
    error_type  TEXT NOT NULL,
    PRIMARY KEY (run_id, question_id, error_type)
);

CREATE INDEX IF NOT EXISTS idx_result_errors_type ON result_errors(error_type);

CREATE VIEW IF NOT EXISTS v_results AS
SELECT r.*, runs.adapter, q.subject, q.difficulty
FROM results r
JOIN runs ON runs.run_id = r.run_id
JOIN questions q ON q.question_id = r.question_id AND q.dataset_sha256 = runs.dataset_sha256;
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def register_questions(conn: sqlite3.Connection, sha256: str, questions: list[Question]) -> None:
    conn.executemany(
        "INSERT OR IGNORE INTO questions VALUES (?, ?, ?, ?, ?, ?, ?)",
        [(sha256, q.id, q.subject, q.difficulty, q.prompt, len(q.key_facts),
          len(q.misconceptions)) for q in questions],
    )
    conn.commit()


def create_run(conn: sqlite3.Connection, adapter: str, dataset_path: str, sha256: str,
               config: dict | None = None) -> str:
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
    conn.execute(
        "INSERT INTO runs (run_id, adapter, dataset_path, dataset_sha256, started_at, config_json)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        (run_id, adapter, dataset_path, sha256, _now(), json.dumps(config or {}, sort_keys=True)),
    )
    conn.commit()
    return run_id


def get_run(conn: sqlite3.Connection, run_id: str) -> dict | None:
    cur = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
    row = cur.fetchone()
    if row is None:
        return None
    return dict(zip([c[0] for c in cur.description], row, strict=True))


def finish_run(conn: sqlite3.Connection, run_id: str, status: str = "complete") -> None:
    conn.execute("UPDATE runs SET finished_at = ?, status = ? WHERE run_id = ?",
                 (_now(), status, run_id))
    conn.commit()


def completed_question_ids(conn: sqlite3.Connection, run_id: str) -> set[str]:
    rows = conn.execute("SELECT question_id FROM results WHERE run_id = ?", (run_id,))
    return {r[0] for r in rows}


def save_result(conn: sqlite3.Connection, run_id: str, row: dict, errors: list[str]) -> None:
    cols = ["run_id", "question_id", "response", "model", "latency_ms", "input_tokens",
            "output_tokens", "stop_reason", "score", "coverage", "instruction_ok",
            "n_misconceptions", "details_json", "adapter_error", "judge_score",
            "judge_rationale", "created_at"]
    values = {**{c: None for c in cols}, **row, "run_id": run_id, "created_at": _now()}
    conn.execute(
        f"INSERT OR REPLACE INTO results ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [values[c] for c in cols],
    )
    conn.execute("DELETE FROM result_errors WHERE run_id = ? AND question_id = ?",
                 (run_id, row["question_id"]))
    conn.executemany("INSERT INTO result_errors VALUES (?, ?, ?)",
                     [(run_id, row["question_id"], e) for e in sorted(set(errors))])
    conn.commit()


def load_results(path: str | Path, run_ids: list[str] | None = None) -> pd.DataFrame:
    with closing(connect(path)) as conn:
        query = "SELECT * FROM v_results"
        params: list = []
        if run_ids:
            query += f" WHERE run_id IN ({', '.join('?' * len(run_ids))})"
            params = list(run_ids)
        df = pd.read_sql_query(query, conn, params=params)
        errs = pd.read_sql_query("SELECT * FROM result_errors", conn)
    labels: dict[tuple[str, str], list[str]] = {}
    for e in errs.itertuples(index=False):
        labels.setdefault((e.run_id, e.question_id), []).append(e.error_type)
    keys = zip(df["run_id"], df["question_id"], strict=True)
    df["errors"] = [sorted(labels.get(key, [])) for key in keys]
    return df


def load_runs(path: str | Path) -> pd.DataFrame:
    with closing(connect(path)) as conn:
        return pd.read_sql_query("SELECT * FROM runs ORDER BY started_at", conn)


ERROR_BREAKDOWN_SQL = """
SELECT runs.adapter, q.subject, e.error_type, COUNT(*) AS n
FROM result_errors e
JOIN runs ON runs.run_id = e.run_id
JOIN questions q ON q.question_id = e.question_id AND q.dataset_sha256 = runs.dataset_sha256
{where}
GROUP BY runs.adapter, q.subject, e.error_type
ORDER BY runs.adapter, q.subject, e.error_type
"""


def error_breakdown(path: str | Path, run_ids: list[str] | None = None) -> pd.DataFrame:
    where, params = "", []
    if run_ids:
        where = f"WHERE e.run_id IN ({', '.join('?' * len(run_ids))})"
        params = list(run_ids)
    with closing(connect(path)) as conn:
        return pd.read_sql_query(ERROR_BREAKDOWN_SQL.format(where=where), conn, params=params)
