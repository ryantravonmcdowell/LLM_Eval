"""Load a JSONL question set and audit it before anything is graded against it.

A rubric that is wrong grades every model wrong, so the dataset is checked the
same way the models are: each reference answer must pass its own rubric, and
each misconception's example must be caught by its own patterns. A broken
pattern is found here instead of silently scoring every answer zero.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

from bioeval.schema import Question, SchemaError, question_from_dict
from bioeval.scoring.factuality import fact_present, misconception_present
from bioeval.scoring.instruction import check_constraints
from bioeval.textnorm import jaccard, normalize, shingles

NEAR_DUPLICATE_THRESHOLD = 0.8


@dataclass(frozen=True)
class Issue:
    line: int | None
    question_id: str | None
    check: str
    severity: str  # "error" | "warning"
    message: str


@dataclass
class DatasetReport:
    path: str
    sha256: str
    n_records: int
    questions: list[Question]
    issues: list[Issue]

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def issues_frame(self) -> pd.DataFrame:
        cols = ["line", "question_id", "check", "severity", "message"]
        return pd.DataFrame([asdict(i) for i in self.issues], columns=cols)

    def coverage_frame(self) -> pd.DataFrame:
        rows = [{"subject": q.subject, "difficulty": q.difficulty} for q in self.questions]
        if not rows:
            return pd.DataFrame()
        return pd.crosstab(
            pd.DataFrame(rows)["subject"], pd.DataFrame(rows)["difficulty"], margins=True
        )

    def summary(self) -> dict:
        valid = len(self.questions)
        checks = Counter(i.check for i in self.issues)
        return {
            "path": self.path,
            "sha256": self.sha256,
            "records": self.n_records,
            "valid_questions": valid,
            "invalid_records": self.n_records - valid,
            "errors": len(self.errors),
            "warnings": len(self.warnings),
            "issues_by_check": dict(sorted(checks.items())),
            "key_facts": sum(len(q.key_facts) for q in self.questions),
            "misconceptions": sum(len(q.misconceptions) for q in self.questions),
        }


def _regex_issues(q: Question, line: int) -> list[Issue]:
    out = []
    groups = [("key_facts", f.id, f.patterns) for f in q.key_facts]
    groups += [("misconceptions", m.id, m.patterns) for m in q.misconceptions]
    for section, item_id, patterns in groups:
        for p in patterns:
            try:
                re.compile(p)
            except re.error as exc:
                out.append(
                    Issue(line, q.id, "invalid_regex", "error",
                          f"{section}.{item_id}: {p!r} does not compile ({exc})")
                )
    return out


def _consistency_issues(q: Question, line: int) -> list[Issue]:
    """Each reference answer must pass its own rubric."""
    out = []
    ids = [f.id for f in q.key_facts] + [m.id for m in q.misconceptions]
    for dup, n in Counter(ids).items():
        if n > 1:
            out.append(Issue(line, q.id, "duplicate_item_id", "error",
                             f"rubric item id {dup!r} used {n} times"))

    ref = normalize(q.reference_answer)
    for f in q.key_facts:
        if not fact_present(f, ref):
            out.append(Issue(line, q.id, "reference_missing_fact", "error",
                             f"reference answer does not match key fact {f.id!r}"))
    for m in q.misconceptions:
        if misconception_present(m, ref):
            out.append(Issue(line, q.id, "reference_has_misconception", "error",
                             f"reference answer matches misconception {m.id!r}"))
        if not misconception_present(m, normalize(m.example)):
            out.append(Issue(line, q.id, "misconception_example_mismatch", "error",
                             f"example for {m.id!r} is not caught by its own patterns"))
    for v in check_constraints(q.reference_answer, q.constraints):
        out.append(Issue(line, q.id, "reference_violates_constraints", "error", v))
    if not q.misconceptions:
        out.append(Issue(line, q.id, "no_misconceptions", "warning",
                         "no misconceptions defined; factual errors cannot be detected"))
    return out


def audit_records(records: list[tuple[int, str]], path: str = "<memory>") -> DatasetReport:
    """Audit raw ``(line_number, text)`` pairs. Separated from I/O for testing."""
    digest = hashlib.sha256("\n".join(t for _, t in records).encode("utf-8")).hexdigest()
    questions: list[Question] = []
    issues: list[Issue] = []
    seen: dict[str, int] = {}
    n_records = 0

    for line, text in records:
        if not text.strip():
            continue
        n_records += 1
        try:
            raw = json.loads(text)
        except json.JSONDecodeError as exc:
            issues.append(Issue(line, None, "json_parse", "error", str(exc)))
            continue
        qid = raw.get("id") if isinstance(raw, dict) else None
        try:
            q = question_from_dict(raw)
        except SchemaError as exc:
            issues.append(Issue(line, qid, "schema", "error", str(exc)))
            continue
        if q.id in seen:
            issues.append(Issue(line, q.id, "duplicate_id", "error",
                                f"id already used on line {seen[q.id]}"))
            continue
        seen[q.id] = line

        regex_problems = _regex_issues(q, line)
        issues.extend(regex_problems)
        if not regex_problems:
            issues.extend(_consistency_issues(q, line))
        questions.append(q)

    sh = {q.id: shingles(q.prompt) for q in questions}
    for a, b in combinations(questions, 2):
        sim = jaccard(sh[a.id], sh[b.id])
        if sim >= NEAR_DUPLICATE_THRESHOLD:
            issues.append(Issue(seen[b.id], b.id, "near_duplicate_prompt", "warning",
                                f"prompt is {sim:.0%} similar to {a.id!r}"))

    return DatasetReport(path, digest, n_records, questions, issues)


def load_dataset(path: str | Path) -> DatasetReport:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    return audit_records(list(enumerate(lines, start=1)), str(path))


def load_questions(path: str | Path) -> list[Question]:
    """Load a dataset and refuse to continue if it has errors."""
    report = load_dataset(path)
    if not report.ok:
        first = report.errors[0]
        raise ValueError(
            f"{path}: {len(report.errors)} dataset error(s); first on line {first.line} "
            f"[{first.check}] {first.message}. Run `bioeval validate {path}` for the full list."
        )
    return report.questions
