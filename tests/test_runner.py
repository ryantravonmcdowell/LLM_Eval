import json
import sqlite3

import pytest

from bioeval import storage
from bioeval.adapters import AdapterConfigError, AdapterError, Completion, MockAdapter
from bioeval.judge import LLMJudge
from bioeval.runner import run_evaluation


class FlakyAdapter:
    """Answers with the reference, but fails on chosen questions."""

    def __init__(self, questions, fail_ids=(), fatal_ids=()):
        self.name = "flaky"
        self._ref = {q.id: q.reference_answer for q in questions}
        self.fail_ids, self.fatal_ids = set(fail_ids), set(fatal_ids)

    def generate(self, request):
        if request.question_id in self.fatal_ids:
            raise AdapterConfigError("invalid API key")
        if request.question_id in self.fail_ids:
            raise AdapterError("HTTP 503")
        return Completion(text=self._ref[request.question_id], model="flaky-1", latency_ms=5.0)


class JudgeStub:
    name = "stub"

    def __init__(self, score=4):
        self.score = score

    def generate(self, request):
        return Completion(text=f'Sure: {{"score": {self.score}, "errors": [], '
                               f'"rationale": "fine"}}', model="stub")


def count(db, sql, *params):
    with sqlite3.connect(db) as conn:
        return conn.execute(sql, params).fetchone()[0]


def test_full_run_is_stored(tmp_path, dataset, questions):
    db = tmp_path / "r.db"
    summary = run_evaluation(db, dataset, MockAdapter(questions, "medium"), progress=lambda m: None)
    assert summary.attempted == len(questions) and summary.failed == 0
    assert count(db, "SELECT COUNT(*) FROM results") == len(questions)
    assert count(db, "SELECT status FROM runs") == "complete"
    df = storage.load_results(db)
    assert set(df["subject"]) and df["score"].between(0, 1).all()
    assert summary.mean_score == pytest.approx(df["score"].mean(), abs=1e-4)


def test_resume_skips_finished_questions(tmp_path, dataset, questions):
    db = tmp_path / "r.db"
    adapter = MockAdapter(questions, "strong")
    first = run_evaluation(db, dataset, adapter, limit=10, progress=lambda m: None)
    second = run_evaluation(db, dataset, adapter, resume_run_id=first.run_id,
                            progress=lambda m: None)
    assert (second.attempted, second.skipped) == (len(questions) - 10, 10)
    stored = count(db, "SELECT COUNT(*) FROM results WHERE run_id = ?", first.run_id)
    assert stored == len(questions)


def test_resume_refuses_a_different_model(tmp_path, dataset, questions):
    db = tmp_path / "r.db"
    first = run_evaluation(db, dataset, MockAdapter(questions, "strong"), limit=2,
                           progress=lambda m: None)
    with pytest.raises(ValueError, match="cannot resume"):
        run_evaluation(db, dataset, MockAdapter(questions, "weak"), resume_run_id=first.run_id,
                       progress=lambda m: None)


def test_failed_calls_are_recorded_not_fatal(tmp_path, dataset, questions):
    db = tmp_path / "r.db"
    bad = [questions[0].id, questions[1].id]
    summary = run_evaluation(db, dataset, FlakyAdapter(questions, fail_ids=bad),
                             progress=lambda m: None)
    assert summary.failed == 2
    assert count(db, "SELECT status FROM runs") == "partial"
    assert count(db, "SELECT COUNT(*) FROM result_errors WHERE error_type = 'adapter_error'") == 2
    assert count(db, "SELECT COUNT(*) FROM results WHERE score IS NULL") == 2


def test_config_errors_abort_the_run(tmp_path, dataset, questions):
    db = tmp_path / "r.db"
    with pytest.raises(AdapterConfigError):
        run_evaluation(db, dataset, FlakyAdapter(questions, fatal_ids=[questions[0].id]),
                       workers=1, progress=lambda m: None)
    assert count(db, "SELECT status FROM runs") == "aborted"


def test_judge_scores_are_stored(tmp_path, dataset, questions):
    db = tmp_path / "r.db"
    run_evaluation(db, dataset, MockAdapter(questions, "strong"), judge=LLMJudge(JudgeStub(4)),
                   limit=5, progress=lambda m: None)
    df = storage.load_results(db)
    assert (df["judge_score"] == 0.75).all()
    assert (df["judge_rationale"] == "fine").all()


def test_results_carry_rubric_details(tmp_path, dataset, questions):
    db = tmp_path / "r.db"
    run_evaluation(db, dataset, MockAdapter(questions, "weak"), progress=lambda m: None)
    df = storage.load_results(db)
    wrong = df[df["n_misconceptions"] > 0]
    assert not wrong.empty
    for row in wrong.itertuples():
        assert "factual_error" in row.errors
        assert json.loads(row.details_json)["misconceptions"]


def test_refuses_a_broken_dataset(tmp_path, questions):
    from bioeval.dataset import audit_records

    broken = audit_records([(1, "{oops")])
    with pytest.raises(ValueError, match="dataset with errors"):
        run_evaluation(tmp_path / "r.db", broken, MockAdapter(questions))
