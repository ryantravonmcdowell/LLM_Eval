import pytest
from conftest import jsonl, record

from bioeval.dataset import audit_records, load_questions
from bioeval.schema import SUBJECTS


def checks(report):
    return sorted(i.check for i in report.issues)


class TestBundledDataset:
    def test_is_clean(self, dataset):
        assert dataset.errors == [], dataset.issues_frame().to_string()
        assert dataset.warnings == []

    def test_size_and_breadth(self, questions):
        assert len(questions) >= 30
        assert {q.subject for q in questions} == SUBJECTS
        assert all(q.misconceptions for q in questions)

    def test_every_format_constraint_is_exercised(self, questions):
        formats = {q.constraints.format for q in questions}
        assert {"numbered_list", "bullets", "single_sentence"} <= formats


class TestAuditCatchesBrokenRecords:
    def test_clean_record(self):
        report = audit_records(jsonl(record()))
        assert report.ok and not report.issues

    def test_bad_json(self):
        report = audit_records(jsonl("{not json"))
        assert checks(report) == ["json_parse"]
        assert report.summary()["invalid_records"] == 1

    def test_schema_error_keeps_line_number(self):
        report = audit_records(jsonl(record(), record(id="t-002", subject="astrology")))
        (issue,) = report.errors
        assert (issue.check, issue.line, issue.question_id) == ("schema", 2, "t-002")

    def test_duplicate_id(self):
        report = audit_records(jsonl(record(), record()))
        assert "duplicate_id" in checks(report)

    def test_invalid_regex(self):
        bad = record(key_facts=[{"id": "x", "description": "d", "patterns": ["(unclosed"]}])
        assert checks(audit_records(jsonl(bad))) == ["invalid_regex"]

    def test_reference_must_contain_every_key_fact(self):
        bad = record(reference_answer="It is 3:1.")  # no reasoning step
        report = audit_records(jsonl(bad))
        assert checks(report) == ["reference_missing_fact"]
        assert "dominance" in report.errors[0].message

    def test_reference_must_not_contain_a_misconception(self):
        bad = record(reference_answer="It is 3:1, not 1:2:1, because dominance masks it.")
        assert "reference_has_misconception" in checks(audit_records(jsonl(bad)))

    def test_misconception_example_must_trigger_its_own_pattern(self):
        bad = record(misconceptions=[{"id": "m", "description": "d", "patterns": [r"1:2:1"],
                                      "example": "Half are dominant."}])
        assert checks(audit_records(jsonl(bad))) == ["misconception_example_mismatch"]

    def test_reference_must_obey_its_own_constraints(self):
        bad = record(constraints={"format": "numbered_list"})
        assert checks(audit_records(jsonl(bad))) == ["reference_violates_constraints"]

    def test_near_duplicate_prompts_warn(self):
        twin = record(id="t-002", prompt=record()["prompt"].rstrip("?"))
        report = audit_records(jsonl(record(), twin))
        assert report.ok
        assert checks(report) == ["near_duplicate_prompt"]

    def test_missing_misconceptions_warn(self):
        report = audit_records(jsonl(record(misconceptions=[])))
        assert report.ok and checks(report) == ["no_misconceptions"]


def test_load_questions_refuses_a_broken_file(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="dataset error"):
        load_questions(path)
