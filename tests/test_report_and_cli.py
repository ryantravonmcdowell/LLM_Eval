import json

import pandas as pd
import pytest

from bioeval.adapters import Completion
from bioeval.cli import main
from bioeval.report import build_report, markdown_table
from bioeval.runner import run_evaluation


class Injector:
    """Returns hostile HTML as every answer, to prove the report escapes it."""

    name = "injector"

    def generate(self, request):
        return Completion(text='<script>alert("x")</script> 3:1', model="inj")


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    assert main(["demo", "--out-dir", str(out)]) == 0
    return out


def test_demo_writes_all_outputs(demo_dir):
    assert (demo_dir / "results.db").exists()
    assert (demo_dir / "leaderboard.md").read_text(encoding="utf-8").startswith("| model |")
    html = (demo_dir / "report.html").read_text(encoding="utf-8")
    for section in ["Leaderboard", "Is the gap real?", "Score by subject", "Error taxonomy",
                    "Errors by subject", "Lowest-scoring answers"]:
        assert f"<h2>{section}</h2>" in html


def test_leaderboard_ranks_mock_models(demo_dir, capsys):
    assert main(["leaderboard", "--db", str(demo_dir / "results.db")]) == 0
    out = capsys.readouterr().out
    assert out.index("mock-strong") < out.index("mock-medium") < out.index("mock-weak")
    assert "Paired bootstrap" in out


def test_report_escapes_model_output(tmp_path, dataset):
    db = tmp_path / "r.db"
    run_evaluation(db, dataset, Injector(), limit=3, progress=lambda m: None)
    html = build_report(db)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_markdown_table():
    md = markdown_table(pd.DataFrame({"model": ["a"], "score": [0.5], "x": [None]}))
    assert md.splitlines() == ["| model | score | x |", "|---|---|---|", "| a | 0.5 | - |"]


def test_validate_command(capsys):
    assert main(["validate"]) == 0
    assert "OK: dataset is valid." in capsys.readouterr().out


def test_validate_json_and_failure(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{oops\n", encoding="utf-8")
    assert main(["validate", str(bad), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["errors"] == 1
    assert payload["issues"][0]["check"] == "json_parse"


def test_run_and_report_commands(tmp_path, capsys):
    db, html = tmp_path / "r.db", tmp_path / "r.html"
    assert main(["run", "--model", "mock:strong", "--model", "mock:weak", "--db", str(db),
                 "--limit", "8"]) == 0
    assert main(["report", "--db", str(db), "--out", str(html)]) == 0
    assert html.exists()


def test_missing_file_is_a_clean_error(tmp_path, capsys):
    assert main(["validate", str(tmp_path / "nope.jsonl")]) == 2
    assert "error:" in capsys.readouterr().err
