import numpy as np
import pandas as pd
import pytest

from bioeval.judge import JudgeParseError, LLMJudge, build_judge_prompt, parse_verdict
from bioeval.ranking import bradley_terry, pairwise_outcomes, rank_models
from bioeval.stats import bootstrap_mean, paired_bootstrap, spearman


class TestJudge:
    def test_parse_plain_json(self):
        v = parse_verdict('{"score": 5, "errors": [], "rationale": "Complete."}')
        assert (v.raw_score, v.score, v.rationale) == (5, 1.0, "Complete.")

    def test_parse_json_inside_prose(self):
        v = parse_verdict('Here is my grade:\n{"score": 2, "errors": ["factual_error"]}\nThanks')
        assert v.score == 0.25 and v.errors == ("factual_error",)

    def test_unknown_error_labels_are_dropped(self):
        v = parse_verdict('{"score": 3, "errors": ["made_up", "reasoning_gap"]}')
        assert v.errors == ("reasoning_gap",)

    @pytest.mark.parametrize("text", ["no json here", '{"score": 9}', '{"score": "high"}',
                                      '{"score": true}', "{broken"])
    def test_rejects_bad_output(self, text):
        with pytest.raises(JudgeParseError):
            parse_verdict(text)

    def test_prompt_contains_rubric_and_answer(self, simple_question):
        prompt = build_judge_prompt(simple_question, "It is 3:1.")
        assert simple_question.reference_answer in prompt
        assert "<answer>\nIt is 3:1.\n</answer>" in prompt
        assert "gives 1:2:1" in prompt

    def test_retries_then_gives_up(self, simple_question):
        class Garbage:
            name = "garbage"
            calls = 0

            def generate(self, request):
                Garbage.calls += 1
                from bioeval.adapters import Completion
                return Completion(text="I think it is fine", model="g")

        with pytest.raises(JudgeParseError, match="after 2 tries"):
            LLMJudge(Garbage(), retries=1).grade(simple_question, "3:1")
        assert Garbage.calls == 2


def _scores(table):
    return pd.DataFrame([{"model": m, "question_id": q, "score": s}
                         for m, row in table.items() for q, s in row.items()])


class TestRanking:
    def test_recovers_true_order(self):
        rng = np.random.default_rng(0)
        true_skill = {"a": 0.9, "b": 0.6, "c": 0.3}
        rows = [
            {"model": m, "question_id": q, "score": float(np.clip(s + rng.normal(0, 0.15), 0, 1))}
            for q in range(200)
            for m, s in true_skill.items()
        ]
        ranked = rank_models(pd.DataFrame(rows))
        assert ranked["model"].tolist() == ["a", "b", "c"]

    def test_ties_count_half_even_when_mixed_with_wins(self):
        # Regression: pandas stored ties as NaN once wins were present, and NaN
        # "is not None", so every tie was credited to the second model.
        scores = _scores({
            "alpha": {"q1": 1.0, "q2": 1.0, "q3": 1.0, "q4": 1.0},
            "beta": {"q1": 0.0, "q2": 1.0, "q3": 1.0, "q4": 1.0},
        })
        outcomes = pairwise_outcomes(scores)
        assert outcomes["winner"].isna().sum() == 3
        strength = bradley_terry(outcomes, prior=0)
        # 1 win + 3 ties = 2.5 vs 1.5 points, so alpha must come out ahead
        assert strength["alpha"] > strength["beta"]
        ranked = rank_models(scores)
        assert ranked.iloc[0]["model"] == "alpha"
        assert ranked.set_index("model").loc["beta", "ties"] == 3

    def test_undefeated_model_has_finite_rating(self):
        scores = _scores({"a": {"q1": 1.0, "q2": 1.0}, "b": {"q1": 0.0, "q2": 0.0}})
        ranked = rank_models(scores)
        assert np.isfinite(ranked["rating"]).all()
        assert ranked.iloc[0]["model"] == "a" and ranked.iloc[0]["win_rate"] == 1.0

    def test_single_model_has_no_ranking(self):
        assert rank_models(_scores({"a": {"q1": 1.0}})).empty


class TestStats:
    def test_bootstrap_interval_brackets_the_mean(self):
        ci = bootstrap_mean([0.2, 0.4, 0.6, 0.8, 1.0])
        assert ci.estimate == pytest.approx(0.6)
        assert ci.low < ci.estimate < ci.high

    def test_bootstrap_is_reproducible(self):
        x = np.random.default_rng(1).random(40)
        assert bootstrap_mean(x, seed=7) == bootstrap_mean(x, seed=7)

    def test_bootstrap_ignores_nan_and_handles_empty(self):
        assert bootstrap_mean([1.0, np.nan, 1.0]).estimate == 1.0
        assert np.isnan(bootstrap_mean([]).estimate)

    def test_paired_bootstrap_detects_a_real_gap(self):
        rng = np.random.default_rng(2)
        base = rng.random(60)
        c = paired_bootstrap(base + 0.1, base)
        assert c.significant and c.mean_diff == pytest.approx(0.1) and c.p_value < 0.01
        # a p-value finer than 1/resamples is reported as a bound, never as 0
        assert c.p_value_text == "< 0.0002"

    def test_paired_bootstrap_no_gap(self):
        x = np.random.default_rng(3).random(30)
        c = paired_bootstrap(x, x)
        assert not c.significant and c.p_value == 1.0 and c.p_value_text == "1.0000"

    def test_paired_requires_same_length(self):
        with pytest.raises(ValueError):
            paired_bootstrap([1, 2], [1])

    def test_spearman(self):
        assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
        assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
        assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))
