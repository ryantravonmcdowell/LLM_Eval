import pytest

from bioeval.schema import Constraints
from bioeval.scoring import ErrorType, RubricWeights, check_constraints, score_response
from bioeval.scoring.instruction import count_sentences, is_refusal


def test_every_reference_answer_scores_perfectly(questions):
    for q in questions:
        card = score_response(q, q.reference_answer)
        assert card.score == 1.0, (q.id, card.errors, card.violations)
        assert card.errors == ()


def test_every_misconception_is_caught_and_penalised(questions):
    for q in questions:
        for m in q.misconceptions:
            card = score_response(q, f"{q.reference_answer} {m.example}")
            assert ErrorType.FACTUAL_ERROR in card.errors, (q.id, m.id)
            assert m.id in card.facts.misconceptions


class TestRubric:
    def test_correct_answer(self, simple_question):
        card = score_response(simple_question, "The ratio is 3:1 since dominance masks the allele.")
        assert card.score == 1.0 and card.errors == ()

    def test_missing_reasoning_is_a_reasoning_gap(self, simple_question):
        card = score_response(simple_question, "The ratio is 3:1.")
        assert card.errors == (ErrorType.REASONING_GAP,)
        assert card.coverage == 0.5
        assert card.score == pytest.approx(0.5)

    def test_formatting_alone_earns_nothing(self, simple_question):
        card = score_response(simple_question, "Mendel studied peas in a monastery garden.")
        assert card.instruction_ok and card.coverage == 0.0
        assert card.score == 0.0

    def test_format_slip_costs_less_than_missing_content(self, simple_question):
        slip = score_response(simple_question, "3:1, as dominance masks it. " + "x " * 60)
        thin = score_response(simple_question, "The ratio is 3:1.")
        assert slip.score == pytest.approx(0.875) and slip.score > thin.score

    def test_missing_final_claim_is_a_missing_fact(self, simple_question):
        card = score_response(simple_question, "Dominant alleles mask recessive ones.")
        assert card.errors == (ErrorType.MISSING_KEY_FACT,)

    def test_misconception_halves_the_score(self, simple_question):
        clean = score_response(simple_question, "3:1, because dominance masks it.")
        wrong = score_response(simple_question, "3:1, because dominance masks it. Or 1:2:1.")
        assert ErrorType.FACTUAL_ERROR in wrong.errors
        assert wrong.score == pytest.approx(clean.score / 2)

    def test_refusal_scores_zero(self, simple_question):
        for text in ["", "   ", "I'm unable to answer that question.", "As an AI, I cannot help."]:
            card = score_response(simple_question, text)
            assert card.score == 0.0
            assert card.errors == (ErrorType.REFUSAL_OR_EMPTY,)

    def test_answer_that_mentions_refusing_is_not_a_refusal(self):
        assert not is_refusal("Cells cannot divide without a spindle.")

    def test_long_answer_with_a_hedge_is_graded_on_content(self):
        hedge = "I can't provide exact figures, but " + "the ratio is 3:1 because of dominance " * 8
        assert not is_refusal(hedge)

    def test_curly_apostrophes(self):
        assert is_refusal("I’m unable to answer this.")

    def test_too_long_is_an_instruction_violation(self, simple_question):
        long = "The ratio is 3:1 because dominance masks it. " + "word " * 60
        card = score_response(simple_question, long)
        assert card.errors == (ErrorType.INSTRUCTION_VIOLATION,)
        assert not card.instruction_ok

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            RubricWeights(fact=0.9, instruction=0.3)

    def test_custom_weights(self, simple_question):
        card = score_response(simple_question, "The ratio is 3:1.",
                              RubricWeights(fact=1.0, instruction=0.0))
        assert card.score == 0.5


class TestInstructionChecks:
    def test_numbered_list(self):
        ok = "1. Prophase\n2. Metaphase\n3. Anaphase"
        assert check_constraints(ok, Constraints(format="numbered_list")) == []
        assert check_constraints("1) a\n2) b", Constraints(format="numbered_list")) == []

    @pytest.mark.parametrize("text", ["Prophase, metaphase, anaphase.", "2. a\n3. b",
                                      "1. a\n3. b", "1. only one item"])
    def test_bad_numbered_lists(self, text):
        assert check_constraints(text, Constraints(format="numbered_list"))

    def test_bullets(self):
        assert check_constraints("- a\n- b", Constraints(format="bullets")) == []
        assert check_constraints("* a\n• b", Constraints(format="bullets")) == []
        assert check_constraints("- only one", Constraints(format="bullets"))

    def test_single_sentence(self):
        assert count_sentences("Glycolysis yields 2 ATP (4 made, e.g. by substrate-level "
                               "phosphorylation, minus 2 used).") == 1
        assert count_sentences("One. Two.") == 2
        assert check_constraints("Yes. No.", Constraints(format="single_sentence"))

    def test_word_limits(self):
        c = Constraints(max_words=3, min_words=2)
        assert check_constraints("one two three", c) == []
        assert "too long" in check_constraints("one two three four", c)[0]
        assert "too short" in check_constraints("one", c)[0]
