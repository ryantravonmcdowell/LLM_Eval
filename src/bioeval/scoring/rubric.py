"""Combine fact coverage, misconceptions and instruction checks into one score."""

from __future__ import annotations

from dataclasses import dataclass, field

from bioeval.schema import Question
from bioeval.scoring.factuality import FactResult, score_facts
from bioeval.scoring.instruction import check_constraints, is_refusal
from bioeval.scoring.taxonomy import ErrorType


@dataclass(frozen=True)
class RubricWeights:
    """How the composite score is built.

    ``score = coverage * (fact + instruction * compliance) * (1 - penalty)``
    where ``penalty = min(1, misconception_penalty * n_misconceptions)``.

    Content comes first: formatting compliance scales the content score rather
    than adding to it, so a well-formatted answer with nothing correct in it
    scores zero. A single misconception halves the score by default, so an
    answer that is complete but wrong cannot outscore a shorter correct one.
    """

    fact: float = 0.75
    instruction: float = 0.25
    misconception_penalty: float = 0.5

    def __post_init__(self):
        if abs(self.fact + self.instruction - 1.0) > 1e-9:
            raise ValueError("fact and instruction weights must sum to 1")
        if not 0 <= self.misconception_penalty <= 1:
            raise ValueError("misconception_penalty must be in [0, 1]")


@dataclass(frozen=True)
class ScoreCard:
    question_id: str
    score: float
    coverage: float
    instruction_ok: bool
    violations: tuple[str, ...]
    facts: FactResult | None
    errors: tuple[ErrorType, ...] = field(default_factory=tuple)

    @property
    def error_labels(self) -> list[str]:
        return [e.value for e in self.errors]


def score_response(
    question: Question, text: str, weights: RubricWeights | None = None
) -> ScoreCard:
    weights = weights or RubricWeights()
    if is_refusal(text):
        return ScoreCard(
            question_id=question.id,
            score=0.0,
            coverage=0.0,
            instruction_ok=False,
            violations=("no substantive answer",),
            facts=None,
            errors=(ErrorType.REFUSAL_OR_EMPTY,),
        )

    facts = score_facts(question, text)
    violations = tuple(check_constraints(text, question.constraints))
    compliance = max(0.0, 1.0 - 0.5 * len(violations))
    penalty = min(1.0, weights.misconception_penalty * len(facts.misconceptions))
    raw = facts.coverage * (weights.fact + weights.instruction * compliance)
    score = round(raw * (1.0 - penalty), 4)

    errors: list[ErrorType] = []
    if facts.misconceptions:
        errors.append(ErrorType.FACTUAL_ERROR)
    if facts.missing_facts:
        errors.append(ErrorType.MISSING_KEY_FACT)
    if facts.missing_reasoning:
        errors.append(ErrorType.REASONING_GAP)
    if violations:
        errors.append(ErrorType.INSTRUCTION_VIOLATION)

    return ScoreCard(
        question_id=question.id,
        score=score,
        coverage=round(facts.coverage, 4),
        instruction_ok=not violations,
        violations=violations,
        facts=facts,
        errors=tuple(errors),
    )
