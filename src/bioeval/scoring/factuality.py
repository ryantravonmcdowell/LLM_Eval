"""Key-fact coverage and misconception detection."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from bioeval.schema import KeyFact, Misconception, Question
from bioeval.textnorm import compile_pattern, normalize


@lru_cache(maxsize=4096)
def _compiled(pattern: str):
    return compile_pattern(pattern)


def matches(patterns: tuple[str, ...], normalized_text: str) -> bool:
    return any(_compiled(p).search(normalized_text) for p in patterns)


def fact_present(fact: KeyFact, normalized_text: str) -> bool:
    return matches(fact.patterns, normalized_text)


def misconception_present(m: Misconception, normalized_text: str) -> bool:
    return matches(m.patterns, normalized_text)


@dataclass(frozen=True)
class FactResult:
    covered: tuple[str, ...]
    missing_facts: tuple[str, ...]
    missing_reasoning: tuple[str, ...]
    misconceptions: tuple[str, ...]
    coverage: float  # weight-adjusted share of key facts present, 0..1


def score_facts(question: Question, text: str) -> FactResult:
    norm = normalize(text)
    covered, missing_f, missing_r = [], [], []
    got = total = 0.0
    for fact in question.key_facts:
        total += fact.weight
        if fact_present(fact, norm):
            covered.append(fact.id)
            got += fact.weight
        elif fact.kind == "reasoning":
            missing_r.append(fact.id)
        else:
            missing_f.append(fact.id)
    hits = tuple(m.id for m in question.misconceptions if misconception_present(m, norm))
    return FactResult(
        covered=tuple(covered),
        missing_facts=tuple(missing_f),
        missing_reasoning=tuple(missing_r),
        misconceptions=hits,
        coverage=got / total if total else 0.0,
    )
