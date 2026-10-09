"""Instruction-following checks: length, format, refusals."""

from __future__ import annotations

import re

from bioeval.schema import Constraints
from bioeval.textnorm import word_count

_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+\S")
_BULLET = re.compile(r"^\s*[-*•]\s+\S")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")
_REFUSAL = re.compile(
    r"\bi(?:\s+am|'m)\s+(?:not\s+able|unable)\s+to\s+(?:help|answer|assist|provide|respond)"
    r"|\bi\s+(?:can(?:no|')t|won't|will\s+not)\s+(?:help|answer|assist|provide|respond)"
    r"|\bas\s+an\s+ai\b",
    re.IGNORECASE,
)
REFUSAL_WINDOW_CHARS = 200
REFUSAL_MAX_WORDS = 50


def is_refusal(text: str) -> bool:
    """Empty, or a short answer that opens by declining.

    A long answer that merely contains a hedge ("I can't provide exact figures,
    but...") is graded on its content instead of being zeroed.
    """
    if not text.strip():
        return True
    head = text[:REFUSAL_WINDOW_CHARS].replace("’", "'")
    return bool(_REFUSAL.search(head)) and word_count(text) < REFUSAL_MAX_WORDS


def count_sentences(text: str) -> int:
    flat = " ".join(line.strip() for line in text.strip().splitlines() if line.strip())
    return len([s for s in _SENTENCE_SPLIT.split(flat) if s.strip()])


def _numbered_ok(lines: list[str]) -> bool:
    numbers = [int(m.group(1)) for m in (_NUMBERED.match(ln) for ln in lines) if m]
    return len(numbers) >= 2 and numbers == list(range(1, len(numbers) + 1))


def check_constraints(text: str, constraints: Constraints) -> list[str]:
    """Return human-readable violations; an empty list means the answer complies."""
    violations: list[str] = []
    words = word_count(text)
    if constraints.max_words and words > constraints.max_words:
        violations.append(f"too long: {words} words > limit {constraints.max_words}")
    if constraints.min_words and words < constraints.min_words:
        violations.append(f"too short: {words} words < minimum {constraints.min_words}")

    lines = [ln for ln in text.splitlines() if ln.strip()]
    fmt = constraints.format
    if fmt == "numbered_list" and not _numbered_ok(lines):
        violations.append("expected a numbered list (1., 2., ...) with at least two items")
    elif fmt == "bullets" and sum(bool(_BULLET.match(ln)) for ln in lines) < 2:
        violations.append("expected at least two bullet points")
    elif fmt == "single_sentence" and count_sentences(text) != 1:
        violations.append(f"expected one sentence, found {count_sentences(text)}")
    return violations
