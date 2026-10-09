"""Text normalisation shared by the dataset checks and the scorers.

Model output and hand-written reference answers spell the same thing many ways
(5′→3′, 5'-to-3', 5' to 3'). Normalising both sides before pattern matching keeps
the rubric patterns short and readable instead of enumerating every glyph.
"""

from __future__ import annotations

import re
import unicodedata

_REPLACEMENTS = (
    ("->", " to "),
    ("→", " to "),
    ("′", "'"),
    ("’", "'"),
    ("‘", "'"),
    ("“", '"'),
    ("”", '"'),
    ("−", "-"),
    ("–", "-"),
    ("—", "-"),
    ("×", "x"),
    ("²", "^2"),
    ("³", "^3"),
    ("β", "beta"),
    ("α", "alpha"),
    ("µ", "u"),
    ("μ", "u"),
)

_SPACES = re.compile(r"[ \t ]+")


def normalize(text: str) -> str:
    """Lower-case, unify glyph variants and collapse horizontal whitespace.

    Line breaks are kept so list detection still works on normalised text.
    Replacements run before NFKC because NFKC would turn ``²`` into a bare ``2``.
    """
    for old, new in _REPLACEMENTS:
        text = text.replace(old, new)
    text = unicodedata.normalize("NFKC", text).lower()
    return _SPACES.sub(" ", text).strip()


def compile_pattern(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE | re.MULTILINE)


def word_count(text: str) -> int:
    return len(re.findall(r"\w+(?:['-]\w+)*", text))


def shingles(text: str, size: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", normalize(text))
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)
