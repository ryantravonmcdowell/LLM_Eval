"""Typed records for one benchmark question and its rubric.

A question carries everything needed to grade an answer without a human in the
loop: the facts a correct answer must contain, the misconceptions it must not
contain, and the formatting constraints the prompt imposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUBJECTS = frozenset(
    {
        "molecular_biology",
        "genetics",
        "cell_biology",
        "physiology",
        "immunology",
        "microbiology",
        "biochemistry",
        "ecology",
        "evolution",
    }
)
DIFFICULTIES = frozenset({"easy", "medium", "hard"})
FACT_KINDS = frozenset({"fact", "reasoning"})
FORMATS = frozenset({"numbered_list", "bullets", "single_sentence"})


class SchemaError(ValueError):
    """Raised when a record cannot be turned into a :class:`Question`."""

    def __init__(self, field_path: str, message: str):
        super().__init__(f"{field_path}: {message}")
        self.field_path = field_path
        self.message = message


@dataclass(frozen=True)
class KeyFact:
    """Something a correct answer must state.

    ``kind="reasoning"`` marks an intermediate step (e.g. "q^2 = 0.16") rather
    than a final claim, so a miss is reported as a reasoning gap, not an omission.
    """

    id: str
    description: str
    patterns: tuple[str, ...]
    kind: str = "fact"
    weight: float = 1.0


@dataclass(frozen=True)
class Misconception:
    """A known wrong claim. ``example`` must itself match ``patterns``."""

    id: str
    description: str
    patterns: tuple[str, ...]
    example: str


@dataclass(frozen=True)
class Constraints:
    max_words: int | None = None
    min_words: int | None = None
    format: str | None = None

    def instruction_text(self) -> str:
        parts = []
        if self.format == "numbered_list":
            parts.append("Format your answer as a numbered list.")
        elif self.format == "bullets":
            parts.append("Format your answer as bullet points.")
        elif self.format == "single_sentence":
            parts.append("Answer in a single sentence.")
        if self.max_words:
            parts.append(f"Use at most {self.max_words} words.")
        if self.min_words:
            parts.append(f"Use at least {self.min_words} words.")
        return " ".join(parts)


@dataclass(frozen=True)
class Question:
    id: str
    subject: str
    difficulty: str
    prompt: str
    reference_answer: str
    key_facts: tuple[KeyFact, ...]
    misconceptions: tuple[Misconception, ...] = ()
    constraints: Constraints = field(default_factory=Constraints)

    def full_prompt(self) -> str:
        extra = self.constraints.instruction_text()
        return f"{self.prompt}\n\n{extra}" if extra else self.prompt


def _require(record: dict, key: str, kind: type | tuple[type, ...], path: str) -> Any:
    if key not in record or record[key] is None:
        raise SchemaError(f"{path}{key}", "missing required field")
    value = record[key]
    if not isinstance(value, kind):
        expected = kind.__name__ if isinstance(kind, type) else "/".join(k.__name__ for k in kind)
        raise SchemaError(f"{path}{key}", f"expected {expected}, got {type(value).__name__}")
    if isinstance(value, str) and not value.strip():
        raise SchemaError(f"{path}{key}", "must not be empty")
    return value


def _patterns(record: dict, path: str) -> tuple[str, ...]:
    pats = _require(record, "patterns", list, path)
    if not pats:
        raise SchemaError(f"{path}patterns", "must contain at least one pattern")
    for i, p in enumerate(pats):
        if not isinstance(p, str) or not p:
            raise SchemaError(f"{path}patterns[{i}]", "must be a non-empty string")
    return tuple(pats)


def question_from_dict(record: dict) -> Question:
    """Parse one JSON object into a :class:`Question`, raising :class:`SchemaError`."""
    if not isinstance(record, dict):
        raise SchemaError("<record>", "expected a JSON object")

    qid = _require(record, "id", str, "")
    subject = _require(record, "subject", str, "")
    if subject not in SUBJECTS:
        raise SchemaError("subject", f"unknown subject {subject!r}")
    difficulty = _require(record, "difficulty", str, "")
    if difficulty not in DIFFICULTIES:
        raise SchemaError("difficulty", f"unknown difficulty {difficulty!r}")

    facts_raw = _require(record, "key_facts", list, "")
    if not facts_raw:
        raise SchemaError("key_facts", "a question needs at least one key fact")
    facts = []
    for i, f in enumerate(facts_raw):
        path = f"key_facts[{i}]."
        if not isinstance(f, dict):
            raise SchemaError(path[:-1], "expected an object")
        kind = f.get("kind", "fact")
        if kind not in FACT_KINDS:
            raise SchemaError(f"{path}kind", f"unknown kind {kind!r}")
        weight = f.get("weight", 1.0)
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise SchemaError(f"{path}weight", "must be a positive number")
        facts.append(
            KeyFact(
                id=_require(f, "id", str, path),
                description=_require(f, "description", str, path),
                patterns=_patterns(f, path),
                kind=kind,
                weight=float(weight),
            )
        )

    miscs = []
    for i, m in enumerate(record.get("misconceptions") or []):
        path = f"misconceptions[{i}]."
        if not isinstance(m, dict):
            raise SchemaError(path[:-1], "expected an object")
        miscs.append(
            Misconception(
                id=_require(m, "id", str, path),
                description=_require(m, "description", str, path),
                patterns=_patterns(m, path),
                example=_require(m, "example", str, path),
            )
        )

    c = record.get("constraints") or {}
    if not isinstance(c, dict):
        raise SchemaError("constraints", "expected an object")
    fmt = c.get("format")
    if fmt is not None and fmt not in FORMATS:
        raise SchemaError("constraints.format", f"unknown format {fmt!r}")
    for key in ("max_words", "min_words"):
        val = c.get(key)
        if val is not None and (not isinstance(val, int) or isinstance(val, bool) or val <= 0):
            raise SchemaError(f"constraints.{key}", "must be a positive integer")
    constraints = Constraints(
        max_words=c.get("max_words"), min_words=c.get("min_words"), format=fmt
    )

    return Question(
        id=qid,
        subject=subject,
        difficulty=difficulty,
        prompt=_require(record, "prompt", str, ""),
        reference_answer=_require(record, "reference_answer", str, ""),
        key_facts=tuple(facts),
        misconceptions=tuple(miscs),
        constraints=constraints,
    )
