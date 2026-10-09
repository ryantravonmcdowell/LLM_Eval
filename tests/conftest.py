from __future__ import annotations

import json

import pytest

from bioeval.cli import default_dataset
from bioeval.dataset import load_dataset
from bioeval.schema import question_from_dict


@pytest.fixture(scope="session")
def dataset():
    return load_dataset(default_dataset())


@pytest.fixture(scope="session")
def questions(dataset):
    return dataset.questions


def record(**overrides) -> dict:
    """A small valid question record; override any field to break it."""
    base = {
        "id": "t-001",
        "subject": "genetics",
        "difficulty": "easy",
        "prompt": "What is the ratio from a monohybrid cross of two heterozygotes?",
        "reference_answer": (
            "The phenotypic ratio is 3:1 because the dominant allele masks the recessive one."
        ),
        "key_facts": [
            {"id": "ratio", "description": "3:1", "patterns": [r"3 ?: ?1"]},
            {"id": "dominance", "kind": "reasoning", "description": "dominance masks",
             "patterns": [r"mask", r"dominan"]},
        ],
        "misconceptions": [
            {"id": "one_two_one", "description": "gives 1:2:1",
             "patterns": [r"\b1 ?: ?2 ?: ?1\b"], "example": "The ratio is 1:2:1."}
        ],
        "constraints": {"max_words": 40},
    }
    base.update(overrides)
    return base


@pytest.fixture
def simple_question():
    return question_from_dict(record())


def jsonl(*records) -> list[tuple[int, str]]:
    return [(i, r if isinstance(r, str) else json.dumps(r)) for i, r in enumerate(records, 1)]
