"""A deterministic offline model, for demos, tests and CI.

It answers from the reference answer and then degrades it the way real models
fail: it drops sentences (omissions), asserts a known misconception (factual
error), flattens list formatting (instruction violation) or refuses. Each
profile sets how often each failure happens, so the pipeline can be shown to
rank a "strong" model above a "weak" one without any API key.

Failures are seeded from ``(seed, profile, question_id)``, so a run is exactly
reproducible and adding a question does not reshuffle the others.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from bioeval.adapters.base import Completion, GenerationRequest
from bioeval.schema import Question


@dataclass(frozen=True)
class Profile:
    drop_sentence: float  # chance each sentence after the first is dropped
    misconception: float  # chance a known misconception is asserted
    break_format: float  # chance list formatting is flattened into prose
    refuse: float  # chance of refusing outright
    pad: float  # chance of adding filler that can breach word limits


PROFILES: dict[str, Profile] = {
    "strong": Profile(drop_sentence=0.05, misconception=0.0, break_format=0.0,
                      refuse=0.0, pad=0.05),
    "medium": Profile(drop_sentence=0.25, misconception=0.15, break_format=0.25,
                      refuse=0.02, pad=0.2),
    "weak": Profile(drop_sentence=0.5, misconception=0.4, break_format=0.6,
                    refuse=0.06, pad=0.4),
}

_FILLER = (
    " It is worth noting that biology is a complex and fascinating field, and many of these "
    "processes interact with one another in ways that scientists are still actively studying "
    "today, so the full picture is considerably more nuanced than any brief summary can capture."
)
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(])")
_LIST_MARK = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")


class MockAdapter:
    def __init__(self, questions: list[Question], profile: str = "strong", seed: int = 0):
        if profile not in PROFILES:
            raise ValueError(f"unknown mock profile {profile!r}; choose from {sorted(PROFILES)}")
        self.profile_name = profile
        self.profile = PROFILES[profile]
        self.seed = seed
        self.name = f"mock-{profile}"
        self._by_id = {q.id: q for q in questions}

    def _rng(self, question_id: str) -> random.Random:
        key = f"{self.seed}|{self.profile_name}|{question_id}".encode()
        return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))

    def generate(self, request: GenerationRequest) -> Completion:
        q = self._by_id.get(request.question_id)
        if q is None:
            return Completion(text="I don't know.", model=self.name)
        rng = self._rng(q.id)
        p = self.profile

        if rng.random() < p.refuse:
            return Completion(text="I'm unable to answer that question.", model=self.name)

        text = q.reference_answer
        is_list = q.constraints.format in ("numbered_list", "bullets")
        if is_list:
            items = [ln for ln in text.splitlines() if ln.strip()]
            kept = [items[0]] + [it for it in items[1:] if rng.random() >= p.drop_sentence]
            if rng.random() < p.break_format:
                text = " ".join(_LIST_MARK.sub("", it) for it in kept)
            else:
                text = "\n".join(kept)
        else:
            sentences = _SPLIT.split(text)
            kept = [sentences[0]] + [s for s in sentences[1:] if rng.random() >= p.drop_sentence]
            text = " ".join(kept)

        if q.misconceptions and rng.random() < p.misconception:
            wrong = rng.choice(q.misconceptions).example
            text = f"{text}\n- {wrong}" if is_list and "\n" in text else f"{text} {wrong}"
        if rng.random() < p.pad:
            text += _FILLER

        tokens = len(text.split())
        return Completion(text=text, model=self.name, latency_ms=0.0,
                          input_tokens=len(request.prompt.split()), output_tokens=tokens)
