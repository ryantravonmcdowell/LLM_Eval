"""Optional LLM-as-judge grading, used as a second opinion next to the rubric.

The pattern rubric is transparent and free but literal: it can miss a correct
answer phrased in an unexpected way. A judge model reads the answer the way a
human reviewer would, given the same reference answer and rubric. The report
shows how far the two agree, so neither is taken on trust.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from bioeval.adapters.base import Adapter, GenerationRequest
from bioeval.schema import Question
from bioeval.scoring.taxonomy import ErrorType

JUDGE_SYSTEM = (
    "You are a meticulous biology examiner grading another model's answer. "
    "Grade only against the reference answer and rubric provided. "
    "Reply with a single JSON object and nothing else."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_VALID_ERRORS = {e.value for e in ErrorType} - {ErrorType.ADAPTER_ERROR.value}


@dataclass(frozen=True)
class JudgeVerdict:
    score: float  # normalised to 0..1 from the 1-5 scale
    raw_score: int
    errors: tuple[str, ...]
    rationale: str


class JudgeParseError(ValueError):
    pass


def build_judge_prompt(question: Question, answer: str) -> str:
    facts = "\n".join(f"- [{f.kind}] {f.description}" for f in question.key_facts)
    miscs = "\n".join(f"- {m.description}" for m in question.misconceptions) or "- (none listed)"
    instructions = question.constraints.instruction_text() or "(none)"
    labels = ", ".join(sorted(_VALID_ERRORS))
    return f"""Question:
{question.prompt}

Instructions given to the model: {instructions}

Reference answer:
{question.reference_answer}

Key points a complete answer contains:
{facts}

Known misconceptions (any of these is a factual error):
{miscs}

Answer to grade:
<answer>
{answer}
</answer>

Score the answer from 1 to 5:
5 = correct, complete and follows the instructions
4 = correct with a minor omission or minor instruction slip
3 = partially correct; important points missing
2 = mostly incorrect or contains a misconception
1 = wrong, empty or a refusal

Return JSON with exactly these keys:
{{"score": <integer 1-5>,
  "errors": [<zero or more of: {labels}>],
  "rationale": "<one or two sentences>"}}"""


def parse_verdict(text: str) -> JudgeVerdict:
    match = _JSON_OBJECT.search(text)
    if not match:
        raise JudgeParseError(f"no JSON object in judge output: {text[:120]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"invalid JSON from judge: {exc}") from exc
    raw = data.get("score")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not 1 <= raw <= 5:
        raise JudgeParseError(f"score must be a number from 1 to 5, got {raw!r}")
    raw = int(round(raw))
    errors = tuple(e for e in data.get("errors") or [] if e in _VALID_ERRORS)
    rationale = str(data.get("rationale", "")).strip()
    return JudgeVerdict(score=(raw - 1) / 4, raw_score=raw, errors=errors, rationale=rationale)


class LLMJudge:
    def __init__(self, adapter: Adapter, retries: int = 1):
        self.adapter = adapter
        self.retries = retries
        self.name = f"judge:{adapter.name}"

    def grade(self, question: Question, answer: str) -> JudgeVerdict:
        request = GenerationRequest(
            question_id=question.id, prompt=build_judge_prompt(question, answer),
            system=JUDGE_SYSTEM,
        )
        last: Exception | None = None
        for _ in range(self.retries + 1):
            try:
                return parse_verdict(self.adapter.generate(request).text)
            except JudgeParseError as exc:
                last = exc
        raise JudgeParseError(f"judge output unparseable after {self.retries + 1} tries: {last}")
