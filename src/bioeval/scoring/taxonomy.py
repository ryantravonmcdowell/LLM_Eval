"""Error taxonomy used to label every graded response.

The categories mirror how a human reviewer writes feedback: is the answer
wrong, incomplete, badly reasoned, non-compliant, or missing altogether?
"""

from __future__ import annotations

from enum import Enum


class ErrorType(str, Enum):
    FACTUAL_ERROR = "factual_error"  # states a known misconception
    MISSING_KEY_FACT = "missing_key_fact"  # omits a required final claim
    REASONING_GAP = "reasoning_gap"  # omits a required intermediate step
    INSTRUCTION_VIOLATION = "instruction_violation"  # breaks length/format rules
    REFUSAL_OR_EMPTY = "refusal_or_empty"  # no substantive answer
    ADAPTER_ERROR = "adapter_error"  # the model call itself failed

    @property
    def description(self) -> str:
        return _DESCRIPTIONS[self]


_DESCRIPTIONS = {
    ErrorType.FACTUAL_ERROR: "Asserts a known misconception or incorrect claim.",
    ErrorType.MISSING_KEY_FACT: "Leaves out a fact a correct answer must contain.",
    ErrorType.REASONING_GAP: "Skips an intermediate step needed to justify the answer.",
    ErrorType.INSTRUCTION_VIOLATION: "Ignores a length or format instruction in the prompt.",
    ErrorType.REFUSAL_OR_EMPTY: "Declines to answer or returns nothing.",
    ErrorType.ADAPTER_ERROR: "The model API call failed; no answer was graded.",
}
