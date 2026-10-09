from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

DEFAULT_SYSTEM_PROMPT = (
    "You are answering a biology exam question. Be accurate and concise, "
    "and follow any formatting or length instructions exactly."
)


class AdapterError(RuntimeError):
    """One model call failed after retries. The run records it and moves on."""


class AdapterConfigError(AdapterError):
    """The adapter is misconfigured (bad key, unknown model, malformed request).

    Retrying every remaining question would fail the same way, so the runner
    stops instead of recording hundreds of identical failures.
    """


@dataclass(frozen=True)
class GenerationRequest:
    question_id: str
    prompt: str
    system: str = DEFAULT_SYSTEM_PROMPT


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    latency_ms: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


@runtime_checkable
class Adapter(Protocol):
    name: str

    def generate(self, request: GenerationRequest) -> Completion: ...
