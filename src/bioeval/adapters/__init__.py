"""Model adapters: one interface, several backends.

``make_adapter("mock:weak")``, ``make_adapter("openai:<model>")`` and
``make_adapter("anthropic:<model>")`` all return an object with the same
``generate(request) -> Completion`` method, so the runner never needs to know
which provider it is talking to.
"""

from __future__ import annotations

from bioeval.adapters.base import (
    Adapter,
    AdapterConfigError,
    AdapterError,
    Completion,
    GenerationRequest,
)
from bioeval.adapters.mock import PROFILES, MockAdapter
from bioeval.adapters.openai_compat import OpenAICompatibleAdapter
from bioeval.schema import Question

__all__ = [
    "PROFILES",
    "Adapter",
    "AdapterConfigError",
    "AdapterError",
    "Completion",
    "GenerationRequest",
    "MockAdapter",
    "OpenAICompatibleAdapter",
    "make_adapter",
]


def make_adapter(spec: str, questions: list[Question] | None = None, **kwargs) -> Adapter:
    """Build an adapter from a ``kind:argument`` spec string.

    ``anthropic:<model>@<effort>`` sets the Claude effort level, e.g.
    ``anthropic:claude-opus-5-5@low``.
    """
    kind, _, arg = spec.partition(":")
    if kind == "mock":
        if questions is None:
            raise ValueError("the mock adapter needs the question set")
        return MockAdapter(questions, profile=arg or "strong", **kwargs)
    if kind == "openai":
        if not arg:
            raise ValueError("usage: openai:<model>")
        return OpenAICompatibleAdapter(model=arg, **kwargs)
    if kind == "anthropic":
        if not arg:
            raise ValueError("usage: anthropic:<model>[@effort]")
        from bioeval.adapters.claude import AnthropicAdapter

        model, _, effort = arg.partition("@")
        return AnthropicAdapter(model=model, effort=effort or None, **kwargs)
    raise ValueError(f"unknown adapter kind {kind!r}; expected mock, openai or anthropic")
