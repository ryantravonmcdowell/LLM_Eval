"""Adapter for Claude models through the official ``anthropic`` Python SDK.

Install with ``pip install "bio-llm-eval[anthropic]"``. The SDK reads
credentials from the environment (``ANTHROPIC_API_KEY`` or an ``ant auth login``
profile) and already retries rate limits and server errors with backoff.

Two choices here are deliberate for an *evaluation* harness:

* No server-side model fallback. A fallback would let a different model answer
  when the requested one declines, and the result would be graded under the
  wrong name. A refusal is recorded as a refusal, which the rubric scores.
* No sampling parameters. Current Claude models reject ``temperature``; output
  depth is set with ``effort`` instead, and recorded with the run.
"""

from __future__ import annotations

import time
from typing import Any

from bioeval.adapters.base import AdapterConfigError, AdapterError, Completion, GenerationRequest


class AnthropicAdapter:
    def __init__(
        self,
        model: str,
        effort: str | None = None,
        max_tokens: int = 16000,
        max_retries: int = 3,
        client: Any = None,
    ):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}" + (f"@{effort}" if effort else "")
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise AdapterConfigError(
                    'the Claude adapter needs the SDK: pip install "bio-llm-eval[anthropic]"'
                ) from exc
            client = anthropic.Anthropic(max_retries=max_retries)
        self._client = client

    @staticmethod
    def _translate(exc: Exception) -> AdapterError:
        import anthropic

        fatal = (
            anthropic.AuthenticationError,
            anthropic.PermissionDeniedError,
            anthropic.NotFoundError,
            anthropic.BadRequestError,
        )
        if isinstance(exc, fatal):
            return AdapterConfigError(f"{type(exc).__name__}: {exc}")
        return AdapterError(f"{type(exc).__name__}: {exc}")

    def generate(self, request: GenerationRequest) -> Completion:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": request.system,
            "messages": [{"role": "user", "content": request.prompt}],
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}

        start = time.perf_counter()
        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            if not _is_api_error(exc):
                raise  # a bug in this code, not a failed API call
            raise self._translate(exc) from exc
        latency = (time.perf_counter() - start) * 1000

        # Check why generation stopped before reading content: a refusal comes
        # back as HTTP 200 and must be graded as no answer, not as an empty one.
        if response.stop_reason == "refusal":
            text = ""
        else:
            text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=getattr(response, "model", self.model),
            latency_ms=round(latency, 1),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            stop_reason=response.stop_reason,
        )


def _is_api_error(exc: Exception) -> bool:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return False
    return isinstance(exc, anthropic.APIError)
