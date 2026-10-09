"""Adapter for any server that speaks the OpenAI chat-completions protocol.

That covers OpenAI itself and most self-hosted inference servers (vLLM,
llama.cpp's server, Ollama's compatible endpoint), so one small adapter reaches
many models. It uses only the standard library: one POST per question, with
exponential backoff on rate limits and server errors.
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request

from bioeval.adapters.base import AdapterConfigError, AdapterError, Completion, GenerationRequest

RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})
FATAL_STATUS = frozenset({400, 401, 403, 404, 422})


class OpenAICompatibleAdapter:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff: float = 1.0,
    ):
        self.model = model
        self.name = f"openai:{model}"
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get(api_key_env)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff

    def _payload(self, request: GenerationRequest) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        return payload

    def _post(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        last: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:300]
                if exc.code in FATAL_STATUS:
                    msg = f"HTTP {exc.code} from {self.base_url}: {body}"
                    raise AdapterConfigError(msg) from exc
                if exc.code not in RETRYABLE_STATUS:
                    raise AdapterError(f"HTTP {exc.code}: {body}") from exc
                last = exc
                retry_after = exc.headers.get("retry-after") if exc.headers else None
                delay = float(retry_after) if retry_after and retry_after.isdigit() else None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last, delay = exc, None
            if attempt < self.max_retries:
                time.sleep(delay if delay is not None
                           else self.backoff * 2**attempt + random.uniform(0, self.backoff))
        raise AdapterError(f"gave up after {self.max_retries + 1} attempts: {last}")

    def generate(self, request: GenerationRequest) -> Completion:
        start = time.perf_counter()
        data = self._post(self._payload(request))
        latency = (time.perf_counter() - start) * 1000
        try:
            choice = data["choices"][0]
            text = choice["message"].get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError(f"unexpected response shape: {str(data)[:200]}") from exc
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", self.model),
            latency_ms=round(latency, 1),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            stop_reason=choice.get("finish_reason"),
        )
