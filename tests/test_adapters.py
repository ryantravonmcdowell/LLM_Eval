import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from bioeval.adapters import (
    AdapterConfigError,
    AdapterError,
    GenerationRequest,
    MockAdapter,
    OpenAICompatibleAdapter,
    make_adapter,
)
from bioeval.adapters.claude import AnthropicAdapter
from bioeval.scoring import score_response

REQ = GenerationRequest(question_id="q1", prompt="What is DNA?")


class TestMockAdapter:
    def test_is_deterministic(self, questions):
        a = MockAdapter(questions, "weak", seed=3)
        b = MockAdapter(questions, "weak", seed=3)
        for q in questions:
            r = GenerationRequest(question_id=q.id, prompt=q.prompt)
            assert a.generate(r).text == b.generate(r).text

    def test_profiles_are_ordered_by_quality(self, questions):
        def mean(profile):
            m = MockAdapter(questions, profile)
            return sum(score_response(q, m.generate(GenerationRequest(q.id, q.prompt)).text).score
                       for q in questions) / len(questions)

        assert mean("strong") > mean("medium") > mean("weak")

    def test_unknown_profile(self, questions):
        with pytest.raises(ValueError, match="unknown mock profile"):
            MockAdapter(questions, "genius")


class _Server:
    """A local HTTP server that replays a scripted list of (status, body) replies."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append({"path": self.path, "headers": dict(self.headers),
                                       "body": json.loads(self.rfile.read(length))})
                status, body = outer.replies.pop(0)
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_port}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


OK_BODY = {
    "model": "test-model-001",
    "choices": [{"message": {"content": "DNA is a nucleic acid."}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 6},
}


@pytest.fixture
def server():
    servers = []

    def start(replies):
        s = _Server(replies)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.close()


def _adapter(url, **kw):
    return OpenAICompatibleAdapter(model="test-model", base_url=url, api_key="sk-test",
                                   backoff=0, **kw)


class TestOpenAICompatibleAdapter:
    def test_success(self, server):
        s = server([(200, OK_BODY)])
        c = _adapter(s.url).generate(REQ)
        assert c.text == "DNA is a nucleic acid."
        assert (c.model, c.input_tokens, c.output_tokens, c.stop_reason) == (
            "test-model-001", 12, 6, "stop")
        sent = s.requests[0]
        assert sent["path"] == "/v1/chat/completions"
        assert sent["headers"]["Authorization"] == "Bearer sk-test"
        assert [m["role"] for m in sent["body"]["messages"]] == ["system", "user"]
        assert sent["body"]["temperature"] == 0.0

    def test_retries_server_errors_then_succeeds(self, server):
        s = server([(503, {"error": "busy"}), (429, {"error": "slow down"}), (200, OK_BODY)])
        assert _adapter(s.url, max_retries=3).generate(REQ).text.startswith("DNA")
        assert len(s.requests) == 3

    def test_gives_up_after_max_retries(self, server):
        s = server([(503, {})] * 3)
        with pytest.raises(AdapterError, match="gave up after 3 attempts"):
            _adapter(s.url, max_retries=2).generate(REQ)

    def test_auth_failure_is_fatal_and_not_retried(self, server):
        s = server([(401, {"error": "bad key"})])
        with pytest.raises(AdapterConfigError, match="401"):
            _adapter(s.url, max_retries=5).generate(REQ)
        assert len(s.requests) == 1

    def test_malformed_response(self, server):
        s = server([(200, {"unexpected": True})])
        with pytest.raises(AdapterError, match="unexpected response shape"):
            _adapter(s.url).generate(REQ)


class _FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _fake_client(stop_reason="end_turn", blocks=None):
    blocks = blocks if blocks is not None else [
        SimpleNamespace(type="thinking", thinking=""),
        SimpleNamespace(type="text", text="DNA stores "),
        SimpleNamespace(type="text", text="genetic information."),
    ]
    response = SimpleNamespace(
        content=blocks, stop_reason=stop_reason, model="claude-test",
        usage=SimpleNamespace(input_tokens=20, output_tokens=9),
    )
    return SimpleNamespace(messages=_FakeMessages(response))


class TestAnthropicAdapter:
    def test_joins_text_blocks_and_skips_thinking(self):
        client = _fake_client()
        c = AnthropicAdapter("claude-test", client=client).generate(REQ)
        assert c.text == "DNA stores genetic information."
        assert (c.input_tokens, c.output_tokens, c.stop_reason) == (20, 9, "end_turn")

    def test_request_shape(self):
        client = _fake_client()
        AnthropicAdapter("claude-test", effort="low", client=client).generate(REQ)
        sent = client.messages.calls[0]
        assert sent["model"] == "claude-test"
        assert sent["system"] == REQ.system
        assert sent["messages"] == [{"role": "user", "content": REQ.prompt}]
        assert sent["output_config"] == {"effort": "low"}
        # current Claude models reject sampling parameters, and an eval must not
        # silently hand a declined question to a different model
        assert "temperature" not in sent and "fallbacks" not in sent

    def test_refusal_is_graded_as_no_answer(self):
        client = _fake_client(stop_reason="refusal",
                              blocks=[SimpleNamespace(type="text", text="partial")])
        c = AnthropicAdapter("claude-test", client=client).generate(REQ)
        assert c.text == "" and c.stop_reason == "refusal"

    def test_make_adapter_parses_effort(self):
        a = make_adapter("anthropic:claude-test@high", client=_fake_client())
        assert (a.model, a.effort, a.name) == ("claude-test", "high", "anthropic:claude-test@high")


@pytest.mark.parametrize("spec", ["nope:x", "openai:", "anthropic:"])
def test_make_adapter_rejects_bad_specs(spec):
    with pytest.raises(ValueError):
        make_adapter(spec)
