"""Provider selection and the mutation agent's fallbacks, against a fake OpenAI client (no network)."""
from types import SimpleNamespace

import openai
import pytest

try:
    import httpx2 as httpx  # openai 3.x builds its errors on httpx2
except ImportError:
    import httpx

import run_evolution as rev

CRUSOE = rev.PROVIDERS["crusoe"]


def completion(content, finish_reason="stop"):
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=content))])


def bad_request():
    response = httpx.Response(400, request=httpx.Request("POST", rev.CRUSOE_BASE_URL))
    return openai.BadRequestError("response_format is not supported", response=response, body=None)


class FakeCompletions:
    """Replays canned replies (or raises canned errors) and records every request."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


@pytest.fixture(autouse=True)
def clean_state(monkeypatch):
    for env in ("LLM_PROVIDER", "CRUSOE_MODEL", "LLM_MAX_TOKENS", "LIQUID_MAX_TOKENS"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("CRUSOE_API_KEY", "test-key")
    rev._FIRST_ATTEMPT.clear()


@pytest.fixture
def fake(monkeypatch):
    def install(*replies):
        completions = FakeCompletions(*replies)
        monkeypatch.setattr(openai, "OpenAI", lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=completions)))
        return completions
    return install


def test_crusoe_wins_auto_detection(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    spec, reason = rev.resolve_provider("")
    assert spec.id == "crusoe" and reason == ""


def test_crusoe_without_key_names_the_variable(monkeypatch):
    monkeypatch.delenv("CRUSOE_API_KEY")
    spec, reason = rev.resolve_provider("crusoe")
    assert spec is None and "CRUSOE_API_KEY" in reason


def test_crusoe_model_default_and_override(monkeypatch):
    # IDs as Crusoe's /v1/models lists them, which is not always how its docs spell them
    assert rev.provider_model(CRUSOE) == "deepseek-ai/Deepseek-V4-Flash"
    monkeypatch.setenv("CRUSOE_MODEL", "nvidia/Nemotron-3.5-Lightning-30B-A3B")
    assert rev.provider_model(CRUSOE) == "nvidia/Nemotron-3.5-Lightning-30B-A3B"


def test_request_shape(fake):
    calls = fake(completion('{"code": "x"}'))
    assert rev.call_mutation_agent("prompt", "deepseek-ai/Deepseek-V4-Flash", CRUSOE) == {"code": "x"}
    (request,) = calls.requests
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["max_tokens"] == 32768
    assert request["response_format"] == {"type": "json_object"}
    assert "extra_body" not in request and "extra_headers" not in request


def test_empty_json_reply_falls_back_once_per_run(fake):
    calls = fake(completion(None), completion('{"code": "a"}'), completion('{"code": "b"}'))
    assert rev.call_mutation_agent("p", "m", CRUSOE) == {"code": "a"}
    assert ["response_format" in r for r in calls.requests] == [True, False]
    # The next generation goes straight to the variant that worked
    assert rev.call_mutation_agent("p", "m", CRUSOE) == {"code": "b"}
    assert ["response_format" in r for r in calls.requests] == [True, False, False]


def test_rejected_json_mode_is_not_sent_again(fake):
    calls = fake(bad_request(), completion('{"code": "a"}'), completion('{"code": "b"}'))
    rev.call_mutation_agent("p", "m", CRUSOE)
    rev.call_mutation_agent("p", "m", CRUSOE)
    assert ["response_format" in r for r in calls.requests] == [True, False, False]


def test_truncated_reply_is_not_retried(fake):
    calls = fake(completion("", finish_reason="length"))
    with pytest.raises(ValueError, match="truncated"):
        rev.call_mutation_agent("p", "m", CRUSOE)
    assert len(calls.requests) == 1


def test_empty_reply_on_last_variant_raises(fake):
    fake(completion(None), completion("  "))
    with pytest.raises(ValueError, match="empty completion"):
        rev.call_mutation_agent("p", "m", CRUSOE)


def test_double_escaped_code_is_unescaped(fake, monkeypatch):
    code = 'import torch\\nx = "μs"\\nprint(x)'
    monkeypatch.setattr(rev, "call_mutation_agent", lambda prompt, model, spec: {"code": code})
    seed = rev.KernelVariant("root", "v0", 0, None, "baseline", "", 1.0, 1.0, "success", {}, "pass", 0.0)
    variant = rev.mutate_kernel_with_agent(rev.PROBLEMS["vectorsum_v2"], rev.HARDWARE_CATALOG["b200"], seed, 1,
                                           provider=CRUSOE, gates=rev.Gates(sanitize="off", cpu_check=False))
    assert variant.code == 'import torch\nx = "μs"\nprint(x)'
