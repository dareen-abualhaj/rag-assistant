from types import SimpleNamespace
from unittest.mock import MagicMock
from src.llm import LLMUnavailableError
import httpx
import pytest
from openai import (
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from src.llm import (
    LLMConfigError,
    LLMContextTooLongError,
    LLMError,
    LLMUnavailableError,
    generate,
)

MSGS = [{"role": "user", "content": "hi"}]


def _req():
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _resp(status, headers=None):
    return httpx.Response(status, request=_req(), headers=headers or {})


def _ok(text="hello"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _client(*effects):
    c = MagicMock()
    c.chat.completions.create.side_effect = list(effects)
    return c


def _run(client, sleeps=None):
    sleeps = sleeps if sleeps is not None else []
    return generate(MSGS, model="m", client=client, sleep=sleeps.append)


def test_success_first_try():
    c = _client(_ok("  answer  "))
    assert _run(c) == "answer"
    assert c.chat.completions.create.call_count == 1


def test_retries_on_429_then_succeeds_with_backoff():
    err = RateLimitError("rate limited", response=_resp(429), body=None)
    c = _client(err, err, _ok())
    sleeps = []
    assert _run(c, sleeps) == "hello"
    assert c.chat.completions.create.call_count == 3
    assert sleeps == [2.0, 4.0]


def test_honours_retry_after_header():
    err = RateLimitError("rate limited", response=_resp(429, {"retry-after": "5"}), body=None)
    c = _client(err, _ok())
    sleeps = []
    _run(c, sleeps)
    assert sleeps == [5.0]


def test_timeout_every_time_raises_unavailable():
    c = _client(*[APITimeoutError(request=_req()) for _ in range(4)])
    with pytest.raises(LLMUnavailableError):
        _run(c)
    assert c.chat.completions.create.call_count == 4


def test_5xx_is_retried():
    err = InternalServerError("boom", response=_resp(500), body=None)
    c = _client(err, _ok())
    assert _run(c) == "hello"


def test_401_raises_config_error_without_retry():
    c = _client(AuthenticationError("bad key", response=_resp(401), body=None))
    with pytest.raises(LLMConfigError, match="Invalid API key"):
        _run(c)
    assert c.chat.completions.create.call_count == 1


def test_404_mentions_model_id():
    c = _client(NotFoundError("nope", response=_resp(404), body=None))
    with pytest.raises(LLMConfigError, match="not found"):
        _run(c)


def test_token_limit_raises_context_too_long():
    err = BadRequestError("maximum context length exceeded: too many tokens",
                          response=_resp(400), body=None)
    with pytest.raises(LLMContextTooLongError):
        _run(_client(err))


def test_other_400_raises_generic_error():
    err = BadRequestError("invalid parameter foo", response=_resp(400), body=None)
    with pytest.raises(LLMError) as exc:
        _run(_client(err))
    assert not isinstance(exc.value, LLMContextTooLongError)



def test_missing_model_raises(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    with pytest.raises(LLMConfigError, match="LLM_MODEL"):
        generate(MSGS, client=MagicMock())


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(LLMConfigError, match="GROQ_API_KEY"):
        generate(MSGS, model="m")


def test_empty_response_is_retried_then_raises():
    c = _client(*[_ok("   ") for _ in range(4)])
    with pytest.raises(LLMUnavailableError):
        _run(c)
    assert c.chat.completions.create.call_count == 4


def test_empty_response_then_success():
    c = _client(_ok(""), _ok("answer"))
    assert _run(c) == "answer"        