"""Regression tests for OpenRouter HTTP-200 error envelopes."""

from types import SimpleNamespace

import pytest

from providers.openai_compatible import OpenAICompatibleProvider
from providers.openrouter import OpenRouterProvider
from providers.shared import ModelCapabilities, ProviderType

CHAT_MODEL = "test/chat-model"


def _chat_success(content: str = "ok") -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=3, completion_tokens=2, total_tokens=5)
    return SimpleNamespace(
        choices=[choice],
        model=CHAT_MODEL,
        id="chatcmpl-ok",
        created=123,
        usage=usage,
    )


def _chat_error_envelope(code: int, message: str, error_type: str) -> SimpleNamespace:
    return SimpleNamespace(
        id="gen-error",
        created=123,
        model=CHAT_MODEL,
        choices=None,
        usage=None,
        error={
            "message": message,
            "code": code,
            "metadata": {"error_type": error_type},
        },
    )


def _not_called(**_kwargs):
    raise AssertionError("unexpected client method call")


def _counting_create(response):
    calls = {"count": 0}

    def create(**_kwargs):
        calls["count"] += 1
        return response

    return create, calls


def _client_with_chat_create(create):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        responses=SimpleNamespace(create=_not_called, retrieve=_not_called),
    )


def test_openrouter_429_error_envelope_surfaces_message_and_retries(monkeypatch):
    """Dropping envelope detection should lose the upstream message and skip 429 retries."""

    monkeypatch.setattr("providers.base.time.sleep", lambda _: None)
    upstream_message = "openai/gpt-6-astra is temporarily rate-limited upstream. Please retry shortly."
    create, calls = _counting_create(_chat_error_envelope(429, upstream_message, "rate_limit_exceeded"))
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = _client_with_chat_create(create)

    with pytest.raises(RuntimeError) as excinfo:
        provider.generate_content("hello", model_name=CHAT_MODEL)

    error_text = str(excinfo.value)
    assert upstream_message in error_text
    assert "Error code: 429" in error_text
    assert "rate_limit_exceeded" in error_text
    assert "error envelope (HTTP 200)" in error_text
    assert calls["count"] > 1


def test_openrouter_502_error_envelope_surfaces_message(monkeypatch):
    """Dropping envelope detection should replace a provider 502 with a generic choices failure."""

    monkeypatch.setattr("providers.base.time.sleep", lambda _: None)
    upstream_message = "You have no credits remaining. Add credits to continue using the API."
    create, _calls = _counting_create(_chat_error_envelope(502, upstream_message, "provider_unavailable"))
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = _client_with_chat_create(create)

    with pytest.raises(RuntimeError) as excinfo:
        provider.generate_content("hello", model_name=CHAT_MODEL)

    error_text = str(excinfo.value)
    assert upstream_message in error_text
    assert "Error code: 502" in error_text
    assert "provider_unavailable" in error_text


def test_openrouter_normal_chat_response_still_works():
    """Treating all OpenRouter responses as envelopes would break normal choices extraction."""

    create, calls = _counting_create(_chat_success("normal response"))
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = _client_with_chat_create(create)

    result = provider.generate_content("hello", model_name=CHAT_MODEL)

    assert result.content == "normal response"
    assert result.provider == ProviderType.OPENROUTER
    assert calls["count"] == 1


def test_non_openrouter_provider_error_envelope_helper_is_noop():
    """Removing the provider guard would change native OpenAI-compatible behavior."""

    class NativeProvider(OpenAICompatibleProvider):
        FRIENDLY_NAME = "Native"

        def get_provider_type(self):
            return ProviderType.OPENAI

        def get_capabilities(self, model_name):
            return ModelCapabilities(
                provider=ProviderType.OPENAI,
                model_name=model_name,
                friendly_name="Native model",
            )

        def validate_model_name(self, model_name):
            return True

        def list_models(self, **kwargs):
            return ["gpt-test"]

    provider = NativeProvider(api_key="test-key")
    response = SimpleNamespace(
        choices=None,
        error={"message": "native provider extra error field", "code": 429},
    )

    provider._raise_for_error_envelope(response, endpoint="chat/completions")


def test_openrouter_responses_error_envelope_surfaces_message(monkeypatch):
    """Responses endpoint envelopes should be detected before output_text extraction."""

    monkeypatch.setattr("providers.base.time.sleep", lambda _: None)
    upstream_message = "openai/gpt-6-astra is temporarily rate-limited upstream."
    response = SimpleNamespace(
        id="resp-error",
        model="openai/gpt-6-astra",
        usage=None,
        error={
            "message": upstream_message,
            "code": 429,
            "metadata": {"error_type": "rate_limit_exceeded"},
        },
    )
    create, calls = _counting_create(response)
    client = SimpleNamespace(
        responses=SimpleNamespace(create=create, retrieve=_not_called),
        chat=SimpleNamespace(completions=SimpleNamespace(create=_not_called)),
    )
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = client

    with pytest.raises(RuntimeError) as excinfo:
        provider._generate_with_responses_endpoint(
            model_name="openai/gpt-6-astra",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

    error_text = str(excinfo.value)
    assert upstream_message in error_text
    assert "Error code: 429" in error_text
    assert "responses error envelope (HTTP 200)" in error_text
    assert calls["count"] > 1
