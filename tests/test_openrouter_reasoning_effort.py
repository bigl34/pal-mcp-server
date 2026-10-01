import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

import utils.model_restrictions
from providers.openai_compatible import OpenAICompatibleProvider
from providers.openrouter import OpenRouterProvider
from providers.registries.base import CapabilityModelRegistry
from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import ModelCapabilities, ModelResponse, ProviderType


class MockOpenRouterProvider(OpenAICompatibleProvider):
    FRIENDLY_NAME = "OpenRouter Test"

    def __init__(self, api_key):
        super().__init__(api_key)
        self.capabilities = ModelCapabilities(
            provider=ProviderType.OPENROUTER,
            model_name="test/model",
            friendly_name=self.FRIENDLY_NAME,
            supports_extended_thinking=True,
        )
        self.ignored_providers = ["novita"]
        self.provider_only = None

    def get_provider_type(self):
        return ProviderType.OPENROUTER

    def get_capabilities(self, model_name):
        return self.capabilities

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["test/model"]

    def _openrouter_ignored_providers(self):
        return self.ignored_providers

    def _openrouter_provider_only(self, model_name):
        return self.provider_only


class MockDirectOpenAIProvider(MockOpenRouterProvider):
    FRIENDLY_NAME = "OpenAI Test"

    def __init__(self, api_key):
        super().__init__(api_key)
        self.capabilities.provider = ProviderType.OPENAI

    def get_provider_type(self):
        return ProviderType.OPENAI


def _chat_response(model="test/model"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="OK"), finish_reason="stop")],
        usage=None,
        model=model,
        id="chatcmpl-test",
        created=123,
    )


def _provider_with_client(provider_class=MockOpenRouterProvider):
    provider = provider_class("test-key")
    provider._client = Mock()
    provider._client.chat.completions.create.return_value = _chat_response()
    return provider


@pytest.mark.parametrize(
    ("thinking_mode", "effort"),
    [
        ("minimal", "minimal"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("max", "xhigh"),
    ],
)
def test_openrouter_chat_maps_thinking_mode_to_reasoning_effort(thinking_mode, effort):
    provider = _provider_with_client()

    provider.generate_content("test", model_name="test/model", thinking_mode=thinking_mode)

    assert provider._client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "provider": {"data_collection": "deny", "zdr": True, "ignore": ["novita"]},
        "reasoning": {"effort": effort},
    }


def test_openrouter_chat_omits_reasoning_when_thinking_mode_is_none():
    provider = _provider_with_client()

    provider.generate_content("test", model_name="test/model", thinking_mode=None)

    assert provider._client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "provider": {"data_collection": "deny", "zdr": True, "ignore": ["novita"]}
    }


def test_openrouter_chat_omits_reasoning_without_extended_thinking():
    provider = _provider_with_client()
    provider.capabilities.supports_extended_thinking = False

    provider.generate_content("test", model_name="test/model", thinking_mode="high")

    assert provider._client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "provider": {"data_collection": "deny", "zdr": True, "ignore": ["novita"]}
    }


def test_openrouter_chat_omits_reasoning_without_routing_capabilities():
    provider = _provider_with_client()
    provider.capabilities = None

    provider.generate_content("test", model_name="test/model", thinking_mode="high")

    assert provider._client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "provider": {"data_collection": "deny", "zdr": True, "ignore": ["novita"]}
    }


def test_openrouter_chat_rejects_invalid_thinking_mode_before_http():
    provider = _provider_with_client()

    with pytest.raises(ValueError, match="Unsupported OpenRouter thinking mode"):
        provider.generate_content("test", model_name="test/model", thinking_mode="bogus")

    provider._client.chat.completions.create.assert_not_called()


def test_direct_openai_chat_does_not_add_openrouter_reasoning():
    provider = _provider_with_client(MockDirectOpenAIProvider)

    provider.generate_content("test", model_name="test/model", thinking_mode="high")

    assert "extra_body" not in provider._client.chat.completions.create.call_args.kwargs


def test_openrouter_responses_path_does_not_add_chat_reasoning():
    provider = _provider_with_client()
    provider.capabilities.use_openai_response_api = True
    response = ModelResponse(content="OK", model_name="test/model", provider=ProviderType.OPENROUTER)

    with patch.object(provider, "_generate_with_responses_endpoint", return_value=response) as responses_endpoint:
        result = provider.generate_content("test", model_name="test/model", thinking_mode="high")

    assert result is response
    provider._client.chat.completions.create.assert_not_called()
    assert "extra_body" not in responses_endpoint.call_args.kwargs


def test_default_thinking_mode_validation_and_registry_loading(tmp_path):
    for invalid_mode in ("bogus", ["high"]):
        with pytest.raises(
            ValueError,
            match="default_thinking_mode must be one of: minimal, low, medium, high, max",
        ):
            ModelCapabilities(
                provider=ProviderType.OPENROUTER,
                model_name="test/model",
                friendly_name="Test model",
                default_thinking_mode=invalid_mode,
            )

    config_path = tmp_path / "openrouter_models.json"
    config_path.write_text(
        json.dumps({"models": [{"model_name": "test/model", "default_thinking_mode": "high"}]}),
        encoding="utf-8",
    )
    registries = [
        CapabilityModelRegistry(
            env_var_name="PAL_TEST_OPENROUTER_MODELS",
            default_filename="openrouter_models.json",
            provider=ProviderType.OPENROUTER,
            friendly_prefix="OpenRouter ({model})",
            config_path=str(config_path),
        ),
        OpenRouterModelRegistry(config_path=str(config_path)),
    ]

    assert [registry.resolve("test/model").default_thinking_mode for registry in registries] == ["high", "high"]


def test_pinned_openrouter_row_preserves_provider_options_with_reasoning(tmp_path, monkeypatch):
    config_path = tmp_path / "openrouter_models.json"
    config_path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "model_name": "test/pinned-model",
                        "supports_extended_thinking": True,
                        "provider_only": ["together"],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    registry = OpenRouterModelRegistry(config_path=str(config_path))
    monkeypatch.setattr(OpenRouterProvider, "_registry", registry)
    monkeypatch.setattr(utils.model_restrictions, "_restriction_service", None)
    provider = OpenRouterProvider("test-key")
    provider._client = Mock()
    provider._client.chat.completions.create.return_value = _chat_response("test/pinned-model")

    provider.generate_content("test", model_name="test/pinned-model", thinking_mode="high")

    assert provider._client.chat.completions.create.call_args.kwargs["extra_body"] == {
        "provider": {
            "data_collection": "deny",
            "zdr": True,
            "only": ["together"],
            "allow_fallbacks": False,
        },
        "reasoning": {"effort": "high"},
    }


def test_extract_usage_surfaces_reasoning_tokens_without_inflating_output():
    provider = _provider_with_client()
    response = SimpleNamespace(
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=42),
        )
    )

    usage = provider._extract_usage(response)

    assert usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30, "reasoning_tokens": 42}


def test_extract_usage_omits_reasoning_tokens_without_details():
    provider = _provider_with_client()
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    )

    usage = provider._extract_usage(response)

    assert usage == {"input_tokens": 10, "output_tokens": 20, "total_tokens": 30}
