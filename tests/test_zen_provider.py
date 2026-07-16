"""Tests for the OpenCode Zen provider."""

import json
import os
import time
from unittest.mock import patch

import pytest

from providers.registries.zen import ZenModelRegistry
from providers.registry import ModelProviderRegistry
from providers.shared import ProviderType
from providers.zen import ZenProvider


def write_zen_config(path, *, model_overrides=None):
    model = {
        "model_name": "gpt-5.5",
        "aliases": ["zen-gpt5.5", "zen-gpt55"],
        "context_window": 400000,
        "max_output_tokens": 272000,
        "supports_extended_thinking": True,
        "supports_system_prompts": True,
        "supports_streaming": False,
        "supports_function_calling": True,
        "supports_json_mode": True,
        "supports_images": True,
        "supports_temperature": True,
        "use_openai_response_api": True,
        "default_reasoning_effort": "high",
        "intelligence_score": 20,
        "endpoint_family": "responses",
        "billing_tier": "paid",
        "retention_policy": "retained_30d",
        "zdr_fallback_eligible": False,
        "openrouter_equivalents": ["openai/gpt-5.5"],
    }
    if model_overrides:
        model.update(model_overrides)

    path.write_text(
        json.dumps(
            {
                "models": [
                    model,
                    {
                        "model_name": "deepseek-v4-flash",
                        "aliases": ["zen-deepseek"],
                        "context_window": 128000,
                        "max_output_tokens": 32000,
                        "supports_extended_thinking": False,
                        "supports_system_prompts": True,
                        "supports_streaming": False,
                        "supports_function_calling": True,
                        "supports_json_mode": True,
                        "supports_images": False,
                        "supports_temperature": True,
                        "intelligence_score": 12,
                        "endpoint_family": "chat_completions",
                        "billing_tier": "paid",
                        "retention_policy": "zero",
                        "zdr_fallback_eligible": True,
                        "openrouter_equivalents": ["deepseek/deepseek-v4-flash"],
                    },
                ]
            }
        )
    )
    return path


def write_safety_cache(path, *, models=None, generated_at=None):
    records = {
        "gpt-5.5": {
            "model_id": "gpt-5.5",
            "display_name": "GPT 5.5",
            "billing_tier": "paid",
            "retention_policy": "retained_30d",
            "runtime_allowed": True,
            "zdr_fallback_eligible": False,
            "reasons": ["OpenAI API requests are retained for 30 days"],
        },
        "deepseek-v4-flash": {
            "model_id": "deepseek-v4-flash",
            "display_name": "DeepSeek V4 Flash",
            "billing_tier": "paid",
            "retention_policy": "zero",
            "runtime_allowed": True,
            "zdr_fallback_eligible": True,
            "reasons": [],
        },
    }
    if models:
        records.update(models)

    path.write_text(
        json.dumps(
            {
                "generated_at": generated_at if generated_at is not None else time.time(),
                "models": records,
            }
        )
    )
    return path


def zen_env(cache_path):
    return {
        "ZEN_SAFETY_CACHE_PATH": str(cache_path),
        "ZEN_SAFETY_CACHE_TTL_SECONDS": "3600",
        "ZEN_SAFETY_REFRESH_DISABLED": "1",
    }


class TestZenProvider:
    def setup_method(self):
        self._original_registry = ZenProvider._registry
        ZenProvider._registry = None

    def teardown_method(self):
        ZenProvider._registry = self._original_registry

    def test_provider_initialization_uses_zen_base_url(self, tmp_path):
        config_path = write_zen_config(tmp_path / "zen_models.json")
        cache_path = write_safety_cache(tmp_path / "zen_safety.json")

        with patch.dict(os.environ, {"ZEN_MODELS_CONFIG_PATH": str(config_path), **zen_env(cache_path)}):
            provider = ZenProvider(api_key="test-zen-key")

        assert provider.api_key == "test-zen-key"
        assert provider.base_url == "https://opencode.ai/zen/v1"
        assert provider.FRIENDLY_NAME == "OpenCode Zen"
        assert provider.get_provider_type() == ProviderType.ZEN

    def test_registry_loads_aliases_and_responses_metadata(self, tmp_path):
        config_path = write_zen_config(tmp_path / "zen_models.json")
        cache_path = write_safety_cache(tmp_path / "zen_safety.json")

        with patch.dict(os.environ, zen_env(cache_path)):
            registry = ZenModelRegistry(config_path=str(config_path))

        caps = registry.resolve("zen-gpt55")
        assert caps is not None
        assert caps.provider == ProviderType.ZEN
        assert caps.model_name == "gpt-5.5"
        assert caps.use_openai_response_api is True
        assert registry.resolve("zen-deepseek").model_name == "deepseek-v4-flash"
        assert registry.get_entry("deepseek-v4-flash")["zdr_fallback_eligible"] is True

    def test_provider_resolves_aliases_from_zen_registry(self, tmp_path):
        config_path = write_zen_config(tmp_path / "zen_models.json")
        cache_path = write_safety_cache(tmp_path / "zen_safety.json")

        with patch.dict(os.environ, {"ZEN_MODELS_CONFIG_PATH": str(config_path), **zen_env(cache_path)}):
            provider = ZenProvider(api_key="test-zen-key")

        assert provider._resolve_model_name("zen-gpt55") == "gpt-5.5"
        assert provider.validate_model_name("zen-deepseek") is True
        assert provider.validate_model_name("claude-opus-4-8") is False

    def test_registry_requires_fresh_safety_manifest(self, tmp_path):
        config_path = write_zen_config(tmp_path / "zen_models.json")
        missing_cache = tmp_path / "missing_zen_safety.json"

        with patch.dict(os.environ, zen_env(missing_cache)):
            with pytest.raises(ValueError, match="Zen safety manifest"):
                ZenModelRegistry(config_path=str(config_path))

    def test_registry_rejects_free_looking_alias(self, tmp_path):
        config_path = write_zen_config(
            tmp_path / "zen_models.json",
            model_overrides={
                "model_name": "paid-new-model",
                "aliases": ["paid-new-model-free"],
                "endpoint_family": "chat_completions",
                "retention_policy": "zero",
                "zdr_fallback_eligible": True,
                "openrouter_equivalents": ["vendor/paid-new-model"],
            },
        )
        cache_path = write_safety_cache(
            tmp_path / "zen_safety.json",
            models={
                "paid-new-model": {
                    "model_id": "paid-new-model",
                    "display_name": "Paid New Model",
                    "billing_tier": "paid",
                    "retention_policy": "zero",
                    "runtime_allowed": True,
                    "zdr_fallback_eligible": True,
                    "reasons": [],
                }
            },
        )

        with patch.dict(os.environ, zen_env(cache_path)):
            with pytest.raises(ValueError, match="free"):
                ZenModelRegistry(config_path=str(config_path))

    def test_registry_rejects_docs_derived_free_model_not_in_static_denylist(self, tmp_path):
        config_path = write_zen_config(
            tmp_path / "zen_models.json",
            model_overrides={
                "model_name": "new-alpha-model",
                "aliases": ["zen-new-alpha"],
                "endpoint_family": "chat_completions",
                "billing_tier": "paid",
                "retention_policy": "zero",
                "zdr_fallback_eligible": True,
                "openrouter_equivalents": ["vendor/new-alpha-model"],
            },
        )
        cache_path = write_safety_cache(
            tmp_path / "zen_safety.json",
            models={
                "new-alpha-model": {
                    "model_id": "new-alpha-model",
                    "display_name": "New Alpha Model",
                    "billing_tier": "free",
                    "retention_policy": "free_retained",
                    "runtime_allowed": False,
                    "zdr_fallback_eligible": False,
                    "reasons": ["pricing table says Free"],
                }
            },
        )

        with patch.dict(os.environ, zen_env(cache_path)):
            with pytest.raises(ValueError, match="not runtime allowed"):
                ZenModelRegistry(config_path=str(config_path))

    def test_registry_blocks_stale_safety_manifest(self, tmp_path):
        config_path = write_zen_config(tmp_path / "zen_models.json")
        cache_path = write_safety_cache(tmp_path / "zen_safety.json", generated_at=time.time() - 7200)

        with patch.dict(os.environ, zen_env(cache_path)):
            with pytest.raises(ValueError, match="stale"):
                ZenModelRegistry(config_path=str(config_path))

    def test_registry_rejects_zdr_fallback_for_retained_model(self, tmp_path):
        config_path = write_zen_config(
            tmp_path / "zen_models.json",
            model_overrides={"zdr_fallback_eligible": True},
        )
        cache_path = write_safety_cache(tmp_path / "zen_safety.json")

        with patch.dict(os.environ, zen_env(cache_path)):
            with pytest.raises(ValueError, match="ZDR fallback"):
                ZenModelRegistry(config_path=str(config_path))

    @pytest.mark.parametrize("endpoint_family", ["messages", "gemini_model_endpoint"])
    def test_registry_rejects_runtime_models_on_unsupported_endpoint_families(self, tmp_path, endpoint_family):
        config_path = write_zen_config(
            tmp_path / "zen_models.json",
            model_overrides={
                "model_name": "paid-new-model",
                "aliases": ["zen-paid-new-model"],
                "endpoint_family": endpoint_family,
                "retention_policy": "zero",
                "zdr_fallback_eligible": False,
                "openrouter_equivalents": [],
            },
        )
        cache_path = write_safety_cache(
            tmp_path / "zen_safety.json",
            models={
                "paid-new-model": {
                    "model_id": "paid-new-model",
                    "display_name": "Paid New Model",
                    "billing_tier": "paid",
                    "retention_policy": "zero",
                    "runtime_allowed": True,
                    "zdr_fallback_eligible": False,
                    "reasons": [],
                }
            },
        )

        with patch.dict(os.environ, zen_env(cache_path)):
            with pytest.raises(ValueError, match="unsupported endpoint family"):
                ZenModelRegistry(config_path=str(config_path))


class TestZenProviderRegistration:
    def setup_method(self):
        self.registry = ModelProviderRegistry()
        self._original_providers = self.registry._providers.copy()
        self._original_initialized = self.registry._initialized_providers.copy()
        ModelProviderRegistry.clear_cache()
        for provider_type in ProviderType:
            ModelProviderRegistry.unregister_provider(provider_type)

    def teardown_method(self):
        self.registry._providers.clear()
        self.registry._providers.update(self._original_providers)
        self.registry._initialized_providers.clear()
        self.registry._initialized_providers.update(self._original_initialized)
        ModelProviderRegistry.clear_cache()

    def test_configure_providers_registers_zen_from_api_key(self, tmp_path):
        from server import configure_providers

        config_path = write_zen_config(tmp_path / "zen_models.json")
        cache_path = write_safety_cache(tmp_path / "zen_safety.json")

        with patch.dict(
            os.environ,
            {
                "ZEN_API_KEY": "test-zen-key",
                "ZEN_MODELS_CONFIG_PATH": str(config_path),
                **zen_env(cache_path),
                "GEMINI_API_KEY": "",
                "OPENAI_API_KEY": "",
                "OPENROUTER_API_KEY": "",
                "CUSTOM_API_URL": "",
                "XAI_API_KEY": "",
                "DIAL_API_KEY": "",
            },
            clear=True,
        ):
            configure_providers()

            available = ModelProviderRegistry.get_available_providers()
            assert ProviderType.ZEN in available
            provider = ModelProviderRegistry.get_provider(ProviderType.ZEN)
            assert isinstance(provider, ZenProvider)
