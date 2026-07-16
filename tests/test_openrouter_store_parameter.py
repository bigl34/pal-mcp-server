"""Tests for OpenRouter store parameter handling in responses endpoint.

Regression tests for GitHub Issue #348: OpenAI "store" parameter validation error
for certain models via OpenRouter.

OpenRouter's /responses endpoint rejects store:true via Zod validation. This is an
endpoint-level limitation, not model-specific. These tests verify that:
- OpenRouter provider omits the store parameter
- Direct OpenAI provider includes store: true
"""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import utils.model_restrictions
from providers.openai_compatible import OpenAICompatibleProvider
from providers.openrouter import OpenRouterProvider
from providers.registries.base import CapabilityModelRegistry
from providers.registries.openrouter import OpenRouterModelRegistry
from providers.shared import ModelCapabilities, ProviderType


class MockOpenRouterProvider(OpenAICompatibleProvider):
    """Mock provider that simulates OpenRouter behavior."""

    FRIENDLY_NAME = "OpenRouter Test"

    def get_provider_type(self):
        return ProviderType.OPENROUTER

    def get_capabilities(self, model_name):
        return ModelCapabilities(
            provider=ProviderType.OPENROUTER,
            model_name=model_name,
            friendly_name="OpenRouter Test",
            supports_temperature=True,
            default_reasoning_effort="high",
        )

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["openai/gpt-5-pro", "openai/gpt-5.1-codex"]


class MockOpenAIProvider(OpenAICompatibleProvider):
    """Mock provider that simulates direct OpenAI behavior."""

    FRIENDLY_NAME = "OpenAI Test"

    def get_provider_type(self):
        return ProviderType.OPENAI

    def get_capabilities(self, model_name):
        return ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name=model_name,
            friendly_name="OpenAI Test",
            supports_temperature=True,
            default_reasoning_effort="high",
        )

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["gpt-5-pro", "gpt-5.1-codex"]


class MockZenProvider(OpenAICompatibleProvider):
    """Mock provider that simulates OpenCode Zen behavior."""

    FRIENDLY_NAME = "OpenCode Zen Test"

    def get_provider_type(self):
        return ProviderType.ZEN

    def get_capabilities(self, model_name):
        return ModelCapabilities(
            provider=ProviderType.ZEN,
            model_name=model_name,
            friendly_name="OpenCode Zen Test",
            supports_temperature=True,
            default_reasoning_effort="high",
        )

    def validate_model_name(self, model_name):
        return True

    def list_models(self, **kwargs):
        return ["gpt-5.5", "gpt-5.5-pro"]


class TestLogicalOpenAIProfiles(unittest.TestCase):
    """Logical PAL profiles may target a different upstream OpenAI model."""

    def test_registry_loads_logical_profile_and_preserves_max_effort(self):
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "openai_models.json"
            config_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_name": "gpt-5.6-sol-pro",
                                "provider_model_name": "gpt-5.6-sol",
                                "aliases": ["gpt-5.5-pro"],
                                "use_openai_response_api": True,
                                "default_reasoning_mode": "pro",
                                "default_reasoning_effort": "max",
                                "host_dedup_frontends": ["codex"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            registry = CapabilityModelRegistry(
                env_var_name="PAL_TEST_OPENAI_MODELS",
                default_filename="openai_models.json",
                provider=ProviderType.OPENAI,
                friendly_prefix="OpenAI ({model})",
                config_path=str(config_path),
            )
            capabilities = registry.resolve("gpt-5.5-pro")

        self.assertEqual(capabilities.model_name, "gpt-5.6-sol-pro")
        self.assertEqual(capabilities.provider_model_name, "gpt-5.6-sol")
        self.assertEqual(capabilities.default_reasoning_mode, "pro")
        self.assertEqual(capabilities.default_reasoning_effort, "max")
        self.assertEqual(capabilities.host_dedup_frontends, ["codex"])

    def test_invalid_reasoning_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "default_reasoning_mode"):
            ModelCapabilities(
                provider=ProviderType.OPENAI,
                model_name="logical-model",
                friendly_name="Logical model",
                default_reasoning_mode="turbo",
            )

    def test_responses_use_wire_model_pro_max_and_preserve_logical_identity(self):
        captured_params = {}
        capabilities = ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="gpt-5.6-sol-pro",
            provider_model_name="gpt-5.6-sol",
            friendly_name="OpenAI (GPT-5.6 Sol Pro)",
            supports_temperature=False,
            use_openai_response_api=True,
            default_reasoning_mode="pro",
            default_reasoning_effort="max",
        )

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            response = Mock()
            response.output_text = "OK"
            response.usage = None
            response.model = "gpt-5.6-sol"
            response.id = "resp-test"
            response.created_at = 123
            return response

        client = Mock()
        client.responses.create = capture_create

        with (
            patch.object(MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: client)),
            patch.object(MockOpenAIProvider, "get_capabilities", return_value=capabilities),
            patch.object(MockOpenAIProvider, "_resolve_model_name", return_value="gpt-5.6-sol-pro"),
        ):
            result = MockOpenAIProvider("test-key").generate_content(
                "Reply with OK.",
                model_name="gpt-5.5-pro",
            )

        self.assertEqual(captured_params["model"], "gpt-5.6-sol")
        self.assertEqual(captured_params["reasoning"], {"mode": "pro", "effort": "max"})
        self.assertTrue(captured_params["store"])
        self.assertTrue(captured_params["background"])
        self.assertEqual(result.model_name, "gpt-5.6-sol-pro")
        self.assertEqual(result.metadata["provider_model_name"], "gpt-5.6-sol")

    def test_chat_completions_use_wire_model_and_preserve_logical_identity(self):
        captured_params = {}
        capabilities = ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="logical-chat-model",
            provider_model_name="upstream-chat-model",
            friendly_name="Logical chat model",
        )

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            message = Mock(content="OK")
            choice = Mock(message=message, finish_reason="stop")
            return Mock(
                choices=[choice],
                usage=None,
                model="upstream-chat-model",
                id="chat-test",
                created=123,
            )

        client = Mock()
        client.chat.completions.create = capture_create

        with (
            patch.object(MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: client)),
            patch.object(MockOpenAIProvider, "get_capabilities", return_value=capabilities),
            patch.object(MockOpenAIProvider, "_resolve_model_name", return_value="logical-chat-model"),
        ):
            result = MockOpenAIProvider("test-key").generate_content(
                "Reply with OK.",
                model_name="chat-alias",
            )

        self.assertEqual(captured_params["model"], "upstream-chat-model")
        self.assertEqual(result.model_name, "logical-chat-model")
        self.assertEqual(result.metadata["provider_model_name"], "upstream-chat-model")

    def test_responses_send_standard_mode_and_omit_unset_mode(self):
        captured_params = []

        def capture_create(**kwargs):
            captured_params.append(kwargs)
            response = Mock(output_text="OK", usage=None)
            return response

        client = Mock()
        client.responses.create = capture_create

        with patch.object(MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: client)):
            provider = MockOpenAIProvider("test-key")
            provider._generate_with_responses_endpoint(
                model_name="logical-model",
                provider_model_name="upstream-model",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                capabilities=ModelCapabilities(
                    provider=ProviderType.OPENAI,
                    model_name="logical-model",
                    friendly_name="Logical model",
                    default_reasoning_mode="standard",
                    default_reasoning_effort="max",
                ),
            )
            provider._generate_with_responses_endpoint(
                model_name="logical-model",
                provider_model_name="upstream-model",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
                capabilities=ModelCapabilities(
                    provider=ProviderType.OPENAI,
                    model_name="logical-model",
                    friendly_name="Logical model",
                    default_reasoning_effort="max",
                ),
            )

        self.assertEqual(captured_params[0]["reasoning"], {"mode": "standard", "effort": "max"})
        self.assertEqual(captured_params[1]["reasoning"], {"effort": "max"})

    def test_openai_wire_only_allowlist_supports_capabilities_and_listing(self):
        capabilities = ModelCapabilities(
            provider=ProviderType.OPENAI,
            model_name="gpt-5.6-sol-pro",
            provider_model_name="gpt-5.6-sol",
            friendly_name="OpenAI (GPT-5.6 Sol Pro)",
            aliases=["gpt-5.5-pro"],
        )

        class LogicalOpenAIProvider(OpenAICompatibleProvider):
            FRIENDLY_NAME = "Logical OpenAI Test"
            MODEL_CAPABILITIES = {capabilities.model_name: capabilities}

            def get_provider_type(self):
                return ProviderType.OPENAI

        with (
            patch.dict("os.environ", {"OPENAI_ALLOWED_MODELS": "gpt-5.6-sol"}),
            patch.object(utils.model_restrictions, "_restriction_service", None),
        ):
            provider = LogicalOpenAIProvider("test-key")

            self.assertTrue(provider.validate_model_name("gpt-5.5-pro"))
            self.assertEqual(provider.get_capabilities("gpt-5.5-pro").model_name, "gpt-5.6-sol-pro")
            self.assertEqual(
                provider.list_models(respect_restrictions=True),
                ["gpt-5.6-sol-pro", "gpt-5.5-pro"],
            )


class TestStoreParameterHandling(unittest.TestCase):
    """Test store parameter is conditionally included based on provider type.

    **Feature: openrouter-store-parameter-fix, Property 1: OpenRouter requests omit store parameter**
    **Feature: openrouter-store-parameter-fix, Property 2: Direct OpenAI requests include store parameter**
    """

    def test_openrouter_responses_omits_store_parameter(self):
        """Test that OpenRouter provider omits store parameter from responses endpoint.

        **Feature: openrouter-store-parameter-fix, Property 1: OpenRouter requests omit store parameter**
        **Validates: Requirements 1.1, 2.1**

        OpenRouter's /responses endpoint rejects store:true via Zod validation (Issue #348).
        The store parameter should be omitted entirely for OpenRouter requests.
        """
        # Capture the completion_params passed to the API
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            # Return a mock response
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockOpenRouterProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockOpenRouterProvider("test-key")

            # Call the method that builds completion_params
            provider._generate_with_responses_endpoint(
                model_name="openai/gpt-5-pro",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
            )

        # Verify store parameter is NOT in the request
        self.assertNotIn("store", captured_params, "OpenRouter requests should NOT include 'store' parameter")
        self.assertEqual(
            captured_params.get("extra_body"),
            {"provider": {"zdr": True, "data_collection": "deny"}},
            "OpenRouter requests should pass ZDR routing through SDK extra_body",
        )
        self.assertNotIn("provider", captured_params, "OpenRouter routing is not an SDK keyword argument")

    def test_openai_responses_includes_store_parameter(self):
        """Test that direct OpenAI provider includes store parameter in responses endpoint.

        **Feature: openrouter-store-parameter-fix, Property 2: Direct OpenAI requests include store parameter**
        **Validates: Requirements 1.2, 2.2**

        Direct OpenAI API supports the store parameter for stored completions.
        The store parameter should be included with value True for OpenAI requests.
        """
        # Capture the completion_params passed to the API
        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            # Return a mock response
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockOpenAIProvider("test-key")

            # Call the method that builds completion_params
            provider._generate_with_responses_endpoint(
                model_name="gpt-5-pro",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
            )

        # Verify store parameter IS in the request with value True
        self.assertIn("store", captured_params, "OpenAI requests should include 'store' parameter")
        self.assertTrue(captured_params["store"], "OpenAI requests should have store=True")
        self.assertNotIn("provider", captured_params, "Direct OpenAI requests should not include OpenRouter routing")
        self.assertNotIn("extra_body", captured_params, "Direct OpenAI requests should not include OpenRouter routing")

    def test_openai_responses_uses_background_multimodal_and_max_output_tokens(self):
        """Direct OpenAI /responses requests should use current Responses API payload shape."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(
            MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockOpenAIProvider("test-key")
            provider._generate_with_responses_endpoint(
                model_name="gpt-5-pro",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe this image"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                        ],
                    }
                ],
                temperature=0.7,
                max_output_tokens=321,
            )

        self.assertTrue(captured_params["store"], "OpenAI requests should keep store=True")
        self.assertTrue(captured_params["background"], "OpenAI /responses requests should use background mode")
        self.assertEqual(captured_params["max_output_tokens"], 321)
        self.assertNotIn("max_completion_tokens", captured_params)
        self.assertEqual(
            captured_params["input"],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Describe this image"},
                        {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
                    ],
                }
            ],
        )

    def test_openai_background_responses_poll_until_completed(self):
        """Queued background responses should be retrieved until completion before output extraction."""

        mock_client_instance = Mock()
        queued = Mock(status="queued", id="resp-123")
        in_progress = Mock(status="in_progress", id="resp-123")
        completed = Mock(status="completed", id="resp-123")
        completed.output_text = "Done"
        completed.usage = None

        mock_client_instance.responses.create.return_value = queued
        mock_client_instance.responses.retrieve.side_effect = [in_progress, completed]

        with (
            patch.object(
                MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
            ),
            patch("providers.openai_compatible.time.sleep"),
        ):
            provider = MockOpenAIProvider("test-key")
            result = provider._generate_with_responses_endpoint(
                model_name="gpt-5-pro",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
            )

        self.assertEqual(result.content, "Done")
        self.assertEqual(mock_client_instance.responses.retrieve.call_count, 2)
        first_call = mock_client_instance.responses.retrieve.call_args_list[0]
        second_call = mock_client_instance.responses.retrieve.call_args_list[1]
        self.assertEqual(first_call.args, ("resp-123",))
        self.assertEqual(second_call.args, ("resp-123",))

    def test_zen_responses_do_not_use_background_polling(self):
        """OpenCode Zen /responses calls should stay synchronous because retrieve(id) is not exposed."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_response = Mock()
            mock_response.status = "completed"
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with patch.object(MockZenProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)):
            provider = MockZenProvider("test-key")
            provider._generate_with_responses_endpoint(
                model_name="gpt-5.5",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.7,
            )

        self.assertNotIn("background", captured_params, "Zen /responses requests should not use background polling")
        self.assertNotIn("store", captured_params, "Zen /responses requests should not request stored responses")
        mock_client_instance.responses.retrieve.assert_not_called()

    def test_failed_zen_sync_response_includes_response_error_detail(self):
        """Synchronous Zen /responses failures should include response.error details."""

        mock_client_instance = Mock()
        failed = Mock(status="failed", id="resp-zen-failed")
        failed.error = Mock(
            code="provider_error",
            message="Zen upstream rejected the request",
        )
        failed.incomplete_details = None
        mock_client_instance.responses.create.return_value = failed

        with patch.object(MockZenProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)):
            provider = MockZenProvider("test-key")
            with self.assertRaisesRegex(RuntimeError, "provider_error"):
                provider._generate_with_responses_endpoint(
                    model_name="gpt-5.5",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.7,
                )

        mock_client_instance.responses.retrieve.assert_not_called()

    def test_failed_background_response_includes_response_error_detail(self):
        """Terminal failed background responses should include response.error details."""

        mock_client_instance = Mock()
        queued = Mock(status="queued", id="resp-quota")
        failed = Mock(status="failed", id="resp-quota")
        failed.error = Mock(
            code="insufficient_quota",
            message="You exceeded your current quota",
        )
        failed.incomplete_details = None

        mock_client_instance.responses.create.return_value = queued
        mock_client_instance.responses.retrieve.return_value = failed

        with (
            patch.object(
                MockOpenAIProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
            ),
            patch("providers.openai_compatible.time.sleep"),
        ):
            provider = MockOpenAIProvider("test-key")
            with self.assertRaisesRegex(RuntimeError, "insufficient_quota"):
                provider._generate_with_responses_endpoint(
                    model_name="gpt-5-pro",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.7,
                )

        self.assertEqual(mock_client_instance.responses.retrieve.call_count, 1)

    def test_openrouter_chat_completions_include_zdr_provider_routing(self):
        """Test that OpenRouter chat completions force ZDR provider routing."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_response.model = kwargs["model"]
            mock_response.id = "chatcmpl-test"
            mock_response.created = 123
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create = capture_create

        with patch.object(
            MockOpenRouterProvider, "client", new_callable=lambda: property(lambda self: mock_client_instance)
        ):
            provider = MockOpenRouterProvider("test-key")
            provider.generate_content("test", model_name="openai/gpt-5-pro")

        self.assertEqual(
            captured_params.get("extra_body"),
            {"provider": {"zdr": True, "data_collection": "deny"}},
            "OpenRouter chat requests should pass ZDR routing through SDK extra_body",
        )
        self.assertNotIn("provider", captured_params, "OpenRouter routing is not an SDK keyword argument")


class TestOpenRouterNonZdrException(unittest.TestCase):
    """Test the explicit fable-5 exception to OpenRouter ZDR routing."""

    def setUp(self):
        self.original_registry = OpenRouterProvider._registry
        self.original_restriction_service = utils.model_restrictions._restriction_service
        utils.model_restrictions._restriction_service = None

    def tearDown(self):
        OpenRouterProvider._registry = self.original_registry
        utils.model_restrictions._restriction_service = self.original_restriction_service

    def configure_registry(self, config_path):
        OpenRouterProvider._registry = OpenRouterModelRegistry(config_path=str(config_path))
        return OpenRouterProvider._registry

    def write_config(self, directory):
        config_path = Path(directory) / "openrouter_models.json"
        config_path.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "model_name": "openrouter/fusion",
                            "aliases": ["fusion", "fusion-quality", "openrouter-fusion"],
                            "context_window": 128000,
                            "max_output_tokens": 128000,
                            "supports_temperature": False,
                            "temperature_constraint": "fixed",
                            "allow_non_zdr": True,
                        },
                        {
                            "model_name": "anthropic/claude-fable-5",
                            "aliases": ["fable-5", "claude-fable-5"],
                            "context_window": 1000000,
                            "max_output_tokens": 128000,
                            "supports_temperature": False,
                            "temperature_constraint": "fixed",
                            "allow_non_zdr": True,
                        },
                        {
                            "model_name": "~anthropic/claude-fable-latest",
                            "aliases": ["fable-latest", "fable", "claude-fable-latest"],
                            "context_window": 1000000,
                            "max_output_tokens": 128000,
                            "supports_temperature": False,
                            "temperature_constraint": "fixed",
                            "allow_non_zdr": True,
                        },
                        {
                            "model_name": "deepseek/deepseek-v4-pro",
                            "aliases": ["deepseek"],
                            "context_window": 1048576,
                            "max_output_tokens": 65536,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_openrouter_registry_accepts_allow_non_zdr_extra(self):
        """OpenRouter config can carry allow_non_zdr without polluting capabilities."""

        with TemporaryDirectory() as temp_dir:
            registry = self.configure_registry(self.write_config(temp_dir))

            fable = registry.resolve("fable-5")
            fable_latest = registry.resolve("fable-latest")

        self.assertEqual(fable.model_name, "anthropic/claude-fable-5")
        self.assertEqual(fable_latest.model_name, "~anthropic/claude-fable-latest")
        self.assertFalse(hasattr(fable, "allow_non_zdr"))
        self.assertEqual(registry.resolve("fusion").model_name, "openrouter/fusion")
        self.assertEqual(registry.resolve("fable-latest").model_name, "~anthropic/claude-fable-latest")
        self.assertEqual(registry.resolve("fable").model_name, "~anthropic/claude-fable-latest")
        self.assertEqual(registry.get_entry("anthropic/claude-fable-5"), {"allow_non_zdr": True})
        self.assertEqual(registry.get_entry("~anthropic/claude-fable-latest"), {"allow_non_zdr": True})
        self.assertEqual(registry.get_entry("openrouter/fusion"), {"allow_non_zdr": True})
        self.assertEqual(registry.get_entry("deepseek/deepseek-v4-pro"), {"allow_non_zdr": False})

    def test_allow_non_zdr_rejected_by_generic_registry(self):
        """allow_non_zdr is OpenRouter-specific and not a global capability."""

        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "generic_models.json"
            config_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model_name": "generic-model",
                                "aliases": ["generic"],
                                "context_window": 1000,
                                "max_output_tokens": 100,
                                "allow_non_zdr": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "allow_non_zdr"):
                CapabilityModelRegistry(
                    env_var_name="PAL_TEST_GENERIC_MODELS",
                    default_filename="generic_models.json",
                    provider=ProviderType.OPENAI,
                    friendly_prefix="Generic ({model})",
                    config_path=str(config_path),
                )

    def test_fable_5_chat_completions_omit_only_zdr(self):
        """fable-5 keeps data_collection deny while omitting strict ZDR."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_response.model = kwargs["model"]
            mock_response.id = "chatcmpl-test"
            mock_response.created = 123
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create = capture_create

        with TemporaryDirectory() as temp_dir:
            self.configure_registry(self.write_config(temp_dir))
            with patch.object(
                OpenRouterProvider,
                "client",
                new_callable=lambda: property(lambda self: mock_client_instance),
            ):
                provider = OpenRouterProvider("test-key")
                provider.generate_content("test", model_name="fable-5")

        self.assertEqual(
            captured_params.get("extra_body"),
            {"provider": {"data_collection": "deny"}},
            "fable-5 should omit only provider.zdr and keep data_collection=deny",
        )
        self.assertEqual(captured_params["model"], "anthropic/claude-fable-5")
        self.assertNotIn("tool_choice", captured_params)

    def test_fable_latest_chat_completions_omit_only_zdr(self):
        """fable-latest resolves to the OpenRouter router while keeping the exception."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_response.model = kwargs["model"]
            mock_response.id = "chatcmpl-test"
            mock_response.created = 123
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create = capture_create

        with TemporaryDirectory() as temp_dir:
            self.configure_registry(self.write_config(temp_dir))
            with patch.object(
                OpenRouterProvider,
                "client",
                new_callable=lambda: property(lambda self: mock_client_instance),
            ):
                provider = OpenRouterProvider("test-key")
                provider.generate_content("test", model_name="fable-latest")

        self.assertEqual(
            captured_params.get("extra_body"),
            {"provider": {"data_collection": "deny"}},
            "fable-latest should omit only provider.zdr and keep data_collection=deny",
        )
        self.assertEqual(captured_params["model"], "~anthropic/claude-fable-latest")
        self.assertNotIn("tool_choice", captured_params)

    def test_fusion_chat_completions_omit_zdr_and_force_tool_choice(self):
        """Fusion keeps data_collection deny, omits strict ZDR, and forces tool invocation."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_response.model = kwargs["model"]
            mock_response.id = "chatcmpl-test"
            mock_response.created = 123
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create = capture_create

        with TemporaryDirectory() as temp_dir:
            self.configure_registry(self.write_config(temp_dir))
            with patch.object(
                OpenRouterProvider,
                "client",
                new_callable=lambda: property(lambda self: mock_client_instance),
            ):
                provider = OpenRouterProvider("test-key")
                provider.generate_content("test", model_name="fusion")

        self.assertEqual(captured_params["model"], "openrouter/fusion")
        self.assertEqual(
            captured_params.get("extra_body"),
            {"provider": {"data_collection": "deny"}},
            "Fusion should omit only provider.zdr and keep data_collection=deny",
        )
        self.assertEqual(captured_params.get("tool_choice"), "required")
        self.assertNotIn("plugins", captured_params)
        self.assertNotIn("analysis_models", captured_params)

    def test_non_exception_openrouter_model_keeps_zdr(self):
        """Other OpenRouter models still force ZDR and data_collection denial."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_choice = Mock()
            mock_choice.message = mock_message
            mock_choice.finish_reason = "stop"
            mock_response = Mock()
            mock_response.choices = [mock_choice]
            mock_response.usage = None
            mock_response.model = kwargs["model"]
            mock_response.id = "chatcmpl-test"
            mock_response.created = 123
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create = capture_create

        with TemporaryDirectory() as temp_dir:
            self.configure_registry(self.write_config(temp_dir))
            with patch.object(
                OpenRouterProvider,
                "client",
                new_callable=lambda: property(lambda self: mock_client_instance),
            ):
                provider = OpenRouterProvider("test-key")
                provider.generate_content("test", model_name="deepseek")

        self.assertEqual(
            captured_params.get("extra_body"),
            {"provider": {"data_collection": "deny", "zdr": True}},
            "non-exception OpenRouter models should keep strict ZDR routing",
        )
        self.assertNotIn("tool_choice", captured_params)

    def test_fable_5_responses_endpoint_omits_only_zdr(self):
        """The fable-5 exception also applies to /responses payloads."""

        captured_params = {}

        def capture_create(**kwargs):
            captured_params.update(kwargs)
            mock_response = Mock()
            mock_response.output_text = "Test response"
            mock_response.usage = None
            return mock_response

        mock_client_instance = Mock()
        mock_client_instance.responses.create = capture_create

        with TemporaryDirectory() as temp_dir:
            self.configure_registry(self.write_config(temp_dir))
            with patch.object(
                OpenRouterProvider,
                "client",
                new_callable=lambda: property(lambda self: mock_client_instance),
            ):
                provider = OpenRouterProvider("test-key")
                provider._generate_with_responses_endpoint(
                    model_name="anthropic/claude-fable-5",
                    messages=[{"role": "user", "content": "test"}],
                    temperature=0.7,
                )

        self.assertEqual(
            captured_params.get("extra_body"),
            {"provider": {"data_collection": "deny"}},
            "fable-5 /responses requests should omit only provider.zdr",
        )


if __name__ == "__main__":
    unittest.main()
