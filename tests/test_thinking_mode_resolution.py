from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from providers.shared import ModelCapabilities, ModelResponse, ProviderType
from tools.analyze import AnalyzeTool
from tools.consensus import ConsensusTool
from tools.shared.base_models import ToolRequest
from tools.simple.base import SimpleTool


class ThinkingModeRequest(ToolRequest):
    prompt: str


class ThinkingModeSimpleTool(SimpleTool):
    def get_name(self):
        return "thinking_mode_test"

    def get_description(self):
        return "Thinking mode test tool"

    def get_tool_fields(self):
        return {"prompt": {"type": "string"}}

    def get_system_prompt(self):
        return "Test system prompt"

    def get_request_model(self):
        return ThinkingModeRequest

    async def prepare_prompt(self, request):
        return request.prompt


def _model_context(default_thinking_mode=None, responses=None):
    capabilities = ModelCapabilities(
        provider=ProviderType.OPENROUTER,
        model_name="test/model",
        friendly_name="Test model",
        supports_extended_thinking=True,
        default_thinking_mode=default_thinking_mode,
    )
    provider = Mock()
    provider.get_provider_type.return_value = ProviderType.OPENROUTER
    if responses is None:
        provider.generate_content.return_value = ModelResponse(
            content="OK",
            model_name="test/model",
            provider=ProviderType.OPENROUTER,
            metadata={"finish_reason": "STOP"},
        )
    else:
        provider.generate_content.side_effect = responses
    context = SimpleNamespace(provider=provider, capabilities=capabilities, model_name="test/model")
    return context, provider


@pytest.mark.parametrize(
    ("request_value", "configured_default", "expected"),
    [
        ("medium", "high", "medium"),
        ("low", "max", "low"),
        (None, "high", "high"),
        (None, None, "medium"),
    ],
)
@pytest.mark.asyncio
async def test_simple_tool_resolves_thinking_mode_precedence(request_value, configured_default, expected):
    tool = ThinkingModeSimpleTool()
    context, provider = _model_context(configured_default)
    arguments = {"prompt": "test", "model": "test/model", "_model_context": context}
    if request_value is not None:
        arguments["thinking_mode"] = request_value

    await tool.execute(arguments)

    assert provider.generate_content.call_args.kwargs["thinking_mode"] == expected


def test_thinking_mode_requests_preserve_omitted_and_explicit_values():
    omitted = ThinkingModeRequest(prompt="test")
    explicit = ThinkingModeRequest(prompt="test", thinking_mode="medium")

    assert omitted.thinking_mode is None
    assert "thinking_mode" not in omitted.model_fields_set
    assert explicit.thinking_mode == "medium"
    assert "thinking_mode" in explicit.model_fields_set


@pytest.mark.parametrize(
    ("request_model", "configured_default", "expected"),
    [
        (ToolRequest(), None, "high"),
        (ToolRequest(thinking_mode="medium"), "high", "medium"),
        (ToolRequest(), "max", "max"),
    ],
)
@pytest.mark.asyncio
async def test_workflow_expert_resolves_thinking_mode_precedence(request_model, configured_default, expected):
    tool = AnalyzeTool()
    context, provider = _model_context(configured_default)
    provider.generate_content.return_value = ModelResponse(
        content="{}",
        model_name="test/model",
        provider=ProviderType.OPENROUTER,
    )
    tool._model_context = context
    tool._current_model_name = "test/model"
    tool.get_validated_temperature = Mock(return_value=(0.3, []))

    await tool._call_expert_analysis({}, request_model)

    assert provider.generate_content.call_args.kwargs["thinking_mode"] == expected


def test_workflow_thinking_mode_is_null_safe_without_model_context():
    tool = AnalyzeTool()
    tool._model_context = None

    assert tool.get_request_thinking_mode(ToolRequest()) == "high"

    del tool._model_context
    assert tool.get_request_thinking_mode(ToolRequest()) == "high"


def test_workflow_thinking_mode_falls_back_when_capabilities_raise():
    class RaisingModelContext:
        @property
        def capabilities(self):
            raise ValueError("capabilities unavailable")

    tool = AnalyzeTool()
    tool._model_context = RaisingModelContext()

    assert tool.get_request_thinking_mode(ToolRequest()) == tool.get_expert_thinking_mode()


@pytest.mark.asyncio
async def test_simple_tool_retry_reuses_resolved_thinking_mode():
    responses = [
        ModelResponse(
            content="",
            model_name="test/model",
            provider=ProviderType.OPENROUTER,
            metadata={"finish_reason": "STOP"},
        ),
        ModelResponse(
            content="OK",
            model_name="test/model",
            provider=ProviderType.OPENROUTER,
            metadata={"finish_reason": "STOP"},
        ),
    ]
    tool = ThinkingModeSimpleTool()
    context, provider = _model_context("max", responses)

    await tool.execute({"prompt": "test", "model": "test/model", "_model_context": context})

    assert provider.generate_content.call_count == 2
    assert [call.kwargs["thinking_mode"] for call in provider.generate_content.call_args_list] == ["max", "max"]


@pytest.mark.asyncio
async def test_consensus_keeps_medium_with_model_default_max():
    tool = ConsensusTool()
    context, provider = _model_context("max")
    provider.generate_content.return_value = ModelResponse(
        content="OK",
        model_name="test/model",
        provider=ProviderType.OPENROUTER,
    )
    provider.get_provider_type.return_value = ProviderType.OPENROUTER
    tool.get_model_provider = Mock(return_value=provider)
    tool.validate_and_correct_temperature = Mock(return_value=(0.3, []))
    request = SimpleNamespace(relevant_files=[], images=None, step="test")

    with patch("utils.model_context.ModelContext", return_value=context):
        await tool._consult_model(
            {"model": "test/model", "stance": "neutral"},
            request,
            original_proposal="test",
        )

    assert provider.generate_content.call_args.kwargs["thinking_mode"] == "medium"
