import json

import pytest
from pydantic import ValidationError

from clink import get_registry
from clink.agents import AgentOutput, CLIAgentError
from clink.parsers.base import ParsedCLIResponse
from tools.clink import MAX_RESPONSE_CHARS, CLinkRequest, CLinkTool
from tools.shared.exceptions import ToolExecutionError
from utils.conversation_memory import get_thread


def test_clink_tool_is_marked_side_effectful():
    tool = CLinkTool()
    assert tool.get_annotations()["readOnlyHint"] is False


@pytest.mark.asyncio
async def test_clink_tool_execute(monkeypatch):
    tool = CLinkTool()
    captured_kwargs = {}

    async def fake_run(**kwargs):
        captured_kwargs.update(kwargs)
        return AgentOutput(
            parsed=ParsedCLIResponse(content="Hello from Gemini", metadata={"model_used": "gemini-2.5-pro"}),
            sanitized_command=["gemini", "-o", "json"],
            returncode=0,
            stdout='{"response": "Hello from Gemini"}',
            stderr="",
            duration_seconds=0.42,
            parser_name="gemini_json",
            output_file_content=None,
            requested_model="gemini-2.5-pro",
            effective_model="gemini-2.5-pro",
            configured_model=None,
            model_source="request",
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    def fake_create_agent(client):
        return DummyAgent()

    monkeypatch.setattr("tools.clink.create_agent", fake_create_agent)

    arguments = {
        "prompt": "Summarize the project",
        "cli_name": "gemini",
        "role": "default",
        "model": "gemini-2.5-pro",
        "absolute_file_paths": [],
        "images": [],
    }

    results = await tool.execute(arguments)
    assert len(results) == 1

    payload = json.loads(results[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert "Hello from Gemini" in payload["content"]
    metadata = payload.get("metadata", {})
    assert metadata.get("cli_name") == "gemini"
    assert metadata.get("command") == ["gemini", "-o", "json"]
    assert metadata.get("requested_model") == "gemini-2.5-pro"
    assert metadata.get("effective_model") == "gemini-2.5-pro"
    assert metadata.get("configured_model") is None
    assert metadata.get("model_source") == "request"
    assert metadata.get("model_used") == "gemini-2.5-pro"
    assert metadata.get("model_mismatch") is False
    assert captured_kwargs["model"] == "gemini-2.5-pro"
    assert "gemini CLI agent" in captured_kwargs["prompt"]


def test_registry_lists_roles():
    registry = get_registry()
    clients = registry.list_clients()
    assert {"codex", "gemini"}.issubset(set(clients))
    roles = registry.list_roles("gemini")
    assert "default" in roles
    assert "default" in registry.list_roles("codex")
    codex_client = registry.get_client("codex")
    # Verify codex enables live web search via -c web_search="live". The older
    # --enable web_search_request form was deprecated in codex-cli 0.146.0 (it emits an
    # error item on every run), and --search is unsupported by `codex exec`.
    assert codex_client.config_args[:4] == [
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "-c",
        'web_search="live"',
    ]
    if "-m" in codex_client.config_args:
        model_index = codex_client.config_args.index("-m")
        assert codex_client.config_args[model_index + 1] == "gpt-5.5"


def test_clink_schema_exposes_model_field():
    tool = CLinkTool()
    schema = tool.get_input_schema()

    assert "model" in schema["properties"]
    assert schema["properties"]["model"]["type"] == "string"
    assert schema["properties"]["model"]["maxLength"] == 256
    assert "model" not in schema["required"]


def test_clink_request_model_validation_trims_empty_and_rejects_malformed():
    assert CLinkRequest(prompt="hi", model="  ").model is None
    assert CLinkRequest(prompt="hi", model="fable").model == "fable"
    assert CLinkRequest(prompt="hi", model="native").model == "native"

    for value in ("gpt 5.5", "gpt\t5.5", "-bad", "x" * 257, 123):
        with pytest.raises(ValidationError):
            CLinkRequest(prompt="hi", model=value)


def test_clink_metadata_alias_aware_model_mismatch():
    tool = CLinkTool()
    client = tool._registry.get_client("claude")
    role = client.get_role("default")
    result = AgentOutput(
        parsed=ParsedCLIResponse(content="ok", metadata={"model_used": "claude-fable-5-20260101"}),
        sanitized_command=["claude"],
        returncode=0,
        stdout="{}",
        stderr="",
        duration_seconds=0.1,
        parser_name="claude_json",
        requested_model="fable",
        effective_model="fable",
        configured_model=None,
        model_source="request",
    )

    metadata = tool._build_success_metadata(client, role, result)

    assert metadata["requested_model"] == "fable"
    assert metadata["effective_model"] == "fable"
    assert metadata["model_used"] == "claude-fable-5-20260101"
    assert metadata["model_mismatch"] is False


def test_clink_metadata_unknown_model_mismatch_is_unknown():
    tool = CLinkTool()
    client = tool._registry.get_client("codex")
    role = client.get_role("default")
    result = AgentOutput(
        parsed=ParsedCLIResponse(content="ok", metadata={"model_used": "provider-specific-model"}),
        sanitized_command=["codex"],
        returncode=0,
        stdout="{}",
        stderr="",
        duration_seconds=0.1,
        parser_name="codex_jsonl",
        requested_model="my-alias",
        effective_model="my-alias",
        configured_model=None,
        model_source="request",
    )

    metadata = tool._build_success_metadata(client, role, result)

    assert metadata["model_mismatch"] == "unknown"


@pytest.mark.asyncio
async def test_clink_recovery_metadata_is_surfaced_and_persisted(monkeypatch):
    tool = CLinkTool()

    class DummyAgent:
        async def run(self, **kwargs):
            del kwargs
            return AgentOutput(
                parsed=ParsedCLIResponse(content="Recovered Codex result", metadata={"model_used": "gpt-5.5"}),
                sanitized_command=["codex", "exec", "--json"],
                returncode=124,
                stdout='{"type":"item.completed"}',
                stderr="",
                duration_seconds=1.5,
                parser_name="codex_jsonl",
                recovery_metadata={
                    "recovered": True,
                    "reason": "parseable_output_after_nonzero_exit",
                    "original_return_code": 124,
                },
            )

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    result = await tool.execute(
        {
            "prompt": "Delegate this",
            "cli_name": "codex",
            "role": "default",
            "absolute_file_paths": [],
            "images": [],
        }
    )
    payload = json.loads(result[0].text)

    expected_recovery = {
        "recovered": True,
        "reason": "parseable_output_after_nonzero_exit",
        "original_return_code": 124,
    }
    assert payload["metadata"]["recovery"] == expected_recovery

    continuation_id = payload["continuation_offer"]["continuation_id"]
    thread = get_thread(continuation_id)
    assert thread is not None
    assistant_turn = thread.turns[-1]
    assert assistant_turn.role == "assistant"
    assert assistant_turn.model_metadata["recovery"] == expected_recovery


@pytest.mark.asyncio
async def test_clink_auth_error_surfaces_without_continuation_offer(monkeypatch):
    tool = CLinkTool()

    class DummyAgent:
        async def run(self, **kwargs):
            raise CLIAgentError(
                "Claude subscription auth failed",
                returncode=1,
                stdout="oauth_org_not_allowed",
                stderr="",
                metadata={"auth_error": True},
            )

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Hello",
        "cli_name": "claude",
        "role": "default",
        "absolute_file_paths": [],
        "images": [],
        "continuation_id": "abc",
    }

    with pytest.raises(ToolExecutionError) as exc_info:
        await tool.execute(arguments)

    payload = json.loads(exc_info.value.payload)
    assert payload["status"] == "error"
    assert payload["metadata"]["auth_error"] is True
    assert "continuation_offer" not in payload


@pytest.mark.asyncio
async def test_clink_tool_defaults_to_first_cli(monkeypatch):
    tool = CLinkTool()

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content="Default CLI response", metadata={"events": ["foo"]}),
            sanitized_command=["gemini"],
            returncode=0,
            stdout='{"response": "Default CLI response"}',
            stderr="",
            duration_seconds=0.1,
            parser_name="gemini_json",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Hello",
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    metadata = payload.get("metadata", {})
    assert metadata.get("cli_name") == tool._default_cli_name
    assert metadata.get("events_removed_for_normal") is True


@pytest.mark.asyncio
async def test_clink_tool_truncates_large_output(monkeypatch):
    tool = CLinkTool()

    summary_section = "<SUMMARY>This is the condensed summary.</SUMMARY>"
    long_text = "A" * (MAX_RESPONSE_CHARS + 500) + summary_section

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content=long_text, metadata={"events": ["event1", "event2"]}),
            sanitized_command=["codex"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.2,
            parser_name="codex_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Summarize",
        "cli_name": tool._default_cli_name,
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert payload["content"].strip() == "This is the condensed summary."
    metadata = payload.get("metadata", {})
    assert metadata.get("output_summarized") is True
    assert metadata.get("events_removed_for_normal") is True
    assert metadata.get("output_original_length") == len(long_text)


@pytest.mark.asyncio
async def test_clink_tool_truncates_without_summary(monkeypatch):
    tool = CLinkTool()

    long_text = "B" * (MAX_RESPONSE_CHARS + 1000)

    async def fake_run(**kwargs):
        return AgentOutput(
            parsed=ParsedCLIResponse(content=long_text, metadata={"events": ["event"]}),
            sanitized_command=["codex"],
            returncode=0,
            stdout="{}",
            stderr="",
            duration_seconds=0.2,
            parser_name="codex_jsonl",
            output_file_content=None,
        )

    class DummyAgent:
        async def run(self, **kwargs):
            return await fake_run(**kwargs)

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())

    arguments = {
        "prompt": "Summarize",
        "cli_name": tool._default_cli_name,
        "absolute_file_paths": [],
        "images": [],
    }

    result = await tool.execute(arguments)
    payload = json.loads(result[0].text)
    assert payload["status"] in {"success", "continuation_available"}
    assert "exceeding the configured clink limit" in payload["content"]
    metadata = payload.get("metadata", {})
    assert metadata.get("output_truncated") is True
    assert metadata.get("events_removed_for_normal") is True
    assert metadata.get("output_original_length") == len(long_text)
