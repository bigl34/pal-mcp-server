import asyncio
import json
import logging
import shutil
from pathlib import Path

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.claude import ClaudeAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole


class DummyProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.stdin_data: bytes | None = None
        self.args: list[str] | None = None
        self.env: dict[str, str] | None = None

    async def communicate(self, input_data):
        self.stdin_data = input_data
        return self._stdout, self._stderr


@pytest.fixture()
def claude_agent():
    prompt_path = Path("systemprompts/clink/default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="claude",
        executable=["claude"],
        internal_args=["--print", "--output-format", "json"],
        config_args=["--permission-mode", "acceptEdits"],
        env={},
        timeout_seconds=30,
        parser="claude_json",
        runner="claude",
        default_model="fable",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return ClaudeAgent(client), role


async def _run_agent_with_process(monkeypatch, agent, role, process, *, system_prompt="System prompt", model=None):
    async def fake_create_subprocess_exec(*args, **kwargs):
        process.args = list(args)
        process.env = kwargs.get("env")
        return process

    def fake_which(executable_name):
        return f"/usr/bin/{executable_name}"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)

    return await agent.run(
        role=role,
        prompt="Respond with 42",
        system_prompt=system_prompt,
        files=[],
        images=[],
        model=model,
    )


@pytest.mark.asyncio
async def test_claude_agent_injects_system_prompt(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": "42",
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert "--append-system-prompt" in result.sanitized_command
    idx = result.sanitized_command.index("--append-system-prompt")
    assert result.sanitized_command[idx + 1] == "System prompt"
    assert process.stdin_data.decode().startswith("Respond with 42")


@pytest.mark.asyncio
async def test_claude_agent_recovers_error_payload(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "result": "API Error",
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=2)

    with pytest.raises(CLIAgentError) as exc_info:
        await _run_agent_with_process(monkeypatch, agent, role, process)

    assert exc_info.value.returncode == 2
    assert exc_info.value.stdout == stdout_payload.decode()
    assert "error payload" in str(exc_info.value)


@pytest.mark.asyncio
async def test_claude_agent_propagates_unparseable_output(monkeypatch, claude_agent):
    agent, role = claude_agent
    process = DummyProcess(stdout=b"", returncode=1)

    with pytest.raises(CLIAgentError):
        await _run_agent_with_process(monkeypatch, agent, role, process)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_model", "config_args", "expected_source", "expected_effective", "expected_configured", "model_flag"),
    [
        ("opus", ["--permission-mode", "acceptEdits", "--model", "sonnet"], "request", "opus", "sonnet", "opus"),
        (None, ["--permission-mode", "acceptEdits", "--model", "sonnet"], "config", "sonnet", "sonnet", "sonnet"),
        (None, ["--permission-mode", "acceptEdits"], "client_default", "fable", None, "fable"),
        ("native", ["--permission-mode", "acceptEdits", "--model", "sonnet"], "native", None, "sonnet", None),
        (None, ["--permission-mode", "acceptEdits", "--model", "native"], "native", None, "native", None),
    ],
)
async def test_claude_agent_model_precedence_and_native(
    monkeypatch,
    caplog,
    claude_agent,
    request_model,
    config_args,
    expected_source,
    expected_effective,
    expected_configured,
    model_flag,
):
    agent, role = claude_agent
    caplog.set_level(logging.INFO)
    original_config_args = list(config_args)
    agent.client.config_args = config_args
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process, model=request_model)

    assert agent.client.config_args == original_config_args
    assert result.requested_model == request_model
    assert result.configured_model == expected_configured
    assert result.effective_model == expected_effective
    assert result.model_source == expected_source
    assert result.sanitized_command == list(process.args)

    command = result.sanitized_command
    assert command.count("--model") == (1 if model_flag else 0)
    if model_flag:
        assert command[command.index("--model") + 1] == model_flag
        assert command.index("--model") < command.index("--append-system-prompt")
    else:
        assert "--model" not in command

    if expected_source == "config":
        assert "config" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_args", "expected_configured"),
    [
        (["--permission-mode", "acceptEdits", "--model", "sonnet"], "sonnet"),
        (["--permission-mode", "acceptEdits", "--model=opus"], "opus"),
    ],
)
async def test_claude_model_flag_forms_are_normalized(monkeypatch, claude_agent, config_args, expected_configured):
    agent, role = claude_agent
    agent.client.config_args = config_args
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    command = result.sanitized_command
    assert result.configured_model == expected_configured
    assert command.count("--model") == 1
    assert command[command.index("--model") + 1] == expected_configured
    assert all(token not in command for token in ("--model=opus", "--model=sonnet"))


@pytest.mark.asyncio
async def test_claude_continue_short_flag_does_not_consume_following_model_flag(monkeypatch, claude_agent):
    agent, role = claude_agent
    agent.client.config_args = ["-c", "--model", "sonnet"]
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    command = result.sanitized_command
    assert "-c" in command
    assert result.model_source == "config"
    assert result.configured_model == "sonnet"
    assert result.effective_model == "sonnet"
    assert command.count("--model") == 1
    assert command[command.index("--model") + 1] == "sonnet"


@pytest.mark.asyncio
async def test_claude_multiple_distinct_config_models_warns_and_last_wins(monkeypatch, caplog, claude_agent):
    agent, role = claude_agent
    agent.client.config_args = ["--model", "sonnet", "--permission-mode", "acceptEdits", "--model=opus"]
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.configured_model == "opus"
    assert result.effective_model == "opus"
    assert result.sanitized_command.count("--model") == 1
    assert "multiple distinct model values" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config_args",
    [
        ["--permission-mode", "acceptEdits", "--model"],
        ["--permission-mode", "acceptEdits", "--model", "--permission-mode"],
        ["--permission-mode", "acceptEdits", "--model=-bad"],
    ],
)
async def test_claude_bad_config_model_flags_are_stripped_with_warning(monkeypatch, caplog, claude_agent, config_args):
    agent, role = claude_agent
    agent.client.config_args = config_args
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.configured_model is None
    assert result.effective_model == "fable"
    assert result.sanitized_command.count("--model") == 1
    assert result.sanitized_command[result.sanitized_command.index("--model") + 1] == "fable"
    assert "--model=-bad" not in result.sanitized_command
    assert "Ignoring" in caplog.text


@pytest.mark.asyncio
async def test_claude_glue_and_role_model_tokens_are_not_normalized(monkeypatch, claude_agent):
    agent, role = claude_agent
    agent.client.config_args = ["--models", "kept", "--modelvalue"]
    role.role_args = ["--model", "role-token"]
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload)

    result = await _run_agent_with_process(monkeypatch, agent, role, process, model="opus")

    assert "--models" in result.sanitized_command
    assert "--modelvalue" in result.sanitized_command
    assert result.sanitized_command.count("--model") == 2
    first_model_index = result.sanitized_command.index("--model")
    assert result.sanitized_command[first_model_index + 1] == "opus"
    assert result.sanitized_command[-2:] == ["--model", "role-token"]


@pytest.mark.asyncio
async def test_claude_env_strips_subscription_conflicting_keys(monkeypatch, claude_agent):
    agent, role = claude_agent
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CODE_API_KEY_HELPER", "helper")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth")
    monkeypatch.setenv("CLAUDE_CODE_MAX_THINKING_TOKENS", "1000")
    agent.client.env = {
        "ANTHROPIC_BASE_URL": "https://example.invalid",
        "CLAUDE_CODE_API_KEY": "from-client",
        "CLAUDE_CODE_OAUTH_TOKEN": "client-oauth",
        "CLAUDE_CODE_CUSTOM": "custom",
    }
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload)

    await _run_agent_with_process(monkeypatch, agent, role, process)

    assert process.env is not None
    assert "ANTHROPIC_API_KEY" not in process.env
    assert "ANTHROPIC_BASE_URL" not in process.env
    assert "CLAUDE_CODE_USE_BEDROCK" not in process.env
    assert "CLAUDE_CODE_API_KEY" not in process.env
    assert "CLAUDE_CODE_API_KEY_HELPER" not in process.env
    assert process.env["CLAUDE_CODE_OAUTH_TOKEN"] == "client-oauth"
    assert process.env["CLAUDE_CODE_MAX_THINKING_TOKENS"] == "1000"
    assert process.env["CLAUDE_CODE_CUSTOM"] == "custom"


@pytest.mark.asyncio
async def test_claude_zero_exit_error_payload_fails(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps({"type": "result", "is_error": True, "result": "API Error"}).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=0)

    with pytest.raises(CLIAgentError) as exc_info:
        await _run_agent_with_process(monkeypatch, agent, role, process)

    assert exc_info.value.returncode == 0
    assert "error payload" in str(exc_info.value)


@pytest.mark.asyncio
async def test_claude_nonzero_success_payload_recovers(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {"type": "result", "subtype": "success", "is_error": False, "result": "Recovered"}
    ).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=2)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 2
    assert result.parsed.content == "Recovered"
    assert result.parsed.metadata["is_error"] is False
    assert result.parsed.metadata["is_error_explicit"] is True


@pytest.mark.asyncio
async def test_claude_nonzero_without_positive_success_signal_fails(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "Not enough"}).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=2)

    with pytest.raises(CLIAgentError) as exc_info:
        await _run_agent_with_process(monkeypatch, agent, role, process)

    assert exc_info.value.returncode == 2
    assert "did not report success" in str(exc_info.value)


@pytest.mark.asyncio
async def test_claude_nonzero_success_subtype_without_is_error_fails(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps({"type": "result", "subtype": "success", "result": "Not explicit"}).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=2)

    with pytest.raises(CLIAgentError) as exc_info:
        await _run_agent_with_process(monkeypatch, agent, role, process)

    assert exc_info.value.returncode == 2
    assert "did not report success" in str(exc_info.value)


@pytest.mark.asyncio
async def test_claude_nonzero_string_false_is_error_fails(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {"type": "result", "subtype": "success", "is_error": "false", "result": "String false"}
    ).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=2)

    with pytest.raises(CLIAgentError) as exc_info:
        await _run_agent_with_process(monkeypatch, agent, role, process)

    assert exc_info.value.returncode == 2
    assert "error payload" in str(exc_info.value)


@pytest.mark.asyncio
async def test_claude_zero_exit_success_without_subtype_passes(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps({"type": "result", "is_error": False, "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=0)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.parsed.content == "42"
    assert result.parsed.metadata.get("subtype") is None


@pytest.mark.asyncio
async def test_claude_zero_exit_payload_without_is_error_passes(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps({"type": "result", "result": "42"}).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=0)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.parsed.content == "42"
    assert result.parsed.metadata["is_error"] is False
    assert result.parsed.metadata["is_error_explicit"] is False


@pytest.mark.asyncio
async def test_claude_successful_content_can_mention_oauth_error_code(monkeypatch, claude_agent):
    agent, role = claude_agent
    stdout_payload = json.dumps(
        {
            "type": "result",
            "is_error": False,
            "result": "The oauth_org_not_allowed code means your org disallows this auth mode.",
        }
    ).encode()
    process = DummyProcess(stdout=stdout_payload, returncode=0)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert "oauth_org_not_allowed" in result.parsed.content


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stdout_payload", "stderr_payload", "returncode"),
    [
        (
            json.dumps(
                {
                    "type": "result",
                    "is_error": True,
                    "error": {"code": "oauth_org_not_allowed", "message": "Subscription account required"},
                }
            ).encode(),
            b"",
            0,
        ),
        (b"non-json oauth_org_not_allowed failure", b"", 1),
        (json.dumps({"type": "result", "is_error": False, "result": "ignored"}).encode(), b"OAUTH_ORG_NOT_ALLOWED", 1),
    ],
)
async def test_claude_oauth_org_not_allowed_reports_subscription_reauth(
    monkeypatch, claude_agent, stdout_payload, stderr_payload, returncode
):
    agent, role = claude_agent
    process = DummyProcess(stdout=stdout_payload, stderr=stderr_payload, returncode=returncode)

    with pytest.raises(CLIAgentError) as exc_info:
        await _run_agent_with_process(monkeypatch, agent, role, process)

    message = str(exc_info.value)
    assert "claude logout" in message
    assert "claude login" in message
    assert "ANTHROPIC_API_KEY" in message
    assert exc_info.value.metadata["auth_error"] is True
    assert exc_info.value.metadata["auth_error_code"] == "oauth_org_not_allowed"
