import asyncio
import shutil
from pathlib import Path

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.gemini import GeminiAgent
from clink.models import ResolvedCLIClient, ResolvedCLIRole


class DummyProcess:
    def __init__(self, *, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.args: list[str] | None = None
        self.env: dict[str, str] | None = None

    async def communicate(self, _input):
        return self._stdout, self._stderr


@pytest.fixture()
def gemini_agent():
    prompt_path = Path("systemprompts/clink/gemini_default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="gemini",
        executable=["gemini"],
        internal_args=[],
        config_args=[],
        env={},
        timeout_seconds=30,
        parser="gemini_json",
        runner="gemini",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return GeminiAgent(client), role


async def _run_agent_with_process(monkeypatch, agent, role, process, *, model=None):
    async def fake_create_subprocess_exec(*args, **kwargs):
        process.args = list(args)
        process.env = kwargs.get("env")
        return process

    def fake_which(executable_name):
        return f"/usr/bin/{executable_name}"

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(shutil, "which", fake_which)
    return await agent.run(role=role, prompt="do something", files=[], images=[], model=model)


@pytest.mark.asyncio
async def test_gemini_agent_recovers_tool_error(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    error_json = """{
  "error": {
    "type": "FatalToolExecutionError",
    "message": "Error executing tool replace: Failed to edit",
    "code": "edit_expected_occurrence_mismatch"
  }
}"""
    stderr = ("Error: Failed to edit, expected 1 occurrence but found 2.\n" + error_json).encode()
    process = DummyProcess(stderr=stderr, returncode=54)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 54
    assert result.parsed.metadata["cli_error_recovered"] is True
    assert result.parsed.metadata["cli_error_code"] == "edit_expected_occurrence_mismatch"
    assert "Gemini CLI reported a tool failure" in result.parsed.content


@pytest.mark.asyncio
async def test_gemini_agent_propagates_unrecoverable_error(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    stderr = b"Plain failure without structured payload"
    process = DummyProcess(stderr=stderr, returncode=54)

    with pytest.raises(CLIAgentError):
        await _run_agent_with_process(monkeypatch, agent, role, process)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_args", "expected_configured"),
    [
        (["--yolo", "--model", "gemini-2.5-pro"], "gemini-2.5-pro"),
        (["--yolo", "--model=gemini-2.5-pro"], "gemini-2.5-pro"),
        (["--yolo", "-m", "gemini-2.5-pro"], "gemini-2.5-pro"),
        (["--yolo", "-m=gemini-2.5-pro"], "gemini-2.5-pro"),
    ],
)
async def test_gemini_model_flag_forms_are_stripped_and_canonicalized(
    monkeypatch, gemini_agent, config_args, expected_configured
):
    agent, role = gemini_agent
    agent.client.config_args = config_args
    stdout = b'{"response":"ok"}'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.model_source == "config"
    assert result.configured_model == expected_configured
    assert result.effective_model == expected_configured
    command = result.sanitized_command
    assert command.count("--model") == 1
    assert command[command.index("--model") + 1] == expected_configured
    assert "--model=gemini-2.5-pro" not in command
    assert "-m=gemini-2.5-pro" not in command
    assert "-m" not in command


@pytest.mark.asyncio
async def test_gemini_explicit_model_yields_valid_command(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    agent.client.config_args = ["--yolo"]
    stdout = b'{"response":"ok"}'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process, model="gemini-2.5-pro")

    assert result.model_source == "request"
    assert result.requested_model == "gemini-2.5-pro"
    assert result.effective_model == "gemini-2.5-pro"
    assert result.sanitized_command[-2:] == ["--model", "gemini-2.5-pro"]


@pytest.mark.asyncio
async def test_gemini_native_request_suppresses_config_model(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    agent.client.config_args = ["--yolo", "--model", "gemini-2.5-pro"]
    stdout = b'{"response":"ok"}'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process, model="native")

    assert result.model_source == "native"
    assert result.configured_model == "gemini-2.5-pro"
    assert result.effective_model is None
    assert "--model" not in result.sanitized_command


@pytest.mark.asyncio
async def test_gemini_native_without_model_leaves_args_untouched(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    agent.client.config_args = ["--yolo"]
    stdout = b'{"response":"ok"}'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.model_source == "native"
    assert result.effective_model is None
    assert result.sanitized_command == ["/usr/bin/gemini", "--yolo"]


@pytest.mark.asyncio
async def test_gemini_env_is_not_sanitized(monkeypatch, gemini_agent):
    agent, role = gemini_agent
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    agent.client.env = {"CLAUDE_CODE_API_KEY": "from-client"}
    stdout = b'{"response":"ok"}'
    process = DummyProcess(stdout=stdout)

    await _run_agent_with_process(monkeypatch, agent, role, process)

    assert process.env is not None
    assert process.env["ANTHROPIC_API_KEY"] == "from-env"
    assert process.env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert process.env["CLAUDE_CODE_API_KEY"] == "from-client"
