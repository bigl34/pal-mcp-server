import asyncio
import shutil
from pathlib import Path

import pytest

from clink.agents.base import CLIAgentError
from clink.agents.codex import CodexAgent
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
def codex_agent():
    prompt_path = Path("systemprompts/clink/codex_default.txt").resolve()
    role = ResolvedCLIRole(name="default", prompt_path=prompt_path, role_args=[])
    client = ResolvedCLIClient(
        name="codex",
        executable=["codex"],
        internal_args=["exec"],
        config_args=["--json", "--dangerously-bypass-approvals-and-sandbox"],
        env={},
        timeout_seconds=30,
        parser="codex_jsonl",
        runner="codex",
        roles={"default": role},
        output_to_file=None,
        working_dir=None,
    )
    return CodexAgent(client), role


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
async def test_codex_agent_recovers_jsonl(monkeypatch, codex_agent):
    agent, role = codex_agent
    stdout = b"""
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"Hello from Codex"}}
{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}
"""
    process = DummyProcess(stdout=stdout, returncode=124)
    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.returncode == 124
    assert "Hello from Codex" in result.parsed.content
    assert result.parsed.metadata["usage"]["output_tokens"] == 5


@pytest.mark.asyncio
async def test_codex_agent_propagates_invalid_json(monkeypatch, codex_agent):
    agent, role = codex_agent
    stdout = b"not json"
    process = DummyProcess(stdout=stdout, returncode=1)

    with pytest.raises(CLIAgentError):
        await _run_agent_with_process(monkeypatch, agent, role, process)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config_args", "expected_configured"),
    [
        (["--json", "--model", "gpt-5.5"], "gpt-5.5"),
        (["--json", "--model=gpt-5.5"], "gpt-5.5"),
        (["--json", "-m", "gpt-5.5"], "gpt-5.5"),
        (["--json", "-m=gpt-5.5"], "gpt-5.5"),
    ],
)
async def test_codex_model_flag_forms_are_stripped_and_canonicalized(
    monkeypatch, codex_agent, config_args, expected_configured
):
    agent, role = codex_agent
    agent.client.config_args = config_args
    stdout = b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.model_source == "config"
    assert result.configured_model == expected_configured
    assert result.effective_model == expected_configured
    assert agent.client.config_args == config_args
    command = result.sanitized_command
    assert command.count("-m") == 1
    assert command[command.index("-m") + 1] == expected_configured
    assert "--model" not in command
    assert "--model=gpt-5.5" not in command
    assert "-m=gpt-5.5" not in command


@pytest.mark.asyncio
async def test_codex_config_model_beats_native_default(monkeypatch, codex_agent):
    agent, role = codex_agent
    agent.client.config_args = ["--json", "-m", "gpt-5.5"]
    stdout = b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.model_source == "config"
    assert result.configured_model == "gpt-5.5"
    assert result.effective_model == "gpt-5.5"
    assert result.sanitized_command[-2:] == ["-m", "gpt-5.5"]


@pytest.mark.asyncio
async def test_codex_native_without_model_leaves_args_untouched(monkeypatch, codex_agent):
    agent, role = codex_agent
    original_args = ["--json", "--dangerously-bypass-approvals-and-sandbox"]
    agent.client.config_args = list(original_args)
    stdout = b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.model_source == "native"
    assert result.effective_model is None
    assert result.configured_model is None
    assert result.sanitized_command == ["/usr/bin/codex", "exec", *original_args]


@pytest.mark.asyncio
async def test_codex_does_not_corrupt_c_config_value_before_m_flag(monkeypatch, codex_agent):
    agent, role = codex_agent
    agent.client.config_args = ["--json", "-c", 'model_reasoning_effort="xhigh"', "-m", "gpt-5.5"]
    stdout = b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert "-c" in result.sanitized_command
    c_index = result.sanitized_command.index("-c")
    assert result.sanitized_command[c_index + 1] == 'model_reasoning_effort="xhigh"'
    assert result.sanitized_command[-2:] == ["-m", "gpt-5.5"]


@pytest.mark.asyncio
@pytest.mark.parametrize("config_args", [["--model=-bad"], ["-m=-bad"], ["-m"], ["-m", "--json"]])
async def test_codex_bad_config_model_flags_are_stripped_without_value(monkeypatch, caplog, codex_agent, config_args):
    agent, role = codex_agent
    agent.client.config_args = config_args
    stdout = b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
    process = DummyProcess(stdout=stdout)

    result = await _run_agent_with_process(monkeypatch, agent, role, process)

    assert result.model_source == "native"
    assert result.configured_model is None
    assert result.effective_model is None
    assert "-m" not in result.sanitized_command
    assert "--model=-bad" not in result.sanitized_command
    assert "-m=-bad" not in result.sanitized_command
    assert "Ignoring" in caplog.text


@pytest.mark.asyncio
async def test_codex_env_is_not_sanitized(monkeypatch, codex_agent):
    agent, role = codex_agent
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    agent.client.env = {"CLAUDE_CODE_API_KEY": "from-client"}
    stdout = b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
    process = DummyProcess(stdout=stdout)

    await _run_agent_with_process(monkeypatch, agent, role, process)

    assert process.env is not None
    assert process.env["ANTHROPIC_API_KEY"] == "from-env"
    assert process.env["CLAUDE_CODE_USE_BEDROCK"] == "1"
    assert process.env["CLAUDE_CODE_API_KEY"] == "from-client"
