import asyncio
import json
import os
import signal
import sys
import time

import pytest
from pydantic import ValidationError

from clink import get_registry
from clink.agents import AgentOutput, CLIAgentError, create_agent
from clink.effort import REASONING_EFFORTS, effort_args
from clink.parsers.base import ParsedCLIResponse
from tools.clink import CLinkRequest, CLinkTool
from tools.shared.exceptions import ToolExecutionError


def test_effort_args_per_cli():
    assert effort_args("codex", "max") == ["-c", 'model_reasoning_effort="max"']
    assert effort_args("claude", "xhigh") == ["--effort", "xhigh"]


def test_effort_args_rejects_ultra_and_unsupported_cli():
    assert "ultra" not in REASONING_EFFORTS
    with pytest.raises(ValueError):
        effort_args("codex", "ultra")
    with pytest.raises(ValueError):
        effort_args("gemini", "max")


def test_clink_request_validates_reasoning_effort():
    assert CLinkRequest(prompt="x", reasoning_effort=" MAX ").reasoning_effort == "max"
    assert CLinkRequest(prompt="x", reasoning_effort="").reasoning_effort is None
    with pytest.raises(ValidationError):
        CLinkRequest(prompt="x", reasoning_effort="ultra")


def test_clink_schema_exposes_reasoning_effort():
    schema = CLinkTool().get_input_schema()
    assert schema["properties"]["reasoning_effort"]["enum"] == list(REASONING_EFFORTS)


def _dummy_output():
    return AgentOutput(
        parsed=ParsedCLIResponse(content="ok", metadata={}),
        sanitized_command=["codex", "exec"],
        returncode=0,
        stdout="",
        stderr="",
        duration_seconds=0.1,
        parser_name="codex_jsonl",
    )


@pytest.mark.asyncio
async def test_clink_tool_appends_effort_after_role_args(monkeypatch):
    captured = {}

    class DummyAgent:
        async def run(self, **kwargs):
            captured.update(kwargs)
            return _dummy_output()

    monkeypatch.setattr("tools.clink.create_agent", lambda client: DummyAgent())
    tool = CLinkTool()
    results = await tool.execute({"prompt": "hi", "cli_name": "codex", "reasoning_effort": "max"})
    payload = json.loads(results[0].text)
    assert payload["metadata"]["reasoning_effort"] == "max"
    assert captured["role"].role_args[-2:] == ["-c", 'model_reasoning_effort="max"']


@pytest.mark.asyncio
async def test_clink_tool_rejects_effort_for_unsupported_cli(monkeypatch):
    monkeypatch.setattr("tools.clink.create_agent", lambda client: pytest.fail("agent must not run"))
    tool = CLinkTool()
    with pytest.raises(ToolExecutionError):
        await tool.execute({"prompt": "hi", "cli_name": "gemini", "reasoning_effort": "max"})


def _real_client(tmp_path, script: str, timeout: int = 1):
    base = get_registry().get_client("codex")
    return base.model_copy(
        update={
            "name": "codex",
            "executable": [sys.executable, "-c", script],
            "internal_args": [],
            "config_args": [],
            "timeout_seconds": timeout,
            "working_dir": tmp_path,
        }
    )


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
@pytest.mark.asyncio
async def test_timeout_kills_orphaned_child_after_leader_exits(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=sys.stdout)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "sys.exit(0)\n"
    )
    agent = create_agent(_real_client(tmp_path, script))
    role = agent.client.get_role(None)
    started = time.monotonic()
    with pytest.raises(Exception) as exc_info:
        await agent.run(role=role, prompt="", files=[], images=[], model="native")
    assert time.monotonic() - started < 10
    assert "timed out" in str(exc_info.value)
    child_pid = int(pid_file.read_text())
    for _ in range(50):
        if not _pid_alive(child_pid):
            break
        await asyncio.sleep(0.05)
    assert not _pid_alive(child_pid)


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
@pytest.mark.asyncio
async def test_repeated_cancellation_still_reaps_sigterm_ignoring_process(tmp_path):
    pid_file = tmp_path / "leader.pid"
    script = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    agent = create_agent(_real_client(tmp_path, script, timeout=120))
    role = agent.client.get_role(None)
    task = asyncio.create_task(agent.run(role=role, prompt="", files=[], images=[], model="native"))
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        await asyncio.sleep(0.05)
    task.cancel()
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    leader_pid = int(pid_file.read_text())
    assert not _pid_alive(leader_pid)
    assert signal.SIGKILL


def _term_ignoring_child_script(pid_file, leader_tail: str) -> str:
    return (
        "import subprocess, sys, time\n"
        "child = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)'],\n"
        "    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,\n"
        ")\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(0.3)\n"
        f"{leader_tail}\n"
    )


async def _assert_dies(pid: int) -> None:
    for _ in range(100):
        if not _pid_alive(pid):
            return
        await asyncio.sleep(0.05)
    assert not _pid_alive(pid)


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
@pytest.mark.asyncio
async def test_timeout_kills_term_ignoring_child_with_redirected_streams(tmp_path):
    pid_file = tmp_path / "child.pid"
    agent = create_agent(_real_client(tmp_path, _term_ignoring_child_script(pid_file, "time.sleep(60)")))
    role = agent.client.get_role(None)
    with pytest.raises(Exception) as exc_info:
        await agent.run(role=role, prompt="", files=[], images=[], model="native")
    assert "timed out" in str(exc_info.value)
    assert exc_info.value.metadata["kill_escalated"] is True
    await _assert_dies(int(pid_file.read_text()))


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
@pytest.mark.asyncio
async def test_normal_exit_sweeps_leftover_group_members(tmp_path):
    pid_file = tmp_path / "child.pid"
    agent = create_agent(_real_client(tmp_path, _term_ignoring_child_script(pid_file, "sys.exit(0)"), timeout=30))
    role = agent.client.get_role(None)
    started = time.monotonic()
    with pytest.raises(CLIAgentError):
        await agent.run(role=role, prompt="", files=[], images=[], model="native")
    assert time.monotonic() - started < 15
    await _assert_dies(int(pid_file.read_text()))
