import asyncio
import json
import time

import pytest

from tools import consensus_cli_seats as seats_module
from tools.consensus import ConsensusTool
from tools.consensus_cli_seats import (
    CLISeatConfigError,
    CLISeatResult,
    CLISeatRunError,
    assert_argv_allowed,
    build_isolated_environment,
    build_review_client,
    build_review_prompt,
    parse_cli_seats,
    run_cli_seat,
    validate_against_providers,
)


def _seat_payload(**overrides):
    astra = {
        "name": "astra-cli",
        "client": "codex",
        "model": "gpt-6-astra",
        "effort": "max",
        "fallback_model": "gpt-6-astra-pro",
        "cli_enabled": True,
        "host_dedup_frontends": ["codex"],
    }
    fable = {
        "name": "fable-cli",
        "client": "claude",
        "model": "fable",
        "effort": "max",
        "fallback_model": "fable-latest",
        "cli_enabled": True,
        "host_dedup_frontends": ["claude"],
    }
    astra.update(overrides.pop("astra", {}))
    fable.update(overrides.pop("fable", {}))
    return {"seats": [astra, fable]}


def _registry(**overrides):
    return parse_cli_seats(_seat_payload(**overrides), source="test")


class TestSeatConfig:
    def test_valid_config_resolves_names_case_insensitively(self):
        registry = _registry(astra={"aliases": ["astra-sub"]})
        assert registry.resolve("ASTRA-CLI").client == "codex"
        assert registry.resolve("astra-sub").name == "astra-cli"
        assert registry.resolve("gpt-6-astra-pro") is None

    @pytest.mark.parametrize(
        "override, message",
        [
            ({"effort": "ultra"}, "effort"),
            ({"client": "gemini"}, "client"),
            ({"host_dedup_frontends": ["claud"]}, "host_dedup_frontends"),
            ({"fallback_model": ""}, "fallback_model"),
            ({"model": "-m"}, "model"),
            ({"cli_timeout_s": 30, "min_cli_budget_s": 60}, "min_cli_budget_s"),
            ({"cli_enabled": "yes"}, "cli_enabled"),
            ({"aliases": ["fable-cli"]}, "more than once"),
        ],
    )
    def test_invalid_config_fails_loudly(self, override, message):
        with pytest.raises(CLISeatConfigError, match=message):
            _registry(astra=override)

    def test_provider_validation_rejects_unresolvable_fallback_and_shadowing(self):
        registry = _registry()
        with pytest.raises(CLISeatConfigError, match="does not resolve"):
            validate_against_providers(registry, lambda name: name == "fable-latest")
        with pytest.raises(CLISeatConfigError, match="shadows"):
            validate_against_providers(registry, lambda name: name in {"gpt-6-astra-pro", "fable-latest", "fable-cli"})
        validate_against_providers(registry, lambda name: name in {"gpt-6-astra-pro", "fable-latest"})

    def test_packaged_default_config_loads(self):
        registry = seats_module.load_cli_seats()
        assert {seat.name for seat in registry.seats} == {"astra-cli", "fable-cli"}
        assert all(seat.effort == "max" for seat in registry.seats)


class TestReviewInvocation:
    def test_claude_review_argv_is_isolated(self, tmp_path):
        seat = _registry().resolve("fable-cli")
        client = build_review_client(seat, working_dir=str(tmp_path), timeout_seconds=120)
        args = client.config_args
        for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"):
            assert flag in args
        assert args[args.index("--tools") + 1] == ""
        assert args[args.index("--mcp-config") + 1] == '{"mcpServers":{}}'
        assert args[args.index("--effort") + 1] == "max"
        assert client.timeout_seconds == 120
        assert client.working_dir == tmp_path
        assert_argv_allowed([*client.executable, *client.internal_args, *args])

    def test_codex_review_argv_is_isolated(self, tmp_path):
        seat = _registry().resolve("astra-cli")
        client = build_review_client(seat, working_dir=str(tmp_path), timeout_seconds=90)
        args = client.config_args
        assert args[args.index("--sandbox") + 1] == "read-only"
        assert args[args.index("--cd") + 1] == str(tmp_path)
        for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral", "--skip-git-repo-check"):
            assert flag in args
        for value in (
            'model_reasoning_effort="max"',
            'web_search="disabled"',
            "project_doc_max_bytes=0",
            "features.shell_tool=false",
            "features.multi_agent=false",
            "features.apps=false",
            "features.hooks=false",
        ):
            assert value in args
        assert "--dangerously-bypass-approvals-and-sandbox" not in args
        assert_argv_allowed([*client.executable, *client.internal_args, *args])

    @pytest.mark.parametrize(
        "bad_arg",
        [
            "--dangerously-bypass-approvals-and-sandbox",
            "acceptEdits",
            'web_search="live"',
            "ultra",
            "danger-full-access",
        ],
    )
    def test_argv_denylist(self, bad_arg):
        with pytest.raises(CLISeatRunError) as exc_info:
            assert_argv_allowed(["codex", "exec", bad_arg])
        assert exc_info.value.reason == "unsafe_argv"

    def test_environment_is_allowlisted(self):
        env = build_isolated_environment(
            {
                "PATH": "/usr/bin",
                "HOME": "/home/x",
                "CLAUDE_CONFIG_DIR": "/home/x/.claude-alt",
                "OPENROUTER_API_KEY": "secret",
                "ANTHROPIC_API_KEY": "secret",
                "MCP_TIMEOUT": "1",
                "PAL_ASSISTANT_FRONTEND": "claude",
            }
        )
        assert env == {"PATH": "/usr/bin", "HOME": "/home/x", "CLAUDE_CONFIG_DIR": "/home/x/.claude-alt"}


class _FakeProcess:
    def __init__(self, stdout: bytes, returncode: int = 0):
        self._stdout = stdout
        self.returncode = returncode
        self.pid = None
        self.stdin_payload: bytes | None = None

    async def communicate(self, payload):
        self.stdin_payload = payload
        return self._stdout, b""

    async def wait(self):
        return self.returncode


def _patch_spawn(monkeypatch, process):
    captured = {}

    async def fake_exec(*args, **kwargs):
        captured["args"] = list(args)
        captured["kwargs"] = kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr("clink.agents.base.shutil.which", lambda name: f"/usr/bin/{name}")
    return captured


CODEX_OK = (
    b'{"type":"item.completed","item":{"type":"agent_message","text":"Verdict: ship it"}}\n'
    b'{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":5}}\n'
)


class TestRunCliSeat:
    async def test_prompt_goes_to_stdin_with_isolated_env_and_model(self, monkeypatch):
        process = _FakeProcess(CODEX_OK)
        captured = _patch_spawn(monkeypatch, process)
        seat = _registry().resolve("astra-cli")

        result = await run_cli_seat(
            seat,
            system_prompt="STANCE-PROMPT",
            prompt="PROPOSAL-TEXT",
            budget_seconds=60,
            source_env={"PATH": "/usr/bin", "HOME": "/h", "OPENROUTER_API_KEY": "secret"},
        )

        assert result.content == "Verdict: ship it"
        assert result.model_used == "gpt-6-astra"
        args = captured["args"]
        assert args[args.index("-m") + 1] == "gpt-6-astra"
        assert "PROPOSAL-TEXT" not in " ".join(args)
        stdin_text = process.stdin_payload.decode()
        assert "STANCE-PROMPT" in stdin_text and "PROPOSAL-TEXT" in stdin_text
        assert stdin_text.startswith(seats_module.CLI_REVIEW_INSTRUCTION)
        assert captured["kwargs"]["env"] == {"PATH": "/usr/bin", "HOME": "/h"}
        assert captured["kwargs"]["cwd"].startswith(seats_module.tempfile.gettempdir())

    async def test_nonzero_exit_is_a_failure_even_when_output_parses(self, monkeypatch):
        _patch_spawn(monkeypatch, _FakeProcess(CODEX_OK, returncode=1))
        seat = _registry().resolve("astra-cli")
        with pytest.raises(CLISeatRunError) as exc_info:
            await run_cli_seat(seat, system_prompt="s", prompt="p", budget_seconds=60, source_env={})
        assert exc_info.value.reason == "cli_nonzero_exit"

    async def test_empty_output_is_a_failure(self, monkeypatch):
        _patch_spawn(monkeypatch, _FakeProcess(b'{"type":"turn.completed"}\n'))
        seat = _registry().resolve("astra-cli")
        with pytest.raises(CLISeatRunError) as exc_info:
            await run_cli_seat(seat, system_prompt="s", prompt="p", budget_seconds=60, source_env={})
        assert exc_info.value.reason in {"cli_empty_output", "cli_error"}

    def test_review_prompt_layout(self):
        text = build_review_prompt("SYS", "BODY")
        assert text.index("SYS") < text.index("BODY")


def _tool_with_seats(monkeypatch, registry=None):
    tool = ConsensusTool()
    active = registry or _registry()
    monkeypatch.setattr(tool, "_cli_seat_state", lambda: (active, None, frozenset(active.all_names())), raising=False)
    return tool


class _Request:
    step = "Evaluate the proposal"
    relevant_files = []
    images = []


def _install_api(monkeypatch, tool, status="success"):
    calls = []

    async def fake_api(model_config, request, *, original_proposal=None, relevant_files=None, images=None):
        calls.append(model_config["model"])
        if status == "success":
            return {
                "model": model_config["model"],
                "stance": model_config.get("stance", "neutral"),
                "status": "success",
                "verdict": f"api verdict from {model_config['model']}",
                "metadata": {
                    "provider": "openrouter",
                    "requested_model_name": model_config["model"],
                    "model_name": f"vendor/{model_config['model']}",
                    "provider_model_name": f"vendor/{model_config['model']}",
                },
            }
        return {"model": model_config["model"], "stance": "neutral", "status": "error", "error": "402 credits"}

    monkeypatch.setattr(tool, "_consult_api_model", fake_api)
    monkeypatch.setattr(tool, "_build_blinded_prompt", lambda *args, **kwargs: "BLINDED")
    return calls


def _install_cli(monkeypatch, behaviour):
    calls = []

    async def fake_run(seat, *, system_prompt, prompt, budget_seconds):
        calls.append({"seat": seat.name, "budget": budget_seconds, "prompt": prompt})
        return await behaviour(seat)

    monkeypatch.setattr("tools.consensus.run_cli_seat", fake_run)
    return calls


class TestConsultCliSeat:
    async def test_cli_success_metadata_contract(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        api_calls = _install_api(monkeypatch, tool)

        async def ok(seat):
            return CLISeatResult(content="cli verdict", model_used="gpt-6-astra", duration_seconds=1.0)

        cli_calls = _install_cli(monkeypatch, ok)
        result = await tool._consult_model(
            {"model": "astra-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 1500
        )

        assert result["status"] == "success" and result["model"] == "astra-cli"
        metadata = result["metadata"]
        assert metadata["requested_model_name"] == "astra-cli"
        assert metadata["backend"] == "cli"
        assert metadata["provider"] == "cli"
        assert metadata["model_name"] == "gpt-6-astra"
        assert metadata["effective_reasoning_effort"] == "max"
        assert metadata["attempts"][0]["backend"] == "cli"
        assert api_calls == []
        assert cli_calls[0]["budget"] == pytest.approx(1080, abs=1)
        assert cli_calls[0]["prompt"] == "BLINDED"
        assert json.loads(json.dumps(result)) == result

    async def test_cli_failure_falls_back_to_api(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        api_calls = _install_api(monkeypatch, tool)

        async def fail(seat):
            raise CLISeatRunError("usage limit reached", reason="cli_error")

        _install_cli(monkeypatch, fail)
        result = await tool._consult_model(
            {"model": "fable-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 1500
        )

        assert result["status"] == "success"
        assert result["model"] == "fable-cli"
        assert result["verdict"] == "api verdict from fable-latest"
        metadata = result["metadata"]
        assert metadata["requested_model_name"] == "fable-cli"
        assert metadata["backend"] == "api_fallback"
        assert metadata["fallback_reason"] == "cli_error"
        assert metadata["effective_reasoning_effort"] is None
        assert metadata["provider"] == "openrouter"
        assert [attempt["backend"] for attempt in metadata["attempts"]] == ["cli", "api_fallback"]
        assert api_calls == ["fable-latest"]

    async def test_model_unavailable_is_classified(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        _install_api(monkeypatch, tool)

        async def fail(seat):
            raise CLISeatRunError("exited 1: error: model not found: gpt-6-astra", reason="cli_nonzero_exit")

        _install_cli(monkeypatch, fail)
        result = await tool._consult_model(
            {"model": "astra-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 1500
        )
        assert result["metadata"]["fallback_reason"] == "cli_model_unavailable"

    async def test_both_backends_failing_is_an_error_leg_with_metadata(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        _install_api(monkeypatch, tool, status="error")

        async def fail(seat):
            raise CLISeatRunError("boom", reason="cli_timeout")

        _install_cli(monkeypatch, fail)
        result = await tool._consult_model(
            {"model": "astra-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 1500
        )
        assert result["status"] == "error"
        assert "cli_timeout" in result["error"] and "402 credits" in result["error"]
        assert result["metadata"]["requested_model_name"] == "astra-cli"
        assert result["metadata"]["backend"] == "api_fallback"

    async def test_no_fallback_when_panel_time_is_nearly_gone(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        api_calls = _install_api(monkeypatch, tool)
        cli_calls = _install_cli(monkeypatch, lambda seat: pytest.fail("CLI must not start"))

        result = await tool._consult_model(
            {"model": "astra-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 10
        )
        assert result["status"] == "error"
        assert result["metadata"]["fallback_reason"] == "cli_budget_exhausted"
        assert api_calls == [] and cli_calls == []

    async def test_small_budget_goes_straight_to_fallback(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        api_calls = _install_api(monkeypatch, tool)
        cli_calls = _install_cli(monkeypatch, lambda seat: pytest.fail("CLI must not start"))

        result = await tool._consult_model(
            {"model": "astra-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 450
        )
        assert result["metadata"]["fallback_reason"] == "cli_budget_exhausted"
        assert result["metadata"]["backend"] == "api_fallback"
        assert api_calls == ["gpt-6-astra-pro"] and cli_calls == []

    async def test_images_and_disabled_seats_use_fallback(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch, _registry(fable={"cli_enabled": False, "disabled_reason": "probe"}))
        _install_api(monkeypatch, tool)
        _install_cli(monkeypatch, lambda seat: pytest.fail("CLI must not start"))

        with_images = await tool._consult_model(
            {"model": "astra-cli"},
            _Request(),
            original_proposal="p",
            images=["/tmp/x.png"],
            deadline_at=time.monotonic() + 1500,
        )
        assert with_images["metadata"]["fallback_reason"] == "images_unsupported_by_cli"

        disabled = await tool._consult_model(
            {"model": "fable-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 1500
        )
        assert disabled["metadata"]["fallback_reason"] == "cli_isolation_unproven"
        assert disabled["metadata"]["cli_error"] == "probe"

    async def test_cancellation_never_triggers_fallback(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        api_calls = _install_api(monkeypatch, tool)
        started = asyncio.Event()

        async def hang(seat):
            started.set()
            await asyncio.sleep(3600)

        _install_cli(monkeypatch, hang)
        task = asyncio.create_task(
            tool._consult_model(
                {"model": "astra-cli"}, _Request(), original_proposal="p", deadline_at=time.monotonic() + 1500
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert api_calls == []

    async def test_sequential_mode_without_deadline_gets_full_cli_timeout(self, monkeypatch):
        tool = _tool_with_seats(monkeypatch)
        _install_api(monkeypatch, tool)

        async def ok(seat):
            return CLISeatResult(content="v", model_used="fable", duration_seconds=1.0)

        cli_calls = _install_cli(monkeypatch, ok)
        await tool._consult_model({"model": "fable-cli"}, _Request(), original_proposal="p")
        assert cli_calls[0]["budget"] == pytest.approx(1080, abs=1)


class TestSeatHostPolicyAndPanel:
    @pytest.mark.parametrize("frontend, skipped", [("claude", "fable-cli"), ("codex", "astra-cli")])
    def test_host_dedup_skips_matching_seat(self, monkeypatch, frontend, skipped):
        from utils import client_info

        monkeypatch.setenv("PAL_ASSISTANT_FRONTEND", frontend)
        monkeypatch.setattr(client_info, "_client_info_cache", None)
        tool = _tool_with_seats(monkeypatch)
        filtered, skipped_models, _ = tool._apply_host_model_skip_policy(
            [{"model": "astra-cli"}, {"model": "fable-cli"}, {"model": "pro"}]
        )
        assert [item["model"] for item in skipped_models] == [skipped]
        assert skipped not in [item["model"] for item in filtered]

    async def test_timed_out_seat_leg_reports_cli_termination(self, monkeypatch):
        monkeypatch.setenv("CONSENSUS_PANEL_DEADLINE_S", "0.05")
        tool = _tool_with_seats(monkeypatch)

        async def consult(model_config, request, **kwargs):
            if model_config["model"] == "astra-cli":
                await asyncio.sleep(5)
            return {"model": model_config["model"], "stance": "neutral", "status": "success", "verdict": "v"}

        monkeypatch.setattr(tool, "_consult_model", consult)
        response = await tool.execute(
            {
                "step": "Evaluate",
                "step_number": 1,
                "total_steps": 1,
                "next_step_required": False,
                "findings": "f",
                "mode": "parallel",
                "models": [{"model": "astra-cli"}, {"model": "alpha"}, {"model": "beta"}],
            }
        )
        payload = json.loads(response[0].text)
        leg = next(item for item in payload["accumulated_responses"] if item["model"] == "astra-cli")
        assert leg["status"] == "timed_out"
        assert "codex CLI process group was terminated" in leg["error"]
        assert leg["metadata"]["requested_model_name"] == "astra-cli"

    async def test_invalid_seat_config_fails_requests_naming_a_seat(self, monkeypatch):
        tool = ConsensusTool()
        error = CLISeatConfigError("fallback_model 'x' does not resolve")
        monkeypatch.setattr(tool, "_cli_seat_state", lambda: (None, error, frozenset({"astra-cli"})), raising=False)
        with pytest.raises(Exception, match="CLI seat configuration is invalid"):
            await tool.execute(
                {
                    "step": "Evaluate",
                    "step_number": 1,
                    "total_steps": 1,
                    "next_step_required": False,
                    "findings": "f",
                    "mode": "parallel",
                    "models": [{"model": "astra-cli"}, {"model": "alpha"}],
                }
            )


class TestReviewHardening:
    def test_cli_enabled_defaults_off_and_unknown_keys_are_rejected(self):
        payload = _seat_payload()
        del payload["seats"][0]["cli_enabled"]
        registry = parse_cli_seats(payload, source="test")
        assert registry.resolve("astra-cli").cli_enabled is False
        with pytest.raises(CLISeatConfigError, match="unknown keys"):
            _registry(astra={"cli_enabeld": True})
        with pytest.raises(CLISeatConfigError, match="unknown top-level keys"):
            parse_cli_seats({**_seat_payload(), "extra": 1}, source="test")

    def test_environment_allowlist_excludes_config_redirects(self):
        env = build_isolated_environment(
            {"HOME": "/h", "XDG_CONFIG_HOME": "/x", "XDG_DATA_HOME": "/y", "SHELL": "/bin/zsh", "XDG_RUNTIME_DIR": "/r"}
        )
        assert env == {"HOME": "/h", "XDG_RUNTIME_DIR": "/r"}

    def test_tool_activity_is_detected_per_cli(self):
        codex = _registry().resolve("astra-cli")
        claude = _registry().resolve("fable-cli")
        benign = {
            "events": [
                {"item": {"type": "error"}},
                {"item": {"type": "agent_message"}},
                {"item": {"type": "todo_list"}},
            ]
        }
        assert seats_module.detect_tool_activity(codex, benign) is None
        shell = {"events": [{"item": {"type": "command_execution"}}]}
        assert "command_execution" in seats_module.detect_tool_activity(codex, shell)
        assert seats_module.detect_tool_activity(claude, {"raw": {"num_turns": 1}}) is None
        assert "num_turns=3" in seats_module.detect_tool_activity(claude, {"raw": {"num_turns": 3}})

    async def test_codex_tool_activity_fails_the_leg(self, monkeypatch):
        stdout = CODEX_OK + b'{"type":"item.completed","item":{"type":"command_execution","command":"ls"}}\n'
        _patch_spawn(monkeypatch, _FakeProcess(stdout))
        seat = _registry().resolve("astra-cli")
        with pytest.raises(CLISeatRunError) as exc_info:
            await run_cli_seat(seat, system_prompt="s", prompt="p", budget_seconds=60, source_env={})
        assert exc_info.value.reason == "cli_tool_activity"

    async def test_semaphore_permit_survives_timeouts_and_cancellation(self, monkeypatch):
        monkeypatch.setenv("CLI_SEAT_MAX_CONCURRENT", "1")
        semaphore = seats_module._client_semaphore("codex-test")
        await semaphore.acquire()
        assert await seats_module._acquire_within(semaphore, 0.01) is False
        waiter = asyncio.create_task(seats_module._acquire_within(semaphore, 5))
        await asyncio.sleep(0.01)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        semaphore.release()
        assert await seats_module._acquire_within(semaphore, 0.1) is True
        semaphore.release()
        assert not semaphore.locked()

    def test_model_unavailable_needs_a_model_mention(self):
        generic = CLISeatRunError("flag --foo not supported", reason="cli_error")
        specific = CLISeatRunError("the model gpt-9 does not exist", reason="cli_error")
        assert seats_module.classify_cli_failure(generic) == "cli_error"
        assert seats_module.classify_cli_failure(specific) == "cli_model_unavailable"

    async def test_timed_out_seat_leg_keeps_recorded_attempts(self, monkeypatch):
        monkeypatch.setenv("CONSENSUS_PANEL_DEADLINE_S", "0.2")
        tool = _tool_with_seats(monkeypatch)
        _install_api(monkeypatch, tool)

        async def hang(seat):
            await asyncio.sleep(30)

        _install_cli(monkeypatch, hang)
        monkeypatch.setattr("tools.consensus_cli_seats.cli_budget_seconds", lambda seat, deadline_at, now: 120.0)
        response = await tool.execute(
            {
                "step": "Evaluate",
                "step_number": 1,
                "total_steps": 1,
                "next_step_required": False,
                "findings": "f",
                "mode": "parallel",
                "models": [{"model": "astra-cli"}, {"model": "alpha"}],
            }
        )
        payload = json.loads(response[0].text)
        leg = next(item for item in payload["accumulated_responses"] if item["model"] == "astra-cli")
        assert leg["status"] == "timed_out"
        assert leg["metadata"]["backend"] == "cli"
        assert leg["metadata"]["duration_seconds"] is not None
        assert not hasattr(tool, "_seat_leg_progress")
