import json
from types import SimpleNamespace

import pytest

from providers.shared import ModelResponse, ProviderType
from systemprompts import GENERATE_CODE_PROMPT
from tools import chat as chat_module
from tools import consensus_cli_seats as seats_module
from tools.chat import ChatTool
from tools.consensus_cli_seats import (
    CLISeatConfigError,
    CLISeatResult,
    CLISeatRunError,
    effort_for_thinking_mode,
    parse_cli_seats,
)
from tools.shared.exceptions import ToolExecutionError


def _registry():
    return parse_cli_seats(
        {
            "seats": [
                {
                    "name": "astra-cli",
                    "client": "codex",
                    "model": "gpt-6-astra",
                    "effort": "max",
                    "fallback_model": "gpt-6-astra-pro",
                    "cli_enabled": True,
                    "host_dedup_frontends": ["codex"],
                },
                {
                    "name": "fable-cli",
                    "client": "claude",
                    "model": "fable",
                    "effort": "max",
                    "fallback_model": "fable-latest",
                    "cli_enabled": True,
                    "host_dedup_frontends": ["claude"],
                },
            ]
        },
        source="test",
    )


class _FakeProvider:
    def __init__(self, content="api answer", fail=False):
        self.calls = []
        self._content = content
        self._fail = fail

    def get_provider_type(self):
        return ProviderType.OPENROUTER

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if self._fail:
            raise RuntimeError("402 credits exhausted")
        return ModelResponse(
            content=self._content,
            model_name=kwargs["model_name"],
            metadata={"provider_model_name": f"openai/{kwargs['model_name']}"},
        )


def _context(provider, allow_code_generation=True):
    capabilities = SimpleNamespace(
        allow_code_generation=allow_code_generation,
        supports_extended_thinking=True,
        default_thinking_mode=None,
    )
    return SimpleNamespace(provider=provider, capabilities=capabilities, model_name="gpt-6-astra-pro")


@pytest.fixture
def seat_tool(monkeypatch):
    registry = _registry()
    monkeypatch.setattr(
        chat_module,
        "get_cli_seat_state",
        lambda resolver: (registry, None, frozenset(registry.all_names())),
    )
    monkeypatch.setenv("PAL_ASSISTANT_FRONTEND", "claude")
    tool = ChatTool()
    monkeypatch.setattr(tool, "prepare_prompt", _async_value("FULL PROMPT WITH FILES"))
    monkeypatch.setattr(tool, "get_validated_temperature", lambda request, context: (0.5, []))
    return tool


def _async_value(value):
    async def _inner(*args, **kwargs):
        return value

    return _inner


def _install_cli(monkeypatch, behaviour):
    calls = []

    async def fake_run(seat, *, system_prompt, prompt, budget_seconds, **framing):
        calls.append(
            {"seat": seat, "system_prompt": system_prompt, "prompt": prompt, "budget": budget_seconds, **framing}
        )
        return await behaviour(seat)

    monkeypatch.setattr(seats_module, "run_cli_seat", fake_run)
    return calls


def _arguments(tmp_path, **overrides):
    arguments = {
        "prompt": "Brainstorm options",
        "working_directory_absolute_path": str(tmp_path),
        "model": "astra-cli",
    }
    arguments.update(overrides)
    return arguments


async def _run(tool, arguments, provider):
    arguments = {**arguments, "_model_context": _context(provider)}
    result = await tool.execute(arguments)
    return json.loads(result[0].text)


class TestEffortMapping:
    @pytest.mark.parametrize(
        "mode, effort",
        [(None, "high"), ("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high"), ("max", "max")],
    )
    def test_thinking_mode_maps_to_effort(self, mode, effort):
        assert effort_for_thinking_mode(mode) == effort

    def test_unknown_thinking_mode_fails_loudly(self):
        with pytest.raises(CLISeatConfigError, match="no CLI effort"):
            effort_for_thinking_mode("ultra")


class TestSeatResolution:
    def test_seat_names_resolve_and_api_models_do_not(self, seat_tool):
        assert seat_tool.resolve_cli_seat("ASTRA-CLI").name == "astra-cli"
        assert seat_tool.resolve_cli_seat("gpt-6-astra") is None

    def test_declared_but_disabled_seat_fails_loudly(self, monkeypatch):
        monkeypatch.setattr(
            chat_module,
            "get_cli_seat_state",
            lambda resolver: (None, CLISeatConfigError("fallback does not resolve"), frozenset({"astra-cli"})),
        )
        with pytest.raises(ValueError, match="configured but disabled"):
            ChatTool().resolve_cli_seat("astra-cli")

    def test_schema_advertises_seats(self, seat_tool, monkeypatch):
        monkeypatch.setattr(chat_module, "load_cli_seats", _registry)
        description = seat_tool.get_input_schema()["properties"]["model"]["description"]
        assert "astra-cli" in description and "fable-cli" in description


class TestChatSeatExecution:
    async def test_cli_success_uses_high_effort_neutral_framing_and_records_metadata(
        self, seat_tool, monkeypatch, tmp_path
    ):
        calls = _install_cli(
            monkeypatch,
            _async_value(CLISeatResult(content="cli answer", model_used="gpt-6-astra", duration_seconds=2.0)),
        )
        provider = _FakeProvider()

        payload = await _run(seat_tool, _arguments(tmp_path), provider)

        assert payload["status"] == "success"
        assert payload["content"].startswith("cli answer")
        assert provider.calls == []
        call = calls[0]
        assert call["seat"].effort == "high"
        assert call["seat"].cli_timeout_s == chat_module.CHAT_SEAT_CLI_TIMEOUT_S
        assert call["budget"] <= chat_module.CHAT_SEAT_CLI_TIMEOUT_S
        assert call["instruction"] == seats_module.CLI_ADVISOR_INSTRUCTION
        assert call["header"] == seats_module.CLI_ADVISOR_HEADER
        assert call["queue_wait_s"] == chat_module.CHAT_SEAT_QUEUE_WAIT_S
        assert call["prompt"] == "FULL PROMPT WITH FILES"
        assert "continuation_id" not in call["prompt"]
        assert GENERATE_CODE_PROMPT.strip()[:40] not in call["system_prompt"]
        metadata = payload["metadata"]
        assert metadata["cli_seat"] == "astra-cli"
        assert metadata["transport"] == "cli"
        assert metadata["model_used"] == "gpt-6-astra"
        assert metadata["effort"] == "high"
        assert metadata["provider_used"] == "cli"
        assert metadata["same_vendor_as_host"] is False
        assert "continuation_offer" not in payload or payload["continuation_offer"] is None

    @pytest.mark.parametrize("mode, effort", [("minimal", "low"), ("max", "max")])
    async def test_requested_thinking_mode_sets_cli_effort(self, seat_tool, monkeypatch, tmp_path, mode, effort):
        calls = _install_cli(
            monkeypatch, _async_value(CLISeatResult(content="ok", model_used="gpt-6-astra", duration_seconds=1.0))
        )
        await _run(seat_tool, _arguments(tmp_path, thinking_mode=mode), _FakeProvider())
        assert calls[0]["seat"].effort == effort

    async def test_invalid_thinking_mode_fails_before_spawning(self, seat_tool, monkeypatch, tmp_path):
        calls = _install_cli(monkeypatch, _async_value(None))
        with pytest.raises(ToolExecutionError) as exc_info:
            await _run(seat_tool, _arguments(tmp_path, thinking_mode="ultra"), _FakeProvider())
        assert "no CLI effort" in exc_info.value.payload
        assert calls == []

    @pytest.mark.parametrize("reason", ["cli_error", "cli_timeout", "cli_busy"])
    async def test_cli_failure_falls_back_to_api_at_requested_thinking_mode(
        self, seat_tool, monkeypatch, tmp_path, reason
    ):
        async def fail(seat):
            raise CLISeatRunError("boom", reason=reason)

        _install_cli(monkeypatch, fail)
        provider = _FakeProvider()

        payload = await _run(seat_tool, _arguments(tmp_path, thinking_mode="medium"), provider)

        assert payload["content"].startswith("api answer")
        assert provider.calls[0]["model_name"] == "gpt-6-astra-pro"
        assert provider.calls[0]["thinking_mode"] == "medium"
        assert provider.calls[0]["images"] is None
        metadata = payload["metadata"]
        assert metadata["transport"] == "api_fallback"
        assert metadata["fallback_reason"] == reason
        assert metadata["model_used"] == "gpt-6-astra-pro"
        assert metadata["effort"] is None

    async def test_fallback_defaults_to_high_thinking(self, seat_tool, monkeypatch, tmp_path):
        async def fail(seat):
            raise CLISeatRunError("boom", reason="cli_error")

        _install_cli(monkeypatch, fail)
        provider = _FakeProvider()
        await _run(seat_tool, _arguments(tmp_path), provider)
        assert provider.calls[0]["thinking_mode"] == "high"

    async def test_both_backends_failing_is_a_loud_error(self, seat_tool, monkeypatch, tmp_path):
        async def fail(seat):
            raise CLISeatRunError("boom", reason="cli_error")

        _install_cli(monkeypatch, fail)
        with pytest.raises(ToolExecutionError) as exc_info:
            await _run(seat_tool, _arguments(tmp_path), _FakeProvider(fail=True))
        assert "API fallback" in exc_info.value.payload

    async def test_continuation_is_rejected(self, seat_tool, monkeypatch, tmp_path):
        calls = _install_cli(monkeypatch, _async_value(None))
        with pytest.raises(ToolExecutionError) as exc_info:
            await _run(seat_tool, _arguments(tmp_path, continuation_id="abc"), _FakeProvider())
        assert "single-shot" in exc_info.value.payload
        assert calls == []

    async def test_images_are_rejected(self, seat_tool, monkeypatch, tmp_path):
        calls = _install_cli(monkeypatch, _async_value(None))
        with pytest.raises(ToolExecutionError) as exc_info:
            await _run(seat_tool, _arguments(tmp_path, images=["/tmp/x.png"]), _FakeProvider())
        assert "images are not supported" in exc_info.value.payload
        assert calls == []

    async def test_generated_code_is_never_written(self, seat_tool, monkeypatch, tmp_path):
        reply = "Plan:\n<GENERATED-CODE>\nprint('x')\n</GENERATED-CODE>"
        _install_cli(
            monkeypatch, _async_value(CLISeatResult(content=reply, model_used="gpt-6-astra", duration_seconds=1.0))
        )
        payload = await _run(seat_tool, _arguments(tmp_path), _FakeProvider())
        assert not (tmp_path / "pal_generated.code").exists()
        assert "<GENERATED-CODE>" in payload["content"]

    async def test_explicit_same_vendor_seat_is_flagged(self, seat_tool, monkeypatch, tmp_path):
        _install_cli(
            monkeypatch, _async_value(CLISeatResult(content="ok", model_used="claude-fable-5-1", duration_seconds=1.0))
        )
        payload = await _run(seat_tool, _arguments(tmp_path, model="fable-cli"), _FakeProvider())
        assert payload["metadata"]["same_vendor_as_host"] is True

    async def test_api_model_path_is_unchanged(self, seat_tool, monkeypatch, tmp_path):
        calls = _install_cli(monkeypatch, _async_value(None))
        provider = _FakeProvider(content="plain answer")
        payload = await _run(seat_tool, _arguments(tmp_path, model="gpt-6-astra"), provider)
        assert payload["content"].startswith("plain answer")
        assert calls == []
        assert provider.calls[0]["model_name"] == "gpt-6-astra"


class TestSeatPromptTransport:
    async def test_large_prompt_goes_to_stdin_not_argv(self, monkeypatch):
        captured = {}

        class _Process:
            returncode = 0
            pid = None

            async def communicate(self, payload):
                captured["stdin"] = payload
                return (
                    b'{"type":"item.completed","item":{"type":"agent_message","text":"ok"}}\n'
                    b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n',
                    b"",
                )

            async def wait(self):
                return 0

        async def fake_exec(*args, **kwargs):
            captured["args"] = list(args)
            return _Process()

        monkeypatch.setattr(seats_module.asyncio, "create_subprocess_exec", fake_exec)
        monkeypatch.setattr("clink.agents.base.shutil.which", lambda name: f"/usr/bin/{name}")
        large_prompt = "x" * (200 * 1024)

        await seats_module.run_cli_seat(
            _registry().resolve("astra-cli"),
            system_prompt="SYS",
            prompt=large_prompt,
            budget_seconds=60,
            source_env={},
            instruction=seats_module.CLI_ADVISOR_INSTRUCTION,
            instructions_header=seats_module.CLI_ADVISOR_INSTRUCTIONS_HEADER,
            header=seats_module.CLI_ADVISOR_HEADER,
        )

        stdin_text = captured["stdin"].decode()
        assert large_prompt in stdin_text
        assert stdin_text.startswith(seats_module.CLI_ADVISOR_INSTRUCTION)
        assert "=== REQUEST ===" in stdin_text and "PROPOSAL" not in stdin_text
        assert all(len(arg) < 10_000 for arg in captured["args"])


class TestServerBoundary:
    async def test_seat_name_passes_model_check_with_fallback_context(self, monkeypatch):
        import server

        registry = _registry()
        monkeypatch.setattr(
            chat_module,
            "get_cli_seat_state",
            lambda resolver: (registry, None, frozenset(registry.all_names())),
        )
        requested = []

        def fake_provider_for_model(model_name):
            requested.append(model_name)
            return object() if model_name == "gpt-6-astra-pro" else None

        contexts = []

        def fake_model_context(model_name, option=None):
            contexts.append(model_name)
            return SimpleNamespace(capabilities=SimpleNamespace(context_window=400_000))

        captured = {}

        async def fake_execute(arguments):
            captured.update(arguments)
            return []

        monkeypatch.setattr(
            "providers.registry.ModelProviderRegistry.get_provider_for_model", staticmethod(fake_provider_for_model)
        )
        monkeypatch.setattr("utils.model_context.ModelContext", fake_model_context)
        monkeypatch.setattr(server.TOOLS["chat"], "execute", fake_execute)

        await server.handle_call_tool(
            "chat", {"prompt": "hi", "model": "astra-cli", "working_directory_absolute_path": "/tmp"}
        )

        assert requested == ["gpt-6-astra-pro"]
        assert contexts == ["gpt-6-astra-pro"]
        assert captured["_resolved_model_name"] == "astra-cli"


class TestReviewFixes:
    def test_seat_calls_get_no_websearch_instruction(self):
        tool = ChatTool()
        tool._active_cli_seat = None
        assert "WEB SEARCH CAPABILITY" in tool.get_websearch_instruction(tool.get_websearch_guidance())
        tool._active_cli_seat = _registry().resolve("astra-cli")
        try:
            assert tool.get_websearch_instruction(tool.get_websearch_guidance()) == ""
        finally:
            tool._active_cli_seat = None

    async def test_concurrent_plain_call_cannot_clobber_seat_state(self, seat_tool, monkeypatch, tmp_path):
        import asyncio

        seat_started = asyncio.Event()
        plain_done = asyncio.Event()

        async def slow_cli(seat):
            seat_started.set()
            await plain_done.wait()
            return CLISeatResult(
                content="seat answer\n<GENERATED-CODE>\nprint(1)\n</GENERATED-CODE>",
                model_used="gpt-6-astra",
                duration_seconds=1.0,
            )

        _install_cli(monkeypatch, slow_cli)

        async def seat_call():
            return await _run(seat_tool, _arguments(tmp_path), _FakeProvider())

        async def plain_call():
            await seat_started.wait()
            try:
                return await _run(seat_tool, _arguments(tmp_path, model="gpt-6-astra"), _FakeProvider("plain"))
            finally:
                plain_done.set()

        seat_payload, plain_payload = await asyncio.gather(
            asyncio.create_task(seat_call()), asyncio.create_task(plain_call())
        )

        assert seat_payload["metadata"]["transport"] == "cli"
        assert seat_payload["metadata"]["cli_seat"] == "astra-cli"
        assert "transport" not in (plain_payload.get("metadata") or {})
        assert not (tmp_path / "pal_generated.code").exists()

    async def test_api_fallback_is_bounded_by_the_deadline(self, monkeypatch):
        import asyncio
        import time

        seat = _registry().resolve("astra-cli")
        monkeypatch.setattr(seats_module, "MIN_FALLBACK_BUDGET_S", 0.1)

        async def slow_fallback():
            await asyncio.sleep(5)
            return {"status": "success", "text": "late"}

        async def failing_runner(*args, **kwargs):
            raise CLISeatRunError("boom", reason="cli_error")

        outcome = await seats_module.consult_seat_with_fallback(
            seat,
            system_prompt="s",
            build_prompt=lambda: "p",
            deadline_at=time.monotonic() + 0.5,
            api_fallback=slow_fallback,
            runner=failing_runner,
            enforce_fallback_deadline=True,
        )
        assert outcome.status == "error"
        assert "did not answer within" in outcome.error

    async def test_fallback_thinking_mode_is_null_without_thinking_support(self, seat_tool, monkeypatch, tmp_path):
        async def fail(seat):
            raise CLISeatRunError("boom", reason="cli_error")

        _install_cli(monkeypatch, fail)
        provider = _FakeProvider()
        context = _context(provider)
        context.capabilities.supports_extended_thinking = False
        result = await seat_tool.execute({**_arguments(tmp_path), "_model_context": context})
        payload = json.loads(result[0].text)
        assert provider.calls[0]["thinking_mode"] is None
        assert payload["metadata"]["fallback_thinking_mode"] is None

    def test_schema_lists_seats_without_provider_validation(self, monkeypatch):
        def refuse(resolver):
            raise AssertionError("schema generation must not validate against providers")

        monkeypatch.setattr(chat_module, "get_cli_seat_state", refuse)
        monkeypatch.setattr(chat_module, "load_cli_seats", _registry)
        description = ChatTool().get_input_schema()["properties"]["model"]["description"]
        assert "astra-cli" in description


class TestPrecommitFixes:
    def test_seat_with_option_suffix_is_rejected(self, seat_tool):
        with pytest.raises(ValueError, match="takes no ':high' suffix"):
            seat_tool.resolve_cli_seat("astra-cli:high")

    def test_server_guard_rejects_suffix_and_continuation_but_passes_plain_seat(self, seat_tool):
        import server

        assert "takes no ':high' suffix" in server.cli_seat_request_error(seat_tool, {"model": "astra-cli:high"})
        continuation_error = server.cli_seat_request_error(seat_tool, {"model": "astra-cli", "continuation_id": "t1"})
        assert "single-shot" in continuation_error
        assert server.cli_seat_request_error(seat_tool, {"model": "astra-cli"}) is None
        assert server.cli_seat_request_error(seat_tool, {"model": "gpt-6-astra", "continuation_id": "t1"}) is None

    async def test_seat_continuation_is_rejected_before_thread_reconstruction(self, monkeypatch):
        import server

        registry = _registry()
        monkeypatch.setattr(
            chat_module,
            "get_cli_seat_state",
            lambda resolver: (registry, None, frozenset(registry.all_names())),
        )

        async def must_not_reconstruct(arguments):
            raise AssertionError("seat continuation must be rejected before the thread is touched")

        monkeypatch.setattr(server, "reconstruct_thread_context", must_not_reconstruct)

        with pytest.raises(ToolExecutionError) as exc_info:
            await server.handle_call_tool(
                "chat",
                {
                    "prompt": "follow up",
                    "model": "astra-cli",
                    "continuation_id": "existing-thread",
                    "working_directory_absolute_path": "/tmp",
                },
            )
        assert "single-shot" in exc_info.value.payload


class TestSeatConcurrency:
    def test_default_allows_three_concurrent_runs_per_client(self, monkeypatch):
        monkeypatch.delenv("CLI_SEAT_MAX_CONCURRENT", raising=False)
        assert seats_module.cli_seat_concurrency() == 3

    @pytest.mark.parametrize("raw", ["0", "-1", "two"])
    def test_invalid_concurrency_fails_loudly(self, monkeypatch, raw):
        monkeypatch.setenv("CLI_SEAT_MAX_CONCURRENT", raw)
        with pytest.raises(CLISeatConfigError, match="CLI_SEAT_MAX_CONCURRENT"):
            seats_module.cli_seat_concurrency()

    def test_invalid_concurrency_disables_seats_instead_of_silent_fallback(self, monkeypatch):
        monkeypatch.setenv("CLI_SEAT_MAX_CONCURRENT", "two")
        seats_module.clear_cli_seat_state_cache()
        try:
            registry, error, declared_names = seats_module.get_cli_seat_state(lambda name: True)
        finally:
            seats_module.clear_cli_seat_state_cache()
        assert registry is None
        assert "CLI_SEAT_MAX_CONCURRENT" in str(error)
        assert "astra-cli" in declared_names

    async def test_chat_seat_and_consensus_leg_run_side_by_side(self, monkeypatch):
        import asyncio

        monkeypatch.delenv("CLI_SEAT_MAX_CONCURRENT", raising=False)
        semaphore = seats_module._client_semaphore("codex-parallel")
        assert await seats_module._acquire_within(semaphore, 0.05) is True
        assert await seats_module._acquire_within(semaphore, 0.05) is True
        assert await seats_module._acquire_within(semaphore, 0.05) is True
        assert await seats_module._acquire_within(semaphore, 0.05) is False
        for _ in range(3):
            semaphore.release()
        await asyncio.sleep(0)
