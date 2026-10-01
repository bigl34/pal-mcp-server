"""CLI-backed consensus seats: run a panel leg through a subscription CLI with an API fallback.

A seat such as ``astra-cli`` names a CLI client, the model and reasoning effort to request
from it, and the API model to fall back to. Seat runs never reuse the shipped clink client
arguments; they build an isolated, prompt-only review invocation (no tools, no MCP servers,
no hooks or project instruction files, empty working directory, allowlisted environment).
"""

from __future__ import annotations

import asyncio
import importlib.resources
import json
import logging
import math
import os
import re
import tempfile
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clink import get_registry
from clink.agents import AgentOutput, CLIAgentError, create_agent
from clink.agents.base import PROCESS_TERMINATION_GRACE_SECONDS
from clink.effort import REASONING_EFFORTS, effort_args
from clink.models import ResolvedCLIClient, ResolvedCLIRole

logger = logging.getLogger(__name__)

CLI_SEATS_CONFIG_ENV = "CLI_SEATS_CONFIG_PATH"
CLI_SEAT_CONCURRENCY_ENV = "CLI_SEAT_MAX_CONCURRENT"
DEFAULT_CLI_SEAT_CONCURRENCY = 3
CLI_SEATS_DEFAULT_FILENAME = "cli_seats.json"
SUPPORTED_SEAT_CLIENTS = ("codex", "claude")
KNOWN_FRONTENDS = ("claude", "codex", "opencode", "cursor")
SEAT_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{1,63}$")
DEFAULT_CLI_TIMEOUT_S = 1080.0
DEFAULT_FALLBACK_RESERVE_S = 400.0
DEFAULT_MIN_CLI_BUDGET_S = 60.0
CLEANUP_SLACK_S = 10.0 + PROCESS_TERMINATION_GRACE_SECONDS

SEAT_KEYS = frozenset(
    {
        "name",
        "client",
        "model",
        "effort",
        "fallback_model",
        "host_dedup_frontends",
        "aliases",
        "cli_enabled",
        "disabled_reason",
        "cli_timeout_s",
        "fallback_reserve_s",
        "min_cli_budget_s",
    }
)
TOP_LEVEL_KEYS = frozenset({"seats", "_README"})
CODEX_ALLOWED_ITEM_TYPES = frozenset({"agent_message", "reasoning", "error", "todo_list"})

ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "XDG_RUNTIME_DIR",
    "CLAUDE_CONFIG_DIR",
    "CODEX_HOME",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)

CLAUDE_REVIEW_ARGS = (
    "--safe-mode",
    "--tools",
    "",
    "--permission-mode",
    "dontAsk",
    "--permission-prompts",
    "none",
    "--strict-mcp-config",
    "--mcp-config",
    '{"mcpServers":{}}',
    "--disable-slash-commands",
    "--no-session-persistence",
)

CODEX_DISABLED_FEATURES = (
    "shell_tool",
    "unified_exec",
    "apps",
    "multi_agent",
    "multi_agent_v2",
    "goals",
    "image_generation",
    "view_image",
    "plugins",
    "remote_plugin",
    "hooks",
    "browser_use",
    "browser_use_external",
    "computer_use",
    "sleep_tool",
    "skill_search",
    "tool_suggest",
    "code_mode_host",
    "in_app_browser",
    "skill_mcp_dependency_install",
    "workspace_dependencies",
    "memories",
    "realtime_conversation",
    "worktrees",
    "collaboration_modes",
    "default_mode_request_user_input",
)

ARGV_DENYLIST = (
    "--dangerously-bypass-approvals-and-sandbox",
    "--dangerously-skip-permissions",
    "--allow-dangerously-skip-permissions",
    "acceptEdits",
    "bypassPermissions",
    "danger-full-access",
    "workspace-write",
    'web_search="live"',
    'model_reasoning_effort="ultra"',
)
ARGV_DENIED_EXACT = ("ultra",)

CLI_REVIEW_INSTRUCTION = (
    "You are one reviewer on a blinded multi-model consensus panel. Answer only from the text of this "
    "message. Do not run commands, read files, browse, call tools, or delegate to other agents."
)
CLI_REVIEW_INSTRUCTIONS_HEADER = "REVIEWER INSTRUCTIONS"
CLI_REVIEW_HEADER = "PROPOSAL"
CLI_ADVISOR_INSTRUCTION = (
    "You are an independent advisor answering a single request from another AI agent. Answer only from the "
    "text of this message. Do not run commands, read files, browse, call tools, or delegate to other agents."
)
CLI_ADVISOR_INSTRUCTIONS_HEADER = "ADVISOR INSTRUCTIONS"
CLI_ADVISOR_HEADER = "REQUEST"

LEG_ERROR_TEXT_LIMIT = 300
MIN_FALLBACK_BUDGET_S = 30.0
CLI_MODEL_UNAVAILABLE_MARKERS = (
    "model not found",
    "unknown model",
    "model_not_found",
    "does not exist",
    "not supported",
    "invalid model",
    "unsupported model",
)
THINKING_MODE_TO_EFFORT = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
}


class CLISeatConfigError(ValueError):
    """Raised when the CLI seat configuration is invalid."""


class CLISeatRunError(RuntimeError):
    """Raised when a CLI seat run does not produce a usable verdict."""

    def __init__(self, message: str, *, reason: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class CLISeat:
    """One CLI-backed consensus seat."""

    name: str
    client: str
    model: str
    effort: str
    fallback_model: str
    host_dedup_frontends: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    cli_enabled: bool = False
    disabled_reason: str | None = None
    cli_timeout_s: float = DEFAULT_CLI_TIMEOUT_S
    fallback_reserve_s: float = DEFAULT_FALLBACK_RESERVE_S
    min_cli_budget_s: float = DEFAULT_MIN_CLI_BUDGET_S

    def names(self) -> tuple[str, ...]:
        return (self.name, *self.aliases)


@dataclass
class CLISeatRegistry:
    """Seats keyed by lower-cased name and alias."""

    seats: tuple[CLISeat, ...] = ()
    source: str = ""
    _by_name: dict[str, CLISeat] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for seat in self.seats:
            for seat_name in seat.names():
                self._by_name[seat_name.lower()] = seat

    def resolve(self, model_name: str | None) -> CLISeat | None:
        if not model_name:
            return None
        return self._by_name.get(str(model_name).strip().lower())

    def all_names(self) -> list[str]:
        return sorted(self._by_name)


def _positive_number(raw: Any, field_name: str, seat_name: str, default: float) -> float:
    if raw is None:
        return default
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw) or raw <= 0:
        raise CLISeatConfigError(f"CLI seat '{seat_name}': {field_name} must be a positive number")
    return float(raw)


def _string_list(raw: Any, field_name: str, seat_name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        raise CLISeatConfigError(f"CLI seat '{seat_name}': {field_name} must be a list of non-empty strings")
    return tuple(item.strip().lower() for item in raw)


def parse_cli_seats(payload: Any, *, source: str = "") -> CLISeatRegistry:
    """Validate a decoded ``cli_seats.json`` payload and build the registry."""
    if not isinstance(payload, dict) or not isinstance(payload.get("seats"), list):
        raise CLISeatConfigError(f"CLI seat config {source or ''} must be an object with a 'seats' list".strip())
    unknown_top_level = sorted(set(payload) - TOP_LEVEL_KEYS)
    if unknown_top_level:
        raise CLISeatConfigError(f"CLI seat config has unknown top-level keys {unknown_top_level}")

    seats: list[CLISeat] = []
    seen_names: set[str] = set()
    for raw in payload["seats"]:
        if not isinstance(raw, dict):
            raise CLISeatConfigError("Each CLI seat must be an object")
        unknown_keys = sorted(set(raw) - SEAT_KEYS)
        if unknown_keys:
            raise CLISeatConfigError(f"CLI seat {raw.get('name')!r} has unknown keys {unknown_keys}")
        name = str(raw.get("name", "")).strip().lower()
        if not SEAT_NAME_PATTERN.match(name):
            raise CLISeatConfigError(f"CLI seat name {raw.get('name')!r} is invalid")

        client = str(raw.get("client", "")).strip().lower()
        if client not in SUPPORTED_SEAT_CLIENTS:
            raise CLISeatConfigError(
                f"CLI seat '{name}': client must be one of {', '.join(SUPPORTED_SEAT_CLIENTS)}; got {client!r}"
            )

        model = str(raw.get("model", "")).strip()
        if not model or model.startswith("-") or any(char.isspace() for char in model):
            raise CLISeatConfigError(f"CLI seat '{name}': model is missing or malformed")

        effort = str(raw.get("effort", "")).strip().lower()
        if effort not in REASONING_EFFORTS:
            raise CLISeatConfigError(
                f"CLI seat '{name}': effort must be one of {', '.join(REASONING_EFFORTS)}; got {effort!r}"
            )

        fallback_model = str(raw.get("fallback_model", "")).strip()
        if not fallback_model:
            raise CLISeatConfigError(f"CLI seat '{name}': fallback_model is required")

        frontends = _string_list(raw.get("host_dedup_frontends"), "host_dedup_frontends", name)
        unknown_frontends = [frontend for frontend in frontends if frontend not in KNOWN_FRONTENDS]
        if unknown_frontends:
            raise CLISeatConfigError(
                f"CLI seat '{name}': unknown host_dedup_frontends {unknown_frontends}; "
                f"expected {', '.join(KNOWN_FRONTENDS)}"
            )

        aliases = _string_list(raw.get("aliases"), "aliases", name)
        for seat_name in (name, *aliases):
            if seat_name in seen_names:
                raise CLISeatConfigError(f"CLI seat name or alias '{seat_name}' is defined more than once")
            seen_names.add(seat_name)

        cli_enabled = raw.get("cli_enabled", False)
        if not isinstance(cli_enabled, bool):
            raise CLISeatConfigError(f"CLI seat '{name}': cli_enabled must be true or false")
        disabled_reason = raw.get("disabled_reason")
        if disabled_reason is not None and not isinstance(disabled_reason, str):
            raise CLISeatConfigError(f"CLI seat '{name}': disabled_reason must be a string")

        cli_timeout_s = _positive_number(raw.get("cli_timeout_s"), "cli_timeout_s", name, DEFAULT_CLI_TIMEOUT_S)
        fallback_reserve_s = _positive_number(
            raw.get("fallback_reserve_s"), "fallback_reserve_s", name, DEFAULT_FALLBACK_RESERVE_S
        )
        min_cli_budget_s = _positive_number(
            raw.get("min_cli_budget_s"), "min_cli_budget_s", name, DEFAULT_MIN_CLI_BUDGET_S
        )
        if min_cli_budget_s > cli_timeout_s:
            raise CLISeatConfigError(f"CLI seat '{name}': min_cli_budget_s cannot exceed cli_timeout_s")

        seats.append(
            CLISeat(
                name=name,
                client=client,
                model=model,
                effort=effort,
                fallback_model=fallback_model,
                host_dedup_frontends=frontends,
                aliases=aliases,
                cli_enabled=cli_enabled,
                disabled_reason=disabled_reason,
                cli_timeout_s=cli_timeout_s,
                fallback_reserve_s=fallback_reserve_s,
                min_cli_budget_s=min_cli_budget_s,
            )
        )

    return CLISeatRegistry(seats=tuple(seats), source=source)


def _read_default_config_text() -> tuple[str, str]:
    try:
        resource = importlib.resources.files("conf").joinpath(CLI_SEATS_DEFAULT_FILENAME)
        return resource.read_text(encoding="utf-8"), f"conf/{CLI_SEATS_DEFAULT_FILENAME}"
    except (FileNotFoundError, ModuleNotFoundError, TypeError):
        fallback_path = Path(__file__).resolve().parent.parent / "conf" / CLI_SEATS_DEFAULT_FILENAME
        return fallback_path.read_text(encoding="utf-8"), str(fallback_path)


def load_cli_seats(config_path: str | None = None) -> CLISeatRegistry:
    """Load seats from ``config_path``, then ``CLI_SEATS_CONFIG_PATH``, then the packaged default."""
    explicit_path = config_path or os.getenv(CLI_SEATS_CONFIG_ENV)
    if explicit_path:
        path = Path(explicit_path).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise CLISeatConfigError(f"Cannot read CLI seat config {path}: {exc}") from exc
        source = str(path)
    else:
        text, source = _read_default_config_text()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise CLISeatConfigError(f"CLI seat config {source} is not valid JSON: {exc}") from exc
    return parse_cli_seats(payload, source=source)


def validate_against_providers(registry: CLISeatRegistry, resolves_as_api_model: Callable[[str], bool]) -> None:
    """Fail when a fallback does not resolve or a seat name shadows an API model name."""
    for seat in registry.seats:
        if not resolves_as_api_model(seat.fallback_model):
            raise CLISeatConfigError(
                f"CLI seat '{seat.name}': fallback_model '{seat.fallback_model}' does not resolve to an API model"
            )
        for seat_name in seat.names():
            if resolves_as_api_model(seat_name):
                raise CLISeatConfigError(
                    f"CLI seat name or alias '{seat_name}' shadows an API model name; choose a distinct name"
                )


def build_isolated_environment(source_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Copy only the variables a subscription CLI needs to find its login."""
    env_source = os.environ if source_env is None else source_env
    return {key: env_source[key] for key in ENV_ALLOWLIST if key in env_source}


def _review_config_args(seat: CLISeat, working_dir: str) -> list[str]:
    requested_effort_args = effort_args(seat.client, seat.effort)
    if seat.client == "claude":
        return [*requested_effort_args, *CLAUDE_REVIEW_ARGS]

    feature_args: list[str] = []
    for feature in CODEX_DISABLED_FEATURES:
        feature_args.extend(["-c", f"features.{feature}=false"])
    return [
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--cd",
        working_dir,
        *requested_effort_args,
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        'web_search="disabled"',
        "-c",
        "agents.max_threads=1",
        *feature_args,
    ]


def build_review_client(seat: CLISeat, *, working_dir: str, timeout_seconds: int) -> ResolvedCLIClient:
    """Create a fresh review-only client; the shipped clink client args are never reused."""
    base_client = get_registry().get_client(seat.client)
    review_role = ResolvedCLIRole(name="consensus_review", prompt_path=base_client.get_role(None).prompt_path)
    return ResolvedCLIClient(
        name=base_client.name,
        executable=list(base_client.executable),
        working_dir=Path(working_dir),
        internal_args=list(base_client.internal_args),
        config_args=_review_config_args(seat, working_dir),
        env={},
        timeout_seconds=max(1, int(timeout_seconds)),
        parser=base_client.parser,
        runner=base_client.runner,
        default_model=None,
        roles={"default": review_role},
        output_to_file=None,
    )


def assert_argv_allowed(argv: list[str]) -> None:
    """Refuse to launch a review CLI whose argv could grant tools, writes, or delegation."""
    for arg in argv:
        if arg in ARGV_DENIED_EXACT:
            raise CLISeatRunError(f"Refusing to launch CLI seat: argument {arg!r} is denylisted", reason="unsafe_argv")
        for denied in ARGV_DENYLIST:
            if denied in arg:
                raise CLISeatRunError(
                    f"Refusing to launch CLI seat: argument {arg!r} matches denylisted token {denied!r}",
                    reason="unsafe_argv",
                )


def _isolated_agent(client: ResolvedCLIClient, env: dict[str, str]):
    agent = create_agent(client)
    isolated_cls = type(f"Isolated{type(agent).__name__}", (type(agent),), {"_build_environment": lambda self: env})
    return isolated_cls(client)


def build_review_prompt(
    system_prompt: str,
    prompt: str,
    *,
    instruction: str = CLI_REVIEW_INSTRUCTION,
    instructions_header: str = CLI_REVIEW_INSTRUCTIONS_HEADER,
    header: str = CLI_REVIEW_HEADER,
) -> str:
    return f"{instruction}\n\n=== {instructions_header} ===\n{system_prompt}\n\n=== {header} ===\n{prompt}"


_client_semaphores: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[str, asyncio.Semaphore]] = (
    weakref.WeakKeyDictionary()
)


def cli_seat_concurrency() -> int:
    """Concurrent runs allowed per CLI client per server; ``CLI_SEAT_MAX_CONCURRENT`` overrides."""
    raw = os.getenv(CLI_SEAT_CONCURRENCY_ENV, "").strip()
    if not raw:
        return DEFAULT_CLI_SEAT_CONCURRENCY
    if not raw.isdigit() or int(raw) < 1:
        raise CLISeatConfigError(f"{CLI_SEAT_CONCURRENCY_ENV} must be a positive integer; got {raw!r}")
    return int(raw)


def _client_semaphore(client_name: str) -> asyncio.Semaphore:
    loop_semaphores = _client_semaphores.setdefault(asyncio.get_running_loop(), {})
    semaphore = loop_semaphores.get(client_name)
    if semaphore is None:
        semaphore = asyncio.Semaphore(cli_seat_concurrency())
        loop_semaphores[client_name] = semaphore
    return semaphore


async def _acquire_within(semaphore: asyncio.Semaphore, timeout_seconds: float) -> bool:
    """Acquire ``semaphore`` within ``timeout_seconds`` without leaking a permit on timeout or cancellation."""
    acquire_task = asyncio.ensure_future(semaphore.acquire())
    try:
        await asyncio.wait_for(asyncio.shield(acquire_task), timeout=max(0.0, timeout_seconds))
        return True
    except asyncio.TimeoutError:
        _settle_abandoned_acquire(semaphore, acquire_task)
        return False
    except asyncio.CancelledError:
        _settle_abandoned_acquire(semaphore, acquire_task)
        raise


def _settle_abandoned_acquire(semaphore: asyncio.Semaphore, acquire_task: asyncio.Future) -> None:
    if acquire_task.done() and not acquire_task.cancelled() and acquire_task.exception() is None:
        semaphore.release()
        return
    acquire_task.cancel()


@dataclass
class CLISeatResult:
    content: str
    model_used: str
    duration_seconds: float
    usage: dict[str, Any] | None = None


async def run_cli_seat(
    seat: CLISeat,
    *,
    system_prompt: str,
    prompt: str,
    budget_seconds: float,
    source_env: Mapping[str, str] | None = None,
    instruction: str = CLI_REVIEW_INSTRUCTION,
    instructions_header: str = CLI_REVIEW_INSTRUCTIONS_HEADER,
    header: str = CLI_REVIEW_HEADER,
    queue_wait_s: float | None = None,
) -> CLISeatResult:
    """Run one isolated review invocation within ``budget_seconds``; raise ``CLISeatRunError`` on failure."""
    started = time.monotonic()
    semaphore = _client_semaphore(seat.client)
    queue_timeout = budget_seconds if queue_wait_s is None else min(budget_seconds, queue_wait_s)
    acquired = False
    try:
        if not await _acquire_within(semaphore, queue_timeout):
            raise CLISeatRunError(
                f"CLI seat '{seat.name}' waited {queue_timeout:.0f}s for another {seat.client} run to finish",
                reason="cli_busy",
            )
        acquired = True

        remaining = budget_seconds - (time.monotonic() - started)
        if remaining < 1:
            raise CLISeatRunError(f"CLI seat '{seat.name}' has no budget left after queueing", reason="cli_busy")

        with tempfile.TemporaryDirectory(prefix="pal-cli-seat-") as working_dir:
            client = build_review_client(seat, working_dir=working_dir, timeout_seconds=math.floor(remaining))
            agent = _isolated_agent(client, build_isolated_environment(source_env))
            role = client.get_role(None)
            model_resolution = agent._resolve_model(seat.model)
            planned_argv = agent._build_command(role=role, system_prompt=None, config_args=model_resolution.config_args)
            assert_argv_allowed(planned_argv)

            try:
                output: AgentOutput = await agent.run(
                    role=role,
                    prompt=build_review_prompt(
                        system_prompt,
                        prompt,
                        instruction=instruction,
                        instructions_header=instructions_header,
                        header=header,
                    ),
                    system_prompt=None,
                    files=[],
                    images=[],
                    model=seat.model,
                )
            except CLIAgentError as exc:
                reason = "cli_timeout" if exc.metadata.get("timed_out") else "cli_error"
                detail = (exc.stderr or exc.stdout or "").strip()[-400:]
                raise CLISeatRunError(
                    f"{exc}{': ' + detail if detail else ''}",
                    reason=reason,
                    metadata={"return_code": exc.returncode},
                ) from exc
    finally:
        if acquired:
            semaphore.release()

    if output.returncode != 0 or output.recovery_metadata:
        raise CLISeatRunError(
            f"CLI seat '{seat.name}' exited with status {output.returncode}",
            reason="cli_nonzero_exit",
            metadata={"return_code": output.returncode},
        )

    tool_activity = detect_tool_activity(seat, output.parsed.metadata)
    if tool_activity:
        raise CLISeatRunError(
            f"CLI seat '{seat.name}' showed tool activity despite isolation: {tool_activity}",
            reason="cli_tool_activity",
        )

    content = (output.parsed.content or "").strip()
    if not content:
        raise CLISeatRunError(f"CLI seat '{seat.name}' returned no verdict text", reason="cli_empty_output")

    model_used = str(output.parsed.metadata.get("model_used") or seat.model)
    usage = output.parsed.metadata.get("usage")
    return CLISeatResult(
        content=content,
        model_used=model_used,
        duration_seconds=round(output.duration_seconds, 3),
        usage=usage if isinstance(usage, dict) else None,
    )


def detect_tool_activity(seat: CLISeat, parsed_metadata: Mapping[str, Any]) -> str | None:
    """Describe any tool, command, MCP, or delegation activity in a seat's CLI output, or return ``None``."""
    if seat.client == "codex":
        unexpected_types = sorted(
            {
                str(event["item"].get("type"))
                for event in parsed_metadata.get("events") or []
                if isinstance(event, dict)
                and isinstance(event.get("item"), dict)
                and event["item"].get("type") not in CODEX_ALLOWED_ITEM_TYPES
            }
        )
        return f"codex item types {unexpected_types}" if unexpected_types else None
    raw = parsed_metadata.get("raw")
    turns = raw.get("num_turns") if isinstance(raw, dict) else None
    if isinstance(turns, int) and turns > 1:
        return f"claude num_turns={turns}"
    if parsed_metadata.get("permission_denials"):
        return f"claude permission_denials={len(parsed_metadata['permission_denials'])}"
    return None


def cli_budget_seconds(seat: CLISeat, *, deadline_at: float, now: float) -> float:
    """Seconds the CLI may run while leaving the API fallback its reserve and cleanup its slack."""
    return min(seat.cli_timeout_s, deadline_at - now - seat.fallback_reserve_s - CLEANUP_SLACK_S)


_seat_state_cache: dict[str, tuple[CLISeatRegistry | None, CLISeatConfigError | None, frozenset[str]]] = {}
_seat_error_cache: dict[
    str, tuple[float, tuple[CLISeatRegistry | None, CLISeatConfigError | None, frozenset[str]]]
] = {}
SEAT_ERROR_RETRY_S = 60.0


def get_cli_seat_state(
    resolves_as_api_model: Callable[[str], bool],
) -> tuple[CLISeatRegistry | None, CLISeatConfigError | None, frozenset[str]]:
    """Load and validate the seat registry once per config path, shared by every tool.

    Returns ``(registry, error, declared_names)``. A config that parses but fails provider
    validation keeps its declared names so requests naming those seats can fail loudly.
    """
    cache_key = os.getenv(CLI_SEATS_CONFIG_ENV) or "<packaged>"
    cached = _seat_state_cache.get(cache_key)
    if cached is not None:
        return cached
    cached_error = _seat_error_cache.get(cache_key)
    if cached_error is not None and cached_error[0] > time.monotonic():
        return cached_error[1]

    registry: CLISeatRegistry | None = None
    declared_names: frozenset[str] = frozenset()
    try:
        cli_seat_concurrency()
        registry = load_cli_seats()
        declared_names = frozenset(registry.all_names())
        validate_against_providers(registry, resolves_as_api_model)
    except CLISeatConfigError as exc:
        logger.error("CLI seats disabled: %s", exc)
        if not declared_names:
            declared_names = _packaged_seat_names()
        error_state = (None, exc, declared_names)
        _seat_error_cache[cache_key] = (time.monotonic() + SEAT_ERROR_RETRY_S, error_state)
        return error_state
    cached = (registry, None, declared_names)
    _seat_state_cache[cache_key] = cached
    return cached


def _packaged_seat_names() -> frozenset[str]:
    """Seat names from the packaged default, so an unreadable override still fails requests loudly."""
    try:
        text, source = _read_default_config_text()
        return frozenset(parse_cli_seats(json.loads(text), source=source).all_names())
    except (OSError, ValueError):
        return frozenset()


def clear_cli_seat_state_cache() -> None:
    _seat_state_cache.clear()
    _seat_error_cache.clear()


def effort_for_thinking_mode(thinking_mode: str | None, *, default: str = "high") -> str:
    """Map a PAL ``thinking_mode`` onto a CLI reasoning effort; raise on anything unmapped."""
    if thinking_mode is None:
        return default
    normalized = str(thinking_mode).strip().lower()
    effort = THINKING_MODE_TO_EFFORT.get(normalized)
    if effort is None or effort not in REASONING_EFFORTS:
        raise CLISeatConfigError(
            f"thinking_mode {thinking_mode!r} has no CLI effort; use one of {', '.join(THINKING_MODE_TO_EFFORT)}"
        )
    return effort


def describe_exception(exception: BaseException) -> str:
    """Compact, bounded description of a leg failure without echoing provider payloads verbatim."""
    message = str(exception).strip()
    if len(message) > LEG_ERROR_TEXT_LIMIT:
        message = f"{message[:LEG_ERROR_TEXT_LIMIT]}…"
    return f"{type(exception).__name__}: {message}" if message else type(exception).__name__


def classify_cli_failure(error: CLISeatRunError) -> str:
    message = str(error).lower()
    if (
        error.reason in ("cli_error", "cli_nonzero_exit")
        and "model" in message
        and any(marker in message for marker in CLI_MODEL_UNAVAILABLE_MARKERS)
    ):
        return "cli_model_unavailable"
    return error.reason


@dataclass
class SeatConsultation:
    """Outcome of one seat call: the CLI answer, its API fallback, or a failure of both."""

    status: str
    backend: str
    text: str = ""
    model_used: str | None = None
    effort: str | None = None
    fallback_reason: str | None = None
    cli_error: str | None = None
    error: str | None = None
    usage: dict[str, Any] | None = None
    fallback_payload: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0


async def consult_seat_with_fallback(
    seat: CLISeat,
    *,
    system_prompt: str,
    build_prompt: Callable[[], str],
    deadline_at: float | None,
    api_fallback: Callable[[], Any],
    images_present: bool = False,
    runner: Callable[..., Any] | None = None,
    instruction: str = CLI_REVIEW_INSTRUCTION,
    instructions_header: str = CLI_REVIEW_INSTRUCTIONS_HEADER,
    header: str = CLI_REVIEW_HEADER,
    queue_wait_s: float | None = None,
    enforce_fallback_deadline: bool = False,
    progress: dict[str, Any] | None = None,
) -> SeatConsultation:
    """Run ``seat`` through its CLI, falling back to its API model when the CLI cannot answer.

    ``api_fallback`` is an async callable returning ``{"status", "text", "metadata", "error"}``.
    """
    run = runner or run_cli_seat
    started = time.monotonic()
    if deadline_at is None:
        deadline_at = started + seat.cli_timeout_s + seat.fallback_reserve_s + CLEANUP_SLACK_S
    attempts: list[dict[str, Any]] = []
    if progress is not None:
        progress["attempts"] = attempts
        progress["backend"] = "cli"
        progress["started"] = started

    fallback_reason: str | None = None
    cli_error_text: str | None = None
    if images_present:
        fallback_reason = "images_unsupported_by_cli"
    elif not seat.cli_enabled:
        fallback_reason = "cli_isolation_unproven"
        cli_error_text = seat.disabled_reason
    else:
        budget = cli_budget_seconds(seat, deadline_at=deadline_at, now=time.monotonic())
        if budget < seat.min_cli_budget_s:
            fallback_reason = "cli_budget_exhausted"
        else:
            attempt_started = time.monotonic()
            try:
                prompt = build_prompt()
                result = await run(
                    seat,
                    system_prompt=system_prompt,
                    prompt=prompt,
                    budget_seconds=budget,
                    **_framing_kwargs(instruction, instructions_header, header, queue_wait_s),
                )
            except CLISeatRunError as exc:
                fallback_reason = classify_cli_failure(exc)
                cli_error_text = str(exc)[:LEG_ERROR_TEXT_LIMIT]
            except Exception as exc:
                logger.exception("CLI seat %s failed unexpectedly", seat.name)
                fallback_reason = "cli_error"
                cli_error_text = describe_exception(exc)
            else:
                attempts.append(
                    {
                        "backend": "cli",
                        "status": "success",
                        "model": result.model_used,
                        "duration_seconds": round(time.monotonic() - attempt_started, 3),
                    }
                )
                return SeatConsultation(
                    status="success",
                    backend="cli",
                    text=result.content,
                    model_used=result.model_used,
                    effort=seat.effort,
                    usage=result.usage,
                    attempts=attempts,
                    duration_seconds=round(time.monotonic() - started, 3),
                )
            attempts.append(
                {
                    "backend": "cli",
                    "status": "error",
                    "reason": fallback_reason,
                    "error": cli_error_text,
                    "duration_seconds": round(time.monotonic() - attempt_started, 3),
                }
            )

    remaining = deadline_at - time.monotonic()
    if remaining < MIN_FALLBACK_BUDGET_S:
        return SeatConsultation(
            status="error",
            backend="cli",
            fallback_reason=fallback_reason,
            cli_error=cli_error_text,
            error=(
                f"CLI seat failed ({fallback_reason}: {cli_error_text or 'no detail'}) and only "
                f"{max(0.0, remaining):.0f}s remained, too little for the {seat.fallback_model} API fallback"
            )[:LEG_ERROR_TEXT_LIMIT],
            attempts=attempts,
            duration_seconds=round(time.monotonic() - started, 3),
        )

    fallback_started = time.monotonic()
    if progress is not None:
        progress["backend"] = "api_fallback"
    if enforce_fallback_deadline:
        try:
            fallback_payload = await asyncio.wait_for(api_fallback(), timeout=remaining)
        except asyncio.TimeoutError:
            fallback_payload = {
                "status": "error",
                "error": f"API fallback {seat.fallback_model} did not answer within the remaining {remaining:.0f}s",
            }
    else:
        fallback_payload = await api_fallback()
    attempts.append(
        {
            "backend": "api_fallback",
            "status": fallback_payload.get("status"),
            "model": seat.fallback_model,
            "duration_seconds": round(time.monotonic() - fallback_started, 3),
        }
    )

    if fallback_payload.get("status") == "success":
        return SeatConsultation(
            status="success",
            backend="api_fallback",
            text=fallback_payload.get("text", ""),
            model_used=fallback_payload.get("model_used") or seat.fallback_model,
            fallback_reason=fallback_reason,
            cli_error=cli_error_text,
            fallback_payload=fallback_payload,
            attempts=attempts,
            duration_seconds=round(time.monotonic() - started, 3),
        )

    return SeatConsultation(
        status="error",
        backend="api_fallback",
        fallback_reason=fallback_reason,
        cli_error=cli_error_text,
        error=(
            f"CLI seat failed ({fallback_reason}: {cli_error_text or 'no detail'}); API fallback "
            f"{seat.fallback_model} failed: {fallback_payload.get('error')}"
        )[:LEG_ERROR_TEXT_LIMIT],
        fallback_payload=fallback_payload,
        attempts=attempts,
        duration_seconds=round(time.monotonic() - started, 3),
    )


def _framing_kwargs(
    instruction: str, instructions_header: str, header: str, queue_wait_s: float | None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if instruction != CLI_REVIEW_INSTRUCTION:
        kwargs["instruction"] = instruction
    if instructions_header != CLI_REVIEW_INSTRUCTIONS_HEADER:
        kwargs["instructions_header"] = instructions_header
    if header != CLI_REVIEW_HEADER:
        kwargs["header"] = header
    if queue_wait_s is not None:
        kwargs["queue_wait_s"] = queue_wait_s
    return kwargs
