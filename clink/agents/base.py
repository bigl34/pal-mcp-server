"""Execute configured CLI agents for the clink tool and parse output."""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import shutil
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from clink.constants import DEFAULT_STREAM_LIMIT
from clink.models import ResolvedCLIClient, ResolvedCLIRole
from clink.parsers import BaseParser, ParsedCLIResponse, ParserError, get_parser

logger = logging.getLogger("clink.agent")


@dataclass
class ModelResolution:
    """Model selection facts for one CLI invocation."""

    config_args: list[str]
    requested_model: str | None
    configured_model: str | None
    effective_model: str | None
    model_source: str


@dataclass
class AgentOutput:
    """Container returned by CLI agents after successful execution."""

    parsed: ParsedCLIResponse
    sanitized_command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    parser_name: str
    output_file_content: str | None = None
    requested_model: str | None = None
    effective_model: str | None = None
    configured_model: str | None = None
    model_source: str = "native"


class CLIAgentError(RuntimeError):
    """Raised when a CLI agent fails (non-zero exit, timeout, parse errors)."""

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None = None,
        stdout: str = "",
        stderr: str = "",
        metadata: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.metadata = dict(metadata or {})


class BaseCLIAgent:
    """Execute a configured CLI command and parse its output."""

    def __init__(self, client: ResolvedCLIClient):
        self.client = client
        self._parser: BaseParser = get_parser(client.parser)
        self._logger = logging.getLogger(f"clink.runner.{client.name}")

    async def run(
        self,
        *,
        role: ResolvedCLIRole,
        prompt: str,
        system_prompt: str | None = None,
        files: Sequence[str],
        images: Sequence[str],
        model: str | None = None,
    ) -> AgentOutput:
        # Files and images are already embedded into the prompt by the tool; they are
        # accepted here only to keep parity with SimpleTool callers.
        _ = (files, images)
        # The runner simply executes the configured CLI command for the selected role.
        model_resolution = self._resolve_model(model)
        command = self._build_command(
            role=role,
            system_prompt=system_prompt,
            config_args=model_resolution.config_args,
        )
        env = self._build_environment()

        # Resolve executable path for cross-platform compatibility (especially Windows)
        executable_name = command[0]
        resolved_executable = shutil.which(executable_name)
        if resolved_executable is None:
            raise CLIAgentError(
                f"Executable '{executable_name}' not found in PATH for CLI '{self.client.name}'. "
                f"Ensure the command is installed and accessible."
            )
        command[0] = resolved_executable

        sanitized_command = list(command)

        cwd = str(self.client.working_dir) if self.client.working_dir else None
        limit = DEFAULT_STREAM_LIMIT

        stdout_text = ""
        stderr_text = ""
        output_file_content: str | None = None
        start_time = time.monotonic()

        output_file_path: Path | None = None
        command_with_output_flag = list(command)

        if self.client.output_to_file:
            fd, tmp_path = tempfile.mkstemp(prefix="clink-", suffix=".json")
            os.close(fd)
            output_file_path = Path(tmp_path)
            flag_template = self.client.output_to_file.flag_template
            try:
                rendered_flag = flag_template.format(path=str(output_file_path))
            except KeyError as exc:  # pragma: no cover - defensive
                raise CLIAgentError(f"Invalid output flag template '{flag_template}': missing placeholder {exc}")
            command_with_output_flag.extend(shlex.split(rendered_flag))
            sanitized_command = list(command_with_output_flag)

        self._logger.debug("Executing CLI command: %s", " ".join(sanitized_command))
        if cwd:
            self._logger.debug("Working directory: %s", cwd)

        try:
            process = await asyncio.create_subprocess_exec(
                *command_with_output_flag,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                limit=limit,
                env=env,
            )
        except FileNotFoundError as exc:
            raise CLIAgentError(f"Executable not found for CLI '{self.client.name}': {exc}") from exc

        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(prompt.encode("utf-8")),
                timeout=self.client.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise CLIAgentError(
                f"CLI '{self.client.name}' timed out after {self.client.timeout_seconds} seconds",
                returncode=None,
            ) from exc

        duration = time.monotonic() - start_time
        return_code = process.returncode
        stdout_text = stdout_bytes.decode("utf-8", errors="replace")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace")

        if output_file_path and output_file_path.exists():
            output_file_content = output_file_path.read_text(encoding="utf-8", errors="replace")
            if self.client.output_to_file and self.client.output_to_file.cleanup:
                try:
                    output_file_path.unlink()
                except OSError:  # pragma: no cover - best effort cleanup
                    pass

            if output_file_content and not stdout_text.strip():
                stdout_text = output_file_content

        if return_code != 0:
            recovered = self._recover_from_error(
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
                sanitized_command=sanitized_command,
                duration_seconds=duration,
                output_file_content=output_file_content,
                model_resolution=model_resolution,
            )
            if recovered is not None:
                return recovered

        if return_code != 0:
            raise CLIAgentError(
                f"CLI '{self.client.name}' exited with status {return_code}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            )

        try:
            parsed = self._parser.parse(stdout_text, stderr_text)
        except ParserError as exc:
            raise CLIAgentError(
                f"Failed to parse output from CLI '{self.client.name}': {exc}",
                returncode=return_code,
                stdout=stdout_text,
                stderr=stderr_text,
            ) from exc

        self._raise_for_parsed_response(parsed, returncode=return_code, stdout=stdout_text, stderr=stderr_text)

        return self._build_agent_output(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=return_code,
            stdout=stdout_text,
            stderr=stderr_text,
            duration_seconds=duration,
            output_file_content=output_file_content,
            model_resolution=model_resolution,
        )

    def _build_command(
        self,
        *,
        role: ResolvedCLIRole,
        system_prompt: str | None,
        config_args: Sequence[str],
    ) -> list[str]:
        base = list(self.client.executable)
        base.extend(self.client.internal_args)
        base.extend(config_args)
        base.extend(role.role_args)

        return base

    def _build_environment(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self.client.env)
        return env

    # ------------------------------------------------------------------
    # Model selection
    # ------------------------------------------------------------------

    def _resolve_model(self, requested_model: str | None) -> ModelResolution:
        config_args, configured_model = self._normalize_model_config_args(self.client.config_args)
        effective_model: str | None = None
        model_source = "native"

        if requested_model:
            if requested_model == "native":
                model_source = "native"
            else:
                effective_model = requested_model
                model_source = "request"
        elif configured_model:
            if configured_model == "native":
                model_source = "native"
                self._logger.info(
                    "CLI '%s' config model sentinel 'native' suppresses clink's internal default model.",
                    self.client.name,
                )
            else:
                effective_model = configured_model
                model_source = "config"
                self._logger.info(
                    "CLI '%s' using model from config args: %s",
                    self.client.name,
                    configured_model,
                )
        elif self.client.default_model:
            effective_model = self.client.default_model
            model_source = "client_default"

        if effective_model:
            config_args = [*config_args, self._canonical_model_flag(), effective_model]

        return ModelResolution(
            config_args=config_args,
            requested_model=requested_model,
            configured_model=configured_model,
            effective_model=effective_model,
            model_source=model_source,
        )

    def _normalize_model_config_args(self, config_args: Sequence[str]) -> tuple[list[str], str | None]:
        normalized: list[str] = []
        stripped_values: list[str] = []
        supports_short_flag = self._supports_short_model_flag()
        index = 0
        args = list(config_args)

        while index < len(args):
            token = args[index]

            if supports_short_flag and token == "-c":
                normalized.append(token)
                index += 1
                if index < len(args):
                    normalized.append(args[index])
                    index += 1
                continue

            if token == "--model":
                index = self._strip_split_model_flag(args, index, stripped_values, "--model")
                continue

            if token.startswith("--model="):
                self._record_attached_model_value(token, token.removeprefix("--model="), stripped_values)
                index += 1
                continue

            if supports_short_flag and token == "-m":
                index = self._strip_split_model_flag(args, index, stripped_values, "-m")
                continue

            if supports_short_flag and token.startswith("-m="):
                self._record_attached_model_value(token, token.removeprefix("-m="), stripped_values)
                index += 1
                continue

            normalized.append(token)
            index += 1

        distinct_values = set(stripped_values)
        if len(distinct_values) > 1:
            self._logger.warning(
                "Stripped multiple distinct model values from CLI '%s' config args; last value wins.",
                self.client.name,
            )

        configured_model = stripped_values[-1] if stripped_values else None
        return normalized, configured_model

    def _strip_split_model_flag(
        self,
        args: list[str],
        index: int,
        stripped_values: list[str],
        flag: str,
    ) -> int:
        next_index = index + 1
        if next_index >= len(args):
            self._logger.warning("Ignoring dangling %s model flag in CLI '%s' config args.", flag, self.client.name)
            return next_index

        value = args[next_index]
        if value.startswith("-"):
            self._logger.warning(
                "Ignoring %s model flag without a usable value in CLI '%s' config args.",
                flag,
                self.client.name,
            )
            return next_index

        self._record_model_value(flag, value, stripped_values)
        return next_index + 1

    def _record_attached_model_value(self, token: str, value: str, stripped_values: list[str]) -> None:
        self._record_model_value(token, value, stripped_values)

    def _record_model_value(self, flag: str, value: str, stripped_values: list[str]) -> None:
        model = value.strip()
        if not model:
            self._logger.warning(
                "Ignoring %s model flag with empty value in CLI '%s' config args.",
                flag,
                self.client.name,
            )
            return
        if model.startswith("-"):
            self._logger.warning(
                "Ignoring %s model flag with invalid leading '-' value in CLI '%s' config args.",
                flag,
                self.client.name,
            )
            return
        stripped_values.append(model)

    def _supports_short_model_flag(self) -> bool:
        runner_name = (self.client.runner or self.client.name).lower()
        return runner_name in {"codex", "gemini"}

    def _canonical_model_flag(self) -> str:
        runner_name = (self.client.runner or self.client.name).lower()
        if runner_name == "codex":
            return "-m"
        return "--model"

    def _build_agent_output(
        self,
        *,
        parsed: ParsedCLIResponse,
        sanitized_command: list[str],
        returncode: int,
        stdout: str,
        stderr: str,
        duration_seconds: float,
        output_file_content: str | None,
        model_resolution: ModelResolution,
    ) -> AgentOutput:
        return AgentOutput(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            parser_name=self._parser.name,
            output_file_content=output_file_content,
            requested_model=model_resolution.requested_model,
            effective_model=model_resolution.effective_model,
            configured_model=model_resolution.configured_model,
            model_source=model_resolution.model_source,
        )

    def _raise_for_parsed_response(
        self,
        parsed: ParsedCLIResponse,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        """Hook for subclasses to turn parsed zero-exit error payloads into failures."""

        _ = (parsed, returncode, stdout, stderr)

    # ------------------------------------------------------------------
    # Error recovery hooks
    # ------------------------------------------------------------------

    def _recover_from_error(
        self,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
        sanitized_command: list[str],
        duration_seconds: float,
        output_file_content: str | None,
        model_resolution: ModelResolution,
    ) -> AgentOutput | None:
        """Hook for subclasses to convert CLI errors into successful outputs.

        Return an AgentOutput to treat the failure as success, or None to signal
        that normal error handling should proceed.
        """

        _ = model_resolution
        return None
