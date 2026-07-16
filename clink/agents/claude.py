"""Claude-specific CLI agent hooks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from clink.models import ResolvedCLIRole
from clink.parsers.base import ParsedCLIResponse
from clink.parsers.base import ParserError

from .base import AgentOutput, BaseCLIAgent, CLIAgentError, ModelResolution

OAUTH_ORG_NOT_ALLOWED = "oauth_org_not_allowed"
CLAUDE_SUBSCRIPTION_AUTH_MESSAGE = (
    "Claude CLI reported oauth_org_not_allowed. Re-authenticate Claude Code with your Anthropic Pro/Max "
    "subscription by running `claude logout`, then `claude login`. Never set ANTHROPIC_API_KEY for clink's "
    "Claude subscription flow; API/console billing credentials are not a fallback."
)
CLAUDE_ENV_STRIP_PREFIXES = ("ANTHROPIC_", "CLAUDE_CODE_USE_", "CLAUDE_CODE_API_KEY")


class ClaudeAgent(BaseCLIAgent):
    """Claude CLI agent with system-prompt injection support."""

    def _build_command(
        self,
        *,
        role: ResolvedCLIRole,
        system_prompt: str | None,
        config_args: Sequence[str],
    ) -> list[str]:
        command = list(self.client.executable)
        command.extend(self.client.internal_args)
        command.extend(config_args)

        if system_prompt and "--append-system-prompt" not in config_args:
            command.extend(["--append-system-prompt", system_prompt])

        command.extend(role.role_args)
        return command

    def _build_environment(self) -> dict[str, str]:
        env = super()._build_environment()
        for key in list(env):
            if key.startswith(CLAUDE_ENV_STRIP_PREFIXES):
                del env[key]
        return env

    def _raise_for_parsed_response(
        self,
        parsed: ParsedCLIResponse,
        *,
        returncode: int,
        stdout: str,
        stderr: str,
    ) -> None:
        if parsed.metadata.get("is_error") is True:
            self._raise_for_oauth_org_not_allowed(parsed.metadata, stdout=stdout, stderr=stderr, returncode=returncode)
            raise CLIAgentError(
                "Claude CLI reported an error payload",
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )
        self._raise_for_oauth_org_not_allowed(None, stdout="", stderr=stderr, returncode=returncode)

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
        self._raise_for_oauth_org_not_allowed(None, stdout=stdout, stderr=stderr, returncode=returncode)

        try:
            parsed = self._parser.parse(stdout, stderr)
        except ParserError:
            return None

        self._raise_for_oauth_org_not_allowed(parsed.metadata, stdout=stdout, stderr=stderr, returncode=returncode)

        if parsed.metadata.get("is_error") is True:
            raise CLIAgentError(
                "Claude CLI reported an error payload",
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )

        explicit_success = (
            parsed.metadata.get("is_error_explicit") is True
            and parsed.metadata.get("is_error") is False
            and parsed.metadata.get("subtype") == "success"
        )
        if not explicit_success:
            raise CLIAgentError(
                f"Claude CLI exited with status {returncode} and did not report success",
                returncode=returncode,
                stdout=stdout,
                stderr=stderr,
            )

        self._logger.info("Recovering Claude CLI non-zero exit %s from explicit success payload.", returncode)
        return self._build_agent_output(
            parsed=parsed,
            sanitized_command=sanitized_command,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            duration_seconds=duration_seconds,
            output_file_content=output_file_content,
            model_resolution=model_resolution,
        )

    def _raise_for_oauth_org_not_allowed(
        self,
        metadata: dict[str, Any] | None,
        *,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> None:
        if not self._contains_oauth_org_not_allowed(metadata, stdout, stderr):
            return
        raise CLIAgentError(
            CLAUDE_SUBSCRIPTION_AUTH_MESSAGE,
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            metadata={"auth_error": True, "auth_error_code": OAUTH_ORG_NOT_ALLOWED},
        )

    def _contains_oauth_org_not_allowed(self, *values: Any) -> bool:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                if OAUTH_ORG_NOT_ALLOWED in value.lower():
                    return True
                continue
            if isinstance(value, dict):
                if self._contains_oauth_org_not_allowed(*value.keys(), *value.values()):
                    return True
                continue
            if isinstance(value, list):
                if self._contains_oauth_org_not_allowed(*value):
                    return True
        return False
