"""Codex-specific CLI agent hooks."""

from __future__ import annotations

from clink.models import ResolvedCLIClient
from clink.parsers.base import ParserError

from .base import AgentOutput, BaseCLIAgent, ModelResolution


class CodexAgent(BaseCLIAgent):
    """Codex CLI agent with JSONL recovery support."""

    def __init__(self, client: ResolvedCLIClient):
        super().__init__(client)

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
        try:
            parsed = self._parser.parse(stdout, stderr)
        except ParserError:
            return None

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
