"""
Chat tool - General development chat and collaborative thinking

This tool provides a conversational interface for general development assistance,
brainstorming, problem-solving, and collaborative thinking. It supports file context,
images, and conversation continuation for seamless multi-turn interactions.
"""

import asyncio
import contextvars
import dataclasses
import logging
import os
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from pydantic import Field

if TYPE_CHECKING:
    from providers.shared import ModelCapabilities
    from tools.models import ToolModelCategory

from config import TEMPERATURE_BALANCED
from providers.shared import ModelResponse
from systemprompts import CHAT_PROMPT, GENERATE_CODE_PROMPT
from tools.consensus_cli_seats import (
    CLI_ADVISOR_HEADER,
    CLI_ADVISOR_INSTRUCTION,
    CLI_ADVISOR_INSTRUCTIONS_HEADER,
    CLISeat,
    CLISeatConfigError,
    consult_seat_with_fallback,
    describe_exception,
    effort_for_thinking_mode,
    get_cli_seat_state,
    load_cli_seats,
)
from tools.shared.base_models import COMMON_FIELD_DESCRIPTIONS, ToolRequest
from utils.client_info import get_current_client_frontend

from .simple.base import SimpleTool

CHAT_SEAT_DEADLINE_S = 1440.0
CHAT_SEAT_CLI_TIMEOUT_S = 600.0
CHAT_SEAT_QUEUE_WAIT_S = 60.0
CHAT_SEAT_DEFAULT_EFFORT = "high"

_SEAT_METADATA: contextvars.ContextVar = contextvars.ContextVar("pal_chat_seat_metadata", default=None)

# Field descriptions matching the original Chat tool exactly
CHAT_FIELD_DESCRIPTIONS = {
    "prompt": (
        "Your question or idea for collaborative thinking to be sent to the external model. Provide detailed context, "
        "including your goal, what you've tried, and any specific challenges. "
        "WARNING: Large inline code must NOT be shared in prompt. Provide full-path to files on disk as separate parameter."
    ),
    "absolute_file_paths": ("Full, absolute file paths to relevant code in order to share with external model"),
    "images": "Image paths (absolute) or base64 strings for optional visual context.",
    "working_directory_absolute_path": (
        "Absolute path to an existing directory where generated code artifacts can be saved."
    ),
}


class ChatRequest(ToolRequest):
    """Request model for Chat tool"""

    prompt: str = Field(..., description=CHAT_FIELD_DESCRIPTIONS["prompt"])
    absolute_file_paths: Optional[list[str]] = Field(
        default_factory=list,
        description=CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
    )
    images: Optional[list[str]] = Field(default_factory=list, description=CHAT_FIELD_DESCRIPTIONS["images"])
    working_directory_absolute_path: str = Field(
        ...,
        description=CHAT_FIELD_DESCRIPTIONS["working_directory_absolute_path"],
    )


class ChatTool(SimpleTool):
    """
    General development chat and collaborative thinking tool using SimpleTool architecture.

    This tool provides identical functionality to the original Chat tool but uses the new
    SimpleTool architecture for cleaner code organization and better maintainability.

    Migration note: This tool is designed to be a drop-in replacement for the original
    Chat tool with 100% behavioral compatibility.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_recordable_response: Optional[str] = None

    @property
    def _seat_metadata(self) -> Optional[dict[str, Any]]:
        return _SEAT_METADATA.get()

    @_seat_metadata.setter
    def _seat_metadata(self, metadata: Optional[dict[str, Any]]) -> None:
        _SEAT_METADATA.set(metadata)

    def get_name(self) -> str:
        return "chat"

    def get_description(self) -> str:
        return (
            "General chat and collaborative thinking partner for brainstorming, development discussion, "
            "getting second opinions, and exploring ideas. Use for ideas, validations, questions, and thoughtful explanations."
        )

    def get_annotations(self) -> Optional[dict[str, Any]]:
        """Chat writes generated artifacts when code-generation is enabled."""

        return {"readOnlyHint": False}

    def get_system_prompt(self) -> str:
        return CHAT_PROMPT

    def get_capability_system_prompts(self, capabilities: Optional["ModelCapabilities"]) -> list[str]:
        prompts = list(super().get_capability_system_prompts(capabilities))
        if self._active_cli_seat is None and capabilities and capabilities.allow_code_generation:
            prompts.append(GENERATE_CODE_PROMPT)
        return prompts

    def get_default_temperature(self) -> float:
        return TEMPERATURE_BALANCED

    def get_model_category(self) -> "ToolModelCategory":
        """Chat prioritizes fast responses and cost efficiency"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.FAST_RESPONSE

    def get_request_model(self):
        """Return the Chat-specific request model"""
        return ChatRequest

    # === Schema Generation Utilities ===

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema matching the original Chat tool expectations."""

        required_fields = ["prompt", "working_directory_absolute_path"]
        if self.is_effective_auto_mode():
            required_fields.append("model")

        schema = {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": CHAT_FIELD_DESCRIPTIONS["prompt"],
                },
                "absolute_file_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": CHAT_FIELD_DESCRIPTIONS["images"],
                },
                "working_directory_absolute_path": {
                    "type": "string",
                    "description": CHAT_FIELD_DESCRIPTIONS["working_directory_absolute_path"],
                },
                "model": self._model_field_schema_with_seats(),
                "temperature": {
                    "type": "number",
                    "description": COMMON_FIELD_DESCRIPTIONS["temperature"],
                    "minimum": 0,
                    "maximum": 1,
                },
                "thinking_mode": {
                    "type": "string",
                    "enum": ["minimal", "low", "medium", "high", "max"],
                    "description": COMMON_FIELD_DESCRIPTIONS["thinking_mode"],
                },
                "continuation_id": {
                    "type": "string",
                    "description": COMMON_FIELD_DESCRIPTIONS["continuation_id"],
                },
            },
            "required": required_fields,
            "additionalProperties": False,
        }

        return schema

    def _model_field_schema_with_seats(self) -> dict[str, Any]:
        schema = dict(self.get_model_field_schema())
        try:
            registry = load_cli_seats()
        except CLISeatConfigError:
            return schema
        if not registry.seats:
            return schema
        seat_names = ", ".join(seat.name for seat in registry.seats)
        schema["description"] = (
            f"{schema.get('description', '')} Subscription CLI seats ({seat_names}) are also accepted: single-shot, "
            "prompt-only, effort from thinking_mode (default high), API fallback to the seat's fallback model; "
            "no continuation_id or images."
        ).strip()
        return schema

    def get_tool_fields(self) -> dict[str, dict[str, Any]]:
        """Tool-specific field definitions used by SimpleTool scaffolding."""

        return {
            "prompt": {
                "type": "string",
                "description": CHAT_FIELD_DESCRIPTIONS["prompt"],
            },
            "absolute_file_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": CHAT_FIELD_DESCRIPTIONS["absolute_file_paths"],
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": CHAT_FIELD_DESCRIPTIONS["images"],
            },
            "working_directory_absolute_path": {
                "type": "string",
                "description": CHAT_FIELD_DESCRIPTIONS["working_directory_absolute_path"],
            },
        }

    def get_required_fields(self) -> list[str]:
        """Required fields for ChatSimple tool"""
        return ["prompt", "working_directory_absolute_path"]

    # === Hook Method Implementations ===

    async def prepare_prompt(self, request: ChatRequest) -> str:
        """
        Prepare the chat prompt with optional context files.

        This implementation matches the original Chat tool exactly while using
        SimpleTool convenience methods for cleaner code.
        """
        # Use SimpleTool's Chat-style prompt preparation
        return self.prepare_chat_style_prompt(request)

    def _validate_file_paths(self, request) -> Optional[str]:
        """Extend validation to cover the working directory path."""

        files = self.get_request_files(request)
        if files:
            expanded_files: list[str] = []
            for file_path in files:
                expanded = os.path.expanduser(file_path)
                if not os.path.isabs(expanded):
                    return (
                        "Error: All file paths must be FULL absolute paths to real files / folders - DO NOT SHORTEN. "
                        f"Received: {file_path}"
                    )
                expanded_files.append(expanded)
            self.set_request_files(request, expanded_files)

        error = super()._validate_file_paths(request)
        if error:
            return error

        working_directory = request.working_directory_absolute_path
        if working_directory:
            expanded = os.path.expanduser(working_directory)
            if not os.path.isabs(expanded):
                return (
                    "Error: 'working_directory_absolute_path' must be an absolute path (you may use '~' which will be expanded). "
                    f"Received: {working_directory}"
                )
            if not os.path.isdir(expanded):
                return (
                    "Error: 'working_directory_absolute_path' must reference an existing directory. "
                    f"Received: {working_directory}"
                )
        return None

    def format_response(self, response: str, request: ChatRequest, model_info: Optional[dict] = None) -> str:
        """
        Format the chat response to match the original Chat tool exactly.
        """
        self._last_recordable_response = None
        body = response
        recordable_override: Optional[str] = None

        if self._model_supports_code_generation():
            block, remainder, _ = self._extract_generated_code_block(response)
            if block:
                sanitized_text = remainder.strip()
                target_directory = request.working_directory_absolute_path
                try:
                    artifact_path = self._persist_generated_code_block(block, target_directory)
                except Exception as exc:  # pragma: no cover - rare filesystem failures
                    logger.error("Failed to persist generated code block: %s", exc, exc_info=True)
                    warning = (
                        f"WARNING: Unable to write pal_generated.code inside '{target_directory}'. "
                        "Check the path permissions and re-run. The generated code block is included below for manual handling."
                    )

                    history_copy_base = sanitized_text
                    history_copy = self._join_sections(history_copy_base, warning) if history_copy_base else warning
                    recordable_override = history_copy

                    sanitized_warning = history_copy.strip()
                    body = f"{sanitized_warning}\n\n{block.strip()}".strip()
                else:
                    if not sanitized_text:
                        base_message = (
                            "Generated code saved to pal_generated.code.\n"
                            "\n"
                            "CRITICAL: Contains mixed instructions + partial snippets - NOT complete code to copy as-is!\n"
                            "\n"
                            "You MUST:\n"
                            "  1. Read as a proposal from partial context - you may need to read the file in sections\n"
                            "  2. Implement ideas using YOUR complete codebase context and understanding\n"
                            "  3. Never paste wholesale - snippets may be partial with missing lines, pasting will corrupt your code!\n"
                            "  4. Adapt to fit your actual structure and style\n"
                            "  5. Build/lint/test after implementation to verify correctness\n"
                            "\n"
                            "Treat as guidance to implement thoughtfully, not ready-to-paste code."
                        )
                        sanitized_text = base_message

                    instruction = self._build_agent_instruction(artifact_path)
                    body = self._join_sections(sanitized_text, instruction)

        final_output = (
            f"{body}\n\n---\n\nAGENT'S TURN: Evaluate this perspective alongside your analysis to "
            "form a comprehensive solution and continue with the user's request and task at hand."
        )

        if recordable_override is not None:
            self._last_recordable_response = (
                f"{recordable_override}\n\n---\n\nAGENT'S TURN: Evaluate this perspective alongside your analysis to "
                "form a comprehensive solution and continue with the user's request and task at hand."
            )
        else:
            self._last_recordable_response = final_output

        return final_output

    def _record_assistant_turn(
        self, continuation_id: str, response_text: str, request, model_info: Optional[dict]
    ) -> None:
        recordable = self._last_recordable_response if self._last_recordable_response is not None else response_text
        try:
            super()._record_assistant_turn(continuation_id, recordable, request, model_info)
        finally:
            self._last_recordable_response = None

    def _model_supports_code_generation(self) -> bool:
        if self._active_cli_seat is not None:
            return False
        context = getattr(self, "_model_context", None)
        if not context:
            return False

        try:
            capabilities = context.capabilities
        except Exception:  # pragma: no cover - defensive fallback
            return False

        return bool(capabilities.allow_code_generation)

    def _extract_generated_code_block(self, text: str) -> tuple[Optional[str], str, int]:
        matches = list(re.finditer(r"<GENERATED-CODE>.*?</GENERATED-CODE>", text, flags=re.DOTALL | re.IGNORECASE))
        if not matches:
            return None, text, 0

        last_match = matches[-1]
        block = last_match.group(0).strip()

        # Merge the text before and after the final block while trimming excess whitespace
        before = text[: last_match.start()]
        after = text[last_match.end() :]
        remainder = self._join_sections(before, after)

        return block, remainder, len(matches)

    def _persist_generated_code_block(self, block: str, working_directory: str) -> Path:
        expanded = os.path.expanduser(working_directory)
        target_dir = Path(expanded).resolve()
        if not target_dir.is_dir():
            raise FileNotFoundError(f"Absolute working directory path '{working_directory}' does not exist")

        target_file = target_dir / "pal_generated.code"
        if target_file.exists():
            try:
                target_file.unlink()
            except OSError as exc:
                logger.warning("Unable to remove existing pal_generated.code: %s", exc)

        content = block if block.endswith("\n") else f"{block}\n"
        target_file.write_text(content, encoding="utf-8")
        logger.info("Generated code artifact written to %s", target_file)
        return target_file

    @staticmethod
    def _build_agent_instruction(artifact_path: Path) -> str:
        return (
            f"CONTINUING FROM PREVIOUS DISCUSSION: Implementation plan saved to `{artifact_path}`.\n"
            "\n"
            f"CRITICAL WARNING: `{artifact_path}` may contain partial code snippets from another AI with limited context. "
            "Wholesale copy-pasting MAY CORRUPT your codebase with incomplete logic and missing lines.\n"
            "\n"
            "Required workflow:\n"
            "1. For <UPDATED_EXISTING_FILE:...> blocks: Partial excerpts only. Understand the intent and implement using YOUR full context. "
            "DO NOT copy wholesale - adapt ideas to fit actual structure.\n"
            "2. For <NEWFILE:...> blocks: Understand proposal and create properly. Verify completeness (imports, syntax, logic).\n"
            "3. Validation: After ALL changes, verify correctness using available tools (build/compile, linters, tests, type checks, etc.).\n"
            f"4. Cleanup: After you're done reading and applying changes, delete `{artifact_path}` once verified to prevent stale instructions.\n"
            "\n"
            "Treat this as a patch-set requiring manual integration, not ready-to-paste code. You have full codebase context - use it."
        )

    @staticmethod
    def _join_sections(*sections: str) -> str:
        chunks: list[str] = []
        for section in sections:
            if section:
                trimmed = section.strip()
                if trimmed:
                    chunks.append(trimmed)
        return "\n\n".join(chunks)

    def get_websearch_guidance(self) -> str:
        """
        Return Chat tool-style web search guidance.
        """
        return self.get_chat_style_websearch_guidance()

    def get_websearch_instruction(self, tool_specific: Optional[str] = None) -> str:
        if self._active_cli_seat is not None:
            return ""
        return super().get_websearch_instruction(tool_specific)

    # === CLI seat routing ===

    def _resolves_as_api_model(self, model_name: str) -> bool:
        try:
            provider = self.get_model_provider(model_name)
            provider.get_capabilities(model_name)
        except Exception:
            return False
        return True

    def resolve_cli_seat(self, model_name: Optional[str]) -> Optional[CLISeat]:
        """Resolve a subscription CLI seat name; a declared but disabled seat fails loudly."""
        registry, error, declared_names = get_cli_seat_state(self._resolves_as_api_model)
        normalized = (model_name or "").strip().lower()
        base_name, _, option = normalized.partition(":")
        if registry is None:
            if base_name in declared_names:
                raise ValueError(f"CLI seat '{model_name}' is configured but disabled: {error}")
            return None
        if option and registry.resolve(base_name) is not None:
            raise ValueError(f"CLI seat '{base_name}' takes no ':{option}' suffix; set thinking_mode instead.")
        return registry.resolve(normalized)

    def validate_cli_seat_request(self, request, seat: CLISeat) -> None:
        if self.get_request_continuation_id(request):
            raise ValueError(
                f"CLI seat '{seat.name}' is single-shot: continuation_id is not supported. "
                "Start a new call with the full context inlined or passed via absolute_file_paths."
            )
        if self.get_request_images(request):
            raise ValueError(f"CLI seat '{seat.name}' is prompt-only: images are not supported.")
        requested_mode = request.thinking_mode if "thinking_mode" in request.model_fields_set else None
        effort_for_thinking_mode(requested_mode, default=CHAT_SEAT_DEFAULT_EFFORT)

    async def generate_model_response(
        self,
        *,
        request,
        provider,
        prompt: str,
        system_prompt: str,
        temperature: float,
        thinking_mode: Optional[str],
        images: Optional[list],
    ):
        seat = self._active_cli_seat
        self._seat_metadata = None
        if seat is None:
            return await super().generate_model_response(
                request=request,
                provider=provider,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                thinking_mode=thinking_mode,
                images=images,
            )

        requested_mode = request.thinking_mode if "thinking_mode" in request.model_fields_set else None
        effort = effort_for_thinking_mode(requested_mode, default=CHAT_SEAT_DEFAULT_EFFORT)
        chat_seat = dataclasses.replace(
            seat,
            effort=effort,
            cli_timeout_s=min(seat.cli_timeout_s, CHAT_SEAT_CLI_TIMEOUT_S),
            min_cli_budget_s=min(seat.min_cli_budget_s, CHAT_SEAT_CLI_TIMEOUT_S),
        )
        fallback_supports_thinking = self._model_context.capabilities.supports_extended_thinking
        fallback_thinking_mode = (requested_mode or CHAT_SEAT_DEFAULT_EFFORT) if fallback_supports_thinking else None

        async def api_fallback() -> dict[str, Any]:
            try:
                response = await asyncio.to_thread(
                    provider.generate_content,
                    prompt=prompt,
                    model_name=seat.fallback_model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    thinking_mode=fallback_thinking_mode,
                    images=None,
                )
            except Exception as exc:
                return {"status": "error", "error": describe_exception(exc)}
            if not (response.content or "").strip():
                return {"status": "error", "error": "API fallback returned an empty response"}
            response_metadata = response.metadata if isinstance(response.metadata, dict) else {}
            return {
                "status": "success",
                "text": response.content,
                "model_used": response.model_name or seat.fallback_model,
                "metadata": {"provider_model_name": response_metadata.get("provider_model_name")},
                "usage": response.usage,
            }

        outcome = await consult_seat_with_fallback(
            chat_seat,
            system_prompt=system_prompt,
            build_prompt=lambda: prompt,
            deadline_at=time.monotonic() + CHAT_SEAT_DEADLINE_S,
            api_fallback=api_fallback,
            instruction=CLI_ADVISOR_INSTRUCTION,
            instructions_header=CLI_ADVISOR_INSTRUCTIONS_HEADER,
            header=CLI_ADVISOR_HEADER,
            queue_wait_s=CHAT_SEAT_QUEUE_WAIT_S,
            enforce_fallback_deadline=True,
        )
        if outcome.status != "success" or not outcome.text.strip():
            raise ValueError(outcome.error or f"CLI seat '{seat.name}' and its API fallback returned no answer")

        frontend = get_current_client_frontend()
        self._seat_metadata = {
            "cli_seat": seat.name,
            "transport": outcome.backend,
            "model_used": outcome.model_used,
            "effort": outcome.effort if outcome.backend == "cli" else None,
            "fallback_model": seat.fallback_model,
            "fallback_reason": outcome.fallback_reason,
            "same_vendor_as_host": frontend in seat.host_dedup_frontends,
            "attempts": outcome.attempts,
            "duration_seconds": outcome.duration_seconds,
        }
        if outcome.backend == "api_fallback":
            self._seat_metadata["fallback_thinking_mode"] = fallback_thinking_mode
        if outcome.cli_error:
            self._seat_metadata["cli_error"] = outcome.cli_error
        provider_model_name = (
            f"{seat.client}:{outcome.model_used}"
            if outcome.backend == "cli"
            else ((outcome.fallback_payload or {}).get("metadata") or {}).get("provider_model_name")
        )
        return ModelResponse(
            content=outcome.text,
            usage=outcome.usage or (outcome.fallback_payload or {}).get("usage") or {},
            model_name=outcome.model_used or seat.name,
            friendly_name=seat.name,
            metadata={"provider_model_name": provider_model_name} if provider_model_name else {},
        )

    def _create_continuation_offer(self, request, model_info: Optional[dict] = None):
        if self._active_cli_seat is not None:
            return None
        return super()._create_continuation_offer(request, model_info)

    def _parse_response(self, raw_text: str, request, model_info: Optional[dict] = None):
        tool_output = super()._parse_response(raw_text, request, model_info)
        if self._active_cli_seat is not None and self._seat_metadata:
            merged = dict(tool_output.metadata or {})
            merged.update(self._seat_metadata)
            if self._seat_metadata["transport"] == "cli":
                merged["provider_used"] = "cli"
            tool_output.metadata = merged
        return tool_output


logger = logging.getLogger(__name__)
