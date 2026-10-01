"""
Consensus tool - Multi-model consensus with blinded panel consultation

This tool gathers independent verdicts from a roster of models. In the default
parallel mode every model is consulted concurrently in a single call; the
legacy sequential mode consults one model per step so the CLI agent can note
each response before the next consultation.

Key features:
- Parallel panel consultation (default) with a panel deadline
- Sequential step-by-step mode retained for callers that drive the loop
- Blinded consultations: every model sees only the original proposal + files
- Context-aware file embedding
- Support for stance-based analysis (for/against/neutral)
- Final synthesis performed by the calling agent
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, model_validator

if TYPE_CHECKING:
    from tools.models import ToolModelCategory

from mcp.types import TextContent

from clink.agents.base import PROCESS_KILL_REAP_SECONDS, PROCESS_TERMINATION_GRACE_SECONDS
from config import TEMPERATURE_ANALYTICAL
from systemprompts import CONSENSUS_PROMPT
from tools.consensus_cli_seats import (
    CLISeat,
    CLISeatConfigError,
    CLISeatRegistry,
    consult_seat_with_fallback,
    describe_exception,
    get_cli_seat_state,
    run_cli_seat,
)
from tools.shared.base_models import WorkflowRequest
from tools.shared.base_tool import BaseTool
from tools.shared.exceptions import ToolExecutionError
from utils.client_info import get_current_client_frontend
from utils.conversation_memory import MAX_CONVERSATION_TURNS, add_turn, create_thread, get_thread

from .workflow.base import WorkflowTool

logger = logging.getLogger(__name__)

CONSENSUS_CHAIN_STATE_KEY = "consensus_chain_state"

CONSENSUS_MODE_PARALLEL = "parallel"
CONSENSUS_MODE_SEQUENTIAL = "sequential"
CONSENSUS_MODES = (CONSENSUS_MODE_PARALLEL, CONSENSUS_MODE_SEQUENTIAL)

CLI_CLEANUP_DRAIN_MARGIN_S = 2.0
CONSENSUS_PANEL_DEADLINE_ENV = "CONSENSUS_PANEL_DEADLINE_S"
DEFAULT_CONSENSUS_PANEL_DEADLINE_S = 1500.0
PANEL_DRAIN_TIMEOUT_S = max(
    5.0, PROCESS_TERMINATION_GRACE_SECONDS + 2 * PROCESS_KILL_REAP_SECONDS + CLI_CLEANUP_DRAIN_MARGIN_S
)

SYNTHESIS_NEXT_STEPS = (
    "CONSENSUS GATHERING IS COMPLETE. You MUST now synthesize all perspectives and present:\n"
    "1. Key points of AGREEMENT across models\n"
    "2. Key points of DISAGREEMENT and why they differ\n"
    "3. Your final consolidated recommendation\n"
    "4. Specific, actionable next steps for implementation\n"
    "5. Critical risks or concerns that must be addressed\n"
    "6. Finish with exactly one line: VERDICT: <approve|revise|reject>"
)


@dataclass
class ConsensusChainState:
    """Mutable state owned by one consensus continuation chain."""

    original_proposal: str
    models_to_consult: list[dict[str, Any]]
    relevant_files: list[str] = field(default_factory=list)
    images: list[str] = field(default_factory=list)
    accumulated_responses: list[dict[str, Any]] = field(default_factory=list)
    work_history: list[dict[str, Any]] = field(default_factory=list)
    host_frontend: str = "unknown"
    host_skipped_models: list[dict[str, Any]] = field(default_factory=list)
    mode: str = CONSENSUS_MODE_SEQUENTIAL

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot for conversation storage."""
        return {
            "version": 2,
            "original_proposal": self.original_proposal,
            "models_to_consult": self.models_to_consult,
            "relevant_files": self.relevant_files,
            "images": self.images,
            "accumulated_responses": self.accumulated_responses,
            "work_history": self.work_history,
            "host_frontend": self.host_frontend,
            "host_skipped_models": self.host_skipped_models,
            "mode": self.mode,
        }

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> ConsensusChainState:
        """Restore a chain snapshot from conversation storage."""
        stored_mode = metadata.get("mode")
        restored_mode = stored_mode if stored_mode in CONSENSUS_MODES else CONSENSUS_MODE_SEQUENTIAL
        return cls(
            original_proposal=str(metadata.get("original_proposal") or ""),
            models_to_consult=list(metadata.get("models_to_consult") or []),
            relevant_files=list(metadata.get("relevant_files") or []),
            images=list(metadata.get("images") or []),
            accumulated_responses=list(metadata.get("accumulated_responses") or []),
            work_history=list(metadata.get("work_history") or []),
            host_frontend=str(metadata.get("host_frontend") or "unknown"),
            host_skipped_models=list(metadata.get("host_skipped_models") or []),
            mode=restored_mode,
        )


HOST_MODEL_SKIP_ALIASES = {
    "codex": {
        "gpt-6-astra-pro",
        "gpt6-astra-pro",
        "gpt6astrapro",
        "gpt-6-pro",
        "gpt6-pro",
        "astra-pro",
        "gpt-5.6-sol-pro",
        "gpt5.6-sol-pro",
        "gpt5.6solpro",
        "gpt-5.6-pro",
        "gpt5.6-pro",
        "gpt-5.5-pro",
        "gpt5.5-pro",
        "gpt5.5pro",
        "gpt5pro",
        "gpt5-pro",
        "zen-gpt-5.5-pro",
        "zen-gpt5.5-pro",
        "zen-gpt55-pro",
    },
    "claude": {
        "fable-latest",
        "fable-5",
        "fable5",
        "claude-fable-latest",
        "claude-fable-5",
        "anthropic/claude-fable-latest",
        "anthropic/claude-fable-5",
    },
}

# Tool-specific field descriptions for consensus workflow
CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS = {
    "step": (
        "Consensus prompt. Step 1: write the exact proposal/question every model will see (use 'Evaluate…', not meta commentary). "
        "Steps 2+: capture internal notes about the latest model response—these notes are NOT sent to other models."
    ),
    "step_number": (
        "Current step index (starts at 1). Parallel mode: always 1 — the whole panel is consulted in this call. "
        "Sequential mode: step 1 is your analysis + the first model; steps 2+ handle each further model response."
    ),
    "total_steps": (
        "Sequential mode: number of models consulted. Parallel mode: the server sets this to 1. "
        "If you send mode 'parallel' you may still fill in the model count — the response tells you whether "
        "the panel is complete."
    ),
    "next_step_required": (
        "True if more model consultations remain; set false when ready to synthesize. "
        "Parallel mode always returns false — stop and synthesize when you see it."
    ),
    "mode": (
        "'parallel' (default): every model is consulted concurrently in this single call and the response carries "
        "all verdicts in accumulated_responses; only step_number 1 is valid and a later step on the same "
        "continuation_id is rejected. 'sequential': legacy one-model-per-step loop. Send mode explicitly together "
        "with the normal step fields (step_number 1, total_steps = model count, next_step_required true) and stop "
        "when the response reports consensus_complete or next_step_required false; otherwise continue the loop. "
        "When mode is omitted and the request looks like a step loop (total_steps > 1 or next_step_required true), "
        "the server runs sequentially."
    ),
    "findings": (
        "Step 1: your independent analysis for later synthesis (not shared with other models). Steps 2+: summarize the newest model response."
    ),
    "relevant_files": "Optional supporting files that help the consensus analysis. Must be absolute full, non-abbreviated paths.",
    "models": (
        "User-specified list of models to consult (provide at least two entries). "
        "Each entry may include model, stance (for/against/neutral), and stance_prompt. "
        "Each (model, stance) pair must be unique, e.g. [{'model':'gpt5','stance':'for'}, {'model':'pro','stance':'against'}]. "
        "CLI seats (configured in cli_seats.json, e.g. astra-cli, fable-cli) run through a subscription CLI in an "
        "isolated prompt-only review and fall back to their configured API model; each leg's metadata.backend "
        "records which answered."
    ),
    "current_model_index": "0-based index of the next model to consult (managed internally).",
    "model_responses": "Internal log of responses gathered so far.",
    "images": "Optional absolute image paths or base64 references that add helpful visual context.",
}


class ConsensusRequest(WorkflowRequest):
    """Request model for consensus workflow steps"""

    # Required fields for each step
    step: str = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step"])
    step_number: int = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step_number"])
    total_steps: int = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["total_steps"])
    next_step_required: bool = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["next_step_required"])

    # Investigation tracking fields
    findings: str = Field(..., description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["findings"])
    confidence: str = Field(default="exploring", exclude=True, description="Not used")

    # Consensus-specific fields (only needed in step 1)
    mode: Literal["parallel", "sequential"] | None = Field(
        None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["mode"]
    )
    models: list[dict] | None = Field(None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["models"])
    relevant_files: list[str] | None = Field(
        default_factory=list,
        description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["relevant_files"],
    )

    # Internal tracking fields
    current_model_index: int | None = Field(
        0,
        description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["current_model_index"],
    )
    model_responses: list[dict] | None = Field(
        default_factory=list,
        description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["model_responses"],
    )

    # Optional images for visual debugging
    images: list[str] | None = Field(default=None, description=CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["images"])

    # Override inherited fields to exclude them from schema
    temperature: float | None = Field(default=None, exclude=True)
    thinking_mode: str | None = Field(default=None, exclude=True)

    # Not used in consensus workflow
    files_checked: list[str] | None = Field(default_factory=list, exclude=True)
    relevant_context: list[str] | None = Field(default_factory=list, exclude=True)
    issues_found: list[dict] | None = Field(default_factory=list, exclude=True)
    hypothesis: str | None = Field(None, exclude=True)

    @model_validator(mode="after")
    def validate_step_one_requirements(self):
        """Ensure step 1 has required models field and unique model+stance combinations."""
        if self.step_number == 1:
            if not self.models:
                raise ValueError("Step 1 requires 'models' field to specify which models to consult")

            # Check for unique model + stance combinations
            seen_combinations = set()
            for model_config in self.models:
                model_name = model_config.get("model", "")
                stance = model_config.get("stance", "neutral")
                combination = f"{model_name}:{stance}"

                if combination in seen_combinations:
                    raise ValueError(
                        f"Duplicate model + stance combination found: {model_name} with stance '{stance}'. "
                        f"Each model + stance combination must be unique."
                    )
                seen_combinations.add(combination)

        return self


class ConsensusTool(WorkflowTool):
    """
    Consensus workflow tool for step-by-step multi-model consensus gathering.

    This tool implements a structured consensus workflow where the CLI agent first provides
    its own neutral analysis, then consults each specified model individually,
    and finally synthesizes all perspectives into a unified recommendation.
    """

    def __init__(self):
        # Consensus implements its own request-scoped workflow orchestration.
        # Initializing BaseWorkflowMixin would add singleton chain state to the
        # registry-owned tool instance, so initialize only BaseTool metadata.
        BaseTool.__init__(self)

    def get_name(self) -> str:
        return "consensus"

    def get_description(self) -> str:
        return (
            "Builds multi-model consensus through systematic analysis and structured debate. "
            "Use for complex decisions, architectural choices, feature proposals, and technology evaluations. "
            "Consults multiple models with different stances to synthesize comprehensive recommendations."
        )

    def get_system_prompt(self) -> str:
        # For the CLI agent's initial analysis, use a neutral version of the consensus prompt
        return CONSENSUS_PROMPT.replace(
            "{stance_prompt}",
            """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
        )

    def get_default_temperature(self) -> float:
        return TEMPERATURE_ANALYTICAL

    def get_model_category(self) -> ToolModelCategory:
        """Consensus workflow requires extended reasoning"""
        from tools.models import ToolModelCategory

        return ToolModelCategory.EXTENDED_REASONING

    def get_workflow_request_model(self):
        """Return the consensus workflow-specific request model."""
        return ConsensusRequest

    def get_input_schema(self) -> dict[str, Any]:
        """Generate input schema for consensus workflow."""
        from .workflow.schema_builders import WorkflowSchemaBuilder

        # Consensus tool-specific field definitions
        consensus_field_overrides = {
            # Override standard workflow fields that need consensus-specific descriptions
            "step": {
                "type": "string",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step"],
            },
            "step_number": {
                "type": "integer",
                "minimum": 1,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["step_number"],
            },
            "total_steps": {
                "type": "integer",
                "minimum": 1,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["total_steps"],
            },
            "next_step_required": {
                "type": "boolean",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["next_step_required"],
            },
            "findings": {
                "type": "string",
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["findings"],
            },
            "relevant_files": {
                "type": "array",
                "items": {"type": "string"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["relevant_files"],
            },
            # consensus-specific fields (not in base workflow)
            "mode": {
                "type": "string",
                "enum": list(CONSENSUS_MODES),
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["mode"],
            },
            "models": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "model": {"type": "string"},
                        "stance": {"type": "string", "enum": ["for", "against", "neutral"], "default": "neutral"},
                        "stance_prompt": {"type": "string"},
                    },
                    "required": ["model"],
                },
                "description": (
                    "User-specified roster of models to consult (provide at least two entries). "
                    + CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["models"]
                ),
                "minItems": 2,
            },
            "current_model_index": {
                "type": "integer",
                "minimum": 0,
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["current_model_index"],
            },
            "model_responses": {
                "type": "array",
                "items": {"type": "object"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["model_responses"],
            },
            "images": {
                "type": "array",
                "items": {"type": "string"},
                "description": CONSENSUS_WORKFLOW_FIELD_DESCRIPTIONS["images"],
            },
        }

        # Provide guidance on available models similar to single-model tools
        model_description = (
            "When the user names a model, you MUST use that exact value or report the "
            "provider error—never swap in another option (a CLI seat's own configured API fallback is the only "
            "substitution, and it is reported in metadata). Use the `listmodels` tool for the full roster."
        )
        seat_registry = self._cli_seat_registry()
        if seat_registry is not None and seat_registry.seats:
            seat_summaries = "; ".join(
                f"{seat.name} ({seat.client} CLI {seat.model} @ {seat.effort}, fallback {seat.fallback_model})"
                for seat in seat_registry.seats
            )
            model_description = f"{model_description} CLI seats: {seat_summaries}."

        summaries, total, restricted = self._get_ranked_model_summaries()
        remainder = max(0, total - len(summaries))
        if summaries:
            label = "Allowed models" if restricted else "Top models"
            top_line = "; ".join(summaries)
            if remainder > 0:
                top_line = f"{label}: {top_line}; +{remainder} more via `listmodels`."
            else:
                top_line = f"{label}: {top_line}."
            model_description = f"{model_description} {top_line}"
        else:
            model_description = (
                f"{model_description} No models detected—configure provider credentials or use the `listmodels` tool "
                "to inspect availability."
            )

        restriction_note = self._get_restriction_note()
        if restriction_note and (remainder > 0 or not summaries):
            model_description = f"{model_description} {restriction_note}."

        existing_models_desc = consensus_field_overrides["models"]["description"]
        consensus_field_overrides["models"]["description"] = f"{existing_models_desc} {model_description}"

        # Define excluded fields for consensus workflow
        excluded_workflow_fields = [
            "files_checked",  # Not used in consensus workflow
            "relevant_context",  # Not used in consensus workflow
            "issues_found",  # Not used in consensus workflow
            "hypothesis",  # Not used in consensus workflow
            "confidence",  # Not used in consensus workflow
        ]

        excluded_common_fields = [
            "model",  # Consensus uses 'models' field instead
            "temperature",  # Not used in consensus workflow
            "thinking_mode",  # Not used in consensus workflow
        ]

        requires_model = self.requires_model()
        model_field_schema = self.get_model_field_schema() if requires_model else None
        auto_mode = self.is_effective_auto_mode() if requires_model else False

        return WorkflowSchemaBuilder.build_schema(
            tool_specific_fields=consensus_field_overrides,
            model_field_schema=model_field_schema,
            auto_mode=auto_mode,
            tool_name=self.get_name(),
            excluded_workflow_fields=excluded_workflow_fields,
            excluded_common_fields=excluded_common_fields,
            require_model=requires_model,
        )

    def get_required_actions(
        self, step_number: int, confidence: str, findings: str, total_steps: int, request=None
    ) -> list[str]:  # noqa: ARG002
        """Define required actions for each consensus phase.

        Now includes request parameter for continuation-aware decisions.
        Note: confidence parameter is kept for compatibility with base class but not used.
        """
        if step_number == 1:
            # CLI Agent's initial analysis
            return [
                "You've provided your initial analysis. The tool will now consult other models.",
                "Wait for the next step to receive the first model's response.",
            ]
        elif step_number < total_steps - 1:
            # Processing individual model responses
            return [
                "Review the model response provided in this step",
                "Note key agreements and disagreements with previous analyses",
                "Wait for the next model's response",
            ]
        else:
            # Ready for final synthesis
            return [
                "All models have been consulted",
                "Synthesize all perspectives into a comprehensive recommendation",
                "Identify key points of agreement and disagreement",
                "Provide clear, actionable guidance based on the consensus",
            ]

    def should_call_expert_analysis(self, consolidated_findings, request=None) -> bool:
        """Consensus workflow doesn't use traditional expert analysis - it consults models step by step."""
        return False

    def prepare_expert_analysis_context(self, consolidated_findings) -> str:
        """Not used in consensus workflow."""
        return ""

    def requires_expert_analysis(self) -> bool:
        """Consensus workflow handles its own model consultations."""
        return False

    def requires_model(self) -> bool:
        """
        Consensus tool doesn't require model resolution at the MCP boundary.

        Uses it's own set of models

        Returns:
            bool: False
        """
        return False

    # Hook method overrides for consensus-specific behavior

    def prepare_step_data(self, request) -> dict:
        """Prepare consensus-specific step data."""
        step_data = {
            "step": request.step,
            "step_number": request.step_number,
            "findings": request.findings,
            "files_checked": [],  # Not used
            "relevant_files": request.relevant_files or [],
            "relevant_context": [],  # Not used
            "issues_found": [],  # Not used
            "confidence": "exploring",  # Not used, kept for compatibility
            "hypothesis": None,  # Not used
            "images": request.images or [],  # Now used for visual context
        }
        return step_data

    async def handle_work_completion(
        self,
        response_data: dict,
        request,
        arguments: dict,
        chain_state: ConsensusChainState | None = None,
    ) -> dict:  # noqa: ARG002
        """Handle consensus workflow completion - no expert analysis, just final synthesis."""
        chain_state = chain_state or ConsensusChainState(original_proposal=request.step, models_to_consult=[])
        response_data["consensus_complete"] = True
        response_data["status"] = "consensus_workflow_complete"

        # Prepare final synthesis data
        response_data["complete_consensus"] = {
            "initial_prompt": chain_state.original_proposal,
            "models_consulted": [
                m["model"] + ":" + m.get("stance", "neutral") for m in chain_state.accumulated_responses
            ],
            "total_responses": len(chain_state.accumulated_responses),
            "consensus_confidence": "high",  # Consensus complete
        }

        response_data["next_steps"] = SYNTHESIS_NEXT_STEPS

        return response_data

    def handle_work_continuation(
        self,
        response_data: dict,
        request,
        chain_state: ConsensusChainState | None = None,
    ) -> dict:
        """Handle continuation between consensus steps."""
        request_models = getattr(request, "models", None)
        if not isinstance(request_models, list):
            request_models = []
        chain_state = chain_state or ConsensusChainState(
            original_proposal=request.step,
            models_to_consult=list(request_models),
        )
        current_idx = request.current_model_index or 0

        if request.step_number == 1:
            # After CLI Agent's initial analysis, prepare to consult first model
            response_data["status"] = "consulting_models"
            response_data["next_model"] = chain_state.models_to_consult[0] if chain_state.models_to_consult else None
            response_data["next_steps"] = (
                "Your initial analysis is complete. The tool will now consult the specified models."
            )
        elif current_idx < len(chain_state.models_to_consult):
            next_model = chain_state.models_to_consult[current_idx]
            response_data["status"] = "consulting_next_model"
            response_data["next_model"] = next_model
            response_data["models_remaining"] = len(chain_state.models_to_consult) - current_idx
            response_data["next_steps"] = f"Model consultation in progress. Next: {next_model['model']}"
        else:
            response_data["status"] = "ready_for_synthesis"
            response_data["next_steps"] = "All models consulted. Ready for final synthesis."

        return response_data

    async def execute_workflow(self, arguments: dict[str, Any]) -> list:
        """Dispatch a consensus request to the parallel panel or the sequential step loop."""

        # Validate request
        request = self.get_workflow_request_model()(**arguments)

        # Resolve existing continuation_id or create a new one on first step
        continuation_id = request.continuation_id

        if request.step_number == 1:
            _registry, seat_config_error, declared_seat_names = self._cli_seat_state()
            if seat_config_error is not None:
                requested_seats = [
                    str(model_config.get("model", ""))
                    for model_config in (request.models or [])
                    if str(model_config.get("model", "")).strip().lower() in declared_seat_names
                ]
                if requested_seats:
                    raise ValueError(
                        f"CLI seat configuration is invalid, so {', '.join(requested_seats)} cannot run: "
                        f"{seat_config_error}"
                    )
            (
                filtered_models,
                skipped_models,
                frontend,
            ) = self._apply_host_model_skip_policy(request.models or [])

            if skipped_models:
                skipped_labels = ", ".join(item["model"] for item in skipped_models)
                logger.info("Consensus host policy skipped model(s) for %s: %s", frontend, skipped_labels)

            if len(filtered_models) < 2:
                skipped_labels = ", ".join(item["model"] for item in skipped_models) or "none"
                raise ValueError(
                    "Consensus host model policy left fewer than two models to consult "
                    f"for frontend '{frontend}'. Skipped: {skipped_labels}"
                )

            request.models = filtered_models
            arguments["models"] = filtered_models

            mode = self._resolve_mode(request)
            request.mode = mode
            arguments["mode"] = mode

            if not continuation_id:
                clean_args = {k: v for k, v in arguments.items() if k not in ["_model_context", "_resolved_model_name"]}
                continuation_id = create_thread(self.get_name(), clean_args)
                request.continuation_id = continuation_id
                arguments["continuation_id"] = continuation_id

            chain_state = ConsensusChainState(
                original_proposal=request.step,
                models_to_consult=list(request.models or []),
                relevant_files=list(request.relevant_files or []),
                images=list(request.images or []),
                host_frontend=frontend,
                host_skipped_models=skipped_models,
                mode=mode,
            )
            if mode == CONSENSUS_MODE_PARALLEL:
                request.total_steps = 1
                request.next_step_required = False
            else:
                # Set total steps: len(models) (each step includes consultation + response)
                request.total_steps = len(chain_state.models_to_consult)
        else:
            if not continuation_id:
                raise ValueError(f"Consensus step {request.step_number} requires a continuation_id")
            chain_state = self._restore_chain_state(continuation_id)
            if chain_state is None:
                raise ToolExecutionError(
                    json.dumps(self._build_restore_miss_payload(request, continuation_id), ensure_ascii=False)
                )

            if request.mode and request.mode != chain_state.mode:
                logger.warning(
                    "Consensus step %s requested mode '%s' but the continuation was started in '%s' mode; "
                    "using the stored mode",
                    request.step_number,
                    request.mode,
                    chain_state.mode,
                )
            request.mode = chain_state.mode
            arguments["mode"] = chain_state.mode

            if chain_state.mode == CONSENSUS_MODE_PARALLEL:
                raise ToolExecutionError(
                    json.dumps(
                        self._build_parallel_step_rejection_payload(request, continuation_id, chain_state),
                        ensure_ascii=False,
                    )
                )

            request.total_steps = len(chain_state.models_to_consult)

        if chain_state.mode == CONSENSUS_MODE_PARALLEL:
            return await self._execute_parallel_panel(request, arguments, continuation_id, chain_state)

        return await self._execute_sequential_step(request, arguments, continuation_id, chain_state)

    def _resolve_mode(self, request) -> str:
        """Pick the consultation mode for a step-1 request."""
        if request.mode in CONSENSUS_MODES:
            return request.mode

        legacy_loop_signature = request.total_steps > 1 or bool(request.next_step_required)
        if legacy_loop_signature:
            logger.info(
                "consensus: legacy step-loop signature (total_steps=%s, next_step_required=%s) — running sequential",
                request.total_steps,
                request.next_step_required,
            )
            return CONSENSUS_MODE_SEQUENTIAL

        logger.info(
            "consensus: mode omitted and request is not loop-shaped — consulting all %s models in parallel",
            len(request.models or []),
        )
        return CONSENSUS_MODE_PARALLEL

    def _build_restore_miss_payload(self, request, continuation_id: str) -> dict[str, Any]:
        """Error payload for a continuation whose chain state is gone (e.g. server restarted)."""
        return {
            "status": "error",
            "content": (
                f"Consensus state for continuation_id '{continuation_id}' was not found (server restarted?). "
                "If an earlier response reported consensus_complete or next_step_required false, synthesize from "
                "its accumulated_responses — do NOT re-run the panel. If a sequential loop was interrupted part-way, "
                "the verdicts you already hold are a partial panel: name the models that never answered rather than "
                "re-running everything. Only start a new round at step_number 1 if you have no verdicts at all."
            ),
            "next_step_required": False,
            "consensus_complete": False,
            "metadata": {
                "tool_name": self.get_name(),
                "continuation_id": continuation_id,
                "step_number": request.step_number,
            },
        }

    def _build_parallel_step_rejection_payload(
        self,
        request,
        continuation_id: str,
        chain_state: ConsensusChainState,
    ) -> dict[str, Any]:
        """Error payload for a step >= 2 on a continuation whose panel already ran in parallel."""
        panel = self._summarize_panel(chain_state, chain_state.accumulated_responses, deadline_seconds=None)
        if panel["succeeded"] > 0:
            guidance = "Do NOT re-run — synthesize from the accumulated_responses in this payload" + (
                f" (a partial panel: {', '.join(panel['failed_models'])} did not answer; say so)."
                if panel["failed"] > 0
                else "."
            )
        else:
            guidance = (
                "That panel produced no successful verdict, so there is nothing to synthesize from these error "
                "records; if you still need a review, start a new round at step_number 1."
            )
        return {
            "status": "error",
            "mode": CONSENSUS_MODE_PARALLEL,
            "content": (
                f"Consensus already ran the whole panel in parallel mode for continuation_id '{continuation_id}'; "
                f"step_number {request.step_number} is not valid. {guidance}"
            ),
            "next_step_required": False,
            "consensus_complete": panel["failed"] == 0 and panel["succeeded"] > 0,
            "panel": panel,
            "accumulated_responses": chain_state.accumulated_responses,
            "metadata": {
                "tool_name": self.get_name(),
                "continuation_id": continuation_id,
                "mode": CONSENSUS_MODE_PARALLEL,
                "host_frontend": chain_state.host_frontend,
            },
        }

    async def _execute_sequential_step(
        self,
        request,
        arguments: dict[str, Any],
        continuation_id: str | None,
        chain_state: ConsensusChainState,
    ) -> list:
        """Consult exactly one model for this step (legacy one-model-per-step protocol)."""
        # For all steps (1 through total_steps), consult the corresponding model
        if request.step_number <= request.total_steps:
            # Calculate which model to consult for this step
            model_idx = request.step_number - 1  # 0-based index

            if model_idx < len(chain_state.models_to_consult):
                # Track workflow state for conversation memory
                step_data = self.prepare_step_data(request)
                chain_state.work_history.append(step_data)

                # Consult the model for this step
                model_response = await self._consult_model(
                    chain_state.models_to_consult[model_idx],
                    request,
                    original_proposal=chain_state.original_proposal,
                    relevant_files=chain_state.relevant_files,
                    images=chain_state.images,
                )

                # Add to accumulated responses
                chain_state.accumulated_responses.append(model_response)

                # Include the model response in the step data
                response_data = {
                    "status": "model_consulted",
                    "step_number": request.step_number,
                    "total_steps": request.total_steps,
                    "model_consulted": model_response["model"],
                    "model_stance": model_response.get("stance", "neutral"),
                    "model_response": model_response,
                    "current_model_index": model_idx + 1,
                    "next_step_required": request.step_number < request.total_steps,
                }

                # Add CLAI Agent's analysis to step 1
                if request.step_number == 1:
                    response_data["agent_analysis"] = {
                        "initial_analysis": request.step,
                        "findings": request.findings,
                    }
                    response_data["status"] = "analysis_and_first_model_consulted"

                # Check if this is the final step
                if request.step_number == request.total_steps:
                    response_data["status"] = "consensus_workflow_complete"
                    response_data["consensus_complete"] = True
                    response_data["complete_consensus"] = {
                        "initial_prompt": chain_state.original_proposal,
                        "models_consulted": [
                            f"{m['model']}:{m.get('stance', 'neutral')}" for m in chain_state.accumulated_responses
                        ],
                        "total_responses": len(chain_state.accumulated_responses),
                        "consensus_confidence": "high",
                    }
                    response_data["next_steps"] = SYNTHESIS_NEXT_STEPS
                else:
                    response_data["next_steps"] = (
                        f"Model {model_response['model']} has provided its {model_response.get('stance', 'neutral')} "
                        f"perspective. Please analyze this response and call {self.get_name()} again with:\n"
                        f"- step_number: {request.step_number + 1}\n"
                        f"- findings: Summarize key points from this model's response"
                    )

                # Add continuation information and workflow customization
                response_data = self.customize_workflow_response(response_data, request, chain_state)

                # Ensure consensus-specific metadata is attached
                self._add_workflow_metadata(response_data, arguments)

                if continuation_id:
                    self._store_chain_turn(continuation_id, response_data, chain_state)
                    continuation_offer = self._build_continuation_offer(continuation_id)
                    if continuation_offer:
                        response_data["continuation_offer"] = continuation_offer

                return [TextContent(type="text", text=json.dumps(response_data, indent=2, ensure_ascii=False))]

        raise ValueError(
            f"Consensus step_number {request.step_number} exceeds the {request.total_steps} configured model steps"
        )

    async def _execute_parallel_panel(
        self,
        request,
        arguments: dict[str, Any],
        continuation_id: str | None,
        chain_state: ConsensusChainState,
    ) -> list:
        """Consult every panel model concurrently and return all verdicts in one response."""
        step_data = self.prepare_step_data(request)
        chain_state.work_history.append(step_data)

        deadline_seconds = self._panel_deadline_seconds()
        if not chain_state.models_to_consult:
            empty_panel = self._summarize_panel(chain_state, [], deadline_seconds)
            raise ToolExecutionError(
                json.dumps(
                    {
                        "status": "error",
                        "mode": CONSENSUS_MODE_PARALLEL,
                        "content": "Consensus has no models to consult after host policy was applied",
                        "next_step_required": False,
                        "consensus_complete": False,
                        "panel": empty_panel,
                        "accumulated_responses": [],
                        "continuation_id": continuation_id,
                        "metadata": {"tool_name": self.get_name(), "host_frontend": chain_state.host_frontend},
                    },
                    ensure_ascii=False,
                )
            )

        deadline_at = time.monotonic() + deadline_seconds
        seat_progress: dict[int, dict[str, Any]] = {}
        tasks = [
            asyncio.create_task(
                self._consult_model(
                    model_config,
                    request,
                    original_proposal=chain_state.original_proposal,
                    relevant_files=chain_state.relevant_files,
                    images=chain_state.images,
                    deadline_at=deadline_at,
                    seat_progress=seat_progress,
                )
            )
            for model_config in chain_state.models_to_consult
        ]

        try:
            _done, pending = await asyncio.wait(tasks, timeout=deadline_seconds)
            await self._cancel_panel_tasks(pending)
        except asyncio.CancelledError:
            # Persist first, synchronously: a second cancellation must not be able to lose finished legs.
            for task in tasks:
                task.cancel()
            results = self._collect_panel_results(
                chain_state, tasks, deadline_seconds, unfinished_status="cancelled", seat_progress=seat_progress
            )
            self._store_partial_panel_after_cancel(
                request, arguments, continuation_id, chain_state, results, deadline_seconds
            )
            try:
                await self._cancel_panel_tasks(tasks)
            except asyncio.CancelledError:
                logger.info("consensus: cancelled again while draining panel legs; abandoning drain")
            raise

        results = self._collect_panel_results(
            chain_state, tasks, deadline_seconds, unfinished_status="timed_out", seat_progress=seat_progress
        )
        chain_state.accumulated_responses = results

        response_data = self._build_panel_response(request, chain_state, results, deadline_seconds)
        response_data = self.customize_workflow_response(response_data, request, chain_state)
        self._add_workflow_metadata(response_data, arguments)

        panel = response_data["panel"]
        response_data["metadata"]["consensus_complete"] = panel["failed"] == 0

        if continuation_id:
            self._store_chain_turn(continuation_id, response_data, chain_state)
            continuation_offer = self._build_continuation_offer(continuation_id)
            if continuation_offer:
                response_data["continuation_offer"] = continuation_offer

        if panel["succeeded"] == 0:
            failure_payload = {
                "status": "error",
                "mode": CONSENSUS_MODE_PARALLEL,
                "content": f"All {panel['consulted']} consensus models failed",
                "next_step_required": False,
                "consensus_complete": False,
                "panel": panel,
                "accumulated_responses": results,
                "continuation_id": continuation_id,
                "metadata": response_data.get("metadata", {}),
            }
            raise ToolExecutionError(json.dumps(failure_payload, ensure_ascii=False))

        return [TextContent(type="text", text=json.dumps(response_data, indent=2, ensure_ascii=False))]

    def _panel_deadline_seconds(self) -> float:
        """Wall-clock budget for the parallel panel, from the environment or the default."""
        raw_value = os.getenv(CONSENSUS_PANEL_DEADLINE_ENV, "").strip()
        if not raw_value:
            return DEFAULT_CONSENSUS_PANEL_DEADLINE_S
        try:
            parsed_value = float(raw_value)
        except ValueError:
            logger.warning(
                "Ignoring invalid %s=%r; using %ss",
                CONSENSUS_PANEL_DEADLINE_ENV,
                raw_value,
                DEFAULT_CONSENSUS_PANEL_DEADLINE_S,
            )
            return DEFAULT_CONSENSUS_PANEL_DEADLINE_S
        if not math.isfinite(parsed_value) or parsed_value <= 0:
            logger.warning(
                "Ignoring non-positive or non-finite %s=%r; using %ss",
                CONSENSUS_PANEL_DEADLINE_ENV,
                raw_value,
                DEFAULT_CONSENSUS_PANEL_DEADLINE_S,
            )
            return DEFAULT_CONSENSUS_PANEL_DEADLINE_S
        return parsed_value

    async def _cancel_panel_tasks(self, tasks) -> None:
        """Cancel unfinished legs and retrieve their exceptions so nothing is left dangling."""
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        if not unfinished:
            return
        drained, still_pending = await asyncio.wait(unfinished, timeout=PANEL_DRAIN_TIMEOUT_S)
        for task in drained:
            if not task.cancelled():
                task.exception()
        if still_pending:
            logger.warning(
                "consensus: %s panel leg(s) did not acknowledge cancellation within %ss; leaving them to the "
                "provider timeout",
                len(still_pending),
                PANEL_DRAIN_TIMEOUT_S,
            )

    def _collect_panel_results(
        self,
        chain_state: ConsensusChainState,
        tasks,
        deadline_seconds: float,
        *,
        unfinished_status: str,
        seat_progress: dict[int, dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Map every leg to a result dict in roster order, whatever state its task ended in."""
        progress_by_config = seat_progress if seat_progress is not None else {}
        results: list[dict[str, Any]] = []
        for model_config, task in zip(chain_state.models_to_consult, tasks):
            model_name = model_config["model"]
            stance = model_config.get("stance", "neutral")

            if not task.done() or task.cancelled():
                seat = self._resolve_cli_seat(model_name)
                if unfinished_status != "timed_out":
                    unfinished_error = "consultation cancelled before the model responded"
                elif seat is not None:
                    unfinished_error = (
                        f"did not complete within panel deadline {deadline_seconds:g}s; the {seat.client} CLI process "
                        "group was terminated (an API fallback call, if one had started, runs until its HTTP read "
                        "timeout)"
                    )
                else:
                    unfinished_error = (
                        f"did not complete within panel deadline {deadline_seconds:g}s; provider call abandoned "
                        "(thread runs until the HTTP read timeout, cost still incurred)"
                    )
                unfinished_result = {
                    "model": model_name,
                    "stance": stance,
                    "status": unfinished_status,
                    "error": unfinished_error,
                }
                if seat is not None:
                    unfinished_result["metadata"] = self._unfinished_seat_metadata(
                        seat, progress_by_config.get(id(model_config))
                    )
                results.append(unfinished_result)
                continue

            exception = task.exception()
            if exception is not None:
                results.append(
                    {"model": model_name, "stance": stance, "status": "error", "error": describe_exception(exception)}
                )
                continue

            results.append(task.result())
        return results

    def _summarize_panel(
        self,
        chain_state: ConsensusChainState,
        results: list[dict[str, Any]],
        deadline_seconds: float | None,
    ) -> dict[str, Any]:
        """Counts describing how much of the requested panel actually answered."""
        consulted = len(chain_state.models_to_consult)
        skipped = len(chain_state.host_skipped_models)
        succeeded = [item for item in results if item.get("status") == "success"]
        unsuccessful = [item for item in results if item.get("status") != "success"]
        timed_out = [item for item in results if item.get("status") == "timed_out"]
        cancelled = [item for item in results if item.get("status") == "cancelled"]
        failed_models = [f"{item.get('model')}:{item.get('stance', 'neutral')}" for item in unsuccessful]
        return {
            "requested": consulted + skipped,
            "consulted": consulted,
            "succeeded": len(succeeded),
            "failed": len(unsuccessful),
            "timed_out": len(timed_out),
            "cancelled": len(cancelled),
            "skipped_by_host_policy": skipped,
            "failed_models": failed_models,
            "deadline_seconds": deadline_seconds,
        }

    def _build_panel_response(
        self,
        request,
        chain_state: ConsensusChainState,
        results: list[dict[str, Any]],
        deadline_seconds: float,
    ) -> dict[str, Any]:
        """Response body for a parallel panel run (complete or partial)."""
        panel = self._summarize_panel(chain_state, results, deadline_seconds)
        complete = panel["failed"] == 0

        next_steps = SYNTHESIS_NEXT_STEPS
        if not complete:
            next_steps = (
                f"{SYNTHESIS_NEXT_STEPS}\n\n"
                f"WARNING: {panel['failed']} of {panel['consulted']} models failed "
                f"({', '.join(panel['failed_models'])}). Do not present the remaining responses as unanimous; "
                "name the missing models."
            )

        return {
            "status": "consensus_workflow_complete" if complete else "consensus_workflow_partial",
            "mode": CONSENSUS_MODE_PARALLEL,
            "step_number": 1,
            "total_steps": 1,
            "next_step_required": False,
            "panel_size": panel["consulted"],
            "panel": panel,
            "agent_analysis": {
                "initial_analysis": request.step,
                "findings": request.findings,
            },
            "consensus_complete": complete,
            "complete_consensus": {
                "initial_prompt": chain_state.original_proposal,
                "models_consulted": [
                    f"{m['model']}:{m.get('stance', 'neutral')}" for m in chain_state.models_to_consult
                ],
                "total_responses": len(results),
                "consensus_confidence": "high" if complete else "partial",
            },
            "next_steps": next_steps,
        }

    def _store_partial_panel_after_cancel(
        self,
        request,
        arguments: dict[str, Any],
        continuation_id: str | None,
        chain_state: ConsensusChainState,
        results: list[dict[str, Any]],
        deadline_seconds: float,
    ) -> None:
        """Best-effort persistence of the legs that finished before the MCP call was cancelled."""
        if not continuation_id:
            return
        try:
            chain_state.accumulated_responses = results
            response_data = self._build_panel_response(request, chain_state, results, deadline_seconds)
            response_data = self.customize_workflow_response(response_data, request, chain_state)
            self._add_workflow_metadata(response_data, arguments)
            self._store_chain_turn(continuation_id, response_data, chain_state)
        except Exception:
            logger.exception("Failed to store partial consensus panel after cancellation")

    def _apply_host_model_skip_policy(self, models: list[dict]) -> tuple[list[dict], list[dict[str, Any]], str]:
        """Remove models that should not be consulted from the current frontend."""
        frontend = get_current_client_frontend()
        skipped_aliases = HOST_MODEL_SKIP_ALIASES.get(frontend, set())
        filtered_models = []
        skipped_models = []
        for model_config in models:
            model_name = str(model_config.get("model", "")).strip()
            normalized_model_name = model_name.lower()
            seat = self._resolve_cli_seat(model_name)
            if seat is not None:
                if str(frontend).strip().lower() in seat.host_dedup_frontends:
                    skipped_models.append(
                        {
                            "model": model_name,
                            "stance": model_config.get("stance", "neutral"),
                            "frontend": frontend,
                        }
                    )
                else:
                    filtered_models.append(model_config)
                continue

            skip_for_capability = False
            try:
                provider = self.get_model_provider(model_name)
                capabilities = provider.get_capabilities(model_name)
                configured_frontends = {
                    str(configured_frontend).strip().lower()
                    for configured_frontend in getattr(capabilities, "host_dedup_frontends", [])
                }
                skip_for_capability = frontend in configured_frontends
            except Exception:
                # Resolution must remain fail-soft so the compatibility table
                # and normal model validation can still handle the request.
                pass

            if skip_for_capability or normalized_model_name in skipped_aliases:
                skipped_models.append(
                    {
                        "model": model_name,
                        "stance": model_config.get("stance", "neutral"),
                        "frontend": frontend,
                    }
                )
                continue

            filtered_models.append(model_config)

        return filtered_models, skipped_models, frontend

    def _build_continuation_offer(self, continuation_id: str) -> dict[str, Any] | None:
        """Create a continuation offer without exposing prior model responses."""
        try:
            from tools.models import ContinuationOffer

            thread = get_thread(continuation_id)
            if thread and thread.turns:
                remaining_turns = max(0, MAX_CONVERSATION_TURNS - len(thread.turns))
            else:
                remaining_turns = MAX_CONVERSATION_TURNS - 1

            # Provide a neutral note specific to consensus workflow
            note = (
                f"Consensus workflow can continue for {remaining_turns} more exchanges."
                if remaining_turns > 0
                else "Consensus workflow continuation limit reached."
            )

            continuation_offer = ContinuationOffer(
                continuation_id=continuation_id,
                note=note,
                remaining_turns=remaining_turns,
            )
            return continuation_offer.model_dump()
        except Exception:
            return None

    def _restore_chain_state(self, continuation_id: str) -> ConsensusChainState | None:
        """Restore only the state belonging to the requested continuation."""
        thread = get_thread(continuation_id)
        if not thread:
            return None

        for turn in reversed(thread.turns):
            if turn.role != "assistant" or turn.tool_name != self.get_name() or not turn.model_metadata:
                continue
            stored_state = turn.model_metadata.get(CONSENSUS_CHAIN_STATE_KEY)
            if isinstance(stored_state, dict):
                return ConsensusChainState.from_metadata(stored_state)
        return None

    def _store_chain_turn(
        self,
        continuation_id: str,
        response_data: dict[str, Any],
        chain_state: ConsensusChainState,
    ) -> None:
        """Persist a chain snapshot without touching the singleton tool instance."""
        clean_content = self._extract_clean_workflow_content_for_history(response_data)
        state_metadata = chain_state.to_metadata()
        add_turn(
            thread_id=continuation_id,
            role="assistant",
            content=clean_content,
            tool_name=self.get_name(),
            files=list(chain_state.relevant_files),
            images=list(chain_state.images),
            model_metadata={
                CONSENSUS_CHAIN_STATE_KEY: state_metadata,
                # Preserve the generic workflow keys for existing conversation tooling.
                "work_history": state_metadata["work_history"],
                "initial_request": state_metadata["original_proposal"],
            },
        )

    async def _consult_model(
        self,
        model_config: dict,
        request,
        *,
        original_proposal: str | None = None,
        relevant_files: list[str] | None = None,
        images: list[str] | None = None,
        deadline_at: float | None = None,
        seat_progress: dict[int, dict[str, Any]] | None = None,
    ) -> dict:
        """Consult a single panel seat: a CLI seat when the name is registered as one, otherwise an API model."""
        seat = self._resolve_cli_seat(model_config.get("model"))
        if seat is not None:
            return await self._consult_cli_seat(
                seat,
                model_config,
                request,
                original_proposal=original_proposal,
                relevant_files=relevant_files,
                images=images,
                deadline_at=deadline_at,
                seat_progress=seat_progress,
            )
        return await self._consult_api_model(
            model_config,
            request,
            original_proposal=original_proposal,
            relevant_files=relevant_files,
            images=images,
        )

    def _build_blinded_prompt(
        self,
        model_context,
        request,
        *,
        original_proposal: str | None,
        relevant_files: list[str] | None,
    ) -> str:
        """Build the blinded consultation prompt shared by API and CLI legs."""
        # Use continuation_id=None for blinded consensus - each model should only see
        # original prompt + files, not conversation history or other model responses
        # CRITICAL: Use the original proposal from step 1, NOT what's in request.step for steps 2+!
        # Steps 2+ contain summaries/notes that must NEVER be sent to other models
        prompt = original_proposal or request.step
        files_for_consultation = relevant_files if relevant_files is not None else request.relevant_files
        if files_for_consultation:
            file_content, _ = self._prepare_file_content_for_prompt(
                files_for_consultation,
                None,  # Use None instead of request.continuation_id for blinded consensus
                "Context files",
                model_context=model_context,
            )
            if file_content:
                prompt = f"{prompt}\n\n=== CONTEXT FILES ===\n{file_content}\n=== END CONTEXT ==="
        return prompt

    async def _consult_api_model(
        self,
        model_config: dict,
        request,
        *,
        original_proposal: str | None = None,
        relevant_files: list[str] | None = None,
        images: list[str] | None = None,
    ) -> dict:
        """Consult a single API model and return its response."""
        try:
            # Import and create ModelContext once at the beginning
            from utils.model_context import ModelContext

            # Get the provider for this model
            model_name = model_config["model"]
            provider = self.get_model_provider(model_name)

            # Create model context once and reuse for both file processing and temperature validation
            model_context = ModelContext(model_name=model_name)

            prompt = self._build_blinded_prompt(
                model_context,
                request,
                original_proposal=original_proposal,
                relevant_files=relevant_files,
            )

            # Get stance-specific system prompt
            stance = model_config.get("stance", "neutral")
            stance_prompt = model_config.get("stance_prompt")
            system_prompt = self._get_stance_enhanced_prompt(stance, stance_prompt)

            # Validate temperature against model constraints (respects supports_temperature)
            validated_temperature, temp_warnings = self.validate_and_correct_temperature(
                self.get_default_temperature(), model_context
            )

            # Log any temperature corrections
            for warning in temp_warnings:
                logger.warning(warning)

            # Call the model with validated temperature.
            # Run the sync provider SDK call on a worker thread so a blocked
            # HTTP read (e.g. zombie TCP after host suspend) cannot freeze the
            # asyncio event loop — see FORK.md.
            response = await asyncio.to_thread(
                provider.generate_content,
                prompt=prompt,
                model_name=model_name,
                system_prompt=system_prompt,
                temperature=validated_temperature,
                thinking_mode="medium",
                images=images if images is not None else (request.images if request.images else None),
            )

            response_metadata = response.metadata if isinstance(response.metadata, dict) else {}
            canonical_model_name = response.model_name or model_name
            provider_model_name = response_metadata.get("provider_model_name") or canonical_model_name

            return {
                "model": model_name,
                "stance": stance,
                "status": "success",
                "verdict": response.content,
                "metadata": {
                    "provider": provider.get_provider_type().value,
                    "requested_model_name": model_name,
                    "model_name": canonical_model_name,
                    "provider_model_name": provider_model_name,
                },
            }

        except Exception as e:
            logger.exception("Error consulting model %s", model_config)
            return {
                "model": model_config.get("model", "unknown"),
                "stance": model_config.get("stance", "neutral"),
                "status": "error",
                "error": str(e),
            }

    def _resolves_as_api_model(self, model_name: str) -> bool:
        try:
            provider = self.get_model_provider(model_name)
            provider.get_capabilities(model_name)
        except Exception:
            return False
        return True

    def _cli_seat_state(self) -> tuple[CLISeatRegistry | None, CLISeatConfigError | None, frozenset[str]]:
        """Shared seat registry state; see ``tools.consensus_cli_seats.get_cli_seat_state``."""
        return get_cli_seat_state(self._resolves_as_api_model)

    def _cli_seat_registry(self) -> CLISeatRegistry | None:
        return self._cli_seat_state()[0]

    def _resolve_cli_seat(self, model_name: str | None) -> CLISeat | None:
        registry = self._cli_seat_registry()
        if registry is None:
            return None
        return registry.resolve(model_name)

    def _cli_seat_base_metadata(self, seat: CLISeat) -> dict[str, Any]:
        return {
            "requested_model_name": seat.name,
            "cli_seat": seat.name,
            "cli_client": seat.client,
            "cli_model": seat.model,
            "fallback_model": seat.fallback_model,
            "requested_reasoning_effort": seat.effort,
        }

    def _unfinished_seat_metadata(self, seat: CLISeat, progress: dict[str, Any] | None) -> dict[str, Any]:
        """Metadata for a seat leg that was cancelled or timed out, including attempts recorded so far."""
        progress = progress or {}
        started = progress.get("started")
        return {
            **self._cli_seat_base_metadata(seat),
            "backend": progress.get("backend", "cli"),
            "effective_reasoning_effort": None,
            "attempts": list(progress.get("attempts") or []),
            "duration_seconds": round(time.monotonic() - started, 3) if started is not None else None,
        }

    async def _consult_cli_seat(
        self,
        seat: CLISeat,
        model_config: dict,
        request,
        *,
        original_proposal: str | None,
        relevant_files: list[str] | None,
        images: list[str] | None,
        deadline_at: float | None,
        seat_progress: dict[int, dict[str, Any]] | None = None,
    ) -> dict:
        """Run a CLI seat, falling back to its API model when the CLI cannot produce a verdict."""
        from utils.model_context import ModelContext

        stance = model_config.get("stance", "neutral")
        base_metadata = self._cli_seat_base_metadata(seat)
        effective_images = images if images is not None else (request.images or None)

        def build_prompt() -> str:
            return self._build_blinded_prompt(
                ModelContext(model_name=seat.fallback_model),
                request,
                original_proposal=original_proposal,
                relevant_files=relevant_files,
            )

        async def api_fallback() -> dict[str, Any]:
            fallback_config = {**model_config, "model": seat.fallback_model}
            fallback_result = await self._consult_api_model(
                fallback_config,
                request,
                original_proposal=original_proposal,
                relevant_files=relevant_files,
                images=images,
            )
            return {
                "status": fallback_result.get("status"),
                "text": fallback_result.get("verdict", ""),
                "metadata": fallback_result.get("metadata") or {},
                "error": fallback_result.get("error"),
            }

        progress: dict[str, Any] = {}
        if seat_progress is not None:
            seat_progress[id(model_config)] = progress
        outcome = await consult_seat_with_fallback(
            seat,
            system_prompt=self._get_stance_enhanced_prompt(stance, model_config.get("stance_prompt")),
            build_prompt=build_prompt,
            deadline_at=deadline_at,
            api_fallback=api_fallback,
            images_present=bool(effective_images),
            runner=run_cli_seat,
            enforce_fallback_deadline=True,
            progress=progress,
        )

        if outcome.status == "success" and outcome.backend == "cli":
            metadata = {
                **base_metadata,
                "provider": "cli",
                "backend": "cli",
                "model_name": outcome.model_used,
                "provider_model_name": f"{seat.client}:{outcome.model_used}",
                "effective_reasoning_effort": seat.effort,
                "fallback_reason": None,
                "attempts": outcome.attempts,
                "duration_seconds": outcome.duration_seconds,
            }
            if outcome.usage:
                metadata["usage"] = outcome.usage
            return {
                "model": seat.name,
                "stance": stance,
                "status": "success",
                "verdict": outcome.text,
                "metadata": metadata,
            }

        if outcome.status == "success":
            metadata = {
                **((outcome.fallback_payload or {}).get("metadata") or {}),
                **base_metadata,
                "backend": "api_fallback",
                "fallback_reason": outcome.fallback_reason,
                "effective_reasoning_effort": None,
                "attempts": outcome.attempts,
                "duration_seconds": outcome.duration_seconds,
            }
            if outcome.cli_error:
                metadata["cli_error"] = outcome.cli_error
            return {
                "model": seat.name,
                "stance": stance,
                "status": "success",
                "verdict": outcome.text,
                "metadata": metadata,
            }

        return {
            "model": seat.name,
            "stance": stance,
            "status": "error",
            "error": outcome.error,
            "metadata": {
                **((outcome.fallback_payload or {}).get("metadata") or {}),
                **base_metadata,
                "backend": outcome.backend,
                "fallback_reason": outcome.fallback_reason,
                "effective_reasoning_effort": None,
                "cli_error": outcome.cli_error,
                "attempts": outcome.attempts,
                "duration_seconds": outcome.duration_seconds,
            },
        }

    def _get_stance_enhanced_prompt(self, stance: str, custom_stance_prompt: str | None = None) -> str:
        """Get the system prompt with stance injection."""
        base_prompt = CONSENSUS_PROMPT

        if custom_stance_prompt:
            return base_prompt.replace("{stance_prompt}", custom_stance_prompt)

        stance_prompts = {
            "for": """SUPPORTIVE PERSPECTIVE WITH INTEGRITY

You are tasked with advocating FOR this proposal, but with CRITICAL GUARDRAILS:

MANDATORY ETHICAL CONSTRAINTS:
- This is NOT a debate for entertainment. You MUST act in good faith and in the best interest of the questioner
- You MUST think deeply about whether supporting this idea is safe, sound, and passes essential requirements
- You MUST be direct and unequivocal in saying "this is a bad idea" when it truly is
- There must be at least ONE COMPELLING reason to be optimistic, otherwise DO NOT support it

WHEN TO REFUSE SUPPORT (MUST OVERRIDE STANCE):
- If the idea is fundamentally harmful to users, project, or stakeholders
- If implementation would violate security, privacy, or ethical standards
- If the proposal is technically infeasible within realistic constraints
- If costs/risks dramatically outweigh any potential benefits

YOUR SUPPORTIVE ANALYSIS SHOULD:
- Identify genuine strengths and opportunities
- Propose solutions to overcome legitimate challenges
- Highlight synergies with existing systems
- Suggest optimizations that enhance value
- Present realistic implementation pathways

Remember: Being "for" means finding the BEST possible version of the idea IF it has merit, not blindly supporting bad ideas.""",
            "against": """CRITICAL PERSPECTIVE WITH RESPONSIBILITY

You are tasked with critiquing this proposal, but with ESSENTIAL BOUNDARIES:

MANDATORY FAIRNESS CONSTRAINTS:
- You MUST NOT oppose genuinely excellent, common-sense ideas just to be contrarian
- You MUST acknowledge when a proposal is fundamentally sound and well-conceived
- You CANNOT give harmful advice or recommend against beneficial changes
- If the idea is outstanding, say so clearly while offering constructive refinements

WHEN TO MODERATE CRITICISM (MUST OVERRIDE STANCE):
- If the proposal addresses critical user needs effectively
- If it follows established best practices with good reason
- If benefits clearly and substantially outweigh risks
- If it's the obvious right solution to the problem

YOUR CRITICAL ANALYSIS SHOULD:
- Identify legitimate risks and failure modes
- Point out overlooked complexities
- Suggest more efficient alternatives
- Highlight potential negative consequences
- Question assumptions that may be flawed

Remember: Being "against" means rigorous scrutiny to ensure quality, not undermining good ideas that deserve support.""",
            "neutral": """BALANCED PERSPECTIVE

Provide objective analysis considering both positive and negative aspects. However, if there is overwhelming evidence
that the proposal clearly leans toward being exceptionally good or particularly problematic, you MUST accurately
reflect this reality. Being "balanced" means being truthful about the weight of evidence, not artificially creating
50/50 splits when the reality is 90/10.

Your analysis should:
- Present all significant pros and cons discovered
- Weight them according to actual impact and likelihood
- If evidence strongly favors one conclusion, clearly state this
- Provide proportional coverage based on the strength of arguments
- Help the questioner see the true balance of considerations

Remember: Artificial balance that misrepresents reality is not helpful. True balance means accurate representation
of the evidence, even when it strongly points in one direction.""",
        }

        stance_prompt = stance_prompts.get(stance, stance_prompts["neutral"])
        return base_prompt.replace("{stance_prompt}", stance_prompt)

    def customize_workflow_response(
        self,
        response_data: dict,
        request,
        chain_state: ConsensusChainState | None = None,
    ) -> dict:
        """Customize response for consensus workflow."""
        # Store model responses in the response for tracking
        if chain_state and chain_state.accumulated_responses:
            response_data["accumulated_responses"] = chain_state.accumulated_responses

        # Add consensus-specific fields
        if chain_state and chain_state.mode == CONSENSUS_MODE_PARALLEL:
            response_data["consensus_workflow_status"] = "ready_for_synthesis"
        elif request.step_number == 1:
            response_data["consensus_workflow_status"] = "initial_analysis_complete"
        elif request.step_number < request.total_steps - 1:
            response_data["consensus_workflow_status"] = "consulting_models"
        else:
            response_data["consensus_workflow_status"] = "ready_for_synthesis"

        # Customize metadata for consensus workflow
        self._customize_consensus_metadata(response_data, request, chain_state)

        return response_data

    def _customize_consensus_metadata(
        self,
        response_data: dict,
        request,
        chain_state: ConsensusChainState | None = None,
    ) -> None:
        """
        Customize metadata for consensus workflow to accurately reflect multi-model nature.

        The default workflow metadata shows the model running Agent's analysis steps,
        but consensus is a multi-model tool that consults different models. We need
        to provide accurate metadata that reflects this.
        """
        if "metadata" not in response_data:
            response_data["metadata"] = {}

        metadata = response_data["metadata"]

        # Always preserve tool_name
        metadata["tool_name"] = self.get_name()
        metadata["host_frontend"] = chain_state.host_frontend if chain_state else "unknown"
        metadata["mode"] = chain_state.mode if chain_state else CONSENSUS_MODE_SEQUENTIAL

        if chain_state and chain_state.host_skipped_models:
            metadata["models_skipped_by_host_policy"] = chain_state.host_skipped_models

        if request.step_number == request.total_steps:
            # Final step - show comprehensive consensus metadata
            models_consulted = []
            if chain_state and chain_state.models_to_consult:
                models_consulted = [f"{m['model']}:{m.get('stance', 'neutral')}" for m in chain_state.models_to_consult]

            metadata.update(
                {
                    "workflow_type": "multi_model_consensus",
                    "models_consulted": models_consulted,
                    "consensus_complete": True,
                    "total_models": len(chain_state.models_to_consult) if chain_state else 0,
                }
            )

            # Remove the misleading single model metadata
            metadata.pop("model_used", None)
            metadata.pop("provider_used", None)

        else:
            # Intermediate steps - show consensus workflow in progress
            models_to_consult = []
            if chain_state and chain_state.models_to_consult:
                models_to_consult = [
                    f"{m['model']}:{m.get('stance', 'neutral')}" for m in chain_state.models_to_consult
                ]

            metadata.update(
                {
                    "workflow_type": "multi_model_consensus",
                    "models_to_consult": models_to_consult,
                    "consultation_step": request.step_number,
                    "total_consultation_steps": request.total_steps,
                }
            )

            # Remove the misleading single model metadata that shows Agent's execution model
            # instead of the models being consulted
            metadata.pop("model_used", None)
            metadata.pop("provider_used", None)

    def _add_workflow_metadata(self, response_data: dict, arguments: dict[str, Any]) -> None:
        """
        Override workflow metadata addition for consensus tool.

        The consensus tool doesn't use single model metadata because it's a multi-model
        workflow. Instead, we provide consensus-specific metadata that accurately
        reflects the models being consulted.
        """
        # Initialize metadata if not present
        if "metadata" not in response_data:
            response_data["metadata"] = {}

        # Add basic tool metadata
        response_data["metadata"]["tool_name"] = self.get_name()

        # The consensus-specific metadata is already added by _customize_consensus_metadata
        # which is called from customize_workflow_response. We don't add the standard
        # single-model metadata (model_used, provider_used) because it's misleading
        # for a multi-model consensus workflow.

        logger.debug(
            f"[CONSENSUS_METADATA] {self.get_name()}: Using consensus-specific metadata instead of single-model metadata"
        )

    # Required abstract methods from BaseTool
    def get_request_model(self):
        """Return the consensus workflow-specific request model."""
        return ConsensusRequest

    async def prepare_prompt(self, request) -> str:  # noqa: ARG002
        """Not used - workflow tools use execute_workflow()."""
        return ""  # Workflow tools use execute_workflow() directly
