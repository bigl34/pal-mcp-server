"""Parallel-panel behaviour of the consensus tool."""

import asyncio
import json
from types import MethodType

import pytest

from tools.consensus import (
    CONSENSUS_MODE_PARALLEL,
    CONSENSUS_MODE_SEQUENTIAL,
    CONSENSUS_PANEL_DEADLINE_ENV,
    ConsensusChainState,
    ConsensusRequest,
    ConsensusTool,
)
from tools.shared.exceptions import ToolExecutionError


def _panel(*names):
    return [{"model": name, "stance": "neutral"} for name in names]


def _step_one(models, **overrides):
    arguments = {
        "step": "Evaluate the proposal.",
        "step_number": 1,
        "total_steps": 1,
        "next_step_required": False,
        "findings": "initial analysis",
        "models": models,
    }
    arguments.update(overrides)
    return arguments


def _success(model_config, verdict=None):
    return {
        "model": model_config["model"],
        "stance": model_config.get("stance", "neutral"),
        "status": "success",
        "verdict": verdict or f"verdict from {model_config['model']}",
    }


def _install_fake_consult(tool, behaviour):
    calls = []

    async def consult_model(
        model_config,
        request,
        *,
        original_proposal=None,
        relevant_files=None,
        images=None,
        deadline_at=None,
        seat_progress=None,
    ):
        del request, relevant_files, images, deadline_at, seat_progress
        calls.append(model_config["model"])
        return await behaviour(model_config, original_proposal)

    tool._consult_model = consult_model
    return calls


async def _run(tool, arguments):
    response = await tool.execute(arguments)
    return json.loads(response[0].text)


class TestParallelPanel:
    async def test_parallel_default_consults_all_models_in_one_call(self):
        tool = ConsensusTool()
        started = {name: asyncio.Event() for name in ("alpha", "beta", "gamma")}
        finish_order = []

        async def behaviour(model_config, original_proposal):
            del original_proposal
            name = model_config["model"]
            started[name].set()
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started.values())), timeout=2)
            delay = {"alpha": 0.03, "beta": 0.01, "gamma": 0.02}[name]
            await asyncio.sleep(delay)
            finish_order.append(name)
            return _success(model_config)

        calls = _install_fake_consult(tool, behaviour)

        data = await _run(tool, _step_one(_panel("alpha", "beta", "gamma")))

        assert calls == ["alpha", "beta", "gamma"]
        assert finish_order == ["beta", "gamma", "alpha"]
        assert data["status"] == "consensus_workflow_complete"
        assert data["mode"] == CONSENSUS_MODE_PARALLEL
        assert data["total_steps"] == 1
        assert data["next_step_required"] is False
        assert data["consensus_complete"] is True
        assert data["panel_size"] == 3
        assert data["panel"]["succeeded"] == 3
        assert data["panel"]["failed"] == 0
        assert [item["model"] for item in data["accumulated_responses"]] == ["alpha", "beta", "gamma"]
        assert "VERDICT:" in data["next_steps"]
        assert "model_consulted" not in data
        assert "model_response" not in data
        assert data["consensus_workflow_status"] == "ready_for_synthesis"
        assert data["metadata"]["mode"] == CONSENSUS_MODE_PARALLEL
        assert data["metadata"]["consensus_complete"] is True
        assert data["complete_consensus"]["consensus_confidence"] == "high"
        assert data["continuation_offer"]["continuation_id"]

    async def test_partial_failure_is_flagged_not_hidden(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            if model_config["model"] == "beta":
                return {"model": "beta", "stance": "neutral", "status": "error", "error": "provider exploded"}
            return _success(model_config)

        _install_fake_consult(tool, behaviour)

        data = await _run(tool, _step_one(_panel("alpha", "beta", "gamma")))

        assert data["status"] == "consensus_workflow_partial"
        assert data["consensus_complete"] is False
        assert data["metadata"]["consensus_complete"] is False
        assert data["panel"]["failed"] == 1
        assert data["panel"]["failed_models"] == ["beta:neutral"]
        assert data["complete_consensus"]["consensus_confidence"] == "partial"
        assert "WARNING: 1 of 3 models failed (beta:neutral)" in data["next_steps"]
        assert data["accumulated_responses"][1]["error"] == "provider exploded"

    async def test_all_failed_raises_tool_execution_error_but_keeps_state(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            return {"model": model_config["model"], "stance": "neutral", "status": "error", "error": "down"}

        _install_fake_consult(tool, behaviour)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(_step_one(_panel("alpha", "beta")))

        payload = json.loads(str(excinfo.value))
        assert payload["status"] == "error"
        assert payload["mode"] == CONSENSUS_MODE_PARALLEL
        assert payload["next_step_required"] is False
        assert payload["consensus_complete"] is False
        assert payload["panel"]["succeeded"] == 0
        assert payload["panel"]["failed"] == 2
        assert [item["model"] for item in payload["accumulated_responses"]] == ["alpha", "beta"]
        restored = tool._restore_chain_state(payload["continuation_id"])
        assert restored is not None
        assert restored.mode == CONSENSUS_MODE_PARALLEL

    async def test_step_two_on_parallel_continuation_is_rejected_without_redispatch(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            return _success(model_config)

        calls = _install_fake_consult(tool, behaviour)
        first = await _run(tool, _step_one(_panel("alpha", "beta")))
        continuation_id = first["continuation_offer"]["continuation_id"]
        assert calls == ["alpha", "beta"]

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                {
                    "step": "notes",
                    "step_number": 2,
                    "total_steps": 2,
                    "next_step_required": False,
                    "findings": "notes",
                    "continuation_id": continuation_id,
                }
            )

        payload = json.loads(str(excinfo.value))
        assert "Do NOT re-run" in payload["content"]
        assert [item["model"] for item in payload["accumulated_responses"]] == ["alpha", "beta"]
        assert payload["panel"]["succeeded"] == 2
        assert calls == ["alpha", "beta"]

    async def test_leg_exception_is_normalised_to_error(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            if model_config["model"] == "beta":
                raise RuntimeError("boom")
            return _success(model_config)

        _install_fake_consult(tool, behaviour)

        data = await _run(tool, _step_one(_panel("alpha", "beta", "gamma")))

        assert data["status"] == "consensus_workflow_partial"
        assert data["accumulated_responses"][1]["status"] == "error"
        assert data["accumulated_responses"][1]["error"] == "RuntimeError: boom"
        assert data["accumulated_responses"][0]["status"] == "success"
        assert data["accumulated_responses"][2]["status"] == "success"

    async def test_legacy_signature_without_mode_runs_sequentially(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            return _success(model_config)

        calls = _install_fake_consult(tool, behaviour)

        data = await _run(
            tool,
            _step_one(_panel("alpha", "beta", "gamma"), total_steps=3, next_step_required=True),
        )

        assert calls == ["alpha"]
        assert data["status"] == "analysis_and_first_model_consulted"
        assert data["next_step_required"] is True
        assert data["total_steps"] == 3
        assert data["metadata"]["mode"] == CONSENSUS_MODE_SEQUENTIAL

    async def test_explicit_parallel_mode_wins_over_loop_shaped_fields(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            return _success(model_config)

        calls = _install_fake_consult(tool, behaviour)

        data = await _run(
            tool,
            _step_one(_panel("alpha", "beta", "gamma"), mode="parallel", total_steps=3, next_step_required=True),
        )

        assert calls == ["alpha", "beta", "gamma"]
        assert data["mode"] == CONSENSUS_MODE_PARALLEL
        assert data["total_steps"] == 1
        assert data["next_step_required"] is False

    async def test_panel_deadline_marks_slow_leg_timed_out(self, monkeypatch):
        tool = ConsensusTool()
        monkeypatch.setenv(CONSENSUS_PANEL_DEADLINE_ENV, "0.05")
        slow_leg_cancelled = asyncio.Event()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            if model_config["model"] == "beta":
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    slow_leg_cancelled.set()
                    raise
            return _success(model_config)

        _install_fake_consult(tool, behaviour)

        data = await _run(tool, _step_one(_panel("alpha", "beta", "gamma")))

        assert slow_leg_cancelled.is_set()
        assert data["status"] == "consensus_workflow_partial"
        assert data["panel"]["timed_out"] == 1
        assert data["panel"]["failed"] == 1
        assert data["panel"]["deadline_seconds"] == 0.05
        assert data["accumulated_responses"][1]["status"] == "timed_out"
        assert "panel deadline 0.05s" in data["accumulated_responses"][1]["error"]
        assert data["accumulated_responses"][0]["status"] == "success"
        assert data["accumulated_responses"][2]["status"] == "success"

    @pytest.mark.parametrize("raw_value", ["soon", "-3", "0", "nan", "inf", "-inf", "1e309"])
    async def test_invalid_deadline_env_falls_back_to_default(self, monkeypatch, raw_value):
        tool = ConsensusTool()
        monkeypatch.setenv(CONSENSUS_PANEL_DEADLINE_ENV, raw_value)
        assert tool._panel_deadline_seconds() == 1500.0

    async def test_valid_deadline_env_is_used(self, monkeypatch):
        tool = ConsensusTool()
        monkeypatch.setenv(CONSENSUS_PANEL_DEADLINE_ENV, "42")
        assert tool._panel_deadline_seconds() == 42.0

    async def test_cancel_during_deadline_drain_still_stores_finished_legs(self, monkeypatch):
        tool = ConsensusTool()
        monkeypatch.setenv(CONSENSUS_PANEL_DEADLINE_ENV, "0.05")
        drain_started = asyncio.Event()
        thread_ids = []

        async def behaviour(model_config, original_proposal):
            del original_proposal
            if model_config["model"] == "beta":
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    drain_started.set()
                    await asyncio.sleep(0.3)
                    raise
            return _success(model_config)

        _install_fake_consult(tool, behaviour)
        real_store = tool._store_chain_turn

        def capture_store(self, continuation_id, response_data, chain_state):
            thread_ids.append(continuation_id)
            real_store(continuation_id, response_data, chain_state)

        tool._store_chain_turn = MethodType(capture_store, tool)

        panel_task = asyncio.create_task(tool.execute(_step_one(_panel("alpha", "beta"))))
        await asyncio.wait_for(drain_started.wait(), timeout=2)
        panel_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await panel_task

        assert thread_ids
        restored = tool._restore_chain_state(thread_ids[-1])
        assert restored.accumulated_responses[0]["status"] == "success"
        assert restored.accumulated_responses[1]["status"] == "cancelled"

    async def test_second_cancel_during_drain_keeps_stored_legs_and_still_cancels(self, monkeypatch):
        tool = ConsensusTool()
        beta_started = asyncio.Event()
        beta_cancel_seen = asyncio.Event()
        thread_ids = []

        async def behaviour(model_config, original_proposal):
            del original_proposal
            if model_config["model"] == "beta":
                beta_started.set()
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    beta_cancel_seen.set()
                    await asyncio.sleep(1)
                    raise
            return _success(model_config)

        _install_fake_consult(tool, behaviour)
        real_store = tool._store_chain_turn

        def capture_store(self, continuation_id, response_data, chain_state):
            thread_ids.append(continuation_id)
            real_store(continuation_id, response_data, chain_state)

        tool._store_chain_turn = MethodType(capture_store, tool)

        panel_task = asyncio.create_task(tool.execute(_step_one(_panel("alpha", "beta"))))
        await asyncio.wait_for(beta_started.wait(), timeout=2)
        await asyncio.sleep(0.01)
        panel_task.cancel()
        await asyncio.wait_for(beta_cancel_seen.wait(), timeout=2)
        panel_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(panel_task, timeout=3)

        assert thread_ids
        restored = tool._restore_chain_state(thread_ids[-1])
        assert restored.accumulated_responses[0]["status"] == "success"
        assert restored.accumulated_responses[1]["status"] == "cancelled"

    async def test_cancelled_partial_then_step_two_is_rejected_with_stored_legs(self):
        tool = ConsensusTool()
        beta_started = asyncio.Event()
        thread_ids = []

        async def behaviour(model_config, original_proposal):
            del original_proposal
            if model_config["model"] == "beta":
                beta_started.set()
                await asyncio.sleep(5)
            return _success(model_config)

        calls = _install_fake_consult(tool, behaviour)
        real_store = tool._store_chain_turn

        def capture_store(self, continuation_id, response_data, chain_state):
            thread_ids.append(continuation_id)
            real_store(continuation_id, response_data, chain_state)

        tool._store_chain_turn = MethodType(capture_store, tool)

        panel_task = asyncio.create_task(tool.execute(_step_one(_panel("alpha", "beta"))))
        await asyncio.wait_for(beta_started.wait(), timeout=2)
        await asyncio.sleep(0.01)
        panel_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await panel_task

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                {
                    "step": "notes",
                    "step_number": 2,
                    "total_steps": 2,
                    "next_step_required": False,
                    "findings": "notes",
                    "continuation_id": thread_ids[-1],
                }
            )

        payload = json.loads(str(excinfo.value))
        assert payload["next_step_required"] is False
        assert payload["consensus_complete"] is False
        assert "partial panel" in payload["content"]
        assert [item["status"] for item in payload["accumulated_responses"]] == ["success", "cancelled"]
        assert calls == ["alpha", "beta"]

    async def test_all_legs_timing_out_raises_with_terminal_flags(self, monkeypatch):
        tool = ConsensusTool()
        monkeypatch.setenv(CONSENSUS_PANEL_DEADLINE_ENV, "0.05")

        async def behaviour(model_config, original_proposal):
            del original_proposal
            await asyncio.sleep(5)
            return _success(model_config)

        _install_fake_consult(tool, behaviour)

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(_step_one(_panel("alpha", "beta")))

        payload = json.loads(str(excinfo.value))
        assert payload["panel"]["timed_out"] == 2
        assert payload["panel"]["succeeded"] == 0
        assert payload["next_step_required"] is False
        assert payload["consensus_complete"] is False

    async def test_empty_roster_is_a_tool_error_not_a_crash(self):
        tool = ConsensusTool()
        chain_state = ConsensusChainState(original_proposal="p", models_to_consult=[], mode="parallel")
        request = ConsensusRequest(
            step="p",
            step_number=1,
            total_steps=1,
            next_step_required=False,
            findings="f",
            models=[{"model": "x"}, {"model": "y"}],
        )

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool._execute_parallel_panel(request, {}, None, chain_state)

        payload = json.loads(str(excinfo.value))
        assert "no models to consult" in payload["content"]
        assert payload["panel"]["consulted"] == 0

    async def test_host_skip_counts_appear_in_panel(self, monkeypatch):
        from utils import client_info

        tool = ConsensusTool()
        monkeypatch.setenv("PAL_ASSISTANT_FRONTEND", "claude")
        monkeypatch.setattr(client_info, "_client_info_cache", None)

        async def behaviour(model_config, original_proposal):
            del original_proposal
            return _success(model_config)

        calls = _install_fake_consult(tool, behaviour)

        data = await _run(tool, _step_one(_panel("alpha", "fable-latest", "beta")))

        assert calls == ["alpha", "beta"]
        assert data["panel"]["skipped_by_host_policy"] == 1
        assert data["panel"]["requested"] == 3
        assert data["panel"]["consulted"] == 2
        assert data["metadata"]["models_skipped_by_host_policy"][0]["model"] == "fable-latest"

    async def test_step_two_after_all_failed_panel_does_not_ask_to_synthesize_errors(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            return {"model": model_config["model"], "stance": "neutral", "status": "error", "error": "down"}

        calls = _install_fake_consult(tool, behaviour)
        with pytest.raises(ToolExecutionError) as first:
            await tool.execute(_step_one(_panel("alpha", "beta")))
        continuation_id = json.loads(str(first.value))["continuation_id"]
        assert json.loads(str(first.value))["metadata"]["consensus_complete"] is False

        with pytest.raises(ToolExecutionError) as second:
            await tool.execute(
                {
                    "step": "notes",
                    "step_number": 2,
                    "total_steps": 2,
                    "next_step_required": False,
                    "findings": "notes",
                    "continuation_id": continuation_id,
                }
            )

        payload = json.loads(str(second.value))
        assert "no successful verdict" in payload["content"]
        assert "synthesize from the accumulated_responses" not in payload["content"]
        assert calls == ["alpha", "beta"]

    async def test_conflicting_mode_on_sequential_continuation_uses_stored_mode(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            del original_proposal
            return _success(model_config)

        calls = _install_fake_consult(tool, behaviour)
        first = await _run(
            tool, _step_one(_panel("alpha", "beta"), mode="sequential", total_steps=2, next_step_required=True)
        )
        assert calls == ["alpha"]

        second = await _run(
            tool,
            {
                "step": "notes",
                "step_number": 2,
                "mode": "parallel",
                "total_steps": 2,
                "next_step_required": False,
                "findings": "notes",
                "continuation_id": first["continuation_offer"]["continuation_id"],
            },
        )

        assert calls == ["alpha", "beta"]
        assert second["status"] == "consensus_workflow_complete"
        assert second["metadata"]["mode"] == CONSENSUS_MODE_SEQUENTIAL
        assert second["model_consulted"] == "beta"

    async def test_mcp_cancellation_stores_finished_legs(self):
        tool = ConsensusTool()
        beta_started = asyncio.Event()
        thread_ids = []

        async def behaviour(model_config, original_proposal):
            del original_proposal
            if model_config["model"] == "beta":
                beta_started.set()
                await asyncio.sleep(5)
            return _success(model_config)

        _install_fake_consult(tool, behaviour)

        real_store = tool._store_chain_turn

        def capture_store(self, continuation_id, response_data, chain_state):
            thread_ids.append(continuation_id)
            real_store(continuation_id, response_data, chain_state)

        tool._store_chain_turn = MethodType(capture_store, tool)

        panel_task = asyncio.create_task(tool.execute(_step_one(_panel("alpha", "beta"))))
        await asyncio.wait_for(beta_started.wait(), timeout=2)
        await asyncio.sleep(0.01)
        panel_task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await panel_task

        assert thread_ids, "partial panel should have been stored before re-raising"
        restored = tool._restore_chain_state(thread_ids[-1])
        assert restored is not None
        assert restored.mode == CONSENSUS_MODE_PARALLEL
        assert restored.accumulated_responses[0]["status"] == "success"
        assert restored.accumulated_responses[1]["status"] == "cancelled"

    async def test_overlapping_parallel_panels_stay_isolated(self):
        tool = ConsensusTool()

        async def behaviour(model_config, original_proposal):
            await asyncio.sleep(0.01)
            return _success(model_config, verdict=f"{original_proposal}:{model_config['model']}")

        _install_fake_consult(tool, behaviour)

        async def run_panel(label):
            return await _run(tool, _step_one(_panel(f"{label}-one", f"{label}-two"), step=label))

        first, second = await asyncio.gather(run_panel("proposal-a"), run_panel("proposal-b"))

        assert first["complete_consensus"]["initial_prompt"] == "proposal-a"
        assert second["complete_consensus"]["initial_prompt"] == "proposal-b"
        assert [item["verdict"] for item in first["accumulated_responses"]] == [
            "proposal-a:proposal-a-one",
            "proposal-a:proposal-a-two",
        ]
        assert [item["verdict"] for item in second["accumulated_responses"]] == [
            "proposal-b:proposal-b-one",
            "proposal-b:proposal-b-two",
        ]
        assert first["continuation_offer"]["continuation_id"] != second["continuation_offer"]["continuation_id"]

    async def test_restore_miss_on_step_two_has_anti_double_spend_guidance(self):
        tool = ConsensusTool()

        with pytest.raises(ToolExecutionError) as excinfo:
            await tool.execute(
                {
                    "step": "notes",
                    "step_number": 2,
                    "total_steps": 2,
                    "next_step_required": False,
                    "findings": "notes",
                    "continuation_id": "12345678-1234-1234-1234-123456789abc",
                }
            )

        payload = json.loads(str(excinfo.value))
        assert "do NOT re-run the panel" in payload["content"]
        assert "partial panel" in payload["content"]
        assert payload["metadata"]["step_number"] == 2


class TestChainStateMode:
    def test_metadata_round_trips_mode(self):
        state = ConsensusChainState(original_proposal="p", models_to_consult=_panel("a", "b"), mode="parallel")
        metadata = state.to_metadata()
        assert metadata["version"] == 2
        assert metadata["mode"] == CONSENSUS_MODE_PARALLEL
        assert ConsensusChainState.from_metadata(metadata).mode == CONSENSUS_MODE_PARALLEL

    def test_v1_metadata_restores_as_sequential(self):
        legacy = {
            "version": 1,
            "original_proposal": "p",
            "models_to_consult": _panel("a", "b"),
        }
        assert ConsensusChainState.from_metadata(legacy).mode == CONSENSUS_MODE_SEQUENTIAL

    def test_unknown_mode_restores_as_sequential(self):
        weird = {"version": 2, "original_proposal": "p", "models_to_consult": _panel("a", "b"), "mode": "turbo"}
        assert ConsensusChainState.from_metadata(weird).mode == CONSENSUS_MODE_SEQUENTIAL


def test_schema_advertises_mode(monkeypatch):
    tool = ConsensusTool()
    monkeypatch.setattr(tool, "_get_ranked_model_summaries", MethodType(lambda self, limit=5: ([], 0, False), tool))
    monkeypatch.setattr(tool, "_get_restriction_note", MethodType(lambda self: None, tool))

    schema = tool.get_input_schema()

    mode_schema = schema["properties"]["mode"]
    assert mode_schema["enum"] == ["parallel", "sequential"]
    assert "default" not in mode_schema
    assert "mode" not in schema.get("required", [])
    assert "consensus_complete" in mode_schema["description"]
