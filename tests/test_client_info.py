"""Tests for MCP client frontend detection."""

from utils.client_info import get_current_client_frontend, get_friendly_name


def test_codex_client_info_maps_to_codex(monkeypatch):
    monkeypatch.delenv("PAL_ASSISTANT_FRONTEND", raising=False)

    assert get_friendly_name("codex-pal-consensus-runner") == "Codex"
    assert get_current_client_frontend("codex-pal-consensus-runner") == "codex"


def test_frontend_env_overrides_client_info(monkeypatch):
    monkeypatch.setenv("PAL_ASSISTANT_FRONTEND", "codex")

    assert get_current_client_frontend("claude-code") == "codex"


def test_claude_code_client_info_maps_to_claude(monkeypatch):
    monkeypatch.delenv("PAL_ASSISTANT_FRONTEND", raising=False)

    assert get_current_client_frontend("claude-code") == "claude"
