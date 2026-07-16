"""Tests for Gemini HTTP timeout handling."""

from providers.gemini import GeminiModelProvider


def test_gemini_timeout_converts_seconds_to_google_sdk_milliseconds():
    assert GeminiModelProvider._to_gemini_timeout_ms(300.0) == 300_000


def test_gemini_timeout_clamps_to_google_minimum_deadline():
    assert GeminiModelProvider._to_gemini_timeout_ms(1.0) == 10_000


def test_gemini_client_uses_converted_timeout(monkeypatch):
    captured = {}

    class FakeHttpOptions:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    class FakeClient:
        def __init__(self, *, api_key, http_options):
            captured["api_key"] = api_key
            captured["http_options"] = http_options

    monkeypatch.setattr("providers.gemini.types.HttpOptions", FakeHttpOptions)
    monkeypatch.setattr("providers.gemini.genai.Client", FakeClient)

    provider = GeminiModelProvider("test-key")
    provider._timeout_override = 300.0

    _ = provider.client

    assert captured["api_key"] == "test-key"
    assert captured["timeout"] == 300_000
