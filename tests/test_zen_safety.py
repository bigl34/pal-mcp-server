"""Tests for dynamic OpenCode Zen safety classification."""

from utils.zen_safety import ZenSafetyPolicy


def test_build_manifest_marks_new_docs_free_model_unsafe():
    models_payload = {
        "data": [
            {"id": "new-alpha-model"},
            {"id": "deepseek-v4-flash"},
        ]
    }
    docs_html = """
    <table>
      <tr><th>Model</th><th>Model ID</th><th>Endpoint</th><th>AI SDK Package</th></tr>
      <tr>
        <td>New Alpha Model</td><td>new-alpha-model</td>
        <td><code>https://opencode.ai/zen/v1/chat/completions</code></td>
        <td><code>@ai-sdk/openai-compatible</code></td>
      </tr>
      <tr>
        <td>DeepSeek V4 Flash</td><td>deepseek-v4-flash</td>
        <td><code>https://opencode.ai/zen/v1/chat/completions</code></td>
        <td><code>@ai-sdk/openai-compatible</code></td>
      </tr>
    </table>
    <table>
      <tr><th>Model</th><th>Input</th><th>Output</th><th>Cached Read</th><th>Cached Write</th></tr>
      <tr><td>New Alpha Model</td><td>Free</td><td>Free</td><td>Free</td><td>-</td></tr>
      <tr><td>DeepSeek V4 Flash</td><td>$0.14</td><td>$0.28</td><td>$0.03</td><td>-</td></tr>
    </table>
    <h2 id="privacy">Privacy</h2>
    <ul>
      <li>New Alpha Model: During its free period, collected data may be used to improve the model.</li>
    </ul>
    """

    manifest = ZenSafetyPolicy.build_manifest(models_payload, docs_html, now=123)

    assert manifest["models"]["new-alpha-model"]["billing_tier"] == "free"
    assert manifest["models"]["new-alpha-model"]["retention_policy"] == "free_retained"
    assert manifest["models"]["new-alpha-model"]["runtime_allowed"] is False
    assert manifest["models"]["new-alpha-model"]["zdr_fallback_eligible"] is False
    assert manifest["models"]["deepseek-v4-flash"]["billing_tier"] == "paid"
    assert manifest["models"]["deepseek-v4-flash"]["retention_policy"] == "zero"
    assert manifest["models"]["deepseek-v4-flash"]["zdr_fallback_eligible"] is True


def test_build_manifest_requires_live_models_listing_presence():
    models_payload = {"data": []}
    docs_html = """
    <table>
      <tr><th>Model</th><th>Model ID</th><th>Endpoint</th><th>AI SDK Package</th></tr>
      <tr>
        <td>DeepSeek V4 Flash</td><td>deepseek-v4-flash</td>
        <td><code>https://opencode.ai/zen/v1/chat/completions</code></td>
        <td><code>@ai-sdk/openai-compatible</code></td>
      </tr>
    </table>
    <table>
      <tr><th>Model</th><th>Input</th><th>Output</th><th>Cached Read</th><th>Cached Write</th></tr>
      <tr><td>DeepSeek V4 Flash</td><td>$0.14</td><td>$0.28</td><td>$0.03</td><td>-</td></tr>
    </table>
    """

    manifest = ZenSafetyPolicy.build_manifest(models_payload, docs_html, now=123)

    assert manifest["models"]["deepseek-v4-flash"]["runtime_allowed"] is False
    assert manifest["models"]["deepseek-v4-flash"]["zdr_fallback_eligible"] is False
    assert "model is not present in the live Zen /models response" in manifest["models"]["deepseek-v4-flash"]["reasons"]


def test_build_manifest_marks_openai_models_runtime_allowed_but_not_zdr_fallback():
    models_payload = {"data": [{"id": "gpt-5.5"}]}
    docs_html = """
    <table>
      <tr><th>Model</th><th>Model ID</th><th>Endpoint</th><th>AI SDK Package</th></tr>
      <tr>
        <td>GPT 5.5</td><td>gpt-5.5</td>
        <td><code>https://opencode.ai/zen/v1/responses</code></td>
        <td><code>@ai-sdk/openai</code></td>
      </tr>
    </table>
    <table>
      <tr><th>Model</th><th>Input</th><th>Output</th><th>Cached Read</th><th>Cached Write</th></tr>
      <tr><td>GPT 5.5</td><td>$5.00</td><td>$30.00</td><td>$0.50</td><td>-</td></tr>
    </table>
    <h2 id="privacy">Privacy</h2>
    <ul><li>OpenAI APIs: Requests are retained for 30 days.</li></ul>
    """

    manifest = ZenSafetyPolicy.build_manifest(models_payload, docs_html, now=123)

    assert manifest["models"]["gpt-5.5"]["billing_tier"] == "paid"
    assert manifest["models"]["gpt-5.5"]["retention_policy"] == "retained_30d"
    assert manifest["models"]["gpt-5.5"]["runtime_allowed"] is True
    assert manifest["models"]["gpt-5.5"]["zdr_fallback_eligible"] is False


def test_free_like_detection_blocks_any_free_substring():
    assert ZenSafetyPolicy.is_free_like("mimo-v2.5-free")
    assert ZenSafetyPolicy.is_free_like("vendor/freePreview")
    assert ZenSafetyPolicy.is_free_like("prefix-freebie")
