"""Safety classification for OpenCode Zen models.

Zen's free-model set can change over time. PAL therefore treats Zen models as
unsafe unless a fresh safety manifest positively classifies them as paid and
non-free.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils.env import get_env

logger = logging.getLogger(__name__)

ZEN_MODELS_URL = "https://opencode.ai/zen/v1/models"
ZEN_DOCS_URL = "https://opencode.ai/docs/zen/"
DEFAULT_CACHE_TTL_SECONDS = 24 * 60 * 60
SUPPORTED_RUNTIME_ENDPOINTS = {"responses", "chat_completions"}
SUPPORTED_FALLBACK_ENDPOINTS = SUPPORTED_RUNTIME_ENDPOINTS
RUNTIME_RETENTION_POLICIES = {"zero", "retained_30d"}
KNOWN_FREE_MODEL_IDS = {
    "big-pickle",
    "deepseek-v4-flash-free",
    "mimo-v2.5-free",
    "minimax-m3-free",
    "nemotron-3-super-free",
    "qwen3.6-plus-free",
}


@dataclass(frozen=True)
class ZenSafetyRecord:
    """Safety metadata for one Zen model ID."""

    model_id: str
    display_name: str = ""
    billing_tier: str = "unknown"
    retention_policy: str = "unknown"
    runtime_allowed: bool = False
    zdr_fallback_eligible: bool = False
    reasons: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, model_id: str, raw: dict[str, Any]) -> ZenSafetyRecord:
        return cls(
            model_id=str(raw.get("model_id") or model_id),
            display_name=str(raw.get("display_name") or ""),
            billing_tier=str(raw.get("billing_tier") or "unknown"),
            retention_policy=str(raw.get("retention_policy") or "unknown"),
            runtime_allowed=bool(raw.get("runtime_allowed", False)),
            zdr_fallback_eligible=bool(raw.get("zdr_fallback_eligible", False)),
            reasons=tuple(str(reason) for reason in raw.get("reasons", []) if reason),
        )


class ZenSafetyPolicy:
    """Load and enforce the dynamic Zen safety manifest."""

    def __init__(
        self,
        *,
        cache_path: Path,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        refresh_disabled: bool = False,
    ) -> None:
        self.cache_path = cache_path
        self.ttl_seconds = ttl_seconds
        self.refresh_disabled = refresh_disabled
        self.manifest = self.load_or_refresh_manifest()

    @classmethod
    def from_env(cls) -> ZenSafetyPolicy:
        cache_path = Path(
            get_env("ZEN_SAFETY_CACHE_PATH") or str(Path.home() / ".cache" / "pal-mcp" / "zen_safety_manifest.json")
        )
        ttl_raw = get_env("ZEN_SAFETY_CACHE_TTL_SECONDS", str(DEFAULT_CACHE_TTL_SECONDS))
        try:
            ttl_seconds = max(1, int(ttl_raw or DEFAULT_CACHE_TTL_SECONDS))
        except ValueError:
            ttl_seconds = DEFAULT_CACHE_TTL_SECONDS

        return cls(
            cache_path=cache_path,
            ttl_seconds=ttl_seconds,
            refresh_disabled=str(get_env("ZEN_SAFETY_REFRESH_DISABLED", "") or "").strip().lower()
            in {"1", "true", "yes", "on"},
        )

    @staticmethod
    def is_free_like(value: str | None) -> bool:
        if not value:
            return False

        lowered = value.strip().lower()
        return lowered in KNOWN_FREE_MODEL_IDS or "free" in lowered

    def load_or_refresh_manifest(self) -> dict[str, Any]:
        if not self.refresh_disabled:
            try:
                manifest = self.fetch_live_manifest()
                self.write_cache(manifest)
                return manifest
            except Exception as exc:  # noqa: BLE001 - cache fallback decides safety
                logger.warning("Unable to refresh Zen safety manifest: %s", exc)

        return self.read_fresh_cache()

    def read_fresh_cache(self) -> dict[str, Any]:
        if not self.cache_path.exists():
            raise ValueError("Zen safety manifest is missing; Zen models are disabled until it is refreshed")

        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Zen safety manifest cannot be read: {exc}") from exc

        generated_at = float(data.get("generated_at") or 0)
        age = time.time() - generated_at
        if age < 0 or age > self.ttl_seconds:
            raise ValueError("Zen safety manifest is stale; Zen models are disabled until it is refreshed")

        if not isinstance(data.get("models"), dict):
            raise ValueError("Zen safety manifest is invalid; missing models map")

        return data

    def write_cache(self, manifest: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fetch_live_manifest(self) -> dict[str, Any]:
        models_payload = self._fetch_json(ZEN_MODELS_URL)
        docs_html = self._fetch_text(ZEN_DOCS_URL)
        return self.build_manifest(models_payload, docs_html)

    def get_record(self, model_id: str) -> ZenSafetyRecord | None:
        models = self.manifest.get("models", {})
        raw = models.get(model_id)
        if not isinstance(raw, dict):
            return None
        return ZenSafetyRecord.from_mapping(model_id, raw)

    def validate_config_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        model_id = str(entry.get("model_name") or "")
        aliases = entry.get("aliases") or []
        if isinstance(aliases, str):
            aliases = [aliases]

        if self.is_free_like(model_id) or any(self.is_free_like(str(alias)) for alias in aliases):
            raise ValueError(f"Zen model '{model_id}' has a free-looking model ID or alias")

        record = self.get_record(model_id)
        if record is None:
            raise ValueError(f"Zen safety manifest has no fresh entry for '{model_id}'")

        if not record.runtime_allowed:
            reason = "; ".join(record.reasons) if record.reasons else "classified as unsafe"
            raise ValueError(f"Zen model '{model_id}' is not runtime allowed: {reason}")

        billing_tier = str(entry.get("billing_tier") or "unknown")
        retention_policy = str(entry.get("retention_policy") or "unknown")
        endpoint_family = str(entry.get("endpoint_family") or "unknown")
        fallback_eligible = bool(entry.get("zdr_fallback_eligible", False))
        openrouter_equivalents = entry.get("openrouter_equivalents") or []

        if billing_tier != "paid" or record.billing_tier != "paid":
            raise ValueError(f"Zen model '{model_id}' is not positively classified as paid")

        if retention_policy not in RUNTIME_RETENTION_POLICIES:
            raise ValueError(f"Zen model '{model_id}' has unsupported retention policy '{retention_policy}'")

        if record.retention_policy not in RUNTIME_RETENTION_POLICIES:
            raise ValueError(
                f"Zen model '{model_id}' has unsafe retention policy in safety manifest: {record.retention_policy}"
            )

        if retention_policy == "zero" and record.retention_policy != "zero":
            raise ValueError(
                f"Zen model '{model_id}' config claims zero retention but live safety policy says "
                f"{record.retention_policy}"
            )

        if endpoint_family not in SUPPORTED_RUNTIME_ENDPOINTS:
            raise ValueError(f"Zen model '{model_id}' uses unsupported endpoint family '{endpoint_family}'")

        if fallback_eligible:
            if retention_policy != "zero" or record.retention_policy != "zero":
                raise ValueError(f"Zen model '{model_id}' is not eligible for ZDR fallback")
            if endpoint_family not in SUPPORTED_FALLBACK_ENDPOINTS:
                raise ValueError(
                    f"Zen model '{model_id}' uses unsupported endpoint family '{endpoint_family}' for ZDR fallback"
                )
            if not record.zdr_fallback_eligible:
                raise ValueError(f"Zen model '{model_id}' is not ZDR fallback eligible in safety manifest")
            if not isinstance(openrouter_equivalents, list):
                raise ValueError(f"Zen model '{model_id}' openrouter_equivalents must be a list")

        return {
            "endpoint_family": endpoint_family,
            "billing_tier": billing_tier,
            "retention_policy": retention_policy,
            "zdr_fallback_eligible": fallback_eligible,
            "openrouter_equivalents": [str(value) for value in openrouter_equivalents],
            "safety_reasons": list(record.reasons),
        }

    @classmethod
    def build_manifest(
        cls, models_payload: dict[str, Any], docs_html: str, *, now: float | None = None
    ) -> dict[str, Any]:
        docs_models = _parse_docs_model_table(docs_html)
        paid_names, free_names = _parse_pricing_table(docs_html)
        privacy_policies = _parse_privacy_exceptions(docs_html)
        api_ids = _extract_model_ids(models_payload)

        docs_by_id = {details["model_id"]: details for details in docs_models.values()}
        records: dict[str, dict[str, Any]] = {}

        all_ids = set(api_ids) | set(docs_by_id)
        for model_id in sorted(all_ids):
            details = docs_by_id.get(model_id, {})
            display_name = str(details.get("display_name") or model_id)
            display_key = _normalise_display_name(display_name)

            reasons: list[str] = []
            billing_tier = "unknown"
            retention_policy = "unknown"

            if model_id in KNOWN_FREE_MODEL_IDS or cls.is_free_like(model_id) or cls.is_free_like(display_name):
                billing_tier = "free"
                retention_policy = "free_retained"
                reasons.append("model ID or display name is free-looking")
            elif display_key in free_names:
                billing_tier = "free"
                retention_policy = "free_retained"
                reasons.append("Zen pricing table marks the model Free")
            elif display_key in paid_names:
                billing_tier = "paid"
                retention_policy = "zero"

            privacy_policy = privacy_policies.get(display_key)
            if privacy_policy:
                retention_policy = privacy_policy["retention_policy"]
                reasons.extend(privacy_policy["reasons"])

            sdk_package = str(details.get("sdk_package") or "")
            if retention_policy == "zero" and model_id.startswith("gpt-") and sdk_package == "@ai-sdk/openai":
                retention_policy = "retained_30d"
                reasons.append("OpenAI API requests are retained for 30 days")
            if retention_policy == "zero" and model_id.startswith("claude-"):
                retention_policy = "retained_30d"
                reasons.append("Anthropic API requests are retained for 30 days")

            available_in_api = model_id in api_ids
            if not available_in_api:
                reasons.append("model is not present in the live Zen /models response")

            runtime_allowed = (
                available_in_api and billing_tier == "paid" and retention_policy in RUNTIME_RETENTION_POLICIES
            )
            endpoint_family = _endpoint_family(str(details.get("endpoint") or ""))
            zdr_fallback_eligible = (
                runtime_allowed and retention_policy == "zero" and endpoint_family in SUPPORTED_FALLBACK_ENDPOINTS
            )

            records[model_id] = {
                "model_id": model_id,
                "display_name": display_name,
                "billing_tier": billing_tier,
                "retention_policy": retention_policy,
                "runtime_allowed": runtime_allowed,
                "zdr_fallback_eligible": zdr_fallback_eligible,
                "reasons": sorted(set(reasons)),
            }

        return {
            "generated_at": now if now is not None else time.time(),
            "sources": {"models": ZEN_MODELS_URL, "docs": ZEN_DOCS_URL},
            "models": records,
        }

    @staticmethod
    def _fetch_json(url: str) -> dict[str, Any]:
        request = urllib.request.Request(url, headers={"User-Agent": "pal-mcp-server/zen-safety"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"failed to fetch {url}: {exc}") from exc

    @staticmethod
    def _fetch_text(url: str) -> str:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 pal-mcp-server/zen-safety"})
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                return response.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"failed to fetch {url}: {exc}") from exc


def _extract_model_ids(models_payload: dict[str, Any]) -> set[str]:
    raw_models = models_payload.get("data", models_payload if isinstance(models_payload, list) else [])
    if not isinstance(raw_models, list):
        return set()

    ids: set[str] = set()
    for item in raw_models:
        if isinstance(item, str):
            ids.add(item)
        elif isinstance(item, dict) and item.get("id"):
            ids.add(str(item["id"]))
    return ids


def _parse_docs_model_table(docs_html: str) -> dict[str, dict[str, str]]:
    models: dict[str, dict[str, str]] = {}
    for cells in _iter_table_rows(docs_html):
        if len(cells) < 4:
            continue

        display_name, model_id, endpoint, sdk_package = cells[:4]
        if "opencode.ai/zen" not in endpoint:
            continue

        models[_normalise_display_name(display_name)] = {
            "display_name": display_name,
            "model_id": model_id,
            "endpoint": endpoint,
            "endpoint_family": _endpoint_family(endpoint),
            "sdk_package": sdk_package,
        }
    return models


def _parse_pricing_table(docs_html: str) -> tuple[set[str], set[str]]:
    paid_names: set[str] = set()
    free_names: set[str] = set()
    for cells in _iter_table_rows(docs_html):
        if len(cells) < 5:
            continue

        model_name = cells[0]
        price_cells = cells[1:]
        if not price_cells:
            continue
        if any(cell.lower() == "free" for cell in price_cells):
            free_names.add(_normalise_display_name(model_name))
        elif any(cell.startswith("$") for cell in price_cells):
            paid_names.add(_normalise_display_name(model_name))
    return paid_names, free_names


def _parse_privacy_exceptions(docs_html: str) -> dict[str, dict[str, Any]]:
    exceptions: dict[str, dict[str, Any]] = {}
    for raw_li in re.findall(r"<li>(.*?)</li>", docs_html, flags=re.IGNORECASE | re.DOTALL):
        text = _strip_tags(raw_li)
        if ":" not in text:
            continue

        label, description = [part.strip() for part in text.split(":", 1)]
        lowered = description.lower()
        label_key = _normalise_display_name(label)
        if "free" in label.lower() or "trial" in lowered or "improve" in lowered or "logged" in lowered:
            exceptions[label_key] = {
                "retention_policy": "free_retained",
                "reasons": [description],
            }
        elif "30 days" in lowered or "retained" in lowered:
            exceptions[label_key] = {
                "retention_policy": "retained_30d",
                "reasons": [description],
            }
    return exceptions


def _iter_table_rows(docs_html: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_row in re.findall(r"<tr>(.*?)</tr>", docs_html, flags=re.IGNORECASE | re.DOTALL):
        cells = [_strip_tags(cell) for cell in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", raw_row, re.I | re.S)]
        if cells:
            rows.append(cells)
    return rows


def _strip_tags(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", value)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalise_display_name(value: str) -> str:
    text = re.sub(r"\([^)]*\)", "", value)
    text = html.unescape(text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _endpoint_family(endpoint: str) -> str:
    if endpoint.endswith("/responses"):
        return "responses"
    if endpoint.endswith("/chat/completions"):
        return "chat_completions"
    if endpoint.endswith("/messages"):
        return "messages"
    if "/models/" in endpoint:
        return "gemini_model_endpoint"
    return "unknown"
