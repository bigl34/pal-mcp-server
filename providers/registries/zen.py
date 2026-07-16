"""Registry loader for OpenCode Zen model capabilities."""

from __future__ import annotations

from utils.zen_safety import ZenSafetyPolicy

from ..shared import ModelCapabilities, ProviderType
from .base import CAPABILITY_FIELD_NAMES, CapabilityModelRegistry

ZEN_EXTRA_KEYS = {
    "endpoint_family",
    "billing_tier",
    "retention_policy",
    "zdr_fallback_eligible",
    "openrouter_equivalents",
    "safety_reasons",
}


class ZenModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/zen_models.json``."""

    def __init__(self, config_path: str | None = None) -> None:
        self._safety_policy = ZenSafetyPolicy.from_env()
        super().__init__(
            env_var_name="ZEN_MODELS_CONFIG_PATH",
            default_filename="zen_models.json",
            provider=ProviderType.ZEN,
            friendly_prefix="OpenCode Zen ({model})",
            config_path=config_path,
        )

    def _extra_keys(self) -> set[str]:
        return ZEN_EXTRA_KEYS

    def _finalise_entry(self, entry: dict) -> tuple[ModelCapabilities, dict]:
        extras = self._safety_policy.validate_config_entry(entry)
        filtered = {key: value for key, value in entry.items() if key in CAPABILITY_FIELD_NAMES}
        filtered.setdefault("provider", self._provider_default())
        capability = ModelCapabilities(**filtered)
        return capability, extras
