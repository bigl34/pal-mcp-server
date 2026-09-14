"""OpenRouter model registry for managing model configurations and aliases."""

from __future__ import annotations

from ..shared import ModelCapabilities, ProviderType
from .base import CAPABILITY_FIELD_NAMES, CapabilityModelRegistry

PROVIDER_PREFERENCE_KEYS = {"ignore"}


class OpenRouterModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/openrouter_models.json``."""

    _ignored_providers: list[str] = []

    def _extra_keys(self) -> set[str]:
        return {"allow_non_zdr"}

    def __init__(self, config_path: str | None = None) -> None:
        super().__init__(
            env_var_name="OPENROUTER_MODELS_CONFIG_PATH",
            default_filename="openrouter_models.json",
            provider=ProviderType.OPENROUTER,
            friendly_prefix="OpenRouter ({model})",
            config_path=config_path,
        )

    @property
    def ignored_providers(self) -> list[str]:
        """Provider slugs every OpenRouter request must exclude via ``provider.ignore``."""

        return list(self._ignored_providers)

    def _parse_settings(self, data: dict) -> list[str]:
        if "provider_preferences" not in data:
            return []
        preferences = data["provider_preferences"]
        if not isinstance(preferences, dict):
            raise ValueError("provider_preferences must be an object")

        unknown_keys = set(preferences.keys()) - PROVIDER_PREFERENCE_KEYS
        if unknown_keys:
            raise ValueError(f"Unsupported provider_preferences keys: {sorted(unknown_keys)}")

        return self._parse_ignore_list(preferences.get("ignore", []))

    def _apply_settings(self, settings: object | None) -> None:
        self._ignored_providers = list(settings or [])

    @staticmethod
    def _parse_ignore_list(raw_ignore: object) -> list[str]:
        if not isinstance(raw_ignore, list):
            raise ValueError("provider_preferences.ignore must be a list of provider slugs")

        ignored: list[str] = []
        for raw_slug in raw_ignore:
            if not isinstance(raw_slug, str):
                raise ValueError("provider_preferences.ignore entries must be strings")
            slug = raw_slug.strip()
            if not slug:
                raise ValueError("provider_preferences.ignore entries must be non-empty")
            if slug not in ignored:
                ignored.append(slug)
        return ignored

    def _finalise_entry(self, entry: dict) -> tuple[ModelCapabilities, dict]:
        provider_override = entry.get("provider")
        if isinstance(provider_override, str):
            entry_provider = ProviderType(provider_override.lower())
        elif isinstance(provider_override, ProviderType):
            entry_provider = provider_override
        else:
            entry_provider = ProviderType.OPENROUTER

        if entry_provider == ProviderType.CUSTOM:
            entry.setdefault("friendly_name", f"Custom ({entry['model_name']})")
        else:
            entry.setdefault("friendly_name", f"OpenRouter ({entry['model_name']})")

        filtered = {k: v for k, v in entry.items() if k in CAPABILITY_FIELD_NAMES}
        filtered.setdefault("provider", entry_provider)
        capability = ModelCapabilities(**filtered)
        return capability, {"allow_non_zdr": bool(entry.get("allow_non_zdr", False))}
