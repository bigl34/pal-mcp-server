"""OpenRouter model registry for managing model configurations and aliases."""

from __future__ import annotations

from ..shared import ModelCapabilities, ProviderType
from .base import CAPABILITY_FIELD_NAMES, CapabilityModelRegistry

PROVIDER_PREFERENCE_KEYS = {"ignore", "require_pins"}


class OpenRouterModelRegistry(CapabilityModelRegistry):
    """Capability registry backed by ``conf/openrouter_models.json``."""

    _ignored_providers: list[str] = []
    _required_pin_prefixes: list[str] = []

    def _extra_keys(self) -> set[str]:
        return {"allow_non_zdr", "provider_only"}

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

    @property
    def required_pin_prefixes(self) -> list[str]:
        return list(self._required_pin_prefixes)

    def _parse_settings(self, data: dict) -> dict[str, list[str]]:
        if "provider_preferences" not in data:
            return {"ignore": [], "require_pins": []}
        preferences = data["provider_preferences"]
        if not isinstance(preferences, dict):
            raise ValueError("provider_preferences must be an object")

        unknown_keys = set(preferences.keys()) - PROVIDER_PREFERENCE_KEYS
        if unknown_keys:
            raise ValueError(f"Unsupported provider_preferences keys: {sorted(unknown_keys)}")

        return {
            "ignore": self._parse_preference_list(preferences.get("ignore", []), "ignore", "provider slugs"),
            "require_pins": self._parse_preference_list(
                preferences.get("require_pins", []), "require_pins", "model prefixes"
            ),
        }

    def _apply_settings(self, settings: object | None) -> None:
        preferences = settings if isinstance(settings, dict) else {}
        self._ignored_providers = list(preferences.get("ignore", []))
        self._required_pin_prefixes = list(preferences.get("require_pins", []))

    @staticmethod
    def _parse_preference_list(raw: object, key: str, entries: str) -> list[str]:
        if not isinstance(raw, list):
            raise ValueError(f"provider_preferences.{key} must be a list of {entries}")

        values: list[str] = []
        for raw_value in raw:
            if not isinstance(raw_value, str):
                raise ValueError(f"provider_preferences.{key} entries must be strings")
            value = raw_value.strip()
            if not value:
                raise ValueError(f"provider_preferences.{key} entries must be non-empty")
            if value not in values:
                values.append(value)
        return values

    def _settings_snapshot(self) -> dict[str, list[str]]:
        return {
            "ignore": list(self._ignored_providers),
            "require_pins": list(self._required_pin_prefixes),
        }

    def _finalise_entry(self, entry: dict) -> tuple[ModelCapabilities, dict]:
        provider_only: list[str] | None = None
        if "provider_only" in entry:
            raw_provider_only = entry["provider_only"]
            if not isinstance(raw_provider_only, list) or not raw_provider_only:
                raise ValueError("provider_only must be a non-empty list of provider slugs")

            provider_only = []
            for raw_slug in raw_provider_only:
                if not isinstance(raw_slug, str) or not raw_slug.strip():
                    raise ValueError("provider_only entries must be non-empty strings")
                slug = raw_slug.strip()
                if slug not in provider_only:
                    provider_only.append(slug)

        model_name = entry["model_name"].removeprefix("~")
        if any(model_name.startswith(prefix) for prefix in self._required_pin_prefixes) and not provider_only:
            raise ValueError(f"{model_name}: BYOK model rows must set provider_only")

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
        return capability, {
            "allow_non_zdr": bool(entry.get("allow_non_zdr", False)),
            "provider_only": provider_only,
        }
