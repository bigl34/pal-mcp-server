"""OpenCode Zen provider implementation."""

from typing import ClassVar, Optional

from .openai_compatible import OpenAICompatibleProvider
from .registries.zen import ZenModelRegistry
from .registry_provider_mixin import RegistryBackedProviderMixin
from .shared import ModelCapabilities, ProviderType


class ZenProvider(RegistryBackedProviderMixin, OpenAICompatibleProvider):
    """Provider for OpenCode Zen's OpenAI-compatible endpoints."""

    FRIENDLY_NAME = "OpenCode Zen"
    REGISTRY_CLASS = ZenModelRegistry
    MODEL_CAPABILITIES: ClassVar[dict[str, ModelCapabilities]] = {}

    def __init__(self, api_key: str, **kwargs):
        """Initialize the Zen provider with the OpenCode Zen API key."""

        self._ensure_registry()
        kwargs.setdefault("base_url", "https://opencode.ai/zen/v1")
        super().__init__(api_key, **kwargs)
        self._invalidate_capability_cache()

    def _lookup_capabilities(
        self,
        canonical_name: str,
        requested_name: Optional[str] = None,
    ) -> Optional[ModelCapabilities]:
        """Look up Zen capabilities from the configured registry."""

        self._ensure_registry()
        return super()._lookup_capabilities(canonical_name, requested_name)

    def _finalise_capabilities(
        self,
        capabilities: ModelCapabilities,
        canonical_name: str,
        requested_name: str,
    ) -> ModelCapabilities:
        """Ensure registry-sourced entries report the Zen provider type."""

        if capabilities.provider != ProviderType.ZEN:
            capabilities.provider = ProviderType.ZEN
        return capabilities

    def get_provider_type(self) -> ProviderType:
        """Identify this provider for restriction and logging logic."""

        return ProviderType.ZEN
