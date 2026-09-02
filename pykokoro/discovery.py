"""Metadata-only discovery of models supported by the PyKokoro runtime."""

from __future__ import annotations

from dataclasses import dataclass

from .model_profiles import (
    canonical_voice_name,
    get_registry_model_profile,
    normalize_language_code,
    registry_support_status,
)
from .model_registry import (
    DownloadPreference,
    ModelRegistryError,
    RegistryClient,
    RuntimeDistribution,
    RuntimeModel,
    distribution_source,
)


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """Runtime capabilities advertised for one canonical registry model."""

    model_id: str
    source: str
    languages: tuple[str, ...]
    voices: tuple[str, ...]
    default_voice: str
    qualities: tuple[str, ...]
    g2p_backend: str | None
    lexicons: tuple[str, ...] | None
    frontend: str
    status: str
    experimental: bool
    runtime_available: bool
    redistribution_allowed: bool
    distribution_id: str | None = None
    provider: str | None = None
    sample_rate: int | None = None
    max_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class ModelDiscoveryResult:
    """Immutable model inventory and the registry provenance used to build it."""

    models: tuple[ModelCapabilities, ...]
    registry_source: str
    cache_fallback: bool
    offline: bool = False
    refreshed: bool = False


def discover_models(
    *,
    offline: bool = False,
    refresh: bool = False,
    preference: DownloadPreference = "auto",
) -> ModelDiscoveryResult:
    """Return model capabilities without downloading model assets or creating sessions.

    Registry metadata follows the same cache and network policy as the runtime asset
    resolver. ``refresh`` only refreshes registry metadata. It never downloads model,
    voice, or auxiliary assets.
    """
    if offline and refresh:
        raise ValueError("offline and refresh cannot be combined")
    if preference not in {"auto", "github", "huggingface", "upstream"}:
        raise ModelRegistryError(f"Unknown download preference: {preference}")

    registry = RegistryClient().load(offline=offline, refresh=refresh)
    capabilities = tuple(
        _capabilities_for_model(model, preference=preference, registry=registry, offline=offline)
        for model in registry.models.values()
    )
    return ModelDiscoveryResult(
        models=tuple(sorted(capabilities, key=lambda item: (item.model_id, item.source))),
        registry_source=registry.source,
        cache_fallback=registry.cache_fallback,
        offline=offline,
        refreshed=refresh,
    )


def _capabilities_for_model(
    model: RuntimeModel,
    *,
    preference: DownloadPreference,
    registry: object,
    offline: bool,
) -> ModelCapabilities:
    status = registry_support_status(model)
    distribution: RuntimeDistribution | None = None
    profile = None

    if model.runtime_available:
        try:
            distribution = model.distribution(preference)
        except ModelRegistryError:
            distribution = None
        try:
            profile = get_registry_model_profile(
                model.model_id,
                preference=preference,
                offline=offline,
                registry=registry,
            )
        except ModelRegistryError:
            profile = None

    experimental = bool(profile and profile.frontend_experimental)
    if status == "ready" and experimental:
        status = "experimental"

    voices = tuple(canonical_voice_name(model.model_id, voice) for voice in model.voices)
    default_voice = canonical_voice_name(model.model_id, model.default_voice)
    source = _source_for_model(model, distribution)
    qualities = _qualities(distribution)

    return ModelCapabilities(
        model_id=model.model_id,
        source=source,
        languages=tuple(normalize_language_code(language) for language in model.language_codes),
        voices=voices,
        default_voice=default_voice,
        qualities=qualities,
        g2p_backend=profile.g2p_backend if profile is not None else None,
        lexicons=profile.named_lexicons if profile is not None else None,
        frontend=model.frontend,
        status=status,
        experimental=experimental,
        runtime_available=model.runtime_available,
        redistribution_allowed=model.redistribution_allowed,
        distribution_id=distribution.id if distribution is not None else None,
        provider=distribution.provider if distribution is not None else None,
        sample_rate=model.sample_rate,
        max_tokens=model.max_tokens,
    )


def _qualities(distribution: RuntimeDistribution | None) -> tuple[str, ...]:
    if distribution is None:
        return ()
    return tuple(
        dict.fromkeys(quality for quality in distribution.qualities if quality is not None)
    )


def _source_for_model(model: RuntimeModel, distribution: RuntimeDistribution | None) -> str:
    if distribution is not None:
        return distribution_source(distribution)
    for candidate in model.distributions:
        try:
            return distribution_source(candidate)
        except ModelRegistryError:
            continue
    configured = model.data.get("model_source", model.data.get("source", ""))
    return str(configured) if configured is not None else ""


__all__ = ["ModelCapabilities", "ModelDiscoveryResult", "discover_models"]
