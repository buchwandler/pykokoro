"""Metadata-only discovery of named pronunciation lexicons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .discovery import discover_models


@dataclass(frozen=True, slots=True)
class LexiconCapabilities:
    """Public metadata for a named lexicon accepted by the runtime."""

    selector: str
    language: str
    locale: str
    asset_id: str | None
    data_backend: str | None
    default: bool
    installed: bool | None
    display_name: str | None = None
    phoneme_encoding: str | None = None
    data_version: str | None = None
    models: tuple[str, ...] = ()
    model_support: str = "unknown"


@dataclass(frozen=True, slots=True)
class LexiconDiscoveryResult:
    """Lexicon inventory and discovery provenance."""

    lexicons: tuple[LexiconCapabilities, ...]
    offline: bool = False
    refreshed: bool = False
    registry_source: str | None = None
    cache_fallback: bool = False


def _normalize_language(language: str) -> str:
    return language.strip().casefold().replace("_", "-")


def _locale_label(language: str) -> str:
    parts = language.split("-")
    if len(parts) == 1:
        return parts[0]
    return "-".join([parts[0].lower(), *[part.upper() for part in parts[1:]]])


def _language_matches(requested: str | None, locale: str) -> bool:
    if requested is None:
        return True
    requested = _normalize_language(requested)
    locale = _normalize_language(locale)
    return locale == requested or (
        "-" not in requested and locale.partition("-")[0] == requested
    )


def _installed_state(asset_id: str | None) -> bool | None:
    if asset_id is None:
        return None
    try:
        from lexphon import DataStore

        return DataStore().verify(asset_id)
    except (ImportError, OSError, RuntimeError, ValueError):
        return None


def _model_capability(
    model_variant: str,
    *,
    offline: bool,
    refresh: bool,
    preference: str,
) -> Any:
    models = discover_models(offline=offline, refresh=refresh, preference=preference).models
    for model in models:
        if model.model_id == model_variant:
            return model
    raise ValueError(f"Unknown model variant: {model_variant}")


def _registry_metadata() -> tuple[tuple[Any, ...], str | None]:
    from kokorog2p.lexicons.registry import iter_lexicon_specs

    return iter_lexicon_specs(), "kokorog2p.lexicons.registry"


def discover_lexicons(
    *,
    language: str | None = None,
    model_variant: str | None = None,
    offline: bool = False,
    refresh: bool = False,
    preference: str = "auto",
) -> LexiconDiscoveryResult:
    """Discover named selectors without installing lexicon or model assets.

    The KokoroG2P pronunciation registry is the selector source of truth. Model
    capability metadata narrows the result only when that capability is known.
    """

    if offline and refresh:
        raise ValueError("offline and refresh cannot be combined")
    if preference not in {"auto", "github", "huggingface", "upstream"}:
        raise ValueError(f"Unknown download preference: {preference}")

    model = None
    if model_variant is not None:
        model = _model_capability(
            model_variant,
            offline=offline,
            refresh=refresh,
            preference=preference,
        )

    specs, registry_source = _registry_metadata()
    entries: list[LexiconCapabilities] = []
    for spec in specs:
        if not _language_matches(language, spec.language):
            continue

        model_support = "unknown"
        models: tuple[str, ...] = ()
        if model is not None:
            named_lexicons = model.lexicons
            if named_lexicons is not None:
                model_support = "known"
                if spec.name not in named_lexicons:
                    continue
                models = (model.model_id,)

        metadata = dict(spec.metadata)
        entries.append(
            LexiconCapabilities(
                selector=spec.name,
                language=spec.language.partition("-")[0],
                locale=_locale_label(spec.language),
                asset_id=spec.id,
                data_backend=spec.backend,
                default=spec.default_priority is not None,
                installed=_installed_state(spec.id),
                display_name=metadata.get("display_name"),
                phoneme_encoding=spec.phoneme_encoding,
                data_version=metadata.get("data_version"),
                models=models,
                model_support=model_support,
            )
        )

    entries.sort(key=lambda item: (item.locale.casefold(), item.selector, item.asset_id or ""))
    return LexiconDiscoveryResult(
        lexicons=tuple(entries),
        offline=offline,
        refreshed=refresh,
        registry_source=registry_source,
    )


__all__ = ["LexiconCapabilities", "LexiconDiscoveryResult", "discover_lexicons"]
