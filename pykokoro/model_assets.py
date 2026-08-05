"""Dependency-light model, config, and voice asset inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asset_constants import HF_CONFIG_FILENAME, HF_MODEL_SUBFOLDER
from .config_types import (
    DEFAULT_MODEL_QUALITY,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_MODEL_VARIANT,
    ModelQuality,
    ModelSource,
    ModelVariant,
)
from .utils import get_user_cache_path
from .model_profiles import get_model_profile


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _model_filename(
    quality: ModelQuality,
    source: ModelSource,
    variant: ModelVariant,
) -> tuple[str, bool]:
    profile = get_model_profile(variant, source)
    try:
        return profile.quality_files[quality], source == "huggingface"
    except KeyError as exc:
        available = ", ".join(profile.quality_files) or "none"
        raise ValueError(
            f"Quality {quality!r} is not available for {source}/{variant}. Available: {available}"
        ) from exc


def _model_asset_dir(source: ModelSource, variant: ModelVariant) -> Path:
    return get_user_cache_path() / "models" / source / variant


def _voices_asset_dir(source: ModelSource, variant: ModelVariant) -> Path:
    return get_user_cache_path() / "voices" / source / variant


def _config_asset_path(variant: ModelVariant) -> Path:
    return get_user_cache_path() / "config" / variant / HF_CONFIG_FILENAME


@dataclass(frozen=True, slots=True)
class ModelAssetPaths:
    source: ModelSource
    variant: ModelVariant
    quality: ModelQuality
    config: Path | None
    model: Path
    voices: Path

    @property
    def missing(self) -> tuple[str, ...]:
        missing: list[str] = []
        required = [("model", self.model), ("voices", self.voices)]
        if self.config is not None:
            required.insert(0, ("config", self.config))
        for name, path in required:
            if not _is_nonempty_file(path):
                missing.append(name)
        return tuple(missing)

    @property
    def complete(self) -> bool:
        return not self.missing


def get_voices_archive_path(
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> Path:
    """Return the canonical combined voice archive path for a source and variant."""
    filename = get_model_profile(variant, source).voices_filename
    return _voices_asset_dir(source, variant) / filename


def get_model_asset_paths(
    *,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> ModelAssetPaths:
    """Return exact config, model, and combined voice paths for one asset set."""
    filename, uses_hf_subfolder = _model_filename(quality, source, variant)
    model = _model_asset_dir(source, variant) / filename
    if uses_hf_subfolder:
        model = model.parent / HF_MODEL_SUBFOLDER / model.name
    return ModelAssetPaths(
        source=source,
        variant=variant,
        quality=quality,
        config=(
            _config_asset_path(variant)
            if get_model_profile(variant, source).vocabulary_source == "downloaded-config"
            else None
        ),
        model=model,
        voices=get_voices_archive_path(source, variant),
    )


def is_model_downloaded(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> bool:
    """Return whether the requested model file is a nonempty regular file."""
    return _is_nonempty_file(
        get_model_asset_paths(quality=quality, source=source, variant=variant).model
    )


def are_voices_downloaded(
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> bool:
    """Return whether the requested combined voice archive is nonempty."""
    return _is_nonempty_file(get_voices_archive_path(source, variant))


def are_models_downloaded(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> bool:
    """Return whether config, model, and voices are all downloaded."""
    return get_model_asset_paths(
        quality=quality,
        source=source,
        variant=variant,
    ).complete
