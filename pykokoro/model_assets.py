"""Dependency-light model, config, and voice asset inspection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .asset_constants import (
    GITHUB_VOICES_FILENAME_V1_0,
    GITHUB_VOICES_FILENAME_V1_1_DE,
    GITHUB_VOICES_FILENAME_V1_1_ZH,
    HF_CONFIG_FILENAME,
    HF_MODEL_SUBFOLDER,
    MODEL_QUALITY_CACHE_FILES_HF_V1_0,
    MODEL_QUALITY_FILES_GITHUB_V1_0,
    MODEL_QUALITY_FILES_GITHUB_V1_1_DE,
    MODEL_QUALITY_FILES_GITHUB_V1_1_ZH,
    MODEL_QUALITY_FILES_HF,
)
from .config_types import (
    DEFAULT_MODEL_QUALITY,
    DEFAULT_MODEL_SOURCE,
    DEFAULT_MODEL_VARIANT,
    ModelQuality,
    ModelSource,
    ModelVariant,
)
from .utils import get_user_cache_path


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
    if source == "huggingface":
        quality_files = (
            MODEL_QUALITY_CACHE_FILES_HF_V1_0 if variant == "v1.0" else MODEL_QUALITY_FILES_HF
        )
        try:
            return quality_files[quality], True
        except KeyError as exc:
            available = ", ".join(quality_files) or "none"
            raise ValueError(
                f"Quality {quality!r} is not available for {source}/{variant}. "
                f"Available: {available}"
            ) from exc
    if source == "github":
        quality_files_by_variant = {
            "v1.0": MODEL_QUALITY_FILES_GITHUB_V1_0,
            "v1.1-zh": MODEL_QUALITY_FILES_GITHUB_V1_1_ZH,
            "v1.1-de": MODEL_QUALITY_FILES_GITHUB_V1_1_DE,
        }
        try:
            return quality_files_by_variant[variant][quality], False
        except KeyError as exc:
            quality_files = quality_files_by_variant.get(variant, {})
            available = ", ".join(quality_files) or "none"
            raise ValueError(
                f"Quality {quality!r} is not available for {source}/{variant}. "
                f"Available: {available}"
            ) from exc
    raise ValueError(f"Unknown model source: {source!r}")


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
    config: Path
    model: Path
    voices: Path

    @property
    def missing(self) -> tuple[str, ...]:
        missing: list[str] = []
        for name, path in (
            ("config", self.config),
            ("model", self.model),
            ("voices", self.voices),
        ):
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
    if source == "huggingface":
        filename = "voices.bin.npz"
    elif source == "github":
        filenames = {
            "v1.0": GITHUB_VOICES_FILENAME_V1_0,
            "v1.1-zh": GITHUB_VOICES_FILENAME_V1_1_ZH,
            "v1.1-de": GITHUB_VOICES_FILENAME_V1_1_DE,
        }
        try:
            filename = filenames[variant]
        except KeyError as exc:
            raise ValueError(
                f"Unknown voice archive combination: source={source!r}, variant={variant!r}"
            ) from exc
    else:
        raise ValueError(
            f"Unknown voice archive combination: source={source!r}, variant={variant!r}"
        )
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
        config=_config_asset_path(variant),
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
