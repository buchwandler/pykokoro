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
from .model_profiles import get_model_profile
from .release_catalog import ReleaseAsset, RemoteModelRelease
from .utils import get_user_cache_path


def _is_nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _model_filename(
    quality: ModelQuality, source: ModelSource, variant: ModelVariant
) -> tuple[str, bool]:
    if source == "github":
        from .asset_constants import (
            MODEL_QUALITY_FILES_GITHUB_V1_0,
            MODEL_QUALITY_FILES_GITHUB_V1_1_ZH,
        )

        legacy = {
            "v1.0": MODEL_QUALITY_FILES_GITHUB_V1_0,
            "v1.1-zh": MODEL_QUALITY_FILES_GITHUB_V1_1_ZH,
        }.get(variant)
        if legacy is not None:
            try:
                return legacy[quality], False
            except KeyError as exc:
                available = ", ".join(legacy)
                raise ValueError(
                    f"Quality {quality!r} is not available for {source}/{variant}. Available: {available}"
                ) from exc
        if variant == "v1.2-de-martin":
            if quality != "fp32":
                raise ValueError(
                    f"Quality {quality!r} is not available for {source}/{variant}. Available: fp32"
                )
            return "kokoro-german-martin-v1.2.onnx", False
        return f"{quality}.onnx", False
    profile = get_model_profile(variant, source)
    try:
        return profile.quality_files[quality], True
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
    source: ModelSource = DEFAULT_MODEL_SOURCE, variant: ModelVariant = DEFAULT_MODEL_VARIANT
) -> Path:
    """Return the compatibility cache path for a combined voice archive."""
    if source == "huggingface":
        filename = "voices.bin.npz"
    else:
        from .asset_constants import GITHUB_VOICES_FILENAME_V1_0, GITHUB_VOICES_FILENAME_V1_1_ZH

        filename = {
            "v1.0": GITHUB_VOICES_FILENAME_V1_0,
            "v1.1-zh": GITHUB_VOICES_FILENAME_V1_1_ZH,
        }.get(variant)
        if filename is None and variant == "v1.2-de-martin":
            filename = "voices-german-martin-v1.2.bin"
        filename = filename or "voices.npz"
    return _voices_asset_dir(source, variant) / filename


def release_asset_path(release: RemoteModelRelease, asset: ReleaseAsset) -> Path:
    """Return the release-identity-aware local path for one published asset."""
    return get_user_cache_path("releases") / release.profile / release.release_tag / asset.name


def installed_manifest_path(release: RemoteModelRelease) -> Path:
    return (
        get_user_cache_path("releases")
        / release.profile
        / release.release_tag
        / "release-manifest.json"
    )


def installed_sidecar_path(release: RemoteModelRelease) -> Path:
    return (
        get_user_cache_path("releases") / release.profile / release.release_tag / "installed.json"
    )


def get_model_asset_paths(
    *,
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> ModelAssetPaths:
    filename, uses_hf_subfolder = _model_filename(quality, source, variant)
    model = _model_asset_dir(source, variant) / filename
    if uses_hf_subfolder:
        model = model.parent / HF_MODEL_SUBFOLDER / model.name
    profile = get_model_profile(variant, source)
    return ModelAssetPaths(
        source=source,
        variant=variant,
        quality=quality,
        config=(
            _config_asset_path(variant)
            if source == "huggingface" and profile.vocabulary_source == "downloaded-config"
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
    return _is_nonempty_file(
        get_model_asset_paths(quality=quality, source=source, variant=variant).model
    )


def are_voices_downloaded(
    source: ModelSource = DEFAULT_MODEL_SOURCE, variant: ModelVariant = DEFAULT_MODEL_VARIANT
) -> bool:
    return _is_nonempty_file(get_voices_archive_path(source, variant))


def are_models_downloaded(
    quality: ModelQuality = DEFAULT_MODEL_QUALITY,
    source: ModelSource = DEFAULT_MODEL_SOURCE,
    variant: ModelVariant = DEFAULT_MODEL_VARIANT,
) -> bool:
    return get_model_asset_paths(quality=quality, source=source, variant=variant).complete
