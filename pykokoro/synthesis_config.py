"""Configuration owned by the request-centric Kokoro synthesis engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from .asset_progress import AssetProgressCallback
from .config_types import ModelQuality, ModelSource, ModelVariant, ProviderType
from .generation_config import GenerationConfig
from .language_routing import LanguageRoutingConfig
from .short_sentence_handler import ShortSentenceConfig
from .voice_level import VoiceLevelConfig
from .voice_manager import VoiceBlend

if TYPE_CHECKING:
    from .tokenizer import EspeakConfig, TokenizerConfig


@dataclass(frozen=True, slots=True)
class SynthesisConfig:
    """Model, frontend, and inference settings for Kokoro speech synthesis."""

    voice: str | VoiceBlend | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    language_routing: LanguageRoutingConfig | None = None

    model_quality: ModelQuality | None = None
    model_source: ModelSource | None = None
    model_variant: ModelVariant | None = None
    model_path: Path | str | None = None
    voices_path: Path | str | None = None
    model_config_path: Path | str | None = None
    release_manifest_path: Path | str | None = None
    model_identity: str | None = None
    provider: ProviderType | None = None
    provider_options: dict[str, Any] | None = None
    session_options: Any | None = None

    asset_progress: AssetProgressCallback | None = None
    tokenizer_config: TokenizerConfig | None = None
    espeak_config: EspeakConfig | None = None
    short_sentence_config: ShortSentenceConfig | None = None
    waveform_validation: Literal["off", "warn", "strict"] = "off"
    inference_audio_diagnostics: bool = False
    inference_cache_enabled: bool = True
    inference_cache_max_bytes: int = 128 * 1024 * 1024
    voice_level: VoiceLevelConfig = field(default_factory=VoiceLevelConfig)
    return_trace: bool = False
    cache_dir: str | None = None
    allow_experimental_frontend: bool = False

    def __post_init__(self) -> None:
        if self.voice is not None and not isinstance(self.voice, (str, VoiceBlend)):
            raise TypeError("voice must be a voice name, VoiceBlend, or None")
        if isinstance(self.voice, str) and not self.voice.strip():
            raise ValueError("voice must be non-empty when supplied")
        if self.language_routing is not None and not isinstance(
            self.language_routing, LanguageRoutingConfig
        ):
            raise TypeError("language_routing must be LanguageRoutingConfig or None")
        if self.waveform_validation not in ("off", "warn", "strict"):
            raise ValueError("waveform_validation must be 'off', 'warn', or 'strict'")
        if isinstance(self.inference_cache_max_bytes, bool) or not isinstance(
            self.inference_cache_max_bytes, int
        ):
            raise TypeError("inference_cache_max_bytes must be an integer")
        if self.inference_cache_max_bytes < 0:
            raise ValueError("inference_cache_max_bytes must be non-negative")
        if not isinstance(self.generation, GenerationConfig):
            raise TypeError("generation must be GenerationConfig")
        if not isinstance(self.voice_level, VoiceLevelConfig):
            raise TypeError("voice_level must be VoiceLevelConfig")


def _manifest_model_name(
    assets: list[dict[str, Any]], *, quality: ModelQuality, manifest_path: Path
) -> str | None:
    models = [
        item for item in assets if item.get("role") == "model" and isinstance(item.get("name"), str)
    ]
    matching = [item for item in models if item.get("quality") == quality]
    if len(matching) == 1:
        return str(matching[0]["name"])
    if len(matching) > 1:
        raise ValueError(
            f"Release manifest {manifest_path} has multiple model assets for quality {quality!r}"
        )
    if len(models) == 1 and models[0].get("quality") in {None, quality}:
        return str(models[0]["name"])
    if models:
        available = sorted(
            {str(item["quality"]) for item in models if isinstance(item.get("quality"), str)}
        )
        suffix = f" Available: {', '.join(available)}" if available else ""
        raise ValueError(
            f"Release manifest {manifest_path} has no model asset for quality {quality!r}.{suffix}"
        )
    return None


def _resolve_manifest_paths(config: SynthesisConfig) -> SynthesisConfig:
    if config.release_manifest_path is None:
        return config
    manifest_path = Path(config.release_manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Release manifest {manifest_path} must contain an object")
    assets = [item for item in data.get("assets", []) if isinstance(item, dict)]
    if config.model_quality is None:
        raise ValueError("Manifest model resolution requires an effective model quality")
    model = _manifest_model_name(assets, quality=config.model_quality, manifest_path=manifest_path)
    voices = next(
        (
            str(item["name"])
            for item in assets
            if item.get("role") == "voices" and item.get("format") == "numpy-npz"
        ),
        next((str(item["name"]) for item in assets if item.get("role") == "voices"), None),
    )
    model_config = next(
        (str(item["name"]) for item in assets if item.get("role") == "config"), None
    )
    if data.get("schema") != 2:
        names = [str(item["name"]) for item in assets if item.get("name")]
        model = model or next((name for name in names if name.endswith(".onnx")), None)
        voices = voices or next((name for name in names if "voice" in name.lower()), None)
        model_config = model_config or next(
            (
                name
                for name in names
                if name.endswith(".json") and "manifest" not in name and "bundle" not in name
            ),
            None,
        )
    if model is None or voices is None:
        raise ValueError(f"Release manifest {manifest_path} lacks model or voice assets")
    base = manifest_path.parent
    return replace(
        config,
        model_path=config.model_path or base / model,
        voices_path=config.voices_path or base / voices,
        model_config_path=config.model_config_path
        or (base / model_config if model_config else None),
    )


def resolve_synthesis_config(
    config: SynthesisConfig,
    *,
    language: str,
    voice: str | VoiceBlend | None = None,
) -> SynthesisConfig:
    """Resolve one request's explicit language and voice to a concrete Kokoro profile."""
    from .model_profiles import (
        get_model_profile,
        model_id_for_voice,
        normalize_language_code,
        profile_for_language,
        profile_for_voice,
    )

    language = normalize_language_code(language)
    effective_voice = voice if voice is not None else config.voice
    source = config.model_source
    variant = config.model_variant
    if variant is None and isinstance(effective_voice, str):
        voice_model = model_id_for_voice(effective_voice)
        if voice_model is not None:
            voice_profile = profile_for_voice(effective_voice)
            if voice_profile is None or voice_profile.variant != voice_model:
                raise ValueError(
                    f"Voice {effective_voice!r} maps to {voice_model!r} but has no compatible profile"
                )
            variant = voice_model
            source = voice_profile.source
    if variant is None and isinstance(effective_voice, str):
        voice_profile = profile_for_voice(effective_voice)
        if (
            voice_profile is not None
            and language in voice_profile.language_codes
            and source in {None, voice_profile.source}
        ):
            variant = voice_profile.variant
            source = voice_profile.source
    if variant is None:
        language_profile = profile_for_language(language)
        if language_profile is not None and source in {None, "github"}:
            variant = language_profile.variant
            source = language_profile.source if source is None else source
        elif language.startswith("zh") and source in {None, "github"}:
            variant = "v1.1-zh"
            source = "github" if source is None else source
        else:
            variant = "v1.0"
            source = "github" if source is None else source
    elif source is None:
        source = "github"

    assert source is not None and variant is not None
    profile = get_model_profile(variant, source)
    if profile.runtime_available is False and config.release_manifest_path is None:
        raise ValueError(
            f"Model profile {variant!r} is present but has no runtime-ready distribution"
        )
    quality = config.model_quality or cast(
        ModelQuality, profile.available_qualities[0] if profile.available_qualities else "fp32"
    )
    resolved = _resolve_manifest_paths(
        replace(
            config,
            model_source=source,
            model_variant=variant,
            model_quality=quality,
            voice=effective_voice,
        )
    )
    if source == "huggingface" and profile.quality_files and quality not in profile.quality_files:
        available = ", ".join(profile.quality_files) or "none"
        raise ValueError(
            f"Quality {quality!r} is not available for {source}/{variant}. Available: {available}"
        )
    if resolved.voice is None:
        resolved = replace(resolved, voice=profile.default_voice)
    elif (
        resolved.voices_path is None
        and isinstance(resolved.voice, str)
        and profile.voice_names
        and resolved.voice not in profile.voice_names
    ):
        available = ", ".join(profile.voice_names)
        raise ValueError(
            f"Voice {resolved.voice!r} is not available for model variant {variant!r}. "
            f"Available voices: {available}"
        )
    return replace(resolved, generation=replace(resolved.generation, lang=language))
