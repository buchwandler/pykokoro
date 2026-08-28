from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from .config_types import (
    ModelQuality,
    ModelSource,
    ModelVariant,
    ProviderType,
)
from .generation_config import GenerationConfig
from .prosody_config import ProsodyConfig
from .short_sentence_handler import ShortSentenceConfig
from .ssmd_config import SSMDRenderConfig
from .tokenizer import EspeakConfig, TokenizerConfig
from .voice_manager import VoiceBlend


@dataclass(frozen=True)
class PipelineConfig:
    """User-facing configuration for the end-to-end pipeline."""

    voice: str | VoiceBlend | None = None
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    ssmd: SSMDRenderConfig = field(default_factory=SSMDRenderConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)

    # Model + provider configuration
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

    # Tokenizer configuration
    tokenizer_config: TokenizerConfig | None = None
    espeak_config: EspeakConfig | None = None
    short_sentence_config: ShortSentenceConfig | None = None
    allow_experimental_frontend: bool = False

    # Span slicing
    overlap_mode: Literal["snap", "strict"] = "snap"

    # Behavior toggles
    return_trace: bool = False
    retain_segment_audio: bool = True
    enable_deprecation_warnings: bool = False

    # Caching
    cache_dir: str | None = None


def _resolve_manifest_paths(cfg: PipelineConfig) -> PipelineConfig:
    if cfg.release_manifest_path is None:
        return cfg
    manifest_path = Path(cfg.release_manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    base = manifest_path.parent
    assets = [item for item in data.get("assets", []) if isinstance(item, dict)]
    model = next((str(item["name"]) for item in assets if item.get("role") == "model"), None)
    voices = next(
        (
            str(item["name"])
            for item in assets
            if item.get("role") == "voices" and item.get("format") == "numpy-npz"
        ),
        next((str(item["name"]) for item in assets if item.get("role") == "voices"), None),
    )
    config = next((str(item["name"]) for item in assets if item.get("role") == "config"), None)
    if data.get("schema") != 2:
        names = [str(item["name"]) for item in assets if item.get("name")]
        model = model or next((name for name in names if name.endswith(".onnx")), None)
        voices = voices or next((name for name in names if "voice" in name.lower()), None)
        config = config or next(
            (
                name
                for name in names
                if name.endswith(".json") and "manifest" not in name and "bundle" not in name
            ),
            None,
        )
    if model is None or voices is None:
        raise ValueError(f"Release manifest {manifest_path} lacks model or voice assets")
    return replace(
        cfg,
        model_path=cfg.model_path or base / model,
        voices_path=cfg.voices_path or base / voices,
        model_config_path=cfg.model_config_path or (base / config if config else None),
    )


def resolve_model_defaults(cfg: PipelineConfig) -> PipelineConfig:
    """Resolve language-aware model and voice defaults into a concrete config.

    ``None`` means automatic selection. Explicit values, including custom asset
    paths, are preserved and incompatible combinations fail before backend or
    G2P construction.
    """
    cfg = _resolve_manifest_paths(cfg)
    from .model_profiles import (
        get_model_profile,
        model_id_for_voice,
        normalize_language_code,
        profile_for_language,
        profile_for_voice,
    )

    if (
        cfg.generation.lang == GenerationConfig().lang
        and isinstance(cfg.voice, str)
        and (voice_profile := profile_for_voice(cfg.voice)) is not None
        and voice_profile.language_codes
    ):
        cfg = replace(
            cfg,
            generation=replace(cfg.generation, lang=voice_profile.language_codes[0]),
        )

    lang = normalize_language_code(cfg.generation.lang)
    source = cfg.model_source
    variant = cfg.model_variant

    if variant is None and isinstance(cfg.voice, str):
        voice_model = model_id_for_voice(cfg.voice)
        if voice_model is not None:
            variant = voice_model
            source = "huggingface"
            if cfg.generation.lang == GenerationConfig().lang:
                cfg = replace(cfg, generation=replace(cfg.generation, lang="ru"))
    lang = normalize_language_code(cfg.generation.lang)
    if variant is None:
        language_profile = profile_for_language(lang)
        if language_profile is not None and source in {None, "github"}:
            variant = language_profile.variant
            source = language_profile.source if source is None else source
        elif lang.startswith("zh") and source in {None, "github"}:
            variant = "v1.1-zh"
            source = "github" if source is None else source
        else:
            variant = "v1.0"
            source = "github" if source is None else source
    elif source is None:
        source = "github"

    assert source is not None
    assert variant is not None
    profile = get_model_profile(variant, source)

    if profile.runtime_available is False and cfg.release_manifest_path is None:
        raise ValueError(
            f"Model profile {variant!r} is present but has no runtime-ready distribution"
        )
    quality = cfg.model_quality or cast(
        ModelQuality, (profile.available_qualities[0] if profile.available_qualities else "fp32")
    )
    if source == "huggingface" and profile.quality_files and quality not in profile.quality_files:
        available = ", ".join(profile.quality_files) or "none"
        raise ValueError(
            f"Quality {quality!r} is not available for {source}/{variant}. Available: {available}"
        )

    voice = cfg.voice
    if voice is None:
        voice = profile.default_voice
    elif (
        cfg.voices_path is None
        and isinstance(voice, str)
        and profile.voice_names
        and voice not in profile.voice_names
    ):
        available = ", ".join(profile.voice_names)
        raise ValueError(
            f"Voice {voice!r} is not available for model variant {variant!r}. "
            f"Available voices: {available}"
        )

    generation = (
        cfg.generation if cfg.generation.lang == lang else replace(cfg.generation, lang=lang)
    )
    return replace(
        cfg,
        voice=voice,
        generation=generation,
        model_quality=quality,
        model_source=source,
        model_variant=variant,
    )
