from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .config_types import (
    DEFAULT_MODEL_SOURCE,
    DEFAULT_MODEL_VARIANT,
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

    voice: str | VoiceBlend = "af"
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    ssmd: SSMDRenderConfig = field(default_factory=SSMDRenderConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)

    # Model + provider configuration
    model_quality: ModelQuality | None = None
    model_source: ModelSource = DEFAULT_MODEL_SOURCE
    model_variant: ModelVariant = DEFAULT_MODEL_VARIANT
    model_path: Path | str | None = None
    voices_path: Path | str | None = None
    model_identity: str | None = None
    provider: ProviderType | None = None
    provider_options: dict[str, Any] | None = None
    session_options: Any | None = None

    # Tokenizer configuration
    tokenizer_config: TokenizerConfig | None = None
    espeak_config: EspeakConfig | None = None
    short_sentence_config: ShortSentenceConfig | None = None

    # Span slicing
    overlap_mode: Literal["snap", "strict"] = "snap"

    # Behavior toggles
    return_trace: bool = False
    retain_segment_audio: bool = True
    enable_deprecation_warnings: bool = False

    # Caching
    cache_dir: str | None = None
