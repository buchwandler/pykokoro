"""Public API for the request-centric Kokoro synthesis engine."""

from __future__ import annotations

from typing import Any

from .exceptions import SynthesisInputTooLongError
from .generation_config import GenerationConfig
from .language_routing import LanguageRoutingConfig
from .short_sentence_handler import ShortSentenceConfig
from .synthesis_config import LongTextSplitMode, SynthesisConfig
from .synthesis_types import (
    LinguisticToken,
    PronunciationOverride,
    RenderedSegment,
    SynthesisSegment,
)
from .tokenizer import EspeakConfig, TokenizerConfig
from .types import WordTiming
from .voice_level import VoiceLevelConfig
from .voice_manager import VoiceBlend

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.9.2"
    __version_tuple__ = (0, 9, 2)


def __getattr__(name: str) -> Any:
    if name in {"AssetProgressEvent", "AssetProgressCallback", "ConsoleAssetProgress"}:
        from .asset_progress import AssetProgressCallback, AssetProgressEvent, ConsoleAssetProgress

        return {
            "AssetProgressEvent": AssetProgressEvent,
            "AssetProgressCallback": AssetProgressCallback,
            "ConsoleAssetProgress": ConsoleAssetProgress,
        }[name]
    if name in {
        "ModelCapabilities",
        "ModelDiscoveryResult",
        "VoiceCapabilities",
        "discover_models",
    }:
        from .discovery import (
            ModelCapabilities,
            ModelDiscoveryResult,
            VoiceCapabilities,
            discover_models,
        )

        return {
            "ModelCapabilities": ModelCapabilities,
            "ModelDiscoveryResult": ModelDiscoveryResult,
            "VoiceCapabilities": VoiceCapabilities,
            "discover_models": discover_models,
        }[name]
    if name in {"LexiconCapabilities", "LexiconDiscoveryResult", "discover_lexicons"}:
        from .lexicon_discovery import (
            LexiconCapabilities,
            LexiconDiscoveryResult,
            discover_lexicons,
        )

        return {
            "LexiconCapabilities": LexiconCapabilities,
            "LexiconDiscoveryResult": LexiconDiscoveryResult,
            "discover_lexicons": discover_lexicons,
        }[name]
    if name == "KokoroSynthesizer":
        from .synthesizer import KokoroSynthesizer

        return KokoroSynthesizer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AssetProgressEvent",
    "AssetProgressCallback",
    "ConsoleAssetProgress",
    "ModelCapabilities",
    "ModelDiscoveryResult",
    "VoiceCapabilities",
    "discover_models",
    "LexiconCapabilities",
    "LexiconDiscoveryResult",
    "discover_lexicons",
    "GenerationConfig",
    "LanguageRoutingConfig",
    "SynthesisConfig",
    "LongTextSplitMode",
    "SynthesisInputTooLongError",
    "VoiceLevelConfig",
    "VoiceBlend",
    "SynthesisSegment",
    "RenderedSegment",
    "PronunciationOverride",
    "LinguisticToken",
    "WordTiming",
    "TokenizerConfig",
    "EspeakConfig",
    "ShortSentenceConfig",
    "KokoroSynthesizer",
    "__version__",
    "__version_tuple__",
]
