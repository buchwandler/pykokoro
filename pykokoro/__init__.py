"""PyKokoro public API with lazy ONNX-backed exports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .generation_config import GenerationConfig
from .language_detection import LanguageDetectionConfig
from .loudness_config import LoudnessConfig
from .prosody_config import ProsodyConfig, ProsodyMethod
from .ssmd_config import SSMDPauseOverrides, SSMDRenderConfig
from .voice_level import VoiceCalibrationKey, VoiceLevelCalibration

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.9.2"
    __version_tuple__ = (0, 9, 2)


if TYPE_CHECKING:
    from .asset_progress import AssetProgressCallback, AssetProgressEvent, ConsoleAssetProgress
    from .discovery import (
        ModelCapabilities,
        ModelDiscoveryResult,
        VoiceCapabilities,
        discover_models,
    )
    from .lexicon_discovery import LexiconCapabilities, LexiconDiscoveryResult, discover_lexicons
    from .pipeline import (
        KokoroPipeline,
        PreparedAudioUnits,
        PreparedFrontend,
        build_pipeline,
        with_spacy_model,
        with_spacy_model_size,
    )
    from .pipeline_config import PipelineConfig, resolve_pipeline_config
    from .release_catalog import (
        available_model_releases,
        download_model_release,
        resolve_model_release,
    )
    from .types import AudioUnitDescriptor, AudioUnitKind, AudioUnitResult, WordTiming


def __getattr__(name: str) -> Any:
    if name in {"available_model_releases", "resolve_model_release", "download_model_release"}:
        from .release_catalog import (
            available_model_releases,
            download_model_release,
            resolve_model_release,
        )

        return {
            "available_model_releases": available_model_releases,
            "resolve_model_release": resolve_model_release,
            "download_model_release": download_model_release,
        }[name]
    if name in {
        "discover_models",
        "ModelCapabilities",
        "ModelDiscoveryResult",
        "VoiceCapabilities",
    }:
        from .discovery import (
            ModelCapabilities,
            ModelDiscoveryResult,
            VoiceCapabilities,
            discover_models,
        )

        return {
            "discover_models": discover_models,
            "ModelCapabilities": ModelCapabilities,
            "ModelDiscoveryResult": ModelDiscoveryResult,
            "VoiceCapabilities": VoiceCapabilities,
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
    if name in {"PipelineConfig", "resolve_pipeline_config"}:
        from .pipeline_config import PipelineConfig, resolve_pipeline_config

        return {
            "PipelineConfig": PipelineConfig,
            "resolve_pipeline_config": resolve_pipeline_config,
        }[name]
    if name in {
        "KokoroPipeline",
        "PreparedAudioUnits",
        "PreparedFrontend",
        "build_pipeline",
        "with_spacy_model",
        "with_spacy_model_size",
    }:
        try:
            from .pipeline import (
                KokoroPipeline,
                PreparedAudioUnits,
                PreparedFrontend,
                build_pipeline,
                with_spacy_model,
                with_spacy_model_size,
            )
        except ModuleNotFoundError as exc:
            if exc.name == "onnxruntime":
                raise RuntimeError(
                    "ONNX-backed pipeline support requires ONNX Runtime; "
                    "install pykokoro[cpu] or a platform provider extra."
                ) from exc
            raise
        return {
            "KokoroPipeline": KokoroPipeline,
            "PreparedAudioUnits": PreparedAudioUnits,
            "PreparedFrontend": PreparedFrontend,
            "build_pipeline": build_pipeline,
            "with_spacy_model": with_spacy_model,
            "with_spacy_model_size": with_spacy_model_size,
        }[name]
    if name in {"AudioUnitDescriptor", "AudioUnitKind", "AudioUnitResult", "WordTiming"}:
        from .types import AudioUnitDescriptor, AudioUnitKind, AudioUnitResult, WordTiming

        return {
            "AudioUnitDescriptor": AudioUnitDescriptor,
            "AudioUnitKind": AudioUnitKind,
            "AudioUnitResult": AudioUnitResult,
            "WordTiming": WordTiming,
        }[name]
    if name in {"AssetProgressEvent", "AssetProgressCallback", "ConsoleAssetProgress"}:
        from .asset_progress import AssetProgressCallback, AssetProgressEvent, ConsoleAssetProgress

        return {
            "AssetProgressEvent": AssetProgressEvent,
            "AssetProgressCallback": AssetProgressCallback,
            "ConsoleAssetProgress": ConsoleAssetProgress,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AssetProgressEvent",
    "AssetProgressCallback",
    "ConsoleAssetProgress",
    "available_model_releases",
    "download_model_release",
    "resolve_model_release",
    "discover_models",
    "ModelCapabilities",
    "ModelDiscoveryResult",
    "VoiceCapabilities",
    "LexiconCapabilities",
    "LexiconDiscoveryResult",
    "discover_lexicons",
    "GenerationConfig",
    "LoudnessConfig",
    "VoiceCalibrationKey",
    "VoiceLevelCalibration",
    "LanguageDetectionConfig",
    "ProsodyConfig",
    "ProsodyMethod",
    "SSMDPauseOverrides",
    "SSMDRenderConfig",
    "KokoroPipeline",
    "AudioUnitDescriptor",
    "AudioUnitKind",
    "AudioUnitResult",
    "WordTiming",
    "PreparedAudioUnits",
    "PreparedFrontend",
    "PipelineConfig",
    "resolve_pipeline_config",
    "__version__",
    "__version_tuple__",
    "build_pipeline",
    "with_spacy_model",
    "with_spacy_model_size",
]
