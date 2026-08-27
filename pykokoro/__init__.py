"""PyKokoro public API with lazy ONNX-backed exports."""

from __future__ import annotations

from typing import Any

from .generation_config import GenerationConfig
from .prosody_config import ProsodyConfig, ProsodyMethod
from .ssmd_config import SSMDPauseOverrides, SSMDRenderConfig

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.8.7"
    __version_tuple__ = (0, 8, 7)


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
    if name == "PipelineConfig":
        from .pipeline_config import PipelineConfig

        return PipelineConfig
    if name in {
        "KokoroPipeline",
        "PreparedAudioUnits",
        "build_pipeline",
        "with_spacy_model",
        "with_spacy_model_size",
    }:
        try:
            from .pipeline import (
                KokoroPipeline,
                PreparedAudioUnits,
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "available_model_releases",
    "download_model_release",
    "resolve_model_release",
    "GenerationConfig",
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
    "PipelineConfig",
    "__version__",
    "__version_tuple__",
    "build_pipeline",
    "with_spacy_model",
    "with_spacy_model_size",
]
