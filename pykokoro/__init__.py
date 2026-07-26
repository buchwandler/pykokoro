"""PyKokoro public API with lazy ONNX-backed exports."""

from __future__ import annotations

from typing import Any

from .generation_config import GenerationConfig

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.6.5"
    __version_tuple__ = (0, 6, 5)


def __getattr__(name: str) -> Any:
    if name == "PipelineConfig":
        from .pipeline_config import PipelineConfig

        return PipelineConfig
    if name in {"KokoroPipeline", "build_pipeline", "with_spacy_model_size"}:
        try:
            from .pipeline import KokoroPipeline, build_pipeline, with_spacy_model_size
        except ModuleNotFoundError as exc:
            if exc.name == "onnxruntime":
                raise RuntimeError(
                    "ONNX-backed pipeline support requires ONNX Runtime; "
                    "install pykokoro[cpu] or a platform provider extra."
                ) from exc
            raise
        return {
            "KokoroPipeline": KokoroPipeline,
            "build_pipeline": build_pipeline,
            "with_spacy_model_size": with_spacy_model_size,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "GenerationConfig",
    "KokoroPipeline",
    "PipelineConfig",
    "__version__",
    "__version_tuple__",
    "build_pipeline",
    "with_spacy_model_size",
]
