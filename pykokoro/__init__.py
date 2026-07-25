"""PyKokoro - pipeline-first API for Kokoro TTS.

Public API:
- build_pipeline(...)
- KokoroPipeline
- PipelineConfig
- GenerationConfig
"""

from .pipeline import KokoroPipeline, build_pipeline, with_spacy_model_size
from .pipeline_config import PipelineConfig
from .generation_config import GenerationConfig

# Version info
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    __version__ = "0.0.0"
    __version_tuple__ = (0, 0, 0)


__all__ = [
    "GenerationConfig",
    "KokoroPipeline",
    "PipelineConfig",
    "__version__",
    "__version_tuple__",
    "build_pipeline",
    "with_spacy_model_size",
]
