"""Dependency-light configuration types and defaults."""

from __future__ import annotations

from typing import Literal

ModelQuality = Literal[
    "fp32", "fp16", "fp16-gpu", "q8", "q8f16", "q4", "q4f16", "uint8", "uint8f16"
]
ModelSource = Literal["huggingface", "github"]
ModelVariant = Literal["v1.0", "v1.1-zh", "v1.1-de"]
ProviderType = Literal["auto", "cpu", "cuda", "openvino", "directml", "coreml"]

DEFAULT_MODEL_SOURCE: ModelSource = "huggingface"
DEFAULT_MODEL_VARIANT: ModelVariant = "v1.0"

LANG_CODE_TO_ONNX: dict[str, str] = {
    "a": "en-us",
    "b": "en-gb",
    "e": "es",
    "f": "fr-fr",
    "h": "hi",
    "i": "it",
    "j": "ja",
    "p": "pt-br",
    "z": "zh",
}
