"""Dependency-light model and voice artifact filenames."""

from __future__ import annotations

from pathlib import Path

HF_MODEL_SUBFOLDER = "onnx"
HF_CONFIG_FILENAME = "config.json"

MODEL_QUALITY_FILES_HF: dict[str, str] = {
    "fp32": "model.onnx",
    "fp16": "model_fp16.onnx",
    "q8": "model_quantized.onnx",
    "q8f16": "model_q8f16.onnx",
    "q4": "model_q4.onnx",
    "q4f16": "model_q4f16.onnx",
    "uint8": "model_uint8.onnx",
    "uint8f16": "model_uint8f16.onnx",
}

MODEL_QUALITY_FILES_GITHUB_V1_0: dict[str, str] = {
    "fp32": "kokoro-v1.0.onnx",
    "fp16": "kokoro-v1.0.fp16.onnx",
    "fp16-gpu": "kokoro-v1.0.fp16-gpu.onnx",
    "q8": "kokoro-v1.0.int8.onnx",
}

MODEL_QUALITY_FILES_GITHUB_V1_1_ZH: dict[str, str] = {
    "fp32": "kokoro-v1.1-zh.onnx",
}

MODEL_QUALITY_FILES_GITHUB_V1_1_DE: dict[str, str] = {
    "fp32": "kokoro-german-v1.1.onnx",
    "q8": "kokoro-german-v1.1.int8.onnx",
}

MODEL_QUALITY_FILES_GITHUB_V1_2_DE_MARTIN: dict[str, str] = {
    "fp32": "kokoro-german-martin-v1.2.onnx",
}


def _timestamped_model_filename(filename: str) -> str:
    path = Path(filename)
    return f"{path.stem}-timestamped{path.suffix}"


MODEL_QUALITY_CACHE_FILES_HF_V1_0: dict[str, str] = {
    quality: _timestamped_model_filename(filename)
    for quality, filename in MODEL_QUALITY_FILES_HF.items()
}

MODEL_QUALITY_FILES = MODEL_QUALITY_FILES_HF

GITHUB_VOICES_FILENAME_V1_0 = "voices-v1.0.bin"
GITHUB_VOICES_FILENAME_V1_1_ZH = "voices-v1.1-zh.bin"
GITHUB_VOICES_FILENAME_V1_1_DE = "voices-german-v1.1.bin"
GITHUB_VOICES_FILENAME_V1_2_DE_MARTIN = "voices-german-martin-v1.2.bin"
