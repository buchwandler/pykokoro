"""Dependency-light model and voice artifact filenames."""

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
    "q8": "kokoro-v1.0.q8.onnx",
    "q8f16": "kokoro-v1.0.q8f16.onnx",
    "q4": "kokoro-v1.0.q4.onnx",
    "q4f16": "kokoro-v1.0.q4f16.onnx",
    "uint8": "kokoro-v1.0.uint8.onnx",
    "uint8f16": "kokoro-v1.0.uint8f16.onnx",
}


MODEL_QUALITY_FILES_GITHUB_V1_1_ZH: dict[str, str] = {
    "fp32": "kokoro-v1.1-zh.onnx",
    "fp16": "kokoro-v1.1-zh.fp16.onnx",
    "q8": "kokoro-v1.1-zh.q8.onnx",
    "int8": "kokoro-v1.1-zh.int8.onnx",
    "q4": "kokoro-v1.1-zh.q4.onnx",
    "q4f16": "kokoro-v1.1-zh.q4f16.onnx",
    "uint8": "kokoro-v1.1-zh.uint8.onnx",
    "bnb4": "kokoro-v1.1-zh.bnb4.onnx",
}


MODEL_QUALITY_FILES_GITHUB_V1_2_DE_MARTIN: dict[str, str] = {
    "fp32": "kokoro-german-martin-v1.2.onnx",
}

MODEL_QUALITY_CACHE_FILES_HF_V1_0 = MODEL_QUALITY_FILES_HF.copy()
MODEL_QUALITY_FILES = MODEL_QUALITY_FILES_HF
GITHUB_VOICES_FILENAME_V1_0 = "voices-v1.0.npz"
GITHUB_VOICES_FILENAME_V1_1_ZH = "voices-v1.1-zh.npz"
GITHUB_VOICES_FILENAME_V1_2_DE_MARTIN = "voices-german-martin-v1.2.bin"
