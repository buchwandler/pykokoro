"""Immutable upstream artifact coordinates used by the download backend.

The revisions below are repository commits, rather than moving ``main`` or a
release tag.  The SHA-256 values are the Git-LFS object digests reported by
Hugging Face for the model files.  Callers may override both fields explicitly
for a separately audited mirror.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactSpec:
    """Pinned revision and optional content digest for one remote artifact."""

    revision: str
    sha256: str | None = None


HF_MODEL_REVISIONS: dict[str, str] = {
    "v1.0": "dd4401a9add81ac692d20e240d22ec9dda82cc29",
    "v1.1-zh": "6cc0f0d2ebe369a68b0df87c2b65c1af8c0ac3e3",
}

HF_CONFIG_REVISIONS: dict[str, str] = {
    "v1.0": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "v1.1-zh": "01e7505bd6a7a2ac4975463114c3a7650a9f7218",
}

# The model digest table covers every model quality exposed by the HF maps.
HF_MODEL_SHA256: dict[tuple[str, str], str] = {
    ("v1.0", "model.onnx"): "651ea8291843a92276a4a003581a215cb07d15e47dde6fcfb1b768f9a1682054",
    ("v1.0", "model_fp16.onnx"): "220724d5c5e0cc01be30f38faa6cf0c895a7cde6e7773e91db2973c8c7e5123c",
    ("v1.0", "model_q4.onnx"): "08dfea732b1c159378be5711f9f366c1bf99d35f9feade2504c8814117b7211a",
    (
        "v1.0",
        "model_q4f16.onnx",
    ): "cc96fd126a19f87b06cf19c206a7e1d7599e5f63bd1b0151c9929e523083596b",
    (
        "v1.0",
        "model_q8f16.onnx",
    ): "db06e3f12daac36b13638ff6f6c4541241705405dac6d23126a7827dadf4c677",
    (
        "v1.0",
        "model_quantized.onnx",
    ): "c0c02b3299fd97c34ea92a98e6d41eaa1a739c8f77bf685aac34bd7b34c1132c",
    (
        "v1.0",
        "model_uint8.onnx",
    ): "b44c5c0d90458a8d9175cea933ff6d8a6351b74f1e651060df48672f5a167e4f",
    (
        "v1.0",
        "model_uint8f16.onnx",
    ): "191cab6b6d0a8d12801a3f8d28ee4429d81200b627d17cd08f9976db65630e2c",
    ("v1.1-zh", "model.onnx"): "94b973941b1852754f979be5d5e20be666d5c81d9bb886b88ae1dc85c9b895ca",
    (
        "v1.1-zh",
        "model_fp16.onnx",
    ): "d59cb626c885c91acb7ecf7250f26f9915ad0af99b770572000d90b598c3dfbc",
    (
        "v1.1-zh",
        "model_q4.onnx",
    ): "dbf3e5a505c4e453303de84d941d51271f5b460956d9b911dea310a90c88b7c2",
    (
        "v1.1-zh",
        "model_q4f16.onnx",
    ): "9ee7ca1ace506c7dc983d0c9a20c7edfcb08c2027645ae4450afa3dbbd8a57ef",
    (
        "v1.1-zh",
        "model_q8f16.onnx",
    ): "d59cb626c885c91acb7ecf7250f26f9915ad0af99b770572000d90b598c3dfbc",
    (
        "v1.1-zh",
        "model_quantized.onnx",
    ): "a39469be791eeaa3089c1ed5e58b8731d1f2462ea0e7dae2bc44388e58f973d8",
    (
        "v1.1-zh",
        "model_uint8.onnx",
    ): "a39469be791eeaa3089c1ed5e58b8731d1f2462ea0e7dae2bc44388e58f973d8",
}

HF_CONFIG_SHA256: dict[str, str] = {
    "v1.0": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "v1.1-zh": "bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890",
}


def hf_model_spec(variant: str, filename: str) -> ArtifactSpec:
    """Return the pinned model coordinates for ``variant`` and ``filename``."""
    return ArtifactSpec(
        revision=HF_MODEL_REVISIONS[variant],
        sha256=HF_MODEL_SHA256.get((variant, filename)),
    )


def hf_config_spec(variant: str) -> ArtifactSpec:
    """Return the pinned config coordinates for ``variant``."""
    return ArtifactSpec(
        revision=HF_CONFIG_REVISIONS[variant],
        sha256=HF_CONFIG_SHA256.get(variant),
    )


def hf_voice_spec(variant: str, filename: str) -> ArtifactSpec:
    """Return the pinned voice revision; per-voice digests may be supplied by callers."""
    del filename
    return ArtifactSpec(revision=HF_MODEL_REVISIONS[variant])
