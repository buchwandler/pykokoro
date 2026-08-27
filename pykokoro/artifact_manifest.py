"""Immutable upstream artifact coordinates used by the download backend."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArtifactSpec:
    """Pinned revision and optional content digest for one remote artifact."""

    revision: str
    sha256: str | None = None


HF_MODEL_REVISIONS: dict[str, str] = {
    "v1.0": "1939ad2a8e416c0acfeecc08a694d14ef25f2231",
    "v1.1-zh": "6cc0f0d2ebe369a68b0df87c2b65c1af8c0ac3e3",
}

HF_CONFIG_REVISIONS: dict[str, str] = {
    "v1.0": "f3ff3571791e39611d31c381e3a41a3af07b4987",
    "v1.1-zh": "01e7505bd6a7a2ac4975463114c3a7650a9f7218",
}

HF_MODEL_SHA256: dict[tuple[str, str], str] = {
    ("v1.0", "model.onnx"): "8fbea51ea711f2af382e88c833d9e288c6dc82ce5e98421ea61c058ce21a34cb",
    ("v1.0", "model_fp16.onnx"): "ba4527a874b42b21e35f468c10d326fdff3c7fc8cac1f85e9eb6c0dfc35c334a",
    ("v1.0", "model_quantized.onnx"): "fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478",
    ("v1.0", "model_q8f16.onnx"): "04c658aec1b6008857c2ad10f8c589d4180d0ec427e7e6118ceb487e215c3cd0",
    ("v1.0", "model_q4.onnx"): "04cf570cf9c4153694f76347ed4b9a48c1b59ff1de0999e6605d123966b197c7",
    ("v1.0", "model_q4f16.onnx"): "d1a508a6a29671ead84fac99c7401fbd3c21a583fc6ed1406d1ec974d53bf45f",
    ("v1.0", "model_uint8.onnx"): "6607a397d77b8514065420b7c1e7320117f7aabfdb45ce15f0050c5b0fe75aea",
    ("v1.0", "model_uint8f16.onnx"): "883333e03c597584b532eebea0f8310f25f0c9ade58fe864792c12d969944a9a",
    ("v1.1-zh", "model.onnx"): "94b973941b1852754f979be5d5e20be666d5c81d9bb886b88ae1dc85c9b895ca",
    ("v1.1-zh", "model_fp16.onnx"): "d59cb626c885c91acb7ecf7250f26f9915ad0af99b770572000d90b598c3dfbc",
    ("v1.1-zh", "model_int8.onnx"): "58b9b997faeaf42b427bac24c8a6246b236b0561311f6b118318cd9d2f47acb1",
    ("v1.1-zh", "model_quantized.onnx"): "a39469be791eeaa3089c1ed5e58b8731d1f2462ea0e7dae2bc44388e58f973d8",
    ("v1.1-zh", "model_q4.onnx"): "dbf3e5a505c4e453303de84d941d51271f5b460956d9b911dea310a90c88b7c2",
    ("v1.1-zh", "model_q4f16.onnx"): "9ee7ca1ace506c7dc983d0c9a20c7edfcb08c2027645ae4450afa3dbbd8a57ef",
    ("v1.1-zh", "model_uint8.onnx"): "a39469be791eeaa3089c1ed5e58b8731d1f2462ea0e7dae2bc44388e58f973d8",
    ("v1.1-zh", "model_bnb4.onnx"): "71d417643add4820933a7ae552074eb3dc646e622cf022bcd69dbbfea4332338",
}

HF_CONFIG_SHA256: dict[str, str] = {
    "v1.0": "5abb01e2403b072bf03d04fde160443e209d7a0dad49a423be15196b9b43c17f",
    "v1.1-zh": "bc333efa5ce4ceff433c8c8e5d027a1eca0166001e4e4a62bea2d26ff7a46890",
}


def hf_model_spec(variant: str, filename: str) -> ArtifactSpec:
    return ArtifactSpec(
        revision=HF_MODEL_REVISIONS[variant],
        sha256=HF_MODEL_SHA256.get((variant, filename)),
    )


def hf_config_spec(variant: str) -> ArtifactSpec:
    return ArtifactSpec(
        revision=HF_CONFIG_REVISIONS[variant],
        sha256=HF_CONFIG_SHA256.get(variant),
    )


def hf_voice_spec(variant: str, filename: str) -> ArtifactSpec:
    del filename
    return ArtifactSpec(revision=HF_MODEL_REVISIONS[variant])
