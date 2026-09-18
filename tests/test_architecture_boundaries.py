from __future__ import annotations

from pathlib import Path


def test_production_does_not_import_onnxruntime_directly() -> None:
    root = Path(__file__).parents[1] / "pykokoro"
    forbidden = ("import onnxruntime", "from onnxruntime", "onnxruntime.InferenceSession")
    for path in root.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(token in source for token in forbidden), path


def test_kokoro_initialization_uses_onnxvoice_boundary() -> None:
    source = (Path(__file__).parents[1] / "pykokoro" / "onnx_backend.py").read_text(
        encoding="utf-8"
    )
    start = source.index("    def _init_kokoro_locked")
    end = source.index("    def warmup", start)
    initializer = source[start:end]
    assert "install_kokoro_model" in initializer
    assert "open_installed_kokoro" in initializer
    assert "open_local_kokoro" in initializer
    assert "_ensure_models" not in initializer


def test_onnx_backend_does_not_reintroduce_legacy_asset_stack():
    source = (Path(__file__).parents[1] / "pykokoro" / "onnx_backend.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "asset_constants",
        "artifact_manifest",
        "from .model_assets",
        "release_catalog",
        "runtime.model_assets",
    )
    assert not any(token in source for token in forbidden)


def test_model_profiles_does_not_import_asset_constants():
    source = (Path(__file__).parents[1] / "pykokoro" / "model_profiles.py").read_text(
        encoding="utf-8"
    )
    assert "asset_constants" not in source
