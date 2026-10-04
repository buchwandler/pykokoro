from __future__ import annotations

import json
from pathlib import Path

import pytest

from pykokoro.synthesis_config import SynthesisConfig, _resolve_manifest_paths

_COMPONENTS = (
    "reference_wavlm",
    "reference_encoders",
    "reference_mapper",
    "prosody",
    "curves",
    "decoder",
    "source_params",
    "config",
)


def _manifest_assets() -> list[dict[str, str]]:
    return [
        {
            "role": "config"
            if component == "config"
            else "metadata"
            if component == "source_params"
            else "model",
            "component": component,
            "name": f"{component}.onnx"
            if component not in {"source_params", "config"}
            else f"{component}.npz"
            if component == "source_params"
            else "config.json",
        }
        for component in _COMPONENTS
    ]


def test_release_manifest_resolves_componentized_reference_artifacts(tmp_path: Path) -> None:
    manifest = tmp_path / "release-manifest.json"
    manifest.write_text(json.dumps({"schema": 2, "assets": _manifest_assets()}), encoding="utf-8")

    resolved = _resolve_manifest_paths(
        SynthesisConfig(
            release_manifest_path=manifest,
            model_variant="en-akinvox-cloning-v1",
            model_quality="fp32",
        )
    )

    assert resolved.model_path is None
    assert resolved.voices_path is None
    assert resolved.model_config_path == tmp_path / "config.json"
    assert set(resolved.model_artifacts or {}) == set(_COMPONENTS)
    for component in _COMPONENTS:
        suffix = (
            "json" if component == "config" else "npz" if component == "source_params" else "onnx"
        )
        assert resolved.model_artifacts[component] == tmp_path / f"{component}.{suffix}"  # type: ignore[index]


def test_bundle_manifest_resolves_components_and_nested_source_parameters(tmp_path: Path) -> None:
    manifest = tmp_path / "bundle.json"
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    data = {
        "components": [
            {"component": component, "filename": f"{component}.onnx", "role": "model"}
            for component in _COMPONENTS[:6]
        ],
        "source_params": {"path": "source-params.npz", "format": "numpy-npz"},
    }
    manifest.write_text(json.dumps(data), encoding="utf-8")

    resolved = _resolve_manifest_paths(
        SynthesisConfig(
            release_manifest_path=manifest,
            model_variant="en-akinvox-cloning-v1",
            model_quality="fp32",
        )
    )

    assert resolved.model_artifacts is not None
    assert resolved.model_artifacts["source_params"] == tmp_path / "source-params.npz"
    assert resolved.model_artifacts["config"] == tmp_path / "config.json"


def test_component_manifest_rejects_missing_runtime_artifact(tmp_path: Path) -> None:
    manifest = tmp_path / "incomplete.json"
    assets = [item for item in _manifest_assets() if item["component"] != "decoder"]
    manifest.write_text(json.dumps({"assets": assets}), encoding="utf-8")

    with pytest.raises(ValueError, match="lacks component artifacts: decoder"):
        _resolve_manifest_paths(
            SynthesisConfig(
                release_manifest_path=manifest,
                model_variant="en-akinvox-cloning-v1",
                model_quality="fp32",
            )
        )
