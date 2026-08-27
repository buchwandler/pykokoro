from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pykokoro.model_registry import RuntimeArtifact, RuntimeDistribution, RuntimeModel
from pykokoro.runtime.dispatcher import UnsupportedRuntimeLayout, create_runtime
from pykokoro.runtime.model_assets import ResolvedRuntimeAssets
from pykokoro.runtime.thai_wayu import ThaiWayuRuntime


def _assets(tmp_path: Path, layout: str = "split-onnx-v1") -> ResolvedRuntimeAssets:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"vocab": {"a": 1}}', encoding="utf-8")
    source = tmp_path / "source-params.npz"
    np.savez(source, weight=np.zeros((1, 9)), bias=np.zeros(1), window=np.ones(20))
    voices = tmp_path / "voices.npz"
    np.savez(voices, default=np.zeros((1, 256), dtype=np.float32))
    paths = {"manifest": manifest, "source_params": source, "voices": voices}
    artifacts = [
        RuntimeArtifact(
            f"model-{component}",
            "model",
            "onnx",
            f"https://example/{component}",
            f"{component}.onnx",
            1,
            "0" * 64,
            quality="fp32",
            component=component,
        )
        for component in ("prosody", "curves", "decoder")
    ]
    artifacts.extend(
        [
            RuntimeArtifact(
                "manifest", "config", "json", "https://example/manifest", manifest.name, 1, "0" * 64
            ),
            RuntimeArtifact(
                "source", "metadata", "numpy-npz", "https://example/source", source.name, 1, "0" * 64
            ),
            RuntimeArtifact(
                "voices", "voices", "numpy-npz", "https://example/voices", voices.name, 1, "0" * 64
            ),
        ]
    )
    materialized = dict(paths)
    for artifact in artifacts:
        path = tmp_path / artifact.local_name
        if artifact.role == "model":
            path.write_bytes(b"0")
        else:
            path = next(value for value in paths.values() if value.name == artifact.local_name)
        materialized[artifact.id] = path
    model = RuntimeModel(
        "thai",
        {"runtime": {"layout": layout, "max_tokens": 510, "default_voice": "default", "voices": ["default"]}},
        (RuntimeDistribution("dist", "github-release", "https", True, tuple(artifacts)),),
    )
    return ResolvedRuntimeAssets(
        "thai", "dist", "github-release", layout, materialized, model, model.distributions[0]
    )


def test_thai_dispatch_uses_registry_component_ids(tmp_path: Path) -> None:
    assets = _assets(tmp_path)
    requested: list[str] = []

    def factory(path: Path):
        requested.append(path.name)
        return object()

    runtime = create_runtime(assets, session_factory=factory)

    assert isinstance(runtime, ThaiWayuRuntime)
    assert requested == ["prosody.onnx", "curves.onnx", "decoder.onnx"]


def test_unknown_runtime_layout_is_explicit(tmp_path: Path) -> None:
    assets = _assets(tmp_path, layout="future-layout")

    with pytest.raises(UnsupportedRuntimeLayout, match="future-layout"):
        create_runtime(assets)
