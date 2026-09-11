from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from pykokoro.model_registry import ModelRegistry
from pykokoro.runtime.model_assets import resolve_runtime_assets


def _registry() -> ModelRegistry:
    def artifact(artifact_id: str, role: str, filename: str) -> dict[str, object]:
        payload = artifact_id.encode()
        return {
            "id": artifact_id,
            "role": role,
            "format": "onnx" if role == "model" else "numpy-npz",
            "url": f"https://example.test/{filename}",
            "local_name": filename,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            **({"quality": "fp32"} if role == "model" else {}),
        }

    return ModelRegistry(
        {
            "schema": 1,
            "runtime_contract": 1,
            "models": {
                "test-model": {
                    "runtime": {"layout": "single-onnx-v1", "voices": ["default"]},
                    "distributions": [
                        {
                            "id": "test-distribution",
                            "provider": "github-release",
                            "transport": "https",
                            "runtime_ready": True,
                            "artifacts": [
                                artifact("model", "model", "model.onnx"),
                                artifact("voices", "voices", "voices.npz"),
                            ],
                        }
                    ],
                }
            },
        },
        "test",
    )


def test_runtime_asset_resolution_logs_download_and_cache_lifecycle(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    def download(artifact, target):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(artifact.id.encode())
        return target

    monkeypatch.setattr("pykokoro.runtime.model_assets.download_artifact", download)
    caplog.set_level(logging.DEBUG, logger="pykokoro.runtime.model_assets")

    resolve_runtime_assets(
        model_id="test-model", quality="fp32", registry=_registry(), cache_dir=tmp_path
    )
    first_messages = [record.message for record in caplog.records]
    assert any("model.resolve.start model_id=test-model" in message for message in first_messages)
    assert any("artifact.download.required" in message for message in first_messages)
    assert any("artifact.materialize.finish" in message for message in first_messages)
    assert any("model.resolve.finish" in message for message in first_messages)

    caplog.clear()
    resolve_runtime_assets(
        model_id="test-model", quality="fp32", registry=_registry(), cache_dir=tmp_path
    )
    assert any("artifact.cache.hit" in record.message for record in caplog.records)
    assert any("model.resolve.finish" in record.message for record in caplog.records)
