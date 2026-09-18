from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pykokoro._onnxvoice as boundary
from pykokoro.asset_progress import AssetProgressEvent
from pykokoro.exceptions import BackendError, ConfigurationError


def test_normalize_kokoro_ref() -> None:
    assert boundary.normalize_kokoro_ref("v1.0") == "kokoro:v1.0"
    assert boundary.normalize_kokoro_ref("kokoro:v1.0") == "kokoro:v1.0"

    with pytest.raises(ConfigurationError, match="Not a Kokoro"):
        boundary.normalize_kokoro_ref("piper:en_US")


def test_normalize_provider_request_supports_legacy_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ONNX_PROVIDER", "gpu")

    providers, options = boundary.normalize_provider_request(None, {"device_id": 2})

    assert providers == "cuda"
    assert options == [{"device_id": 2}]


def test_normalize_provider_request_preserves_provider_specific_options() -> None:
    providers, options = boundary.normalize_provider_request(
        [("CUDAExecutionProvider", {"device_id": "1"}), "cpu"],
        None,
    )

    assert providers == ["CUDAExecutionProvider", "cpu"]
    assert options == [{"device_id": "1"}, {}]


def test_resolve_maps_installation_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = tmp_path / "model.onnx"
    voices = tmp_path / "voices.bin"
    config = tmp_path / "config.json"
    installation = SimpleNamespace(
        id="v1.0",
        ref="kokoro:v1.0",
        sample_rate=24_000,
        metadata={"selected_quality": "fp32"},
        artifacts=(
            SimpleNamespace(role="model", path=model),
            SimpleNamespace(role="voices", path=voices),
            SimpleNamespace(role="config", path=config),
        ),
    )

    class Manager:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs == {"cache_dir": tmp_path, "offline": True}

        def resolve(self, ref: str, **kwargs: object) -> object:
            assert ref == "kokoro:v1.0"
            assert kwargs == {"quality": "fp32", "distribution": "cpu"}
            return installation

    monkeypatch.setattr(boundary, "_onnxvoice", lambda: SimpleNamespace(OnnxVoice=Manager))

    resolved = boundary.resolve_kokoro_model(
        "v1.0",
        quality="fp32",
        distribution="cpu",
        cache_dir=tmp_path,
        offline=True,
    )

    assert resolved.ref == "kokoro:v1.0"
    assert resolved.model_id == "v1.0"
    assert resolved.quality == "fp32"
    assert resolved.sample_rate == 24_000
    assert resolved.model_paths == (model,)
    assert resolved.voices_path == voices
    assert resolved.config_path == config


def test_open_local_forwards_split_artifacts_and_runtime_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def open_local(**kwargs: object) -> str:
        calls.append(kwargs)
        return "runtime"

    monkeypatch.setattr(boundary, "_onnxvoice", lambda: SimpleNamespace(open_local=open_local))

    result = boundary.open_local_kokoro(
        artifacts={"prosody": tmp_path / "prosody.onnx", "curves": tmp_path / "curves.onnx"},
        runtime={"layout": "split"},
        providers="cpu",
        provider_options={"intra_op_num_threads": 2},
        sample_rate=24_000,
    )

    assert isinstance(result, boundary.KokoroRuntimeAdapter)
    assert result.installation is None
    assert calls == [
        {
            "system": "kokoro",
            "model": None,
            "artifacts": {"prosody": tmp_path / "prosody.onnx", "curves": tmp_path / "curves.onnx"},
            "config": None,
            "voices": None,
            "runtime": {"layout": "split"},
            "sample_rate": 24_000,
            "providers": "cpu",
            "provider_options": [{"intra_op_num_threads": 2}],
            "session_options": None,
        }
    ]


def test_open_errors_are_mapped_and_chained(monkeypatch: pytest.MonkeyPatch) -> None:
    class Manager:
        def open(self, *args: object, **kwargs: object) -> object:
            raise RuntimeError("provider unavailable")

    monkeypatch.setattr(boundary, "_onnxvoice", lambda: SimpleNamespace(OnnxVoice=Manager))
    resolved = SimpleNamespace(installation=object())

    with pytest.raises(BackendError, match="Could not open Kokoro runtime") as caught:
        boundary.open_installed_kokoro(resolved)

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_progress_and_diagnostics_are_trace_safe() -> None:
    events: list[AssetProgressEvent] = []
    callback = boundary.adapt_progress(events.append)
    assert callback is not None
    callback(
        SimpleNamespace(
            phase="download_progress",
            ref="kokoro:v1.0",
            distribution="cpu",
            artifact="model.onnx",
            role="model",
            filename="model.onnx",
            completed=3,
            total=5,
            target="cache",
        )
    )

    assert events == [
        AssetProgressEvent(
            phase="download-progress",
            model_id="v1.0",
            distribution_id="",
            artifact_id="model.onnx",
            role="model",
            filename="model.onnx",
            bytes_done=3,
            bytes_total=5,
            target="cache",
        )
    ]


def test_progress_maps_install_lifecycle_and_ignores_unknown_phases() -> None:
    events: list[AssetProgressEvent] = []
    callback = boundary.adapt_progress(events.append)
    assert callback is not None

    callback(SimpleNamespace(phase="install_started", ref="kokoro:v1.0"))
    assert events == []

    callback(
        SimpleNamespace(
            phase="install_completed",
            ref="kokoro:v1.0",
            target="/cache/onnxvoice/kokoro/v1.0",
        )
    )
    assert events == [
        AssetProgressEvent(
            phase="install-complete",
            model_id="v1.0",
            distribution_id="",
            artifact_id="",
            role="artifact",
            filename="",
            bytes_done=0,
            bytes_total=None,
            target="/cache/onnxvoice/kokoro/v1.0",
        )
    ]

    callback(SimpleNamespace(phase="future_phase", ref="kokoro:v1.0"))
    assert len(events) == 1
    diagnostic = SimpleNamespace(
        system="kokoro",
        ref="kokoro:v1.0",
        layout="split",
        sessions=(
            SimpleNamespace(
                component="prosody",
                model_path=Path("prosody.onnx"),
                providers_requested=("CPUExecutionProvider",),
                providers_active=("CPUExecutionProvider",),
                inputs=("input_ids",),
                outputs=("duration",),
            ),
        ),
    )
    runtime = SimpleNamespace(diagnostics=lambda: diagnostic)

    assert boundary.runtime_diagnostics(runtime)["layout"] == "split"
    assert boundary.runtime_diagnostics(runtime)["sessions"][0]["component"] == "prosody"


def test_summarize_inference_is_json_safe() -> None:
    result = SimpleNamespace(
        timings=np.zeros(4, dtype=np.float32),
        outputs={"duration": np.zeros((1, 4), dtype=np.int64)},
    )

    timings, outputs = boundary.summarize_inference(result)

    assert timings == {"shape": [4], "dtype": "float32"}
    assert outputs == {"duration": {"shape": [1, 4], "dtype": "int64"}}
