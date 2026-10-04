from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import pykokoro._onnxvoice as boundary
from pykokoro.asset_progress import AssetProgressEvent
from pykokoro.exceptions import BackendError, ConfigurationError
from pykokoro.reference_voice import ReferenceVoice


def test_normalize_kokoro_ref() -> None:
    assert boundary.normalize_kokoro_ref("v1.0") == "kokoro:v1.0"
    assert boundary.normalize_kokoro_ref("kokoro:v1.0") == "kokoro:v1.0"


def test_backend_forwards_cache_dir_to_managed_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pykokoro.onnx_backend as onnx_backend

    class InstallReached(Exception):
        pass

    calls: list[dict[str, object]] = []

    def install(*args: object, **kwargs: object) -> object:
        calls.append(kwargs)
        raise InstallReached

    monkeypatch.setattr(onnx_backend, "install_kokoro_model", install)
    backend = object.__new__(onnx_backend.Kokoro)
    backend._runtime = None
    backend._audio_generator = None
    backend._provider = None
    backend._use_gpu = False
    backend._model_variant = "v1.0"
    backend._model_source = "github"
    backend._model_quality = "fp32"
    backend._model_path = None
    backend._model_artifacts = None
    backend._voices_path = None
    backend._cache_dir = tmp_path
    backend._asset_progress = None

    with pytest.raises(InstallReached):
        backend._init_kokoro_locked()

    assert calls[0]["cache_dir"] == tmp_path
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
        selected_quality="fp32",
        selected_distribution="github-model-files-v1.0-timestamped-r4",
        storage_id="kokoro--sel-abc123",
        metadata={},
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
            assert kwargs == {
                "quality": "fp32",
                "distribution": "github-model-files-v1.0-timestamped-r4",
            }
            return installation

    monkeypatch.setattr(boundary, "_onnxvoice", lambda: SimpleNamespace(OnnxVoice=Manager))

    resolved = boundary.resolve_kokoro_model(
        "v1.0",
        quality="fp32",
        source="github",
        cache_dir=tmp_path,
        offline=True,
    )

    assert resolved.ref == "kokoro:v1.0"
    assert resolved.model_id == "v1.0"
    assert resolved.quality == "fp32"
    assert resolved.distribution == "github-model-files-v1.0-timestamped-r4"
    assert resolved.storage_id == "kokoro--sel-abc123"
    assert resolved.sample_rate == 24_000
    assert resolved.model_paths == (model,)
    assert resolved.voices_path == voices
    assert resolved.config_path == config


@pytest.mark.parametrize(
    ("metadata", "installation", "runtime_support", "expected"),
    [
        ({}, SimpleNamespace(), None, None),
        ({"runtime": {"timings_output": "durations"}}, SimpleNamespace(), None, True),
        (
            {"onnx_contract": {"timing": {"output": "durations"}}},
            SimpleNamespace(),
            None,
            True,
        ),
        ({}, SimpleNamespace(timing_output="durations"), None, True),
        ({}, SimpleNamespace(), False, False),
    ],
)
def test_runtime_adapter_preserves_tri_state_timing_support(
    metadata: dict[str, object],
    installation: object,
    runtime_support: bool | None,
    expected: bool | None,
    tmp_path: Path,
) -> None:
    runtime = SimpleNamespace()
    if runtime_support is not None:
        runtime.supports_timings = runtime_support
    resolved = boundary.ResolvedKokoroModel(
        ref="kokoro:v1.0",
        model_id="v1.0",
        quality="fp32",
        distribution="github-model-files-v1.0-timestamped-r4",
        storage_id="kokoro--test",
        installation=installation,
        metadata=metadata,
        sample_rate=24_000,
        model_paths=(tmp_path / "model.onnx",),
        voices_path=None,
        config_path=None,
    )
    adapter = boundary.KokoroRuntimeAdapter(runtime, resolved)
    assert adapter.supports_timings is expected


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
            "metadata": None,
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


def test_runtime_diagnostics_exposes_timing_contract_json_safely(tmp_path: Path) -> None:
    resolved = boundary.ResolvedKokoroModel(
        ref="kokoro:v1.0",
        model_id="v1.0",
        quality="fp32",
        distribution="github-model-files-v1.0-timestamped-r4",
        storage_id="storage-1",
        installation=None,
        metadata={"runtime": {"timings_output": "durations"}},
        sample_rate=24_000,
        model_paths=(tmp_path / "model.onnx",),
        voices_path=tmp_path / "voices.bin",
        config_path=None,
    )
    diagnostic = SimpleNamespace(
        system="kokoro",
        ref="kokoro:v1.0",
        layout="single",
        sessions=(SimpleNamespace(outputs=("duration",)),),
    )
    runtime = SimpleNamespace(
        resolved=resolved,
        supports_timings=True,
        timing_layout="special-padded",
        diagnostics=lambda: diagnostic,
    )

    summary = boundary.runtime_diagnostics(runtime)
    json.dumps(summary)
    assert summary["distribution_id"] == "github-model-files-v1.0-timestamped-r4"
    assert summary["timing_contract"] == {
        "supports_timings": True,
        "output": "durations",
        "layout": "special-padded",
    }
    assert summary["sessions"][0]["outputs"] == ["duration"]


def test_summarize_inference_is_json_safe() -> None:
    result = SimpleNamespace(
        timings=np.zeros(4, dtype=np.float32),
        outputs={"duration": np.zeros((1, 4), dtype=np.int64)},
    )

    timings, outputs = boundary.summarize_inference(result)

    assert timings == {"shape": [4], "dtype": "float32", "count": 4, "layout": "unknown"}
    assert outputs == {"duration": {"shape": [1, 4], "dtype": "int64", "count": 4}}


def test_onnxvoice_distribution_for_maps_github_variants() -> None:
    assert (
        boundary.onnxvoice_distribution_for(source="github", variant="v1.0")
        == "github-model-files-v1.0-timestamped-r4"
    )
    assert (
        boundary.onnxvoice_distribution_for(source="github", variant="v1.1-zh")
        == "github-model-files-v1.1"
    )


def test_onnxvoice_distribution_for_raises_on_unmapped_pair() -> None:
    with pytest.raises(ConfigurationError, match="No OnnxVoice distribution mapping"):
        boundary.onnxvoice_distribution_for(source="huggingface", variant="v1.0")

    with pytest.raises(ConfigurationError, match="No OnnxVoice distribution mapping"):
        boundary.onnxvoice_distribution_for(source="github", variant="v2.0")


def test_cache_identity_uses_storage_id_for_managed_installations(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    resolved = boundary.ResolvedKokoroModel(
        ref="kokoro:v1.0",
        model_id="v1.0",
        quality="fp32",
        distribution="github-model-files-v1.0-timestamped-r4",
        storage_id="kokoro--sel-abc123",
        installation=None,
        metadata={},
        sample_rate=24_000,
        model_paths=(model,),
        voices_path=None,
        config_path=None,
    )
    runtime = SimpleNamespace(infer=lambda *a, **kw: None)
    adapter = boundary.KokoroRuntimeAdapter(runtime, resolved)

    assert adapter.cache_identity == ("managed", "kokoro:v1.0", "kokoro--sel-abc123")


def test_cache_identity_uses_paths_for_local_models(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    voices = tmp_path / "voices.bin"
    resolved = boundary.ResolvedKokoroModel(
        ref=None,
        model_id=None,
        quality=None,
        distribution=None,
        storage_id=None,
        installation=None,
        metadata={},
        sample_rate=24_000,
        model_paths=(model,),
        voices_path=voices,
        config_path=None,
    )
    runtime = SimpleNamespace(infer=lambda *a, **kw: None)
    adapter = boundary.KokoroRuntimeAdapter(runtime, resolved)

    assert adapter.cache_identity[0] == "local"
    assert adapter.cache_identity[1] == (str(model.resolve()),)
    assert adapter.cache_identity[2] == str(voices.resolve())


def test_different_distributions_produce_different_cache_identities(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    resolved_a = boundary.ResolvedKokoroModel(
        ref="kokoro:v1.0",
        model_id="v1.0",
        quality="fp32",
        distribution="github-model-files-v1.0-timestamped-r4",
        storage_id="kokoro--sel-aaa",
        installation=None,
        metadata={},
        sample_rate=24_000,
        model_paths=(model,),
        voices_path=None,
        config_path=None,
    )
    resolved_b = boundary.ResolvedKokoroModel(
        ref="kokoro:v1.0",
        model_id="v1.0",
        quality="fp32",
        distribution="github-model-files-v1.1",
        storage_id="kokoro--sel-bbb",
        installation=None,
        metadata={},
        sample_rate=24_000,
        model_paths=(model,),
        voices_path=None,
        config_path=None,
    )
    runtime = SimpleNamespace(infer=lambda *a, **kw: None)
    adapter_a = boundary.KokoroRuntimeAdapter(runtime, resolved_a)
    adapter_b = boundary.KokoroRuntimeAdapter(runtime, resolved_b)

    assert adapter_a.cache_identity != adapter_b.cache_identity


@pytest.mark.parametrize(
    ("timings", "expected_layout", "expected_positions"),
    [
        (np.arange(5, dtype=np.float32), "special-padded", np.arange(1, 4, dtype=np.float32)),
        (np.arange(3, dtype=np.float32), "model-positions", np.arange(3, dtype=np.float32)),
    ],
)
def test_timing_boundary_classifies_supported_layouts(
    timings: np.ndarray, expected_layout: str, expected_positions: np.ndarray
) -> None:
    result = SimpleNamespace(
        timings=timings,
        outputs={"duration": timings.copy()},
        metadata={"runtime_ref": "onnxvoice-0.1.9"},
    )
    payload = boundary.timing_payload_from_result(result, generated_position_count=3)

    assert payload is not None
    assert payload.layout == expected_layout
    assert payload.raw_count == len(timings)
    np.testing.assert_array_equal(payload.model_position_values, expected_positions)
    assert payload.is_valid


def test_timing_boundary_rejects_unknown_count_and_non_finite_values() -> None:
    unknown = boundary.classify_timing(np.ones(4), generated_position_count=3)
    non_finite = boundary.classify_timing(np.array([1.0, np.nan, 1.0]), generated_position_count=3)

    assert unknown.layout == "invalid"
    assert not unknown.is_valid
    assert "unsupported duration count" in (unknown.error or "")
    assert non_finite.layout == "model-positions"
    assert not non_finite.is_valid
    assert "finite" in (non_finite.error or "")


def test_timing_summary_preserves_named_outputs_and_runtime_identity() -> None:
    result = SimpleNamespace(
        timings=np.ones(3, dtype=np.float32),
        outputs={"waveform": np.zeros((1, 8), dtype=np.float32)},
        metadata={"runtime_ref": "onnxvoice-0.1.9"},
    )
    timing, outputs = boundary.summarize_inference(result, input_token_count=3)

    assert timing == {"shape": [3], "dtype": "float32", "count": 3, "layout": "model-positions"}
    assert outputs == {"waveform": {"shape": [1, 8], "dtype": "float32", "count": 8}}
    assert result.metadata["runtime_ref"] == "onnxvoice-0.1.9"


def _reference_voice() -> ReferenceVoice:
    return ReferenceVoice(
        style=np.zeros((1, 256), dtype=np.float32),
        memory=np.ones((1, 2, 192), dtype=np.float32),
        memory_mask=np.ones((1, 2), dtype=np.bool_),
        model_id="en-akinvox-cloning-v1",
        model_fingerprint="model-fingerprint",
    )


def test_runtime_adapter_prepares_and_converts_reference_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RuntimeReferenceState:
        def __init__(self, **values: object) -> None:
            self.values = values

    class Runtime:
        def __init__(self) -> None:
            self.prepared: dict[str, object] | None = None
            self.inference: dict[str, object] | None = None

        def prepare_reference(self, token_ids: object, **kwargs: object) -> SimpleNamespace:
            self.prepared = {"token_ids": token_ids, **kwargs}
            return SimpleNamespace(
                style=np.zeros((1, 256), dtype=np.float32),
                memory=np.ones((1, 2, 192), dtype=np.float32),
                memory_mask=np.ones((1, 2), dtype=np.bool_),
                model_fingerprint="model-fingerprint",
            )

        def infer(self, token_ids: object, **kwargs: object) -> str:
            self.inference = {"token_ids": token_ids, **kwargs}
            return "inferred"

    monkeypatch.setattr(
        boundary,
        "_onnxvoice",
        lambda: SimpleNamespace(KokoroReferenceState=RuntimeReferenceState),
    )
    runtime = Runtime()
    resolved = boundary.ResolvedKokoroModel(
        ref="kokoro:en-akinvox-cloning-v1",
        model_id="en-akinvox-cloning-v1",
        quality="fp32",
        distribution="fixture",
        storage_id="managed-clone",
        installation=None,
        metadata={"runtime": {"layout": "cloning-onnx-v1"}},
        sample_rate=24_000,
        model_paths=(),
        voices_path=None,
        config_path=None,
    )
    adapter = boundary.KokoroRuntimeAdapter(runtime, resolved)
    audio_24k = np.ones(72_000, dtype=np.float32)
    audio_16k = np.ones(48_000, dtype=np.float32)

    state = adapter.prepare_reference([1, 2, 3], audio_24k=audio_24k, audio_16k=audio_16k)
    result = adapter.infer([4, 5], reference=_reference_voice(), speed=1.0, seed=17)

    assert state.model_fingerprint == "model-fingerprint"
    assert runtime.prepared == {
        "token_ids": [1, 2, 3],
        "audio_24k": audio_24k,
        "audio_16k": audio_16k,
    }
    assert result == "inferred"
    assert runtime.inference is not None
    assert runtime.inference["token_ids"] == [4, 5]
    assert runtime.inference["speed"] == 1.0
    assert runtime.inference["seed"] == 17
    assert isinstance(runtime.inference["reference"], RuntimeReferenceState)
    assert runtime.inference["reference"].values["model_fingerprint"] == "model-fingerprint"


def test_open_local_cloning_artifacts_preserves_model_components(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    components = (
        "reference_wavlm",
        "reference_encoders",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
        "source_params",
        "config",
    )
    artifact_paths = {component: tmp_path / f"{component}.bin" for component in components}
    artifact_keys = {
        (
            "metadata:source_params"
            if component == "source_params"
            else "config"
            if component == "config"
            else f"model:{component}"
        ): path
        for component, path in artifact_paths.items()
    }
    calls: list[dict[str, object]] = []

    def open_local(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        artifacts = []
        for key, path in kwargs["artifacts"].items():
            role, _, component = key.partition(":")
            artifacts.append(
                SimpleNamespace(
                    role=role,
                    component=component or None,
                    path=path,
                )
            )
        installation = SimpleNamespace(
            id="local-config",
            ref=None,
            sample_rate=24_000,
            selected_quality=None,
            selected_distribution=None,
            storage_id=None,
            metadata={
                "model_id": "en-akinvox-cloning-v1",
                "runtime": kwargs["runtime"],
            },
            artifacts=tuple(artifacts),
        )
        return SimpleNamespace(installation=installation, infer=lambda *args, **kw: None)

    monkeypatch.setattr(boundary, "_onnxvoice", lambda: SimpleNamespace(open_local=open_local))

    adapter = boundary.open_local_kokoro(
        artifacts=artifact_keys,
        runtime={"layout": "cloning-onnx-v1", "voice_mode": "reference"},
        metadata={"model_id": "en-akinvox-cloning-v1"},
        sample_rate=24_000,
    )

    assert calls[0]["artifacts"] == artifact_keys
    assert adapter.resolved.model_id == "en-akinvox-cloning-v1"
    assert adapter.resolved.voices_path is None
    assert adapter.resolved.model_artifacts == artifact_paths
    assert set(adapter.resolved.model_paths) == {
        artifact_paths[component] for component in components[:6]
    }


def test_reference_backend_opens_components_without_a_voice_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pykokoro.onnx_backend as onnx_backend

    components = (
        "reference_wavlm",
        "reference_encoders",
        "reference_mapper",
        "prosody",
        "curves",
        "decoder",
        "source_params",
        "config",
    )
    model_artifacts = {name: tmp_path / f"{name}.bin" for name in components}
    expected_artifacts = {
        (
            "metadata:source_params"
            if name == "source_params"
            else "config"
            if name == "config"
            else f"model:{name}"
        ): path
        for name, path in model_artifacts.items()
    }
    resolved = boundary.ResolvedKokoroModel(
        ref=None,
        model_id="en-akinvox-cloning-v1",
        quality="fp32",
        distribution=None,
        storage_id=None,
        installation=None,
        metadata={"runtime": {"layout": "cloning-onnx-v1", "voice_mode": "reference"}},
        sample_rate=24_000,
        model_paths=tuple(model_artifacts[name] for name in components[:6]),
        voices_path=None,
        config_path=model_artifacts["config"],
        model_artifacts=model_artifacts,
    )
    reference_calls: list[tuple[object, dict[str, object]]] = []

    def prepare_reference(token_ids: object, **kwargs: object) -> SimpleNamespace:
        reference_calls.append((token_ids, kwargs))
        return SimpleNamespace(
            style=np.zeros((1, 256), dtype=np.float32),
            memory=np.ones((1, 2, 192), dtype=np.float32),
            memory_mask=np.ones((1, 2), dtype=np.bool_),
            model_fingerprint="model-fingerprint",
        )

    runtime = SimpleNamespace(resolved=resolved, prepare_reference=prepare_reference)
    calls: list[dict[str, object]] = []

    def open_local(**kwargs: object) -> SimpleNamespace:
        calls.append(kwargs)
        return runtime

    class AudioGeneratorStub:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    monkeypatch.setattr(onnx_backend, "open_local_kokoro", open_local)
    monkeypatch.setattr(
        onnx_backend,
        "runtime_diagnostics",
        lambda _runtime: {"timing_contract": {"supports_timings": False}},
    )
    monkeypatch.setattr(onnx_backend, "AudioGenerator", AudioGeneratorStub)
    backend = onnx_backend.Kokoro(
        model_artifacts=model_artifacts,
        model_variant="en-akinvox-cloning-v1",
        model_quality="fp32",
    )
    backend._tokenizer = object()

    backend._init_kokoro_locked()

    audio_24k = np.ones(72_000, dtype=np.float32)
    audio_16k = np.ones(48_000, dtype=np.float32)
    voice = backend.prepare_reference_voice(
        [1, 2, 3],
        audio_24k=audio_24k,
        audio_16k=audio_16k,
        name="speaker",
        metadata={"reference_text_sha256": "a" * 64},
    )
    assert calls[0]["artifacts"] == expected_artifacts
    assert calls[0]["runtime"] == {"layout": "cloning-onnx-v1", "voice_mode": "reference"}
    assert calls[0]["metadata"] == {"model_id": "en-akinvox-cloning-v1"}
    assert reference_calls == [
        (
            [1, 2, 3],
            {"audio_24k": audio_24k, "audio_16k": audio_16k},
        )
    ]
    assert voice.model_id == "en-akinvox-cloning-v1"
    assert voice.model_fingerprint == "model-fingerprint"
    assert voice.metadata["reference_text_sha256"] == "a" * 64
    assert backend._voice_manager is None
    assert backend.get_voices() == []
    with pytest.raises(ConfigurationError, match="Static voice styles are unavailable"):
        backend.get_voice_style("af_heart")
