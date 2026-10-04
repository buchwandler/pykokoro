from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import pykokoro._onnxvoice as onnxvoice_boundary
import pykokoro.synthesizer as synthesizer_module
from pykokoro.exceptions import ConfigurationError, InvalidRequestError, InvalidVoiceError
from pykokoro.model_registry import VoiceEnrollerSpec
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesizer import KokoroSynthesizer
from pykokoro.voice_enrollment import InnoEnrollmentOptions
from pykokoro.voice_pack import KokoroVoicePack


def tone(sample_rate: int = 24_000, duration: float = 3.5) -> np.ndarray:
    times = np.arange(round(sample_rate * duration), dtype=np.float64) / sample_rate
    return (0.15 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)


class NoCallG2P:
    def phonemize(self, *args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Inno enrollment must not phonemize a transcript")


class FakeRenderer:
    def __init__(self) -> None:
        self.call: tuple[Any, ...] | None = None

    def enroll_inno_voice(
        self,
        audio: Any,
        config: SynthesisConfig,
        *,
        language: str,
        enroller: str,
        options: InnoEnrollmentOptions,
        name: str | None,
    ) -> KokoroVoicePack:
        self.call = (audio, config, language, enroller, options, name)
        return KokoroVoicePack.from_array(
            np.zeros((510, 1, 256), dtype=np.float32), engine=enroller, name=name
        )


def inno_spec(*, transcript_required: bool = False) -> VoiceEnrollerSpec:
    return VoiceEnrollerSpec(
        id="inno-v0.2",
        kind="zero-shot-tuner",
        transcript_required=transcript_required,
        min_seconds=3.0,
        max_seconds=30.0,
        recommended_seconds=10.0,
        output_format="kokoro-voicepack-v1",
    )


def patch_resolver(monkeypatch: pytest.MonkeyPatch, *, transcript_required: bool = False) -> None:
    monkeypatch.setattr(
        synthesizer_module,
        "resolve_voice_enroller",
        lambda model_id, enroller_id: (
            model_id or "v1.0",
            inno_spec(transcript_required=transcript_required),
        ),
    )


def test_inno_enrollment_uses_original_mono_audio_and_typed_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_resolver(monkeypatch)
    source = tone(44_100)
    stereo = np.column_stack((source, source * 0.5))
    renderer = FakeRenderer()
    synthesizer = KokoroSynthesizer(
        SynthesisConfig(voice="af_heart"), g2p=NoCallG2P(), renderer=renderer
    )

    pack = synthesizer.enroll_voice(
        stereo,
        sample_rate=44_100,
        name="tuned",
        options=InnoEnrollmentOptions(fmax=8_000),
    )

    assert isinstance(pack, KokoroVoicePack)
    assert pack.name == "tuned"
    assert pack.engine == "inno-v0.2"
    assert pack.data.dtype == np.float32
    assert renderer.call is not None
    audio, config, language, enroller, options, name = renderer.call
    assert audio.original_sample_rate == 44_100
    assert audio.mono_audio.shape == source.shape
    np.testing.assert_allclose(audio.mono_audio, source * 0.75, atol=1.0e-7)
    assert config.model_variant == "v1.0"
    assert config.voice is None
    assert language == "en"
    assert enroller == "inno-v0.2"
    assert options.to_runtime_options() == {"fmax": 8_000.0}
    assert name == "tuned"


def test_inno_enrollment_needs_no_transcript_or_g2p(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_resolver(monkeypatch)
    renderer = FakeRenderer()
    synthesizer = KokoroSynthesizer(g2p=NoCallG2P(), renderer=renderer)

    result = synthesizer.enroll_voice(tone(), sample_rate=24_000)

    assert isinstance(result, KokoroVoicePack)
    assert renderer.call is not None


def test_inno_enrollment_rejects_transcript_and_wrong_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_resolver(monkeypatch)
    synthesizer = KokoroSynthesizer(g2p=NoCallG2P(), renderer=FakeRenderer())

    with pytest.raises(InvalidRequestError, match="must be omitted"):
        synthesizer.enroll_voice(tone(), "unneeded transcript", sample_rate=24_000)
    with pytest.raises(InvalidRequestError, match="InnoEnrollmentOptions"):
        synthesizer.enroll_voice(tone(), sample_rate=24_000, options={"fmax": 8000})  # type: ignore[arg-type]
    with pytest.raises(InvalidVoiceError, match="finite positive"):
        InnoEnrollmentOptions(fmax=float("nan"))


def test_inno_enrollment_rejects_enroller_that_requires_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_resolver(monkeypatch, transcript_required=True)
    synthesizer = KokoroSynthesizer(g2p=NoCallG2P(), renderer=FakeRenderer())

    with pytest.raises(InvalidRequestError, match="unexpectedly requires a transcript"):
        synthesizer.enroll_voice(tone(), sample_rate=24_000)


def test_inno_enrollment_reports_reference_duration_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    patch_resolver(monkeypatch)
    synthesizer = KokoroSynthesizer(g2p=NoCallG2P(), renderer=FakeRenderer())

    with pytest.raises(InvalidVoiceError, match="between 3 and 30 seconds"):
        synthesizer.enroll_voice(tone(duration=2.9), sample_rate=24_000)


def test_runtime_adapter_forwards_inno_enrollment_and_wraps_pack() -> None:
    calls: list[tuple[np.ndarray, dict[str, Any]]] = []

    class Runtime:
        def enroll_voice(self, audio: np.ndarray, **kwargs: Any) -> SimpleNamespace:
            calls.append((audio, kwargs))
            return SimpleNamespace(
                data=np.zeros((510, 1, 256), dtype=np.float16),
                metadata={"runtime_version": "0.2"},
            )

    resolved = SimpleNamespace(
        ref="kokoro:v1.0",
        storage_id="model-cache-id",
        model_paths=(),
        voices_path=None,
        config_path=None,
        model_artifacts={},
        metadata={},
        installation=None,
        sample_rate=24_000,
        model_id="v1.0",
        distribution=None,
    )
    runtime = Runtime()
    adapter = onnxvoice_boundary.KokoroRuntimeAdapter(runtime, resolved)
    audio = tone()

    pack = adapter.enroll_voice(
        audio,
        sample_rate=44_100,
        enroller="inno-v0.2",
        options={"fmax": 8_000.0},
        name="voice",
    )

    assert isinstance(pack, KokoroVoicePack)
    assert pack.data.dtype == np.float32
    assert pack.name == "voice"
    assert pack.engine == "inno-v0.2"
    assert pack.metadata["runtime_version"] == "0.2"
    assert calls == [
        (audio, {"sample_rate": 44_100, "enroller": "inno-v0.2", "options": {"fmax": 8_000.0}})
    ]


def test_runtime_adapter_requires_runtime_enrollment_support() -> None:
    resolved = SimpleNamespace(
        ref="kokoro:v1.0",
        storage_id=None,
        model_paths=(),
        voices_path=None,
        config_path=None,
        model_artifacts={},
        metadata={},
        installation=None,
        sample_rate=24_000,
        model_id="v1.0",
        distribution=None,
    )
    adapter = onnxvoice_boundary.KokoroRuntimeAdapter(SimpleNamespace(), resolved)

    with pytest.raises(ConfigurationError, match="does not support voice enrollment"):
        adapter.enroll_voice(tone(), sample_rate=24_000, enroller="inno-v0.2", options={})
