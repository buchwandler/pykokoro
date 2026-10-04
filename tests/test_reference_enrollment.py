from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from pykokoro.exceptions import EmptyTextError, InvalidLanguageError
from pykokoro.prepared_g2p import PreparedSynthesis
from pykokoro.reference_audio import PreparedReferenceAudio, prepare_reference_audio
from pykokoro.reference_voice import ReferenceVoice
from pykokoro.request_renderer import OnnxRequestRenderer
from pykokoro.synthesis_config import SynthesisConfig
from pykokoro.synthesis_types import SynthesisSegment
from pykokoro.synthesizer import KokoroSynthesizer


def tone(sample_rate: int = 24_000, duration: float = 3.5) -> np.ndarray:
    times = np.arange(round(sample_rate * duration), dtype=np.float64) / sample_rate
    return (0.15 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)


class FakeG2P:
    def __init__(self) -> None:
        self.requests: list[SynthesisSegment] = []
        self.configs: list[SynthesisConfig] = []

    def phonemize(self, segment: SynthesisSegment, config: SynthesisConfig) -> PreparedSynthesis:
        self.requests.append(segment)
        self.configs.append(config)
        return PreparedSynthesis(
            request_id=segment.id,
            text=segment.text,
            language=segment.language,
            voice=None,
            phonemes="həˈloʊ",
            token_ids=(1, 2, 3),
        )


class FakeRenderer:
    def __init__(self) -> None:
        self.call: tuple[Any, ...] | None = None

    def enroll_voice(
        self,
        prepared: PreparedSynthesis,
        audio: PreparedReferenceAudio,
        config: SynthesisConfig,
        *,
        name: str | None,
        reference_text_sha256: str,
    ) -> ReferenceVoice:
        self.call = (prepared, audio, config, name, reference_text_sha256)
        return ReferenceVoice(
            style=np.zeros((1, 256), dtype=np.float32),
            memory=np.ones((1, 2, 192), dtype=np.float32),
            memory_mask=np.ones((1, 2), dtype=np.bool_),
            model_id=config.model_variant or "en-akinvox-cloning-v1",
            model_fingerprint="runtime-build-fingerprint",
            name=name,
            metadata={
                "reference_audio_sha256": audio.audio_sha256,
                "reference_text_sha256": reference_text_sha256,
            },
        )


def test_enroll_voice_uses_exact_transcript_and_selects_cloning_profile() -> None:
    g2p = FakeG2P()
    renderer = FakeRenderer()
    original_config = SynthesisConfig(voice="af_heart")
    synthesizer = KokoroSynthesizer(original_config, g2p=g2p, renderer=renderer)
    transcript = 'Dr. Smith said, "Don\'t leave yet, um."'

    voice = synthesizer.enroll_voice(
        tone(), transcript, sample_rate=24_000, language="en-US", name="speaker"
    )

    request = g2p.requests[0]
    prepared, audio, config, name, text_hash = renderer.call or (None, None, None, None, None)
    assert request.text == transcript
    assert request.language == "en-us"
    assert prepared.token_ids == (1, 2, 3)
    assert audio.audio_24k.shape == (84_000,)
    assert config.model_variant == "en-akinvox-cloning-v1"
    assert config.voice is None
    assert name == "speaker"
    assert text_hash == hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    assert voice.model_id == "en-akinvox-cloning-v1"
    assert voice.metadata["reference_text_sha256"] == text_hash
    assert "reference_text" not in voice.metadata
    assert synthesizer.config is original_config


def test_enroll_voice_reads_wav_without_normalizing_transcript(tmp_path: Path) -> None:
    path = tmp_path / "reference.wav"
    sf.write(path, tone(), 24_000, subtype="FLOAT")
    g2p = FakeG2P()
    renderer = FakeRenderer()
    synthesizer = KokoroSynthesizer(g2p=g2p, renderer=renderer)
    transcript = "  Hello, exactly as spoken.  "

    voice = synthesizer.enroll_voice(path, transcript)

    assert g2p.requests[0].text == transcript
    assert (
        voice.metadata["reference_text_sha256"]
        == hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    )
    assert renderer.call is not None
    assert renderer.call[1].original_sample_rate == 24_000


def test_enroll_voice_rejects_empty_transcript_and_non_english() -> None:
    synthesizer = KokoroSynthesizer(g2p=FakeG2P(), renderer=FakeRenderer())

    with pytest.raises(EmptyTextError, match="reference_text"):
        synthesizer.enroll_voice(tone(), "  ", sample_rate=24_000)
    with pytest.raises(InvalidLanguageError, match="English only"):
        synthesizer.enroll_voice(tone(), "Hello.", sample_rate=24_000, language="de")


def test_onnx_request_renderer_bridges_prepared_reference_to_backend() -> None:
    g2p = FakeG2P()
    audio = prepare_reference_audio(tone(), sample_rate=24_000)
    config = SynthesisConfig(model_variant="en-akinvox-cloning-v1")
    request = SynthesisSegment("reference", "Hello there.", "en-us")
    prepared = g2p.phonemize(request, config)

    class Backend:
        def __init__(self) -> None:
            self.call: tuple[object, ...] | None = None

        def prepare_reference_voice(self, token_ids: object, **kwargs: object) -> ReferenceVoice:
            self.call = (token_ids, kwargs)
            return ReferenceVoice(
                style=np.zeros((1, 256), dtype=np.float32),
                memory=np.ones((1, 2, 192), dtype=np.float32),
                memory_mask=np.ones((1, 2), dtype=np.bool_),
                model_id="en-akinvox-cloning-v1",
                model_fingerprint="model-build",
                name=kwargs["name"],
                metadata=kwargs["metadata"],
            )

    backend = Backend()
    renderer = OnnxRequestRenderer(g2p, backend_factory=lambda _: backend)
    text_hash = hashlib.sha256(request.text.encode("utf-8")).hexdigest()

    voice = renderer.enroll_voice(
        prepared,
        audio,
        config,
        name="speaker",
        reference_text_sha256=text_hash,
    )

    assert backend.call is not None
    token_ids, arguments = backend.call
    assert token_ids == (1, 2, 3)
    assert arguments["audio_24k"] is audio.audio_24k
    assert arguments["audio_16k"] is audio.audio_16k
    assert arguments["name"] == "speaker"
    assert arguments["metadata"] == {
        "reference_audio_sha256": audio.audio_sha256,
        "reference_text_sha256": text_hash,
    }
    assert voice.name == "speaker"
