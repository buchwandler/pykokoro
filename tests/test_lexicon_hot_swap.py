from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.tokenizer import TokenizerConfig
from pykokoro.types import PhonemeSegment, Trace


def test_acoustic_key_ignores_lexicon_but_tracks_runtime_settings() -> None:
    base = PipelineConfig(
        generation=GenerationConfig(lang="en-us"),
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
        model_path=Path("model.onnx"),
        voices_path=Path("voices.bin"),
        model_config_path=Path("config.json"),
        provider="cpu",
        provider_options={"device": "CPU"},
        session_options={"threads": 1},
        tokenizer_config=TokenizerConfig(lexicons=("gold",)),
    )
    pipeline = KokoroPipeline(base)

    crane = replace(
        base,
        tokenizer_config=replace(base.tokenizer_config, lexicons=("crane",)),
    )
    installed_only = replace(
        base,
        tokenizer_config=replace(
            base.tokenizer_config, lexicon_data_policy="installed-only"
        ),
    )
    assert pipeline._kokoro_key(base) == pipeline._kokoro_key(installed_only)
    assert pipeline._kokoro_key(base) == pipeline._kokoro_key(crane)

    for changed in (
        replace(base, model_quality="fp16"),
        replace(base, model_source="huggingface"),
        replace(base, model_variant="v1.1-zh"),
        replace(base, model_path=Path("other.onnx")),
        replace(base, voices_path=Path("other.bin")),
        replace(base, model_config_path=Path("other.json")),
        replace(base, provider="cuda"),
        replace(base, provider_options={"device": "GPU"}),
        replace(base, session_options={"threads": 2}),
        replace(base, waveform_validation="strict"),
    ):
        assert pipeline._kokoro_key(base) != pipeline._kokoro_key(changed)


def test_four_lexicons_reuse_one_backend_and_close_once(monkeypatch) -> None:
    instances: list[Any] = []

    class FakeKokoro:
        def __init__(self, **kwargs: object) -> None:
            _ = kwargs
            self.close_calls = 0
            instances.append(self)

        def close(self) -> None:
            self.close_calls += 1

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", FakeKokoro)
    base = PipelineConfig(generation=GenerationConfig(lang="de"))
    pipeline = KokoroPipeline(base)

    for lexicon in ("gold", "crane", "espeak", "olaph"):
        cfg = replace(
            base,
            tokenizer_config=TokenizerConfig(lexicons=(lexicon,)),
        )
        backend, changed = pipeline._ensure_kokoro(cfg)
        assert changed is (lexicon == "gold")
        assert backend is instances[0]

    assert len(instances) == 1
    pipeline.close()
    assert instances[0].close_calls == 1


def test_acoustic_generation_receives_each_request_phoneme_output() -> None:
    class FakeKokoro:
        def __init__(self) -> None:
            self.received: list[tuple[int, list[str]]] = []

        def resolve_voice_style(self, voice: str) -> np.ndarray:
            _ = voice
            return np.zeros((1, 1), dtype=np.float32)

        def generate_raw_audio_segments(
            self,
            segments: list[PhonemeSegment],
            voice_style: np.ndarray,
            speed: float,
            voice_resolver: Any,
            *,
            default_voice_name: str | None = None,
        ) -> list[PhonemeSegment]:
            _ = voice_style, speed, voice_resolver, default_voice_name
            self.received.append((id(self), [segment.phonemes for segment in segments]))
            return segments

    backend = FakeKokoro()
    adapter = OnnxAudioGenerationAdapter(backend)
    config = PipelineConfig(voice="martin", generation=GenerationConfig(lang="de"))

    def segment(phonemes: str) -> PhonemeSegment:
        return PhonemeSegment("id", "source", 0, "Haus", phonemes, [], lang="de")

    adapter.generate([segment("gold-output")], config, Trace())
    adapter.generate([segment("crane-output")], config, Trace())

    assert backend.received == [
        (id(backend), ["gold-output"]),
        (id(backend), ["crane-output"]),
    ]
