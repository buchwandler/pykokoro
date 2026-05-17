from typing import Any

import numpy as np
import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.audio_generation.onnx import OnnxAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.onnx import OnnxAudioPostprocessingAdapter
from pykokoro.stages.phoneme_processing.onnx import OnnxPhonemeProcessorAdapter
from pykokoro.types import PhonemeSegment


class DummyG2PAdapter:
    def phonemize(self, segments, doc, cfg, trace):
        _ = doc
        _ = cfg
        _ = trace
        out = []
        for segment in segments:
            out.append(
                PhonemeSegment(
                    id=f"{segment.id}_ph0",
                    segment_id=segment.id,
                    phoneme_id=0,
                    text=segment.text,
                    phonemes="test",
                    tokens=[1],
                    lang=cfg.generation.lang,
                    char_start=segment.char_start,
                    char_end=segment.char_end,
                    paragraph_idx=segment.paragraph_idx,
                    sentence_idx=segment.sentence_idx,
                    clause_idx=segment.clause_idx,
                )
            )
        return out


class DummyKokoro:
    def __init__(self, *args, **kwargs) -> None:
        _ = args
        _ = kwargs
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def preprocess_segments(self, phoneme_segments, enable_short_sentence):
        _ = enable_short_sentence
        return phoneme_segments

    def resolve_voice_style(self, voice):
        _ = voice
        return np.zeros((1, 1), dtype=np.float32)

    def get_voice_style(self, voice_name: str):
        _ = voice_name
        return np.zeros((1, 1), dtype=np.float32)

    def generate_raw_audio_segments(
        self, phoneme_segments, voice_style, speed, voice_resolver
    ):
        _ = voice_style
        _ = speed
        _ = voice_resolver
        return phoneme_segments

    def postprocess_audio_segments(self, phoneme_segments, trim_silence):
        _ = trim_silence
        return phoneme_segments

    def concatenate_audio_segments(self, processed):
        _ = processed
        return np.zeros(1, dtype=np.float32)


def test_pipeline_context_manager_closes_kokoro(monkeypatch):
    instances = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)

    pipeline = KokoroPipeline(PipelineConfig(voice="af"), g2p=DummyG2PAdapter())
    with pipeline as active:
        active.run("Hello")

    assert instances
    assert instances[0].close_calls == 1


def test_pipeline_context_manager_closes_on_exception(monkeypatch):
    instances = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)

    with pytest.raises(RuntimeError):
        with KokoroPipeline(
            PipelineConfig(voice="af"), g2p=DummyG2PAdapter()
        ) as active:
            active.run("Hello")
            raise RuntimeError("boom")

    assert instances
    assert instances[0].close_calls == 1


def test_pipeline_does_not_close_external_kokoro():
    shared: Any = DummyKokoro()
    pipeline = KokoroPipeline(
        PipelineConfig(voice="af"),
        g2p=DummyG2PAdapter(),
        phoneme_processing=OnnxPhonemeProcessorAdapter(shared),
        audio_generation=OnnxAudioGenerationAdapter(shared),
        audio_postprocessing=OnnxAudioPostprocessingAdapter(shared),
    )
    pipeline.run("Hello")
    pipeline.close()

    assert shared.close_calls == 0


def test_phoneme_processor_passes_random_seed_to_short_sentence_preprocess():
    class SeedRecordingKokoro(DummyKokoro):
        def __init__(self) -> None:
            super().__init__()
            self.random_seed = None

        def preprocess_segments(
            self, phoneme_segments, enable_short_sentence, random_seed=None
        ):
            _ = enable_short_sentence
            self.random_seed = random_seed
            return phoneme_segments

    shared = SeedRecordingKokoro()
    pipeline = KokoroPipeline(
        PipelineConfig(voice="af", generation=GenerationConfig(random_seed=42)),
        g2p=DummyG2PAdapter(),
        phoneme_processing=OnnxPhonemeProcessorAdapter(shared),
        audio_generation=OnnxAudioGenerationAdapter(shared),
        audio_postprocessing=OnnxAudioPostprocessingAdapter(shared),
    )

    pipeline.run("Hello")

    assert shared.random_seed == 42
