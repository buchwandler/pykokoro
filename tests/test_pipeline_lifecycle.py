from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from pykokoro.generation_config import GenerationConfig
from pykokoro.onnx_backend import Kokoro
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

        self.warmup_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def warmup(self) -> None:
        self.warmup_calls += 1

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
        self,
        phoneme_segments,
        voice_style,
        speed,
        voice_resolver,
        *,
        default_voice_name=None,
    ):
        _ = voice_style
        _ = speed
        _ = voice_resolver
        return phoneme_segments

    def postprocess_audio_segments(self, phoneme_segments, trim_silence, prosody_config=None):
        _ = trim_silence, prosody_config
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

    pipeline = KokoroPipeline(
        PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us")),
        g2p=DummyG2PAdapter(),
    )
    with pipeline as active:
        active.run("Hello")

    assert instances
    assert instances[0].close_calls == 1


def test_pipeline_warmup_reuses_backend(monkeypatch) -> None:
    instances = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)
    pipeline = KokoroPipeline(
        PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us")),
        g2p=DummyG2PAdapter(),
    )

    pipeline.warmup()
    pipeline.warmup()

    assert len(instances) == 1
    assert instances[0].warmup_calls == 2
    pipeline.close()


def test_pipeline_context_manager_closes_on_exception(monkeypatch):
    instances = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)

    with (
        pytest.raises(RuntimeError),
        KokoroPipeline(
            PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us")),
            g2p=DummyG2PAdapter(),
        ) as active,
    ):
        active.run("Hello")
        raise RuntimeError("boom")

    assert instances
    assert instances[0].close_calls == 1


def test_pipeline_does_not_close_external_kokoro():
    shared: Any = DummyKokoro()
    pipeline = KokoroPipeline(
        PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us")),
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

        def preprocess_segments(self, phoneme_segments, enable_short_sentence, random_seed=None):
            _ = enable_short_sentence
            self.random_seed = random_seed
            return phoneme_segments

    shared = SeedRecordingKokoro()
    pipeline = KokoroPipeline(
        PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us", random_seed=42)),
        g2p=DummyG2PAdapter(),
        phoneme_processing=OnnxPhonemeProcessorAdapter(shared),
        audio_generation=OnnxAudioGenerationAdapter(shared),
        audio_postprocessing=OnnxAudioPostprocessingAdapter(shared),
    )

    pipeline.run("Hello")

    assert shared.random_seed == 42


def test_kokoro_close_clears_all_resources_and_closes_database_once():
    class FakeDatabase:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    database = FakeDatabase()
    kokoro = object.__new__(Kokoro)
    kokoro._voice_db = database
    kokoro._audio_generator = object()
    kokoro._tokenizer = object()
    kokoro._voice_manager = object()
    kokoro._session = object()

    kokoro.close()
    kokoro.close()

    assert database.close_calls == 1
    assert kokoro._voice_db is None
    assert kokoro._audio_generator is None
    assert kokoro._tokenizer is None
    assert kokoro._voice_manager is None
    assert kokoro._session is None


def test_kokoro_close_is_safe_after_partial_initialization():
    kokoro = object.__new__(Kokoro)
    kokoro._session = object()

    kokoro.close()

    assert kokoro._voice_db is None
    assert kokoro._audio_generator is None
    assert kokoro._tokenizer is None
    assert kokoro._voice_manager is None
    assert kokoro._session is None


def test_pipeline_propagates_asset_progress_to_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    callbacks = []
    instances = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            callbacks.append(kwargs["asset_progress"])
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)

    def first(event):
        pass

    def second(event):
        pass

    pipeline = KokoroPipeline(
        PipelineConfig(
            voice="af",
            generation=GenerationConfig(lang="en-us"),
            asset_progress=first,
        ),
        g2p=DummyG2PAdapter(),
    )

    pipeline._ensure_kokoro(pipeline.config)
    pipeline._ensure_kokoro(
        PipelineConfig(
            voice="af",
            generation=GenerationConfig(lang="en-us"),
            asset_progress=second,
        )
    )

    assert callbacks == [first]
    assert instances[0]._asset_progress is second


def test_owned_default_stages_rebind_when_model_variant_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.model_variant = kwargs["model_variant"]
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)
    first_cfg = PipelineConfig(
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
        voice="af_alloy",
        generation=GenerationConfig(lang="en-us"),
    )
    pipeline = KokoroPipeline(first_cfg, g2p=DummyG2PAdapter())

    first_phoneme, first_audio, first_post = pipeline._resolve_stages(first_cfg)
    second_cfg = replace(
        first_cfg,
        model_variant="v1.1-zh",
        voice="af_maple",
    )
    second_phoneme, second_audio, second_post = pipeline._resolve_stages(second_cfg)

    assert len(instances) == 2
    assert instances[0].model_variant == "v1.0"
    assert instances[1].model_variant == "v1.1-zh"
    assert instances[0].close_calls == 1
    assert first_phoneme is not second_phoneme
    assert first_audio is not second_audio
    assert first_post is not second_post
    assert second_phoneme._kokoro is instances[1]
    assert second_audio._kokoro is instances[1]
    assert second_post._kokoro is instances[1]


def test_owned_default_stages_reuse_for_voice_and_language_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)
    first_cfg = PipelineConfig(
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
        voice="af_alloy",
        generation=GenerationConfig(lang="en-us"),
    )
    pipeline = KokoroPipeline(first_cfg, g2p=DummyG2PAdapter())

    first = pipeline._resolve_stages(first_cfg)
    second = pipeline._resolve_stages(replace(first_cfg, voice="af_sky"))
    third = pipeline._resolve_stages(
        replace(first_cfg, generation=replace(first_cfg.generation, lang="zh"))
    )

    assert len(instances) == 1
    assert second == first
    assert third == first


def test_custom_stages_control_backend_creation_and_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instances: list[Any] = []

    class TrackingKokoro(DummyKokoro):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", TrackingKokoro)
    first_cfg = PipelineConfig(
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
        voice="af_alloy",
        generation=GenerationConfig(lang="en-us"),
    )
    custom_stages = (object(), object(), object())
    fully_custom = KokoroPipeline(
        first_cfg,
        phoneme_processing=custom_stages[0],
        audio_generation=custom_stages[1],
        audio_postprocessing=custom_stages[2],
    )

    assert fully_custom._resolve_stages(first_cfg) == custom_stages
    assert (
        fully_custom._resolve_stages(replace(first_cfg, model_variant="v1.1-zh")) == custom_stages
    )
    assert instances == []

    custom_audio = object()
    partially_custom = KokoroPipeline(
        first_cfg,
        g2p=DummyG2PAdapter(),
        audio_generation=custom_audio,
    )
    first = partially_custom._resolve_stages(first_cfg)
    second = partially_custom._resolve_stages(
        replace(first_cfg, model_variant="v1.1-zh", voice="af_maple")
    )

    assert len(instances) == 2
    assert first[1] is custom_audio
    assert second[1] is custom_audio
    assert first[0] is not second[0]
    assert first[2] is not second[2]


def test_failed_backend_replacement_keeps_previous_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    instances: list[DummyKokoro] = []

    class FailingKokoro(DummyKokoro):
        def __init__(self, *args: object, **kwargs: object) -> None:
            if kwargs["model_variant"] == "v1.1-zh":
                raise RuntimeError("replacement failed")
            super().__init__(*args, **kwargs)
            instances.append(self)

    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", FailingKokoro)
    first_cfg = PipelineConfig(
        model_source="github",
        model_variant="v1.0",
        model_quality="fp32",
        voice="af_alloy",
        generation=GenerationConfig(lang="en-us"),
    )
    pipeline = KokoroPipeline(first_cfg, g2p=DummyG2PAdapter())
    stages = pipeline._resolve_stages(first_cfg)

    with pytest.raises(RuntimeError, match="replacement failed"):
        pipeline._resolve_stages(replace(first_cfg, model_variant="v1.1-zh", voice="af_maple"))

    assert len(instances) == 1
    assert instances[0].close_calls == 0
    assert pipeline._kokoro is instances[0]
    assert stages[0]._kokoro is instances[0]
    assert stages[1]._kokoro is instances[0]
    assert stages[2]._kokoro is instances[0]
