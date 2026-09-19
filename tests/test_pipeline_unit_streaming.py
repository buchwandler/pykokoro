from __future__ import annotations

import pytest


from dataclasses import replace

import numpy as np
from audiocompose import Composer
from audiosig import measure_loudness

from pykokoro import (
    AudioUnitDescriptor,
    AudioUnitResult,
    KokoroPipeline,
    LoudnessConfig,
    PipelineConfig,
    PreparedAudioUnits,
)
from pykokoro.exceptions import ConfigurationError
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import _unit_text_hash
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.types import PhonemeSegment, WordTiming


class CountingG2P:
    def __init__(self) -> None:
        self.calls = 0

    def phonemize(self, segments, doc, cfg, trace):
        self.calls += 1
        return [
            PhonemeSegment(
                id=f"{segment.id}_ph0",
                segment_id=segment.id,
                phoneme_id=0,
                text=segment.text,
                phonemes="a",
                tokens=[],
                lang=cfg.generation.lang,
                char_start=segment.char_start,
                char_end=segment.char_end,
                paragraph_idx=segment.paragraph_idx,
                sentence_idx=segment.sentence_idx,
                clause_idx=segment.clause_idx,
            )
            for segment in segments
        ]


class TwoSegmentG2P:
    def phonemize(self, segments, doc, cfg, trace):
        _ = doc, trace
        source = segments[0]
        return [
            PhonemeSegment(
                id=f"{source.id}-a",
                segment_id=source.id,
                phoneme_id=0,
                text="first",
                phonemes="a",
                tokens=[],
                lang=cfg.generation.lang,
                char_start=source.char_start,
                char_end=source.char_start + 5,
                pause_after=10 / 24000,
            ),
            PhonemeSegment(
                id=f"{source.id}-b",
                segment_id=source.id,
                phoneme_id=1,
                text="second",
                phonemes="a",
                tokens=[],
                lang=cfg.generation.lang,
                char_start=source.char_start + 6,
                char_end=source.char_end,
            ),
        ]


class TimedGenerator:
    def generate(self, phoneme_segments, cfg, trace):
        _ = cfg, trace
        for index, segment in enumerate(phoneme_segments):
            length = 100 + index * 20
            segment.raw_audio = np.ones(length, dtype=np.float32)
            segment.word_timings = [
                WordTiming(
                    segment.text,
                    segment.char_start,
                    segment.char_end,
                    10,
                    length - 20,
                    segment.id,
                )
            ]
        return phoneme_segments


class CountingProcessor:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, phoneme_segments, cfg, trace):
        self.calls += 1
        return phoneme_segments


class CountingGenerator(NoopAudioGenerationAdapter):
    def __init__(self) -> None:
        super().__init__(seconds_per_segment=0.001)
        self.calls = 0

    def generate(self, phoneme_segments, cfg, trace):
        self.calls += 1
        return super().generate(phoneme_segments, cfg, trace)


class ConstantGenerator(CountingGenerator):
    def __init__(self) -> None:
        super().__init__()
        self.seconds_per_segment = 0.2

    def generate(self, phoneme_segments, cfg, trace):
        generated = super().generate(phoneme_segments, cfg, trace)
        for segment in generated:
            if segment.raw_audio is not None:
                segment.raw_audio.fill(0.1)
        return generated


def build_pipeline(
    *,
    g2p: CountingG2P | None = None,
    processor: CountingProcessor | None = None,
    generator: CountingGenerator | None = None,
    config: PipelineConfig | None = None,
) -> KokoroPipeline:
    return KokoroPipeline(
        config or PipelineConfig(generation=GenerationConfig(lang="en-us")),
        g2p=g2p or CountingG2P(),
        phoneme_processing=processor or CountingProcessor(),
        audio_generation=generator or CountingGenerator(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )


def test_public_unit_types_and_exports() -> None:
    assert AudioUnitDescriptor.__name__ == "AudioUnitDescriptor"
    assert AudioUnitResult.__name__ == "AudioUnitResult"
    assert PreparedAudioUnits.__name__ == "PreparedAudioUnits"


def test_descriptors_are_ordered_and_have_clean_text_ranges() -> None:
    pipeline = build_pipeline()
    text = "First paragraph.\n\nSecond paragraph."

    with pipeline.prepare_units(text) as prepared:
        assert [unit.index for unit in prepared.units] == [0, 1]
        assert [unit.paragraph_idx for unit in prepared.units] == [0, 1]
        for unit in prepared.units:
            assert unit.text
            assert unit.text == " ".join(text[unit.char_start : unit.char_end].split())
            assert unit.segment_ids
            assert unit.phoneme_segment_ids
            assert len(unit.text_hash) == 64


def test_results_expose_prepared_and_source_text_coordinates() -> None:
    pipeline = build_pipeline()
    result = pipeline.run("Dr. Smith arrived.")

    assert result.source_text == "Dr. Smith arrived."
    assert result.clean_text != result.source_text
    for segment in result.segments:
        assert segment.text == result.clean_text[segment.char_start : segment.char_end]


def test_prepared_units_retain_text_metadata_after_close() -> None:
    pipeline = build_pipeline()
    prepared = pipeline.prepare_units("Hello world.")

    assert prepared.clean_text == "Hello world."
    assert prepared.source_text == "Hello world."
    prepared.close()
    assert prepared.clean_text == "Hello world."
    assert prepared.source_text == "Hello world."


def test_global_stages_run_once_and_generation_is_deferred() -> None:
    g2p = CountingG2P()
    processor = CountingProcessor()
    generator = CountingGenerator()
    pipeline = build_pipeline(g2p=g2p, processor=processor, generator=generator)

    with pipeline.prepare_units("One.\n\nTwo.") as prepared:
        assert len(prepared.units) == 2
        assert (g2p.calls, processor.calls, generator.calls) == (1, 1, 0)
        first = next(prepared.render())
        assert generator.calls == 1
        first.release_audio()
    assert generator.calls == 1


def test_streamed_audio_matches_run_for_noop_stages() -> None:
    pipeline = build_pipeline()
    text = "First.\n\nSecond."
    legacy = pipeline.run(text)
    pieces: list[np.ndarray] = []
    with pipeline.prepare_units(text) as prepared:
        for result in prepared.render():
            pieces.append(result.audio.copy())
            result.release_audio()
    np.testing.assert_array_equal(legacy.audio, np.concatenate(pieces))


def test_pipeline_timing_contract_without_network() -> None:
    pipeline = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        g2p=TwoSegmentG2P(),
        phoneme_processing=CountingProcessor(),
        audio_generation=TimedGenerator(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )

    job = pipeline.to_audio_job("First second")
    _ = Composer().compose(job)
    for item in job.items:
        if not hasattr(item, "source") or not item.spans:
            continue
        waveform = item.source.load()[0]
        for span in item.spans:
            assert 0 <= span.sample_start <= span.sample_end <= len(waveform)

    result = pipeline.run("First second")
    for segment in result.phoneme_segments:
        assert segment.processed_audio is not None
        for timing in segment.word_timings:
            assert 0 <= timing.start_sample <= timing.end_sample <= len(segment.processed_audio)

    starts = [timing.start_sample for timing in result.word_timings]
    assert starts == sorted(starts)


def test_skip_indices_do_not_generate_and_order_is_descriptor_order() -> None:
    generator = CountingGenerator()
    pipeline = build_pipeline(generator=generator)
    with pipeline.prepare_units("One.\n\nTwo.\n\nThree.") as prepared:
        results = list(prepared.render(skip_indices={1}))
        assert [result.descriptor.index for result in results] == [0, 2]
        assert generator.calls == 2


def test_invalid_and_duplicate_indices_fail_before_generation() -> None:
    generator = CountingGenerator()
    pipeline = build_pipeline(generator=generator)
    with pipeline.prepare_units("One.\n\nTwo.") as prepared:
        with pytest.raises(IndexError):
            prepared.render(indices=[2])
        with pytest.raises(ValueError):
            prepared.render(indices=[0, 0])
        assert generator.calls == 0


def test_render_is_one_pass_and_concurrent_use_is_rejected() -> None:
    pipeline = build_pipeline()
    with pipeline.prepare_units("One.\n\nTwo.") as prepared:
        iterator = prepared.render()
        first = next(iterator)
        with pytest.raises(RuntimeError, match="one render pass"):
            prepared.render()
        iterator.close()
        first.release_audio()


def test_iterator_close_and_prepared_close_release_arrays() -> None:
    pipeline = build_pipeline()
    prepared = pipeline.prepare_units("One.\n\nTwo.")
    iterator = prepared.render()
    result = next(iterator)
    external = result.audio
    iterator.close()
    assert result.audio.size == 0
    assert external.size > 0
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.render()


def test_pipeline_close_closes_prepared_objects() -> None:
    pipeline = build_pipeline()
    prepared = pipeline.prepare_units("One.")
    pipeline.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.render()


class MarkerG2P(CountingG2P):
    def phonemize(self, segments, doc, cfg, trace):
        self.calls += 1
        return [
            PhonemeSegment(
                id=f"{segment.id}_ph0",
                segment_id=segment.id,
                phoneme_id=0,
                text=segment.text,
                phonemes="a",
                tokens=[],
                char_start=segment.char_start,
                char_end=segment.char_end,
                paragraph_idx=segment.paragraph_idx,
            )
            for segment in segments
        ]


def test_markers_have_local_offsets_and_aggregate_offsets() -> None:
    pipeline = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        g2p=MarkerG2P(),
        phoneme_processing=CountingProcessor(),
        audio_generation=CountingGenerator(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )
    text = "@before\nOne.\n@after_one\n\n@before_two\nTwo."
    with pipeline.prepare_units(text) as prepared:
        results = []
        for result in prepared.render():
            results.append((result.descriptor, result.markers, len(result.audio)))
            result.release_audio()
    assert results[0][0].marker_names == ("before",)
    assert results[1][0].marker_names == ("after_one", "before_two")
    assert all("unit_index" in marker for _, markers, _ in results for marker in markers)

    legacy = pipeline.run(text)
    assert [marker["name"] for marker in legacy.markers] == [
        "before",
        "after_one",
        "before_two",
    ]
    assert legacy.markers[2]["sample_offset"] >= results[0][2]


class FailingGenerator(CountingGenerator):
    def generate(self, phoneme_segments, cfg, trace):
        super().generate(phoneme_segments, cfg, trace)
        raise RuntimeError("unit generation failed")


def test_generation_failure_clears_partial_arrays() -> None:
    generator = FailingGenerator()
    pipeline = build_pipeline(generator=generator)
    prepared = pipeline.prepare_units("One.")
    with pytest.raises(RuntimeError, match="unit generation failed"):
        next(prepared.render())
    prepared.close()


def test_effective_configuration_is_snapshot() -> None:
    generation = {"lang": "en-us"}
    pipeline = build_pipeline()
    with pipeline.prepare_units("One.", generation=generation) as prepared:
        generation["lang"] = "de"
        assert prepared.units[0].text_hash


def test_unit_hash_ignores_non_audio_runtime_options() -> None:
    segment = PhonemeSegment(
        id="s0-ph0",
        segment_id="s0",
        phoneme_id=0,
        text="One.",
        phonemes="a",
        tokens=[],
    )
    base = PipelineConfig()

    def identity(cfg: PipelineConfig) -> str:
        return _unit_text_hash(0, 0, 4, "One.", [segment], cfg, ())

    assert identity(base) == identity(replace(base, return_trace=True))
    assert identity(base) == identity(replace(base, retain_segment_audio=False))
    assert identity(base) == identity(replace(base, cache_dir="/machine/cache"))
    assert identity(base) == identity(replace(base, enable_deprecation_warnings=True))


def test_unit_hash_changes_for_audio_semantic_options() -> None:
    segment = PhonemeSegment(
        id="s0-ph0",
        segment_id="s0",
        phoneme_id=0,
        text="One.",
        phonemes="a",
        tokens=[],
    )
    base = PipelineConfig()

    def identity(cfg: PipelineConfig) -> str:
        return _unit_text_hash(0, 0, 4, "One.", [segment], cfg, ())

    assert identity(base) != identity(replace(base, voice="af_sarah"))
    assert identity(base) != identity(replace(base, generation=replace(base.generation, speed=1.1)))
    assert identity(base) != identity(replace(base, model_identity="model-v2"))


def test_empty_input_has_no_units_and_fallback_document_is_renderable() -> None:
    pipeline = build_pipeline()
    with pipeline.prepare_units("") as prepared:
        assert prepared.units == ()
        assert list(prepared.render()) == []


def test_sentence_units_are_ordered_and_lazily_rendered() -> None:
    g2p = CountingG2P()
    processor = CountingProcessor()
    generator = CountingGenerator()
    pipeline = build_pipeline(g2p=g2p, processor=processor, generator=generator)
    text = "First sentence. Second sentence.\n\nThird paragraph."

    with pipeline.prepare_units(text, unit="sentence") as prepared:
        assert prepared.unit_kind == "sentence"
        assert [(unit.paragraph_idx, unit.sentence_idx) for unit in prepared.units] == [
            (0, 0),
            (0, 1),
            (1, 0),
        ]
        assert [unit.unit_kind for unit in prepared.units] == ["sentence"] * 3
        assert [unit.text for unit in prepared.units] == [
            "First sentence.",
            "Second sentence.",
            "Third paragraph.",
        ]
        assert [unit.text for unit in prepared.units] == [
            text[unit.char_start : unit.char_end] for unit in prepared.units
        ]
        assert (g2p.calls, processor.calls, generator.calls) == (1, 1, 0)

        results = list(prepared.render(skip_indices={1}))
        assert [result.descriptor.index for result in results] == [0, 2]
        assert generator.calls == 2


def test_sentence_units_match_paragraph_audio_for_noop_stages() -> None:
    pipeline = build_pipeline()
    text = "First sentence. Second sentence. Third sentence."
    legacy = pipeline.run(text)

    pieces: list[np.ndarray] = []
    with pipeline.prepare_units(text, unit="sentence") as prepared:
        for result in prepared.render():
            pieces.append(result.audio.copy())
            result.release_audio()

    np.testing.assert_array_equal(legacy.audio, np.concatenate(pieces))


def test_sentence_unit_rejects_unsupported_kinds() -> None:
    pipeline = build_pipeline()
    with pytest.raises(ValueError, match="Unsupported audio unit kind"):
        pipeline.prepare_units("One.", unit="clause")  # type: ignore[arg-type]


def test_play_streaming_uses_sentence_units_and_forwards_playback_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int, int | str | None]] = []

    def fake_play(prepared, *, device, queue_size):
        calls.append((prepared.unit_kind, queue_size, device))

    monkeypatch.setattr("pykokoro.playback.play_prepared_units", fake_play)
    pipeline = build_pipeline()
    pipeline.play_streaming("One. Two.", device="test-device", queue_size=3)

    assert calls == [("sentence", 3, "test-device")]


def test_target_lufs_rejects_isolated_and_streaming_units() -> None:
    pipeline = build_pipeline(
        config=PipelineConfig(
            generation=GenerationConfig(lang="en-us"),
            loudness=LoudnessConfig(target_lufs=-18.0),
        )
    )
    with pytest.raises(ConfigurationError, match="complete paragraph result"):
        pipeline.prepare_units("One. Two.", unit="sentence")
    with pytest.raises(ConfigurationError, match="complete waveform"):
        pipeline.play_streaming("One. Two.")


def test_run_applies_complete_target_after_unit_concatenation() -> None:
    pipeline = build_pipeline(
        generator=ConstantGenerator(),
        config=PipelineConfig(
            generation=GenerationConfig(lang="en-us"),
            loudness=LoudnessConfig(target_lufs=-42.0),
            return_trace=True,
        ),
    )
    result = pipeline.run("One. Two.")
    metrics = measure_loudness(result.audio, sample_rate=result.sample_rate)
    assert metrics.integrated_lufs == pytest.approx(-42.0, abs=0.1)
    assert result.trace is not None
    composition_events = [event for event in result.trace.events if event.stage == "composition"]
    assert composition_events
    assert composition_events[0].details["applied_loudness_gain_db"] != 0.0
