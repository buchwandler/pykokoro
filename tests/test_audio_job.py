from __future__ import annotations

import numpy as np
from audiocompose import AudioJob, Composer

from pykokoro.audio_generator import _crop_word_timings, _scale_word_timings
from pykokoro.audio_job import rendered_segments_to_audio_job
from pykokoro.types import BoundaryEvent, PhonemeSegment, WordTiming


def test_rendered_segments_create_quantized_pauses_markers_and_spans() -> None:
    segment = PhonemeSegment(
        id="seg-1",
        segment_id="seg-1",
        phoneme_id=0,
        text="hello",
        phonemes="həloʊ",
        tokens=[1],
        char_start=0,
        char_end=5,
        pause_before=0.1,
        pause_after=0.2,
        processed_audio=np.ones(3, dtype=np.float32),
        word_timings=[WordTiming("hello", 0, 5, 0, 3, "seg-1")],
    )
    job = rendered_segments_to_audio_job(
        [segment],
        boundaries=[BoundaryEvent(0, "marker", attrs={"marker": "start"})],
    )

    assert isinstance(job, AudioJob)
    assert [type(item).__name__ for item in job.items] == ["Silence", "AudioClip", "Silence"]
    composed = Composer().compose(job)
    assert composed.audio.size == 3 + round(0.1 * 24000) + round(0.2 * 24000)
    assert composed.markers[0].name == "start"
    assert composed.spans[0].id == "word:seg-1:0"

def test_multi_segment_spans_remain_clip_local_and_compose_globally() -> None:
    first = PhonemeSegment(
        id="a",
        segment_id="a",
        phoneme_id=0,
        text="a",
        phonemes="a",
        tokens=[1],
        processed_audio=np.zeros(100, dtype=np.float32),
        pause_after=10 / 24000,
        word_timings=[WordTiming("a", 0, 1, 10, 80, "a")],
    )
    second = PhonemeSegment(
        id="b",
        segment_id="b",
        phoneme_id=0,
        text="b",
        phonemes="b",
        tokens=[1],
        processed_audio=np.zeros(120, dtype=np.float32),
        word_timings=[WordTiming("b", 0, 1, 20, 100, "b")],
    )

    job = rendered_segments_to_audio_job([first, second])

    first_clip = job.items[0]
    second_clip = job.items[2]
    assert first_clip.spans[0].sample_start == 10
    assert first_clip.spans[0].sample_end == 80
    assert second_clip.spans[0].sample_start == 20
    assert second_clip.spans[0].sample_end == 100
    composed = Composer().compose(job)
    assert composed.spans[1].sample_start == 130
    assert composed.spans[1].sample_end == 210


def test_invalid_segment_word_timing_fails_before_composition() -> None:
    segment = PhonemeSegment(
        id="invalid",
        segment_id="invalid",
        phoneme_id=0,
        text="bad",
        phonemes="bad",
        tokens=[1],
        processed_audio=np.zeros(10, dtype=np.float32),
        word_timings=[WordTiming("bad", 0, 3, 5, 11, "invalid")],
    )

    try:
        rendered_segments_to_audio_job([segment])
    except ValueError as exc:
        assert "segment='invalid'" in str(exc)
        assert "range=5:11" in str(exc)
        assert "waveform_length=10" in str(exc)
    else:
        raise AssertionError("invalid local timing was accepted")

def test_trim_and_prosody_timing_transforms_remain_composable() -> None:
    cropped = _crop_word_timings(
        [WordTiming("speech", 0, 6, 10, 90, "segment")],
        5,
        95,
    )
    scaled = _scale_word_timings(cropped, 90, 120)
    segment = PhonemeSegment(
        id="segment",
        segment_id="segment",
        phoneme_id=0,
        text="speech",
        phonemes="speech",
        tokens=[1],
        processed_audio=np.zeros(120, dtype=np.float32),
        word_timings=scaled,
    )

    assert [(timing.start_sample, timing.end_sample) for timing in scaled] == [(7, 113)]
    composed = Composer().compose(rendered_segments_to_audio_job([segment]))
    assert composed.spans[0].sample_start == 7
    assert composed.spans[0].sample_end == 113

def test_audio_job_metadata_is_json_safe() -> None:
    segment = PhonemeSegment(
        id="seg-1",
        segment_id="seg-1",
        phoneme_id=0,
        text="hello",
        phonemes="həloʊ",
        tokens=[1],
        processed_audio=np.zeros(2, dtype=np.float32),
        ssmd_metadata={"custom": object()},
    )
    job = rendered_segments_to_audio_job([segment])
    assert job.items[0].metadata["ssmd"]["custom"]


def test_audio_job_persists_and_replays(tmp_path) -> None:
    segment = PhonemeSegment(
        id="kokoro",
        segment_id="kokoro",
        phoneme_id=0,
        text="hi",
        phonemes="haɪ",
        tokens=[1],
        processed_audio=np.ones(4, dtype=np.float32),
    )
    job = rendered_segments_to_audio_job([segment])
    manifest = tmp_path / "job.json"
    saved_manifest = job.save(str(manifest))
    replayed = Composer().compose(AudioJob.load(saved_manifest))
    direct = Composer().compose(job)
    np.testing.assert_array_equal(replayed.audio, direct.audio)


def test_audio_job_accepts_mixed_engine_clips() -> None:
    from audiocompose import AudioBufferSource, AudioClip, OutputPolicy, Silence
    from audiocompose import AudioJob as ComposeJob

    job = ComposeJob(
        items=(
            AudioClip("kokoro", AudioBufferSource(np.ones(2, dtype=np.float32), 24000)),
            Silence("gap", 1 / 24000),
            AudioClip("piper-like", AudioBufferSource(np.zeros(2, dtype=np.float32), 24000)),
        ),
        output=OutputPolicy(sample_rate=24000),
        producer={"engines": ["pykokoro", "piper-like"]},
    )
    result = Composer().compose(job)
    assert result.audio.size == 5
