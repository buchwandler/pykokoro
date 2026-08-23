from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from pykokoro.audio_generator import (
    AudioGenerator,
    _crop_word_timings,
    _join_timestamps,
    _scale_word_timings,
    _translate_word_timings,
)
from pykokoro.constants import MAX_PHONEME_LENGTH
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.types import G2PAlignmentToken, PhonemeSegment, Segment, WordTiming


class _Tokenizer:
    def tokenize(self, phonemes: str) -> list[int]:
        return list(range(len(phonemes)))

    def detokenize(self, tokens: list[int]) -> str:
        return "a" * len(tokens)


class _NamedTimestampSession:
    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="input_ids")]

    def get_outputs(self) -> list[Any]:
        return [
            SimpleNamespace(name="waveform"),
            SimpleNamespace(name="auxiliary"),
            SimpleNamespace(name="pred_dur"),
        ]

    def run(self, output_names: Any, inputs: Any) -> list[np.ndarray]:
        _ = output_names, inputs
        return [
            np.zeros(240, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            np.asarray([3, 1, 1, 2, 2, 2, 0], dtype=np.float32),
        ]


def test_word_timing_seconds_and_sample_transform_helpers() -> None:
    timing = WordTiming("hello", 0, 5, 10, 20, "segment")
    assert timing.start_seconds(10) == 1.0
    assert timing.end_seconds(10) == 2.0
    assert _crop_word_timings([timing], 5, 25)[0].start_sample == 5
    assert _scale_word_timings([timing], 20, 40)[0].end_sample == 40
    assert _translate_word_timings([timing], 7)[0].start_sample == 17


def test_timestamp_output_is_selected_by_name() -> None:
    generator = AudioGenerator(
        cast(Any, _NamedTimestampSession()),
        cast(Any, _Tokenizer()),
    )
    audio, durations = generator._run_onnx("ab", np.zeros((2, 256), dtype=np.float32), 1.0)
    assert len(audio) == 240
    assert durations is not None
    assert durations.tolist() == [3, 1, 1, 2, 2, 2, 0]


def test_model_durations_map_to_source_words_and_clip_to_audio() -> None:
    generator = AudioGenerator(
        cast(Any, _NamedTimestampSession()),
        cast(Any, _Tokenizer()),
    )
    segment = PhonemeSegment(
        id="segment-1",
        segment_id="segment-1",
        phoneme_id=0,
        text="Hello!",
        phonemes="hello",
        tokens=[1, 2, 3, 4, 5],
        char_start=0,
        char_end=6,
        alignment_tokens=[
            G2PAlignmentToken("Hello", "he", "", 0, 5, 2),
            G2PAlignmentToken("!", "!", "", 5, 6, 1),
        ],
    )
    timings = generator._map_pred_dur_to_word_timings(
        segment, np.asarray([3, 1, 1, 2, 2, 2, 0], dtype=np.float32), 240
    )
    assert len(timings) == 1
    assert timings[0].text == "Hello"
    assert 0 <= timings[0].start_sample < timings[0].end_sample <= 240


def test_join_timestamps_merges_source_expansions() -> None:
    tokens = [
        {
            "text": "42",
            "phonemes": "fortytwo",
            "model_token_count": 2,
            "whitespace": "",
            "char_start": 0,
            "char_end": 2,
        },
        {
            "text": "42",
            "phonemes": "more",
            "model_token_count": 1,
            "whitespace": "",
            "char_start": 0,
            "char_end": 2,
        },
    ]
    joined = _join_timestamps(tokens, np.asarray([1, 2, 3, 4, 5], dtype=np.float32))
    assert joined[0]["start_ts"] == 0.0
    assert joined[1]["speech_end_ts"] >= joined[0]["speech_end_ts"]


def test_invalid_or_missing_timing_output_is_safe() -> None:
    class WaveformOnly:
        def get_inputs(self) -> list[Any]:
            return [SimpleNamespace(name="input_ids")]

        def get_outputs(self) -> list[Any]:
            return [SimpleNamespace(name="waveform"), SimpleNamespace(name="other")]

    generator = AudioGenerator(cast(Any, WaveformOnly()), cast(Any, _Tokenizer()))
    segment = PhonemeSegment(
        id="segment-1",
        segment_id="segment-1",
        phoneme_id=0,
        text="hello",
        phonemes="hello",
        tokens=[1],
    )
    assert generator._map_pred_dur_to_word_timings(segment, None, 100) == []
    assert generator._map_pred_dur_to_word_timings(segment, np.asarray([np.nan, 1, 1]), 100) == []


def test_kokorog2p_alignment_reads_token_metadata() -> None:
    class G2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            return list(range(len(phonemes)))

    segment = Segment(id="segment-1", text="Hello", char_start=0, char_end=5)
    raw_token = SimpleNamespace(
        text="Hello",
        char_start=0,
        char_end=5,
        meta={"phonemes": "həˈloʊ", "whitespace": ""},
    )

    tokens = KokoroG2PAdapter._normalize_alignment_tokens([raw_token], segment, G2P(), "1.0")

    assert len(tokens) == 1
    assert tokens[0].text == "Hello"
    assert tokens[0].phonemes == "həˈloʊ"
    assert tokens[0].char_start == 0
    assert tokens[0].char_end == 5


def test_alignment_offsets_are_rebased_to_document_clean_text() -> None:
    class G2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            return list(range(len(phonemes)))

    segment = Segment(id="segment-2", text="Hello", char_start=100, char_end=105)
    raw_token = SimpleNamespace(
        text="Hello",
        char_start=0,
        char_end=5,
        meta={"phonemes": "həloʊ", "whitespace": ""},
    )

    tokens = KokoroG2PAdapter._normalize_alignment_tokens([raw_token], segment, G2P(), "1.0")

    assert tokens[0].char_start == 100
    assert tokens[0].char_end == 105


def test_alignment_partition_counts_whitespace_model_positions_across_batches() -> None:
    alignment = [
        G2PAlignmentToken(
            text="word",
            phonemes="w",
            whitespace=" ",
            model_token_count=1,
        ),
        *[
            G2PAlignmentToken(text="word", phonemes="w", model_token_count=1)
            for _ in range(MAX_PHONEME_LENGTH - 1)
        ],
    ]

    batches = KokoroG2PAdapter._partition_alignment_tokens(alignment, [MAX_PHONEME_LENGTH, 1])

    assert [len(batch) for batch in batches] == [MAX_PHONEME_LENGTH - 1, 1]


def test_strict_timestamp_join_rejects_incomplete_duration_mapping() -> None:
    tokens = [
        {
            "text": "word",
            "phonemes": "word",
            "model_token_count": 4,
            "whitespace": "",
        }
    ]
    durations = np.asarray([1.0, 1.0, 1.0, 0.0], dtype=np.float32)

    assert _join_timestamps(tokens, durations, strict=True) == []
