from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from pykokoro._onnxvoice import KokoroTimingPayload
from pykokoro.audio_generator import (
    AudioGenerator,
    _collect_unit_word_timings,
    _crop_word_timings,
    _join_timestamps,
    _record_short_sentence_timing_alignment,
    _scale_word_timings,
    _translate_word_timings,
    populate_short_sentence_boundary_metadata,
)
from pykokoro.constants import MAX_PHONEME_LENGTH
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.types import (
    G2PAlignmentToken,
    PhonemeSegment,
    Segment,
    WordTiming,
    _exact_timing_geometry,
    _model_span_token_count,
)


class _Tokenizer:
    def tokenize(self, phonemes: str) -> list[int]:
        return list(range(len(phonemes)))

    def detokenize(self, tokens: list[int]) -> str:
        return "a" * len(tokens)


class _NamedTimestampSession:
    supports_timings = True
    sample_rate = 24_000

    def infer(self, token_ids, *, style, speed, seed=None):
        _ = token_ids, style, speed, seed
        return SimpleNamespace(
            audio=np.zeros(240, dtype=np.float32),
            timings=np.asarray([3, 1, 1, 2, 2, 2, 0], dtype=np.float32),
        )


def test_word_timing_seconds_and_sample_transform_helpers() -> None:
    timing = WordTiming("hello", 0, 5, 10, 20, "segment")
    assert timing.start_seconds(10) == 1.0
    assert timing.end_seconds(10) == 2.0
    assert _crop_word_timings([timing], 5, 25)[0].start_sample == 5
    assert _scale_word_timings([timing], 20, 40)[0].end_sample == 40
    assert _translate_word_timings([timing], 7)[0].start_sample == 17


def test_collect_unit_word_timings_translates_copies_and_preserves_segments() -> None:
    first = PhonemeSegment(
        id="a",
        segment_id="a",
        phoneme_id=0,
        text="A",
        phonemes="a",
        tokens=[1],
        processed_audio=np.zeros(100, dtype=np.float32),
        pause_after=1.0,
        word_timings=[WordTiming("A", 0, 1, 10, 80, "a")],
    )
    second = PhonemeSegment(
        id="b",
        segment_id="b",
        phoneme_id=0,
        text="B",
        phonemes="b",
        tokens=[1],
        processed_audio=np.zeros(120, dtype=np.float32),
        word_timings=[WordTiming("B", 2, 3, 20, 100, "b")],
    )
    before_first = list(first.word_timings)
    before_second = list(second.word_timings)

    timings = _collect_unit_word_timings([first, second], sample_rate=10)

    assert [(timing.start_sample, timing.end_sample) for timing in timings] == [
        (10, 80),
        (130, 210),
    ]
    assert first.word_timings == before_first
    assert second.word_timings == before_second


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


def test_join_timestamps_accepts_model_position_payload() -> None:
    tokens = [
        {
            "text": "one",
            "phonemes": "one",
            "model_token_count": 2,
            "model_span_token_count": 2,
            "whitespace": " ",
        },
        {"text": "two", "phonemes": "two", "model_token_count": 1, "whitespace": ""},
    ]
    timing = KokoroTimingPayload(
        values=np.asarray([1, 2, 3], dtype=np.float32),
        layout="model-positions",
        generated_position_count=3,
        raw_count=3,
    )

    joined = _join_timestamps(tokens, timing, strict=True)

    assert len(joined) == 2
    assert joined[0]["start_ts"] == 0.0
    assert joined[1]["start_ts"] >= joined[0]["end_ts"]


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
        meta={
            "phonemes": "həˈloʊ",
            "whitespace": "",
            "pronunciation_source": "provider",
            "pronunciation_provider": "espeak",
            "pronunciation_requested_language": "de-de",
            "pronunciation_source_ipa": "fˈaɪl",
            "pronunciation_language_markers": [{"language": "en", "ipa_offset": 0}],
        },
    )

    tokens = KokoroG2PAdapter._normalize_alignment_tokens([raw_token], segment, G2P(), "1.0")

    assert len(tokens) == 1
    assert tokens[0].text == "Hello"
    assert tokens[0].phonemes == "həˈloʊ"
    assert tokens[0].char_start == 0
    assert tokens[0].char_end == 5
    assert tokens[0].pronunciation_source == "provider"
    assert tokens[0].pronunciation_provider == "espeak"
    assert tokens[0].pronunciation_requested_language == "de-de"
    assert tokens[0].pronunciation_source_ipa == "fˈaɪl"
    assert tokens[0].pronunciation_language_markers == [{"language": "en", "ipa_offset": 0}]
    assert tokens[0].to_dict()["pronunciation_provider"] == "espeak"


def test_kokorog2p_alignment_preserves_static_lexicon_provenance() -> None:
    class G2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            return list(range(len(phonemes)))

    segment = Segment(id="segment-lexicon", text="File", char_start=0, char_end=4)
    raw_token = {
        "text": "File",
        "phonemes": "fˈiːlə",
        "whitespace": "",
        "pronunciation_source": "lexicon",
        "pronunciation_lexicon_id": "de-de:espeak",
    }

    tokens = KokoroG2PAdapter._normalize_alignment_tokens([raw_token], segment, G2P(), "1.0")

    assert tokens[0].pronunciation_source == "lexicon"
    assert tokens[0].pronunciation_lexicon_id == "de-de:espeak"
    assert tokens[0].pronunciation_provider is None


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


def test_explicit_model_span_overrides_compatibility_whitespace_rule() -> None:
    token = G2PAlignmentToken(
        "word", "phonemes", " ", model_token_count=2, model_span_token_count=7
    )
    assert _model_span_token_count(token) == 7
    assert token.to_dict()["model_span_token_count"] == 7


def test_context_normalization_reads_kokorog2p_tokenspan_meta() -> None:
    from kokorog2p.types import TokenSpan

    class FakeG2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            return list(range(len(phonemes)))

    raw_tokens = [
        TokenSpan(
            text="The",
            char_start=0,
            char_end=3,
            meta={"phonemes": "abc", "whitespace": " "},
        ),
        TokenSpan(
            text="who",
            char_start=4,
            char_end=7,
            meta={"phonemes": "de", "whitespace": ""},
        ),
    ]
    phrase_phonemes = "abc de"
    result = SimpleNamespace(
        phonemes=phrase_phonemes,
        token_ids=FakeG2P.phonemes_to_ids(phrase_phonemes, model="1.0"),
        tokens=raw_tokens,
    )

    normalized = KokoroG2PAdapter._normalize_context_result(
        result,
        g2p_module=FakeG2P(),
        model_version="1.0",
    )

    assert [token.text for token in normalized.tokens] == ["The", "who"]
    assert [token.phonemes for token in normalized.tokens] == ["abc", "de"]
    assert [token.whitespace for token in normalized.tokens] == [" ", ""]
    assert [token.char_start for token in normalized.tokens] == [0, 4]
    assert [token.char_end for token in normalized.tokens] == [3, 7]
    assert [token.model_token_count for token in normalized.tokens] == [3, 2]
    assert [token.model_span_token_count for token in normalized.tokens] == [4, 2]
    assert sum(token.model_span_token_count or 0 for token in normalized.tokens) == len(
        normalized.ids
    )


def test_context_normalization_reconciles_prefix_spans_with_whole_phrase_ids() -> None:
    class FakeG2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            # The separator is one model position and punctuation is merged.
            return list(range(len(phonemes) - phonemes.count(".")))

    result = SimpleNamespace(
        phonemes="a b.",
        ids=[1, 2, 3],
        tokens=[
            {"text": "a", "phonemes": "a", "whitespace": " "},
            {"text": "b", "phonemes": "b", "whitespace": "."},
        ],
    )
    normalized = KokoroG2PAdapter._normalize_context_result(
        result, g2p_module=FakeG2P(), model_version="1.0"
    )
    assert [token.model_span_token_count for token in normalized.tokens] == [2, 1]
    assert sum(token.model_span_token_count or 0 for token in normalized.tokens) == len(
        normalized.ids
    )


def test_context_geometry_preserves_zero_width_items_and_uses_contextual_deltas() -> None:
    class FakeG2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            return list(range(len(phonemes.replace(".", ""))))

    tokens = [
        SimpleNamespace(phonemes="a", whitespace=""),
        SimpleNamespace(phonemes="", whitespace="."),
        SimpleNamespace(phonemes="b", whitespace=""),
    ]
    geometry = KokoroG2PAdapter._derive_context_model_geometry(tokens, (1, 2), FakeG2P(), "1.0")
    assert [(item.speech_count, item.span_count) for item in geometry] == [(1, 1), (0, 0), (1, 1)]


def test_context_geometry_allows_empty_phoneme_positive_span() -> None:
    class FakeG2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            return list(range(len(phonemes)))

    tokens = [
        SimpleNamespace(phonemes="a", whitespace=""),
        SimpleNamespace(phonemes="", whitespace="  "),
    ]
    geometry = KokoroG2PAdapter._derive_context_model_geometry(tokens, (1, 2, 3), FakeG2P(), "1.0")
    assert [(item.speech_count, item.span_count) for item in geometry] == [(1, 1), (0, 2)]


def test_context_geometry_rejects_non_monotonic_and_final_sum_mismatch() -> None:
    class NonMonotonicG2P:
        @staticmethod
        def phonemes_to_ids(phonemes: str, *, model: str) -> list[int]:
            _ = model
            return list(range(1 if phonemes == "a" else 0))

    tokens = [
        SimpleNamespace(phonemes="a", whitespace=""),
        SimpleNamespace(phonemes="b", whitespace=""),
    ]
    assert KokoroG2PAdapter._derive_context_model_geometry(
        tokens, (1,), NonMonotonicG2P(), "1.0"
    ) == [None, None]


def test_record_alignment_uses_explicit_zero_span_without_whitespace_plus_one() -> None:
    for span in (0, 1, 2):
        metadata: dict[str, object] = {"generated_token_count": span}
        _record_short_sentence_timing_alignment(
            metadata,
            [
                {
                    "phonemes": "",
                    "whitespace": " ",
                    "model_token_count": 0,
                    "model_span_token_count": span,
                }
            ],
        )
        assert metadata["timing_model_position_count"] == span
        assert metadata["timing_alignment_complete"] is True


def test_exact_timing_geometry_accepts_zero_and_rejects_invalid_counts() -> None:
    assert _exact_timing_geometry({"model_token_count": 0, "model_span_token_count": 0}) == (0, 0)
    assert _exact_timing_geometry({"model_token_count": 1, "model_span_token_count": 0}) is None
    assert _exact_timing_geometry({"model_token_count": -1, "model_span_token_count": 0}) is None


def test_alignment_metadata_records_duration_and_unresolved_geometry_diagnostics() -> None:
    metadata: dict[str, object] = {"generated_token_count": 2}
    _record_short_sentence_timing_alignment(
        metadata,
        [
            {
                "text": ".",
                "phonemes": "",
                "whitespace": "",
                "model_token_count": None,
                "model_span_token_count": None,
            }
        ],
        pred_duration_count=5,
    )
    assert metadata["expected_pred_duration_count"] == 4
    assert metadata["pred_duration_count_delta"] == 1
    assert metadata["timing_failure_detail"] == "unresolved-model-span"
    assert metadata["timing_first_unresolved_token_model_token_count"] is None
    assert metadata["timing_first_unresolved_token_model_span_token_count"] is None


def test_model_free_phrase_timing_produces_target_boundaries() -> None:
    timing_tokens = [
        {
            "text": "Before",
            "phonemes": "ab",
            "whitespace": "",
            "model_token_count": 2,
            "model_span_token_count": 2,
            "is_target": False,
        },
        {
            "text": ".",
            "phonemes": "",
            "whitespace": ".",
            "model_token_count": 0,
            "model_span_token_count": 0,
            "is_target": False,
        },
        {
            "text": "Go",
            "phonemes": "g",
            "whitespace": "",
            "model_token_count": 1,
            "model_span_token_count": 1,
            "is_target": True,
        },
        {
            "text": "!",
            "phonemes": "",
            "whitespace": " ",
            "model_token_count": 0,
            "model_span_token_count": 1,
            "is_target": False,
        },
    ]
    metadata: dict[str, object] = {"generated_token_count": 4}
    _record_short_sentence_timing_alignment(metadata, timing_tokens, pred_duration_count=6)
    joined = _join_timestamps(
        timing_tokens, np.ones(6, dtype=np.float32), strict=True, metadata=metadata
    )
    populate_short_sentence_boundary_metadata(metadata, joined)
    assert metadata["timing_alignment_complete"] is True
    assert joined
    assert metadata["timing_model_position_count"] == 4
    assert metadata["pred_duration_count"] == 6
    assert metadata["target_start_ts"] is not None
    assert metadata["target_end_ts"] is not None
