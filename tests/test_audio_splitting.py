from __future__ import annotations

import logging
import random
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

import pykokoro.short_sentence_handler as short_sentence_handler
from pykokoro.audio_generator import (
    AudioGenerator,
    _join_timestamps,
    populate_short_sentence_boundary_metadata,
)
from pykokoro.constants import MAX_PHONEME_LENGTH
from pykokoro.short_sentence_handler import (
    PhraseResolveMode,
    RandomizedPhraseResolveMode,
    SHORT_SENTENCE_META_KEY,
    ShortSentenceConfig,
)
from pykokoro.types import PhonemeSegment


class DummyTokenizer:
    def __init__(self, factor: int) -> None:
        self.factor = factor

    def tokenize(self, text: str):
        return list(range(len(text) * self.factor))

    def detokenize(self, tokens):
        if not tokens:
            return ""
        return "a" * max(1, len(tokens) // self.factor)


class DummySession:
    def get_inputs(self):
        return [SimpleNamespace(name="input_ids")]

    def get_outputs(self):
        return [SimpleNamespace(name="waveform")]


class TimestampSession(DummySession):
    def get_outputs(self):
        return [SimpleNamespace(name="waveform"), SimpleNamespace(name="pred_dur")]

    def run(self, output_names, inputs):
        _ = output_names, inputs
        audio = np.zeros(8, dtype=np.float32)
        pred_dur = np.array([3, 1, 1, 1, 2, 2, 0], dtype=np.float32)
        return [audio, pred_dur]


def test_split_phonemes_uses_token_count():
    tokenizer = DummyTokenizer(factor=300)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )

    batches = generator.split_phonemes("hi")

    assert len(batches) > 1


def test_preprocess_pipeline_splits_segments():
    tokenizer = DummyTokenizer(factor=MAX_PHONEME_LENGTH)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="hello",
        phonemes="aa",
        tokens=[],
        pause_before=0.3,
        pause_after=0.7,
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=False
    )

    assert len(processed) == 2
    assert processed[0].id == "seg_1_ph0"
    assert processed[1].id == "seg_1_ph1"
    assert processed[0].pause_before == 0.3
    assert processed[0].pause_after == 0.0
    assert processed[1].pause_before == 0.0
    assert processed[1].pause_after == 0.7
    assert len(processed[0].tokens) == MAX_PHONEME_LENGTH
    assert len(processed[1].tokens) == MAX_PHONEME_LENGTH


def test_preprocess_wraps_short_segments_when_enabled():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=ShortSentenceConfig(min_phoneme_length=5),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
        pause_before=0.3,
        pause_after=0.7,
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert len(processed) == 1
    pretext = ShortSentenceConfig().phoneme_pretext
    assert processed[0].phonemes == f"{pretext}abc{pretext}"
    assert len(processed[0].tokens) == len(f"{pretext}abc{pretext}")
    assert processed[0].pause_before == 0.3
    assert processed[0].pause_after == 0.7


def test_preprocess_does_not_wrap_short_segments_when_disabled():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=ShortSentenceConfig(min_phoneme_length=5),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=False
    )

    assert processed[0].phonemes == "abc"
    assert len(processed[0].tokens) == len("abc")


def test_preprocess_uses_custom_short_sentence_pretext():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=ShortSentenceConfig(phoneme_pretext="..."),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "...abc..."


def test_preprocess_does_not_wrap_long_segments():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=ShortSentenceConfig(min_phoneme_length=5),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Hello",
        phonemes="abcde",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "abcde"


def test_preprocess_punctuation_only_segments_follow_short_sentence_override():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="!",
        phonemes="?!",
        tokens=[],
    )

    enabled = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )
    disabled = generator._preprocess_segments(
        [segment], enable_short_sentence_override=False
    )

    assert enabled[0].phonemes == ""
    assert enabled[0].tokens == []
    assert disabled[0].phonemes == "?!"
    assert len(disabled[0].tokens) == len("?!")


def test_preprocess_phrase_mode_uses_phrase_phonemes_and_metadata(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                neutral_phrase="The word, {segment}, appears here.",
                end_phrase="The word is hello. The word is '{segment}'",
            )
        },
        resolve_mode="phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go"
        assert phrase_template == "The word, {segment}, appears here."
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "abc def."
    assert processed[0].tokens == list(range(8))
    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["mode"] == "phrase"
    assert metadata["kind"] == "phrase"
    assert metadata["phrase_template"] == "The word, {segment}, appears here."
    assert metadata["original_token_count"] == 3
    assert metadata["generated_token_count"] == 8


def test_preprocess_randomized_phrase_mode_uses_configured_phrases(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "randomized-phrase": RandomizedPhraseResolveMode(
                neutral_phrases=["The word, {segment}, appears here."],
                end_phrases=["The word is hello. The word is '{segment}'"],
            )
        },
        resolve_mode="randomized-phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert phrase_template == "The word, {segment}, appears here."
        return "abc option.", list(range(10))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].phonemes == "abc option."
    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["mode"] == "randomized-phrase"
    assert metadata["phrase_template"] == "The word, {segment}, appears here."


def test_preprocess_phrase_mode_uses_end_phrase_for_period(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                neutral_phrase="The word, {segment}, appears here.",
                end_phrase="The word is hello. The word is '{segment}'",
            )
        },
        resolve_mode="phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go."
        assert phrase_template == "The word is hello. The word is '{segment}'"
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go.",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_template"] == "The word is hello. The word is '{segment}'"


def test_preprocess_phrase_mode_adds_ranked_phrase_fallbacks(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    default_mode = RandomizedPhraseResolveMode()
    config = ShortSentenceConfig(
        phrase_fallback_tries=3,
        resolve_modes={"phrase": PhraseResolveMode(phrase_selection="neutral")},
        resolve_mode="phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go"
        assert phrase_template == PhraseResolveMode().neutral_phrase
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_fallback_templates"] == default_mode.neutral_phrases[1:4]


def test_preprocess_phrase_fallbacks_skip_used_template_not_index(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    default_mode = RandomizedPhraseResolveMode()
    config = ShortSentenceConfig(
        phrase_fallback_tries=3,
        resolve_modes={
            "phrase": PhraseResolveMode(
                phrase_selection="neutral",
                neutral_phrase=default_mode.neutral_phrases[1],
            )
        },
        resolve_mode="phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go"
        assert phrase_template == default_mode.neutral_phrases[1]
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_fallback_templates"] == [
        default_mode.neutral_phrases[0],
        default_mode.neutral_phrases[2],
        default_mode.neutral_phrases[3],
    ]


def test_preprocess_randomized_phrase_mode_adds_following_fallbacks(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    phrases = [
        "First {segment}.",
        "Second {segment}.",
        "Third {segment}.",
        "Fourth {segment}.",
    ]
    config = ShortSentenceConfig(
        phrase_fallback_tries=3,
        resolve_modes={
            "randomized-phrase": RandomizedPhraseResolveMode(
                phrase_selection="neutral",
                neutral_phrases=phrases,
            )
        },
        resolve_mode="randomized-phrase",
    )

    monkeypatch.setattr(short_sentence_handler.random, "choice", lambda choices: choices[2])

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go"
        assert phrase_template == phrases[2]
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_template"] == phrases[2]
    assert metadata["phrase_fallback_templates"] == [
        phrases[3],
        phrases[0],
        phrases[1],
    ]
    assert metadata["phrase_fallback_tries"] == 3


def test_preprocess_randomized_phrase_mode_uses_seeded_sequence(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    phrases = [
        "First {segment}.",
        "Second {segment}.",
        "Third {segment}.",
        "Fourth {segment}.",
    ]
    config = ShortSentenceConfig(
        resolve_modes={
            "randomized-phrase": RandomizedPhraseResolveMode(
                phrase_selection="neutral",
                neutral_phrases=phrases,
            )
        },
        resolve_mode="randomized-phrase",
    )
    selected_templates: list[str] = []

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        _ = segment
        selected_templates.append(phrase_template)
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segments = [
        PhonemeSegment(
            id=f"seg_{index}",
            segment_id=f"seg_{index}",
            phoneme_id=0,
            text="Go",
            phonemes="abc",
            tokens=[],
        )
        for index in range(4)
    ]

    generator._preprocess_segments(
        segments,
        enable_short_sentence_override=True,
        random_seed=1234,
    )

    expected_rng = random.Random(1234)
    assert selected_templates == [expected_rng.choice(phrases) for _ in segments]


def test_preprocess_phrase_mode_can_force_end_phrase_for_neutral_text(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                phrase_selection="end",
                neutral_phrase="The word, {segment}, appears here.",
                end_phrase="The word is hello. The word is '{segment}'",
            )
        },
        resolve_mode="phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go"
        assert phrase_template == "The word is hello. The word is '{segment}'"
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_template"] == "The word is hello. The word is '{segment}'"


def test_preprocess_phrase_mode_can_force_neutral_phrase_for_period(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "phrase": PhraseResolveMode(
                phrase_selection="neutral",
                neutral_phrase="The word, {segment}, appears here.",
                end_phrase="The word is hello. The word is '{segment}'",
            )
        },
        resolve_mode="phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go."
        assert phrase_template == "The word, {segment}, appears here."
        return "abc def.", list(range(8))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go.",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_template"] == "The word, {segment}, appears here."


def test_preprocess_randomized_phrase_mode_can_force_end_phrases(monkeypatch):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={
            "randomized-phrase": RandomizedPhraseResolveMode(
                phrase_selection="end",
                neutral_phrases=["The word, {segment}, appears here."],
                end_phrases=["The word is hello. The word is '{segment}'"],
            )
        },
        resolve_mode="randomized-phrase",
    )

    def fake_phonemize(segment: PhonemeSegment, phrase_template: str):
        assert segment.text == "Go"
        assert phrase_template == "The word is hello. The word is '{segment}'"
        return "abc option.", list(range(10))

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        fake_phonemize,
    )
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc",
        tokens=[],
    )

    processed = generator._preprocess_segments(
        [segment], enable_short_sentence_override=True
    )

    assert processed[0].ssmd_metadata is not None
    metadata = processed[0].ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["phrase_template"] == "The word is hello. The word is '{segment}'"


def test_postprocess_phrase_mode_leaves_audio_when_timestamps_are_missing():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.concatenate(
        [
            np.ones(10, dtype=np.float32),
            np.zeros(5, dtype=np.float32),
            np.ones(30, dtype=np.float32),
        ]
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "expected_cut_ratio": 10 / 45,
                "silence_threshold": 1e-4,
                "min_silence_seconds": 0.0,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert len(processed[0].processed_audio) == len(audio)


def test_phrase_modes_fall_back_to_wrap_once_without_timestamp_output(
    monkeypatch, capsys
):
    tokenizer = DummyTokenizer(factor=1)
    config = ShortSentenceConfig(
        resolve_modes={"phrase": PhraseResolveMode()},
        resolve_mode="phrase",
    )

    def unexpected_phrase(*args, **kwargs):
        raise AssertionError("phrase mode should not run without timestamp output")

    monkeypatch.setattr(
        short_sentence_handler,
        "phonemize_short_sentence_phrase",
        unexpected_phrase,
    )
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
        short_sentence_config=config,
    )
    segments = [
        PhonemeSegment(
            id=f"seg_{idx}",
            segment_id=f"seg_{idx}",
            phoneme_id=0,
            text="Go",
            phonemes="abc",
            tokens=[],
        )
        for idx in range(2)
    ]

    processed = generator._preprocess_segments(
        segments,
        enable_short_sentence_override=True,
    )

    pretext = ShortSentenceConfig().phoneme_pretext
    assert [segment.phonemes for segment in processed] == [
        f"{pretext}abc{pretext}",
        f"{pretext}abc{pretext}",
    ]
    assert all(segment.ssmd_metadata is not None for segment in processed)
    assert all(
        segment.ssmd_metadata[SHORT_SENTENCE_META_KEY]["kind"] == "wrap"
        for segment in processed
        if segment.ssmd_metadata is not None
    )
    captured = capsys.readouterr()
    assert captured.out.count("Falling back to wrap mode for this run.") == 1


def test_postprocess_phrase_mode_cuts_with_default_energy_valley():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.concatenate(
        [
            np.ones(120, dtype=np.float32),
            np.zeros(480, dtype=np.float32),
            np.ones(480, dtype=np.float32),
            np.zeros(480, dtype=np.float32),
            np.ones(120, dtype=np.float32),
        ]
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "has_left_context": True,
                "has_right_context": True,
                "previous_token_end_ts": 100 / 24000,
                "next_token_start_ts": 1600 / 24000,
                "target_start_ts": 700 / 24000,
                "target_end_ts": 900 / 24000,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert np.array_equal(processed[0].processed_audio, audio[600:1080])


def test_postprocess_randomized_phrase_mode_uses_default_energy_valley():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.concatenate(
        [
            np.ones(120, dtype=np.float32),
            np.zeros(480, dtype=np.float32),
            np.ones(480, dtype=np.float32),
            np.zeros(480, dtype=np.float32),
            np.ones(120, dtype=np.float32),
        ]
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "randomized-phrase",
                "has_left_context": True,
                "has_right_context": True,
                "previous_token_end_ts": 100 / 24000,
                "next_token_start_ts": 1600 / 24000,
                "target_start_ts": 700 / 24000,
                "target_end_ts": 900 / 24000,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert np.array_equal(processed[0].processed_audio, audio[600:1080])


def test_generate_logs_randomized_phrase_target_timestamps(caplog, capsys):
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, TimestampSession()),
        tokenizer=cast(Any, tokenizer),
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="thgo",
        tokens=[1, 2, 3, 4],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "randomized-phrase",
                "timing_tokens": [
                    {
                        "text": "The",
                        "phonemes": "th",
                        "whitespace": " ",
                        "is_target": False,
                    },
                    {
                        "text": "Go",
                        "phonemes": "go",
                        "whitespace": "",
                        "is_target": True,
                    },
                ],
            }
        },
    )

    with caplog.at_level(logging.DEBUG, logger="pykokoro.audio_generator"):
        generator._generate_raw_audio_segments(
            [segment],
            voice_style=np.zeros((16, 256), dtype=np.float32),
            speed=1.0,
            voice_resolver=None,
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "segment='Go'" in caplog.text
    assert "token='Go'" in caplog.text
    assert "start=" in caplog.text
    assert "end=" in caplog.text
    metadata = segment.ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["target_start_ts"] == 0.0625
    assert metadata["target_end_ts"] == 0.175
    assert metadata["has_left_context"] is True
    assert metadata["has_right_context"] is False
    assert metadata["previous_token_end_ts"] == 0.05


def test_join_timestamps_keeps_trailing_whitespace_outside_speech_end():
    tokens = [
        {
            "text": "Go",
            "phonemes": "go",
            "whitespace": " ",
            "is_target": True,
        },
        {
            "text": "on",
            "phonemes": "on",
            "whitespace": "",
            "is_target": False,
        },
    ]
    pred_dur = np.array([3, 1, 1, 2, 2, 2, 0], dtype=np.float32)

    joined = _join_timestamps(tokens, pred_dur)

    assert joined[0]["speech_end_ts"] < joined[0]["end_ts"]
    assert joined[0]["end_ts"] == joined[1]["start_ts"]


def test_boundary_context_ignores_punctuation_tokens():
    metadata: dict[str, object] = {}
    timestamped = [
        {
            "text": "response",
            "start_ts": 1.0,
            "speech_end_ts": 1.5,
            "end_ts": 1.5,
            "is_target": False,
        },
        {
            "text": ".",
            "start_ts": 1.5,
            "speech_end_ts": 1.7,
            "end_ts": 1.8,
            "is_target": False,
        },
        {
            "text": "Yes",
            "start_ts": 1.8,
            "speech_end_ts": 2.1,
            "end_ts": 2.1,
            "is_target": True,
        },
        {
            "text": "!",
            "start_ts": 2.1,
            "speech_end_ts": 2.3,
            "end_ts": 2.3,
            "is_target": True,
        },
        {
            "text": ".",
            "start_ts": 2.3,
            "speech_end_ts": 2.4,
            "end_ts": 2.5,
            "is_target": False,
        },
        {
            "text": "The",
            "start_ts": 2.5,
            "speech_end_ts": 2.8,
            "end_ts": 2.8,
            "is_target": False,
        },
    ]

    populate_short_sentence_boundary_metadata(metadata, timestamped)

    assert metadata["previous_token_end_ts"] == 1.5
    assert metadata["next_token_start_ts"] == 2.5


def test_postprocess_phrase_mode_accepts_quiet_run_overlapping_target_end():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.concatenate(
        [
            np.ones(120, dtype=np.float32),
            np.zeros(120, dtype=np.float32),
            np.ones(120, dtype=np.float32),
            np.zeros(120, dtype=np.float32),
            np.ones(120, dtype=np.float32),
        ]
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "has_left_context": True,
                "has_right_context": True,
                "previous_token_end_ts": 60 / 24000,
                "target_start_ts": 240 / 24000,
                "target_end_ts": 420 / 24000,
                "next_token_start_ts": 480 / 24000,
                "min_silence_seconds": 0.005,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert np.array_equal(processed[0].processed_audio, audio[240:420])


def test_postprocess_end_phrase_mode_keeps_suffix_after_left_boundary():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.concatenate(
        [
            np.ones(120, dtype=np.float32),
            np.zeros(240, dtype=np.float32),
            np.ones(360, dtype=np.float32),
        ]
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go.",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "has_left_context": True,
                "has_right_context": False,
                "previous_token_end_ts": 60 / 24000,
                "target_start_ts": 420 / 24000,
                "target_end_ts": 600 / 24000,
                "min_silence_seconds": 0.005,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert np.array_equal(processed[0].processed_audio, audio[360:])


def test_postprocess_phrase_mode_can_use_energy_valley_cutter():
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    audio = np.concatenate(
        [
            np.ones(120, dtype=np.float32),
            np.zeros(240, dtype=np.float32),
            np.ones(240, dtype=np.float32),
            np.zeros(240, dtype=np.float32),
            np.ones(120, dtype=np.float32),
        ]
    )
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="abc def.",
        tokens=[],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "cutter": "energy-valley",
                "has_left_context": True,
                "has_right_context": True,
                "previous_token_end_ts": 60 / 24000,
                "next_token_start_ts": 900 / 24000,
                "target_start_ts": 420 / 24000,
                "target_end_ts": 600 / 24000,
                "min_silence_seconds": 0.005,
            }
        },
        raw_audio=audio,
    )

    processed = generator._postprocess_audio_segments([segment], trim_silence=False)

    assert processed[0].processed_audio is not None
    assert np.array_equal(processed[0].processed_audio, audio[360:600])


def test_prepare_phrase_audio_falls_back_to_wrap_when_cut_is_uncertain(
    monkeypatch,
    caplog,
):
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    phrase_audio = np.ones(240, dtype=np.float32)
    fallback_audio = np.full(120, 0.5, dtype=np.float32)

    monkeypatch.setattr(
        "pykokoro.audio_generator.cut_short_sentence_phrase_audio",
        lambda audio, metadata: None,
    )

    def fake_run_onnx(phonemes: str, voice_style: np.ndarray, speed: float):
        _ = voice_style, speed
        assert phonemes == "—abc—"
        return fallback_audio, None

    monkeypatch.setattr(generator, "_run_onnx", fake_run_onnx)
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="phrase",
        tokens=[1, 2, 3],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "fallback_phonemes": "—abc—",
                "fallback_tokens": [4, 5, 6],
            }
        },
    )

    with caplog.at_level(logging.WARNING, logger="pykokoro.audio_generator"):
        audio = generator._prepare_short_sentence_phrase_audio(
            segment,
            phrase_audio,
            voice_style=np.zeros((16, 256), dtype=np.float32),
            speed=1.0,
        )

    assert np.array_equal(audio, fallback_audio)
    assert segment.phonemes == "—abc—"
    assert segment.tokens == [4, 5, 6]
    metadata = segment.ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["fallback_used"] == "wrap"
    assert metadata["cut_applied"] is True
    assert "falling back to wrap mode" in caplog.text


def test_prepare_phrase_audio_tries_phrase_fallbacks_before_wrap(
    monkeypatch,
    caplog,
):
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    phrase_audio = np.ones(240, dtype=np.float32)
    retry_audio = np.full(240, 0.25, dtype=np.float32)
    cut_retry_audio = np.full(120, 0.75, dtype=np.float32)

    def fake_cut(audio: np.ndarray, metadata: dict[str, object]):
        _ = audio
        if metadata.get("phrase_template") == "Retry two {segment}":
            return cut_retry_audio
        return None

    def fake_retry(
        segment: PhonemeSegment,
        phrase_template: str,
        base_metadata: dict[str, object],
    ):
        _ = segment, base_metadata
        return short_sentence_handler.ShortSentenceApplication(
            phonemes=f"phonemes for {phrase_template}",
            tokens=[len(phrase_template)],
            metadata={
                "kind": "phrase",
                "phrase_template": phrase_template,
                "timing_tokens": [],
            },
        )

    def fake_run_onnx(phonemes: str, voice_style: np.ndarray, speed: float):
        _ = phonemes, voice_style, speed
        return retry_audio, np.array([1.0, 1.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "pykokoro.audio_generator.cut_short_sentence_phrase_audio",
        fake_cut,
    )
    monkeypatch.setattr(
        "pykokoro.audio_generator.build_short_sentence_phrase_retry",
        fake_retry,
    )
    monkeypatch.setattr(generator, "_run_onnx", fake_run_onnx)
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="phrase",
        tokens=[1, 2, 3],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "phrase_template": "Primary {segment}",
                "phrase_fallback_tries": 3,
                "phrase_fallback_templates": [
                    "Retry one {segment}",
                    "Retry two {segment}",
                ],
                "fallback_phonemes": "wrap",
                "fallback_tokens": [4],
            }
        },
    )

    with caplog.at_level(logging.INFO, logger="pykokoro.audio_generator"):
        audio = generator._prepare_short_sentence_phrase_audio(
            segment,
            phrase_audio,
            voice_style=np.zeros((16, 256), dtype=np.float32),
            speed=1.0,
        )

    assert np.array_equal(audio, cut_retry_audio)
    assert segment.phonemes == "phonemes for Retry two {segment}"
    assert segment.tokens == [len("Retry two {segment}")]
    metadata = segment.ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["fallback_used"] == "phrase"
    assert metadata["cut_applied"] is True
    assert metadata["phrase_template"] == "Retry two {segment}"
    assert "using fallback phrase 'Retry two {segment}'" in caplog.text


def test_prepare_phrase_audio_obeys_phrase_fallback_tries(
    monkeypatch,
):
    tokenizer = DummyTokenizer(factor=1)
    generator = AudioGenerator(
        session=cast(Any, DummySession()),
        tokenizer=cast(Any, tokenizer),
    )
    attempted_templates: list[str] = []

    def fake_cut(audio: np.ndarray, metadata: dict[str, object]):
        _ = audio, metadata
        return None

    def fake_retry(
        segment: PhonemeSegment,
        phrase_template: str,
        base_metadata: dict[str, object],
    ):
        _ = segment, base_metadata
        attempted_templates.append(phrase_template)
        return short_sentence_handler.ShortSentenceApplication(
            phonemes=f"phonemes for {phrase_template}",
            tokens=[len(phrase_template)],
            metadata={
                "kind": "phrase",
                "phrase_template": phrase_template,
                "timing_tokens": [],
            },
        )

    def fake_run_onnx(phonemes: str, voice_style: np.ndarray, speed: float):
        _ = phonemes, voice_style, speed
        return np.ones(240, dtype=np.float32), np.array([1.0, 1.0], dtype=np.float32)

    monkeypatch.setattr(
        "pykokoro.audio_generator.cut_short_sentence_phrase_audio",
        fake_cut,
    )
    monkeypatch.setattr(
        "pykokoro.audio_generator.build_short_sentence_phrase_retry",
        fake_retry,
    )
    monkeypatch.setattr(generator, "_run_onnx", fake_run_onnx)
    segment = PhonemeSegment(
        id="seg_1",
        segment_id="seg_1",
        phoneme_id=0,
        text="Go",
        phonemes="phrase",
        tokens=[1, 2, 3],
        ssmd_metadata={
            SHORT_SENTENCE_META_KEY: {
                "kind": "phrase",
                "phrase_template": "Primary {segment}",
                "phrase_fallback_tries": 1,
                "phrase_fallback_templates": [
                    "Retry one {segment}",
                    "Retry two {segment}",
                    "Retry three {segment}",
                ],
                "fallback_phonemes": "wrap",
                "fallback_tokens": [4],
            }
        },
    )

    audio = generator._prepare_short_sentence_phrase_audio(
        segment,
        np.ones(240, dtype=np.float32),
        voice_style=np.zeros((16, 256), dtype=np.float32),
        speed=1.0,
    )

    assert np.array_equal(audio, np.ones(240, dtype=np.float32))
    assert attempted_templates == ["Retry one {segment}"]
    metadata = segment.ssmd_metadata[SHORT_SENTENCE_META_KEY]
    assert metadata["fallback_used"] == "wrap"
    assert metadata["retry_attempts"] == 1
