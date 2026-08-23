"""Regression tests for the public API preserved from the v0.8.5 baseline."""

from __future__ import annotations

import inspect
import sys
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import pykokoro
from pykokoro.types import (
    AudioResult,
    AudioUnitDescriptor,
    AudioUnitResult,
    PhonemeSegment,
    WordTiming,
)

ROOT = Path(__file__).parents[1]
HISTORICAL_PHONEME_FIELDS = [
    "id",
    "segment_id",
    "phoneme_id",
    "text",
    "phonemes",
    "tokens",
    "lang",
    "char_start",
    "char_end",
    "paragraph_idx",
    "sentence_idx",
    "clause_idx",
    "pause_before",
    "pause_after",
    "ssmd_metadata",
    "voice_name",
    "voice_language",
    "voice_gender",
    "voice_variant",
    "raw_audio",
    "processed_audio",
]


def test_phoneme_segment_085_positional_prefix_is_stable() -> None:
    params = list(inspect.signature(PhonemeSegment).parameters)
    assert params[: len(HISTORICAL_PHONEME_FIELDS)] == HISTORICAL_PHONEME_FIELDS

    segment = PhonemeSegment(
        "ph0",
        "seg0",
        0,
        "Hello",
        "həloʊ",
        [1, 2, 3],
        "de",
        10,
        15,
    )

    assert segment.lang == "de"
    assert segment.char_start == 10
    assert segment.char_end == 15
    assert segment.alignment_tokens == []
    assert segment.word_timings == []


def test_phoneme_segment_timing_state_is_keyword_only_and_not_identity() -> None:
    parameter_map = inspect.signature(PhonemeSegment).parameters
    assert parameter_map["alignment_tokens"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter_map["word_timings"].kind is inspect.Parameter.KEYWORD_ONLY

    field_map = {field.name: field for field in fields(PhonemeSegment)}
    for name in ("alignment_tokens", "word_timings"):
        assert field_map[name].repr is False
        assert field_map[name].compare is False

    base = PhonemeSegment("id", "segment", 0, "Hello", "həloʊ", [1])
    changed = PhonemeSegment(
        "id",
        "segment",
        0,
        "Hello",
        "həloʊ",
        [1],
        alignment_tokens=[],
        word_timings=[WordTiming("Hello", 0, 5, 0, 1, "segment")],
    )
    assert base == changed
    assert "alignment_tokens" not in repr(changed)
    assert "word_timings" not in repr(changed)


def test_existing_top_level_imports_and_result_positional_fields_remain() -> None:
    from pykokoro import (  # noqa: PLC0415
        AudioUnitDescriptor as ImportedDescriptor,
    )
    from pykokoro import (
        AudioUnitResult as ImportedUnitResult,
    )
    from pykokoro import (
        GenerationConfig,
        PipelineConfig,
        ProsodyConfig,
        ProsodyMethod,
        SSMDPauseOverrides,
        SSMDRenderConfig,
    )
    from pykokoro import (
        WordTiming as ImportedWordTiming,
    )

    assert ImportedDescriptor is AudioUnitDescriptor
    assert ImportedUnitResult is AudioUnitResult
    assert ImportedWordTiming is WordTiming
    assert GenerationConfig and PipelineConfig and ProsodyConfig
    assert ProsodyMethod and SSMDPauseOverrides and SSMDRenderConfig

    descriptor = AudioUnitDescriptor(0, 0, 0, 5, "hello", "hash", (), ())
    unit = AudioUnitResult(descriptor, np.ones(2, dtype=np.float32), 24_000, [], [], [], None, {})
    result = AudioResult(np.ones(2, dtype=np.float32), 24_000, [], [], None, {}, [])
    assert unit.descriptor is descriptor
    assert result.sample_rate == 24_000


def test_existing_defaults_remain_paragraph_and_word_timings_survive_release() -> None:
    descriptor = AudioUnitDescriptor(0, 0, 0, 5, "hello", "hash", (), ())
    assert descriptor.unit_kind == "paragraph"
    assert descriptor.sentence_idx is None
    assert list(pykokoro.AudioUnitKind.__args__) == ["paragraph", "sentence"]

    timing = WordTiming("hello", 0, 5, 1, 2, "segment")
    result = AudioResult(np.ones(2, dtype=np.float32), 24_000, word_timings=[timing])
    result.release_audio()
    assert result.word_timings == [timing]


def test_audio_result_play_remains_callable_without_arguments(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setitem(
        sys.modules,
        "pykokoro.playback",
        SimpleNamespace(play_audio=lambda audio, sample_rate, **kwargs: calls.append(kwargs)),
    )

    AudioResult(np.ones(2, dtype=np.float32), 24_000).play()
    assert calls == [{"device": None}]


def test_playback_extras_keep_both_names() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
        import tomli as tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]
    assert optional["playback"] == ["sounddevice"]
    assert optional["sounddevice"] == ["sounddevice"]


def test_pykokoro_public_word_timing_export_is_additive() -> None:
    assert "WordTiming" in pykokoro.__all__
    assert pykokoro.WordTiming is WordTiming
