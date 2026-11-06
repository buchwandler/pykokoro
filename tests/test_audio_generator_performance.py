from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pykokoro.audio_generator import AudioGenerator
from pykokoro.types import G2PAlignmentToken, PhonemeSegment, Trace


class _Input:
    def __init__(self, name: str, type_name: str = "tensor(float)") -> None:
        self.name = name
        self.type = type_name


class _Output:
    def __init__(self, name: str) -> None:
        self.name = name


class _Tokenizer:
    def tokenize(self, phonemes: str) -> list[int]:
        return list(range(1, len(phonemes) + 1))


class _Session:
    supports_timings = True
    sample_rate = 24_000
    cache_identity = ("test", "fp32", "cpu")

    def __init__(self, *, speed_type: str = "tensor(float)") -> None:
        _ = speed_type
        self.calls = 0

    def infer(self, token_ids, *, style, speed, seed=None):
        _ = token_ids, style, speed, seed
        self.calls += 1
        return SimpleNamespace(
            audio=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            timings=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )


def _generator(**kwargs: Any) -> tuple[AudioGenerator, _Session]:
    session = _Session(**kwargs.pop("session_kwargs", {}))
    return AudioGenerator(session, _Tokenizer(), **kwargs), session


class _ContextSession:
    supports_timings = True
    sample_rate = 24_000
    cache_identity = ("context-test", "fp32", "cpu")
    resolved = SimpleNamespace(metadata={"runtime": {"layout": "language-router-onnx-v1"}})

    def __init__(self, responses: list[tuple[int, list[float] | None]]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[int], str | None]] = []

    def infer(self, token_ids, *, style, speed, seed=None, language=None):
        _ = style, speed, seed
        self.calls.append((list(token_ids), language))
        audio_length, timings = self.responses[len(self.calls) - 1]
        return SimpleNamespace(
            audio=np.arange(audio_length, dtype=np.float32),
            timings=None if timings is None else np.asarray(timings, dtype=np.float32),
        )


def _narration_context_segment() -> PhonemeSegment:
    text = "before body after"
    context_tokens = [
        G2PAlignmentToken(
            token,
            token,
            char_start=start,
            char_end=end,
            model_token_count=1,
            model_span_token_count=1,
        )
        for token, start, end in (("before", 0, 6), ("body", 7, 11), ("after", 12, 17))
    ]
    body_token = G2PAlignmentToken(
        "body",
        "body",
        char_start=7,
        char_end=11,
        model_token_count=1,
        model_span_token_count=1,
    )
    return PhonemeSegment(
        id="request:phoneme:0",
        segment_id="request",
        phoneme_id=0,
        text=text,
        phonemes="before body after",
        tokens=[1, 2, 3],
        lang="de-de",
        char_start=0,
        char_end=len(text),
        engine_metadata={
            "__akinvox_narration_context": {
                "context_char_start": 0,
                "context_char_end": len(text),
                "body_char_start": 7,
                "body_char_end": 11,
                "context_before_words": 1,
                "context_after_words": 1,
                "context_text": text,
                "source_word_spans": [[0, 6], [7, 11], [12, 17]],
                "body_word_spans": [[7, 11]],
                "body": {
                    "text": "body",
                    "phonemes": "body",
                    "tokens": [2],
                    "char_start": 7,
                    "char_end": 11,
                    "alignment_tokens": [body_token.to_dict()],
                },
            }
        },
        alignment_tokens=context_tokens,
    )


def test_narration_context_audio_is_trimmed_using_exact_word_timings() -> None:
    session = _ContextSession([(1800, [1.0, 1.0, 1.0])])
    generator = AudioGenerator(session, _Tokenizer(), inference_cache_enabled=False)
    segment = _narration_context_segment()

    result = generator._generate_raw_audio_segments(
        [segment], np.zeros((2, 256), dtype=np.float32), 1.0
    )

    assert len(session.calls) == 1
    assert session.calls[0] == ([1, 2, 3], "de-de")
    assert result[0].text == "body"
    assert result[0].raw_audio is not None
    np.testing.assert_array_equal(result[0].raw_audio, np.arange(600, 1200, dtype=np.float32))
    assert [(timing.char_start, timing.char_end) for timing in result[0].word_timings] == [(7, 11)]
    assert (result[0].word_timings[0].start_sample, result[0].word_timings[0].end_sample) == (
        0,
        600,
    )


def test_narration_context_without_exact_timings_retries_body_only() -> None:
    session = _ContextSession([(1800, None), (600, [1.0])])
    generator = AudioGenerator(session, _Tokenizer(), inference_cache_enabled=False)
    segment = _narration_context_segment()
    trace = Trace()

    result = generator._generate_raw_audio_segments(
        [segment], np.zeros((2, 256), dtype=np.float32), 1.0, trace
    )

    assert session.calls == [([1, 2, 3], "de-de"), ([2], "de-de")]
    assert result[0].text == "body"
    assert result[0].raw_audio is not None
    assert result[0].raw_audio.size == 600
    assert trace.counters["narration_context_body_only_fallback"] == 1
    assert [(timing.char_start, timing.char_end) for timing in result[0].word_timings] == [(7, 11)]


def test_trace_records_ort_timing_and_aggregate_metrics() -> None:
    generator, session = _generator(inference_cache_enabled=False)
    trace = Trace()

    audio, pred_dur = generator._run_onnx("abc", np.zeros((2, 256), dtype=np.float32), 1.0, trace)

    assert session.calls == 1
    assert audio.size == 3
    assert pred_dur is not None
    record = trace.inference[0]
    assert record["call_number"] == 1
    assert record["cache_hit"] is False
    assert record["runtime_ms"] >= 0.0
    assert record["audio_samples"] == 3
    assert record["audio_seconds"] > 0.0
    assert record["rtf"] is not None
    assert trace.inference_summary()["onnx_calls"] == 1
    assert trace.inference_summary()["onnx_cache_misses"] == 1


def test_waveform_diagnostics_are_explicitly_opt_in(monkeypatch) -> None:
    calls = {"noise": 0, "waveform": 0}

    def noise(audio: np.ndarray) -> tuple[bool, dict[str, float]]:
        calls["noise"] += 1
        return False, {}

    def waveform(audio: np.ndarray) -> dict[str, float | int | bool]:
        calls["waveform"] += 1
        return {"samples": int(audio.size)}

    monkeypatch.setattr("pykokoro.audio_generator._is_stationary_broadband_noise", noise)
    monkeypatch.setattr("pykokoro.audio_generator._waveform_metrics", waveform)

    ordinary, _ = _generator(inference_cache_enabled=False)
    ordinary._run_onnx("abc", np.zeros((2, 256), dtype=np.float32), 1.0, Trace())
    assert calls == {"noise": 0, "waveform": 0}

    diagnostic, _ = _generator(
        inference_cache_enabled=False,
        inference_audio_diagnostics=True,
    )
    trace = Trace()
    diagnostic._run_onnx("abc", np.zeros((2, 256), dtype=np.float32), 1.0, trace)
    assert calls == {"noise": 1, "waveform": 1}
    assert trace.inference[0]["audio"]["samples"] == 3


def test_diagnostic_setting_does_not_change_audio() -> None:
    style = np.zeros((2, 256), dtype=np.float32)
    normal, _ = _generator(inference_cache_enabled=False)
    diagnostic, _ = _generator(inference_cache_enabled=False, inference_audio_diagnostics=True)

    normal_audio, normal_duration = normal._run_onnx("abc", style, 1.0)
    diagnostic_audio, diagnostic_duration = diagnostic._run_onnx("abc", style, 1.0)

    np.testing.assert_array_equal(normal_audio, diagnostic_audio)
    np.testing.assert_array_equal(normal_duration, diagnostic_duration)


def test_exact_input_cache_reuses_raw_outputs_and_returns_copies() -> None:
    generator, session = _generator(inference_cache_max_bytes=1024)
    first_trace = Trace()
    first_audio, first_duration = generator._run_onnx(
        "abc", np.zeros((2, 256), dtype=np.float32), 1.0, first_trace
    )
    first_audio[0] = 99.0
    assert first_duration is not None
    assert not first_duration.values.flags.writeable

    second_trace = Trace()
    second_audio, second_duration = generator._run_onnx(
        "abc", np.zeros((2, 256), dtype=np.float32), 1.0, second_trace
    )

    assert session.calls == 1
    assert second_trace.inference[0]["cache_hit"] is True
    np.testing.assert_array_equal(second_audio, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    np.testing.assert_array_equal(second_duration, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert first_trace.inference_summary()["onnx_cache_misses"] == 1
    assert second_trace.inference_summary()["onnx_cache_hits"] == 1


def test_inference_cache_misses_for_changed_model_inputs() -> None:
    generator, session = _generator(inference_cache_max_bytes=1024)
    style = np.zeros((2, 256), dtype=np.float32)

    generator._run_onnx("abc", style, 1.0)
    generator._run_onnx("abcd", style, 1.0)
    generator._run_onnx("abc", np.ones((2, 256), dtype=np.float32), 1.0)
    generator._run_onnx("abc", style, 1.1)

    assert session.calls == 4


def test_inference_cache_key_includes_dtype_and_shape() -> None:
    generator, _ = _generator(inference_cache_max_bytes=1024)
    base = {"tokens": np.array([[1, 2]], dtype=np.int64)}
    different_dtype = {"tokens": np.array([[1, 2]], dtype=np.int32)}
    different_shape = {"tokens": np.array([[1], [2]], dtype=np.int64)}

    assert generator._inference_cache_key(base) != generator._inference_cache_key(different_dtype)
    assert generator._inference_cache_key(base) != generator._inference_cache_key(different_shape)


def test_inference_cache_evicts_by_bytes_and_close_clears_entries() -> None:
    generator, session = _generator(inference_cache_max_bytes=30)
    style = np.zeros((2, 256), dtype=np.float32)

    generator._run_onnx("abc", style, 1.0)
    generator._run_onnx("abcd", style, 1.0)
    generator._run_onnx("abc", style, 1.0)
    assert session.calls == 3

    generator.close()
    assert not generator._inference_cache
    generator._run_onnx("abc", style, 1.0)
    assert session.calls == 4


def test_trace_summary_exposes_short_sentence_counters() -> None:
    trace = Trace()
    for name in (
        "short_sentence_detected",
        "short_sentence_phrase_initial",
        "short_sentence_phrase_retry",
        "short_sentence_cut_strict_success",
        "short_sentence_cut_adaptive_success",
        "short_sentence_cut_failure",
        "short_sentence_wrap_fallback",
    ):
        trace.increment_counter(name)

    summary = trace.inference_summary()
    assert all(summary[name] == 1 for name in trace.counters)


class _PhraseSession(_Session):
    def __init__(self, audio: np.ndarray) -> None:
        super().__init__()
        self.audio = audio.astype(np.float32, copy=False)

    def infer(self, token_ids, *, style, speed, seed=None):
        _ = token_ids, style, speed, seed
        self.calls += 1
        return SimpleNamespace(
            audio=self.audio,
            timings=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        )


def _phrase_metadata(*, cutter: str = "energy-valley", valid: bool = True) -> dict[str, object]:
    target_end = 2400 / 24000 if valid else 1600 / 24000
    return {
        "kind": "phrase",
        "cutter": cutter,
        "target_start_ts": 1600 / 24000,
        "target_end_ts": target_end,
        "previous_token_end_ts": 600 / 24000,
        "next_token_start_ts": 3400 / 24000,
        "has_left_context": True,
        "has_right_context": True,
        "frame_duration_ms": 5,
        "energy_threshold": 0.05,
        "min_silence_seconds": 0.02,
        "phrase_fallback_templates": ["retry {segment}"],
        "phrase_fallback_tries": 1,
        "fallback_phonemes": "—abc—",
        "fallback_tokens": [1, 2, 3],
    }


@pytest.mark.parametrize(
    ("waveform", "cutter"),
    [
        (np.zeros(4000, dtype=np.float32), "energy-valley"),
        (np.ones(4000, dtype=np.float32), "timestamp-adaptive"),
    ],
)
def test_phrase_cut_success_uses_one_model_call(waveform, cutter) -> None:
    session = _PhraseSession(waveform)
    generator = AudioGenerator(session, _Tokenizer(), inference_cache_enabled=False)
    trace = Trace()
    audio, _ = generator._run_onnx("abc", np.zeros((2, 256), dtype=np.float32), 1.0, trace)
    segment = SimpleNamespace(
        text="Hi!",
        lang="en-us",
        phonemes="abc",
        tokens=[1, 2, 3],
        word_timings=[],
        engine_metadata={"__short_sentence": _phrase_metadata(cutter=cutter)},
    )

    result = generator._prepare_short_sentence_phrase_audio(
        segment, audio, np.zeros((2, 256), dtype=np.float32), 1.0, trace=trace
    )

    assert result.size > 0
    assert session.calls == 1
    assert trace.counters.get("short_sentence_cut_failure", 0) == 0


def test_invalid_timestamps_use_one_retry(monkeypatch) -> None:
    from pykokoro.short_sentence_handler import ShortSentenceApplication

    session = _PhraseSession(np.zeros(4000, dtype=np.float32))
    generator = AudioGenerator(session, _Tokenizer(), inference_cache_enabled=False)
    trace = Trace()
    audio, _ = generator._run_onnx("abc", np.zeros((2, 256), dtype=np.float32), 1.0, trace)
    retry_metadata = _phrase_metadata(valid=True)
    monkeypatch.setattr(
        "pykokoro.audio_generator.build_short_sentence_phrase_retry",
        lambda *args, **kwargs: ShortSentenceApplication("abc", [1, 2, 3], retry_metadata),
    )
    segment = SimpleNamespace(
        text="Hi!",
        lang="en-us",
        phonemes="abc",
        tokens=[1, 2, 3],
        word_timings=[],
        engine_metadata={"__short_sentence": _phrase_metadata(valid=False)},
    )

    result = generator._prepare_short_sentence_phrase_audio(
        segment, audio, np.zeros((2, 256), dtype=np.float32), 1.0, trace=trace
    )

    assert result.size > 0
    assert session.calls == 2
    assert trace.inference_summary()["short_sentence_phrase_retry"] == 1


def test_retry_failure_uses_one_retry_and_wrap_fallback(monkeypatch) -> None:
    from pykokoro.short_sentence_handler import ShortSentenceApplication

    session = _PhraseSession(np.zeros(4000, dtype=np.float32))
    generator = AudioGenerator(session, _Tokenizer(), inference_cache_enabled=False)
    trace = Trace()
    audio, _ = generator._run_onnx("abc", np.zeros((2, 256), dtype=np.float32), 1.0, trace)
    retry_metadata = _phrase_metadata(valid=False)
    monkeypatch.setattr(
        "pykokoro.audio_generator.build_short_sentence_phrase_retry",
        lambda *args, **kwargs: ShortSentenceApplication("abc", [1, 2, 3], retry_metadata),
    )
    segment = SimpleNamespace(
        text="Hi!",
        lang="en-us",
        phonemes="abc",
        tokens=[1, 2, 3],
        word_timings=[],
        engine_metadata={"__short_sentence": _phrase_metadata(valid=False)},
    )

    result = generator._prepare_short_sentence_phrase_audio(
        segment, audio, np.zeros((2, 256), dtype=np.float32), 1.0, trace=trace
    )

    assert result.size > 0
    assert session.calls == 3
    assert trace.inference_summary()["short_sentence_phrase_retry"] == 1
    assert trace.inference_summary()["short_sentence_wrap_fallback"] == 1


class _LanguageRouterSession(_Session):
    def __init__(self) -> None:
        super().__init__()
        self.resolved = SimpleNamespace(metadata={"runtime": {"layout": "language-router-onnx-v1"}})
        self.languages: list[str | None] = []

    def infer(self, token_ids, *, style, speed, seed=None, language=None):
        self.languages.append(language)
        return super().infer(token_ids, style=style, speed=speed, seed=seed)


def test_router_inference_cache_is_language_specific_and_traces_route() -> None:
    session = _LanguageRouterSession()
    generator = AudioGenerator(session, _Tokenizer())
    style = np.zeros((2, 256), dtype=np.float32)
    english_trace = Trace()
    german_trace = Trace()
    repeated_english_trace = Trace()

    generator._run_onnx("abc", style, 1.0, english_trace, language="en-US")
    generator._run_onnx("abc", style, 1.0, german_trace, language="de-DE")
    generator._run_onnx("abc", style, 1.0, repeated_english_trace, language="en-GB")

    assert session.calls == 2
    assert session.languages == ["en-US", "de-DE"]
    assert english_trace.inference[0]["acoustic_route"] == "en"
    assert german_trace.inference[0]["acoustic_route"] == "de"
    assert repeated_english_trace.inference[0]["cache_hit"] is True
