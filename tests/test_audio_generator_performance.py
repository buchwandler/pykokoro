from __future__ import annotations

from typing import Any

import numpy as np

from pykokoro.audio_generator import AudioGenerator
from pykokoro.types import Trace


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
    def __init__(self, *, speed_type: str = "tensor(float)") -> None:
        self.calls = 0
        self.inputs = [
            _Input("tokens", "tensor(int64)"),
            _Input("style"),
            _Input("speed", speed_type),
        ]

    def get_inputs(self) -> list[_Input]:
        return self.inputs

    def get_outputs(self) -> list[_Output]:
        return [_Output("audio"), _Output("pred_dur")]

    def run(self, _outputs: Any, _inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls += 1
        return [
            np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
        ]


def _generator(**kwargs: Any) -> tuple[AudioGenerator, _Session]:
    session = _Session(**kwargs.pop("session_kwargs", {}))
    return AudioGenerator(session, _Tokenizer(), **kwargs), session


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
    first_duration[0] = 99.0

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
