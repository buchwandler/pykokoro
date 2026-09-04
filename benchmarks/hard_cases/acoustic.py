from __future__ import annotations

import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .schema import AcousticConstraints


@dataclass(frozen=True, slots=True)
class AcousticDiagnostics:
    status: str
    sample_rate: int
    sample_count: int
    duration_s: float
    finite_audio: bool
    peak_amplitude: float
    rms: float
    dc_offset: float
    word_timing_valid: bool
    pause_timing_valid: bool
    wall_time_s: float
    rtf: float | None
    onnx_calls: int = 0
    onnx_wall_time_ms: float = 0.0
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    audio_path: str | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


def evaluate_audio(
    result: Any, constraints: AcousticConstraints | None = None, *, wall_time_s: float = 0.0
) -> AcousticDiagnostics:
    constraints = constraints or AcousticConstraints()
    audio = np.asarray(getattr(result, "audio", np.array([], dtype=np.float32)))
    sample_rate = int(getattr(result, "sample_rate", 0) or 0)
    finite = bool(np.isfinite(audio).all()) if audio.size else False
    duration = len(audio) / sample_rate if sample_rate > 0 else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    dc = float(np.mean(audio)) if audio.size else 0.0
    failures: list[str] = []
    warnings: list[str] = []
    if sample_rate <= 0:
        failures.append("zero sample rate")
    if not audio.size:
        failures.append("empty waveform")
    if constraints.must_be_finite and not finite:
        failures.append("waveform contains NaN or Inf")
    if constraints.min_duration_s is not None and duration < constraints.min_duration_s:
        failures.append("duration below minimum")
    if constraints.max_duration_s is not None and duration > constraints.max_duration_s:
        failures.append("duration above maximum")
    if constraints.max_peak is not None and peak > constraints.max_peak:
        failures.append("peak amplitude above maximum")
    if duration > 30.0:
        warnings.append("unusually long audio")
    timings = tuple(getattr(result, "word_timings", ()))
    timing_valid = _valid_timings(timings)
    if not timing_valid:
        failures.append("invalid word timings")
    pause_valid = all(
        float(getattr(item, "pause_before", 0.0)) >= 0
        and float(getattr(item, "pause_after", 0.0)) >= 0
        for item in getattr(result, "phoneme_segments", ())
    )
    if not pause_valid:
        failures.append("negative pause timing")
    rtf = wall_time_s / duration if duration > 0 and wall_time_s >= 0 else None
    trace = getattr(result, "trace", None)
    inference = (
        trace.inference_summary()
        if trace is not None and hasattr(trace, "inference_summary")
        else {}
    )
    return AcousticDiagnostics(
        status="pass" if not failures else "fail",
        sample_rate=sample_rate,
        sample_count=int(audio.size),
        duration_s=duration,
        finite_audio=finite,
        peak_amplitude=peak,
        rms=rms,
        dc_offset=dc,
        word_timing_valid=timing_valid,
        pause_timing_valid=pause_valid,
        wall_time_s=wall_time_s,
        rtf=rtf,
        onnx_calls=int(inference.get("onnx_calls", 0) or 0),
        onnx_wall_time_ms=float(inference.get("onnx_runtime_ms", 0.0) or 0.0),
        warnings=tuple(warnings),
        failures=tuple(failures),
    )


def run_acoustic(
    frontend: Any,
    text: str,
    *,
    constraints: AcousticConstraints | None = None,
    render_audio: str | Path | None = None,
) -> AcousticDiagnostics:
    started = time.perf_counter()
    result = frontend.run(text)
    diagnostics = evaluate_audio(result, constraints, wall_time_s=time.perf_counter() - started)
    if render_audio is not None:
        path = Path(render_audio)
        path.parent.mkdir(parents=True, exist_ok=True)
        result_obj = result
        if hasattr(result_obj, "save_wav"):
            result_obj.save_wav(str(path))
        else:
            import soundfile as sf

            sf.write(path, np.asarray(result_obj.audio), int(result_obj.sample_rate))
        diagnostics = replace(diagnostics, audio_path=str(path))
    return diagnostics


def _valid_timings(timings: tuple[Any, ...]) -> bool:
    previous = 0
    for timing in timings:
        start = int(getattr(timing, "start_sample", -1))
        end = int(getattr(timing, "end_sample", -1))
        if start < previous or end < start:
            return False
        previous = end
    return True


__all__ = ["AcousticDiagnostics", "evaluate_audio", "run_acoustic"]
