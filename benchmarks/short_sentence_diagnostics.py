"""Diagnose short-sentence phrase timing without writing audio files."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import logging
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 24000
SCHEMA = "pykokoro.short-sentence-diagnostics.v1"
DEFAULT_TEMPLATE = "The short message read: {segment}"
CARRIER_TEMPLATES = (
    DEFAULT_TEMPLATE,
    "The note contained only these words: {segment}",
    "The conversation stopped after one last reply: {segment}",
    "The question was asked plainly: {segment}",
    "A quiet voice asked: {segment}",
    "The speaker called out: {segment}",
    "The announcement ended with: {segment}",
    "The conversation stopped, {segment}, before someone answered.",
    "The hallway went quiet; {segment}; then footsteps resumed.",
)
SPEED_VALUES = (0.75, 0.90, 1.00, 1.10, 1.25)
CUTTER_VALUES = ("energy-valley", "timestamp-adaptive")
PARAMETER_VALUES: dict[str, tuple[int | float, ...]] = {
    "frame-duration-ms": (2, 5, 10, 20),
    "energy-threshold": (0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15),
    "min-silence-seconds": (0.005, 0.010, 0.020, 0.040, 0.080),
    "search-radius-ms": (10, 20, 35, 50, 75, 100),
    "context-guard-ms": (0, 2, 4, 8, 12, 20),
    "analysis-window-ms": (2, 3, 5, 8, 10, 20),
}
PARAMETER_STAGE = {
    "phrase-template": "context-g2p",
    "phrase-selection": "context-g2p",
    "speed": "onnx-duration",
    "cutter": "cut-search",
    "frame-duration-ms": "cut-search",
    "energy-threshold": "cut-search",
    "min-silence-seconds": "cut-search",
    "search-radius-ms": "cut-search",
    "context-guard-ms": "cut-search",
    "analysis-window-ms": "cut-search",
}


@dataclass(frozen=True)
class DiagnosticSettings:
    phrase_selection: str = "neutral"
    neutral_phrase: str = DEFAULT_TEMPLATE
    end_phrase: str = DEFAULT_TEMPLATE
    frame_duration_ms: int = 5
    energy_threshold: float = 0.05
    silence_threshold: float = 1e-4
    min_silence_seconds: float = 0.02
    cutter: str = "timestamp-adaptive"
    search_radius_ms: float = 35.0
    context_guard_ms: float = 8.0
    analysis_window_ms: float = 5.0
    speed: float = 1.0
    random_seed: int = 0
    min_phoneme_length: int = 30
    phoneme_pretext: str = "—"
    enabled: bool = True
    resolve_mode: str = "phrase"
    phrase_fallback_tries: int = 0

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    axis: str
    value: object
    settings: DiagnosticSettings

    @property
    def parameter_stage(self) -> str:
        return PARAMETER_STAGE.get(self.axis, "context-g2p")

    @property
    def phrase_text(self) -> str:
        return self.settings.neutral_phrase


@dataclass
class ShortSentenceDiagnosticCase:
    case_id: str
    axis: str
    value: object
    settings: dict[str, object]
    phrase_text: str = ""
    phonemes: str = ""
    token_ids: list[int] = field(default_factory=list)
    timing_tokens: list[dict[str, object]] = field(default_factory=list)
    pred_dur: list[float] | None = None
    timestamped_tokens: list[dict[str, object]] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
    audio_samples: int = 0
    runtime_ms: float = 0.0
    cache_hit: bool = False
    succeeded: bool = False
    context_tokens: list[dict[str, object]] = field(default_factory=list)
    invariants: dict[str, bool] = field(default_factory=dict)
    failure_stage: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            **asdict(self),
        }


def build_baseline_settings(*, speed: float = 1.0, random_seed: int = 0) -> DiagnosticSettings:
    return DiagnosticSettings(speed=speed, random_seed=random_seed)


def _case_settings(base: DiagnosticSettings, **changes: object) -> DiagnosticSettings:
    from dataclasses import replace

    return replace(base, **changes)


def _slug(value: object) -> str:
    text = str(value).lower()
    return "-".join(
        part
        for part in "".join(char if char.isalnum() else "-" for char in text).split("-")
        if part
    )


def build_diagnostic_cases(
    *,
    text: str = "who",
    matrix: str = "all",
    speed: float = 1.0,
    random_seed: int = 0,
) -> list[DiagnosticCase]:
    """Build the deterministic baseline plus the requested OFAT matrix."""
    if matrix not in {"default", "alignment", "cutter", "all"}:
        raise ValueError(f"unknown matrix {matrix!r}")
    base = build_baseline_settings(speed=speed, random_seed=random_seed)
    cases: list[DiagnosticCase] = []

    def add(axis: str, value: object, settings: DiagnosticSettings) -> None:
        cases.append(DiagnosticCase(f"{len(cases) + 1:03d}", axis, value, settings))

    add("baseline", "default", base)
    if matrix in {"default", "alignment", "all"}:
        for template in CARRIER_TEMPLATES:
            add(
                "phrase-template",
                _slug(template.removesuffix(" {segment}")),
                _case_settings(base, neutral_phrase=template, end_phrase=template),
            )
        for selection in ("neutral", "auto", "end"):
            add("phrase-selection", selection, _case_settings(base, phrase_selection=selection))
        for value in SPEED_VALUES:
            add("speed", value, _case_settings(base, speed=float(value)))
    if matrix in {"default", "cutter", "all"}:
        for value in CUTTER_VALUES:
            add("cutter", value, _case_settings(base, cutter=value))
        for axis, values in PARAMETER_VALUES.items():
            field_name = axis.replace("-", "_")
            for value in values:
                add(axis, value, _case_settings(base, **{field_name: value}))
    _ = text
    return cases


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _settings_lines(settings: dict[str, object]) -> list[str]:
    labels = {
        "enabled": "short_sentence.enabled",
        "resolve_mode": "short_sentence.resolve_mode",
        "phrase_fallback_tries": "short_sentence.phrase_fallback_tries",
        "phrase_selection": "phrase.phrase_selection",
        "neutral_phrase": "phrase.neutral_phrase",
        "end_phrase": "phrase.end_phrase",
        "frame_duration_ms": "phrase.frame_duration_ms",
        "energy_threshold": "phrase.energy_threshold",
        "silence_threshold": "phrase.silence_threshold",
        "min_silence_seconds": "phrase.min_silence_seconds",
        "cutter": "phrase.cutter",
        "search_radius_ms": "phrase.search_radius_ms",
        "context_guard_ms": "phrase.context_guard_ms",
        "analysis_window_ms": "phrase.analysis_window_ms",
        "speed": "generation.speed",
        "random_seed": "random_seed",
        "min_phoneme_length": "short_sentence.min_phoneme_length",
        "phoneme_pretext": "short_sentence.phoneme_pretext",
    }
    lines = []
    for key, value in settings.items():
        suffix = "  [inactive]" if key == "silence_threshold" else ""
        lines.append(f"  {labels.get(key, key)}: {value!r}{suffix}")
    return lines


def format_header(
    args: argparse.Namespace, *, output_metadata: dict[str, object] | None = None
) -> str:
    metadata = output_metadata or {}
    lines = [
        "=== PyKokoro short-sentence diagnostics ===",
        f"text: {args.text!r}",
        f"voice: {args.voice}",
        f"language: {args.lang}",
        f"model_source: {args.model_source}",
        f"model_variant: {args.model_variant}",
        f"model_path: {metadata.get('model_path', args.model_path or '-')}",
        f"voices_path: {metadata.get('voices_path', args.voices_path or '-')}",
        f"provider: {metadata.get('provider', args.provider or 'auto')}",
        f"sample_rate: {metadata.get('sample_rate', SAMPLE_RATE)}",
        f"speed_baseline: {args.speed}",
        f"random_seed: {args.random_seed}",
        "wav_output: disabled",
        "",
        "ONNX inputs:",
        "  input_ids/tokens: actual session metadata is reported after loading",
        "  style/ref_s: actual session metadata is reported after loading",
        "  speed: actual session metadata is reported after loading",
        "",
        "ONNX outputs:",
        "  actual output names, dtype, shape, and timestamp index are reported after loading",
        "",
        "tokenizer:",
        "  actual backend and vocabulary version are reported after loading",
        "",
        "installed packages:",
    ]
    for package in (
        "pykokoro",
        "kokorog2p",
        "lexphon",
        "phrasplit",
        "ssmd",
        "spokenform",
        "audiosig",
        "onnxruntime",
        "numpy",
    ):
        lines.append(f"  {package}: {_version(package)}")
    return "\n".join(lines)


def _token_dict(token: object) -> dict[str, object]:
    if isinstance(token, dict):
        return dict(token)
    result: dict[str, object] = {}
    for name in (
        "text",
        "phonemes",
        "whitespace",
        "char_start",
        "char_end",
        "model_token_count",
        "model_span_token_count",
    ):
        value = getattr(token, name, None)
        if value is not None:
            result[name] = value
    return result


def compute_geometry_rows(tokens: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    cumulative = 0
    for index, token in enumerate(tokens):
        speech = token.get("model_token_count")
        span = token.get("model_span_token_count")
        valid = isinstance(speech, int) and not isinstance(speech, bool) and speech >= 0
        valid = valid and isinstance(span, int) and not isinstance(span, bool) and span >= speech
        row = {
            "index": index,
            "text": token.get("text", ""),
            "phonemes": token.get("phonemes", ""),
            "whitespace": token.get("whitespace", ""),
            "is_target": bool(token.get("is_target")),
            "model_token_count": speech if valid else None,
            "model_span_token_count": span if valid else None,
            "cumulative_start": cumulative if valid else None,
            "cumulative_speech_end": cumulative + speech if valid else None,
            "cumulative_span_end": cumulative + span if valid else None,
        }
        rows.append(row)
        if valid:
            cumulative += span
    return rows


def _ids_difference(left: list[int], right: list[int]) -> int | None:
    for index, (first, second) in enumerate(zip(left, right, strict=False)):
        if first != second:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def _invariants(
    result: ShortSentenceDiagnosticCase,
    *,
    acoustic_ids: list[int],
) -> dict[str, bool]:
    rows = compute_geometry_rows(result.timing_tokens)
    geometry_valid = all(
        row["model_token_count"] is not None and row["model_span_token_count"] is not None
        for row in rows
    )
    span_sum = sum(
        int(row["model_span_token_count"])
        for row in rows
        if row["model_span_token_count"] is not None
    )
    expected_duration = result.metadata.get("expected_pred_duration_count")
    duration_count = 0 if result.pred_dur is None else len(result.pred_dur)
    cursor = result.metadata.get("timing_final_duration_cursor")
    expected_cursor = result.metadata.get("timing_expected_final_duration_cursor")
    target_rows = [row for row in result.timestamped_tokens if row.get("is_target")]
    previous_end = result.metadata.get("previous_token_end_ts")
    target_start = result.metadata.get("target_start_ts")
    target_end = result.metadata.get("target_end_ts")
    next_start = result.metadata.get("next_token_start_ts")
    boundary_ok = (
        previous_end is None or target_start is None or previous_end <= target_start
    ) and (target_end is None or next_start is None or target_end <= next_start)
    left = result.metadata.get("cut_left")
    right = result.metadata.get("cut_right")
    cut_ok = (
        isinstance(left, int)
        and isinstance(right, int)
        and 0 <= left < right <= result.audio_samples
    )
    return {
        "A_phrase_ids": len(result.token_ids)
        == int(result.metadata.get("generated_token_count", -1)),
        "B_token_geometry": geometry_valid,
        "C_phrase_span_sum": geometry_valid
        and span_sum == int(result.metadata.get("generated_token_count", -1)),
        "D_duration_vector_length": expected_duration is not None
        and duration_count == expected_duration,
        "E_timestamp_cursor": cursor is not None and cursor == expected_cursor,
        "F_target_token_exists": any(
            bool(token.get("is_target")) for token in result.timing_tokens
        ),
        "G_target_timestamps_exist": len(target_rows) > 0,
        "H_legal_boundary_geometry": boundary_ok,
        "I_cut_bounds": cut_ok,
        "ids_equal": result.token_ids == acoustic_ids,
    }


def _classify_failure(
    metadata: dict[str, object], *, succeeded: bool
) -> tuple[str | None, str | None]:
    if succeeded:
        return None, None
    stage = metadata.get("failure_stage")
    reason = metadata.get("timing_failure_reason") or metadata.get("cut_failure_reason")
    if not isinstance(stage, str):
        stage = "cut-search" if metadata.get("cutter_invoked") else "timing-token-build"
    return stage, str(reason) if reason is not None else "unknown"


def _session_metadata(backend: Any) -> dict[str, object]:
    session = getattr(backend, "_session", None)
    inputs = []
    audio_generator = getattr(backend, "_audio_generator", None)
    outputs = []
    providers: list[str] = []
    if session is not None:
        for collection, target in (("get_inputs", inputs), ("get_outputs", outputs)):
            getter = getattr(session, collection, None)
            if callable(getter):
                for item in getter() or []:
                    target.append(
                        {
                            "name": str(getattr(item, "name", "")),
                            "dtype": str(getattr(item, "type", "")),
                            "shape": list(getattr(item, "shape", []) or []),
                        }
                    )
    if session is not None:
        provider_getter = getattr(session, "get_providers", None)
        if callable(provider_getter):
            providers = [str(provider) for provider in provider_getter() or []]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "timestamp_output_index": getattr(audio_generator, "_timestamp_output_index", None),
        "timestamp_output_name": next(
            (
                item["name"]
                for item in outputs
                if str(item["name"]).lower()
                in {"pred_dur", "pred_duration", "durations", "duration"}
            ),
            None,
        ),
        "has_timestamp_output": bool(getattr(audio_generator, "_has_timestamp_output", False)),
        "model_path": str(getattr(backend, "_model_path", "-")),
        "voices_path": str(getattr(backend, "_voices_path", "-")),
        "provider": providers[0] if providers else getattr(backend, "_provider", "auto"),
        "providers": providers,
        "tokenizer_backend": getattr(
            getattr(getattr(backend, "tokenizer", None), "config", None), "backend", None
        ),
        "vocab_version": getattr(backend, "_vocab_version", "unknown"),
    }


def _runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    from pykokoro import GenerationConfig, PipelineConfig
    from pykokoro.pipeline import KokoroPipeline
    from pykokoro.short_sentence_handler import ShortSentenceConfig

    config = PipelineConfig(
        voice=args.voice,
        model_source=args.model_source,
        model_variant=args.model_variant,
        model_path=args.model_path,
        voices_path=args.voices_path,
        provider=args.provider,
        generation=GenerationConfig(lang=args.lang, speed=args.speed, random_seed=args.random_seed),
        short_sentence_config=ShortSentenceConfig(enabled=False),
        return_trace=True,
        retain_segment_audio=True,
    )
    pipeline = KokoroPipeline(config)
    pipeline.warmup()
    prepared = pipeline.prepare_units(args.text, unit="paragraph")
    if prepared._prepared is None or not prepared._prepared.phoneme_segments:
        prepared.close()
        pipeline.close()
        raise RuntimeError("the input did not produce a phoneme segment")
    segment = prepared._prepared.phoneme_segments[0]
    backend = pipeline._kokoro
    if backend is None or getattr(backend, "_audio_generator", None) is None:
        prepared.close()
        pipeline.close()
        raise RuntimeError("the selected runtime does not expose the ONNX AudioGenerator probe")
    return pipeline, prepared, segment, backend


def run_phrase_probe(
    case: DiagnosticCase,
    *,
    segment: Any,
    backend: Any,
    context_phonemizer: Any,
    voice: str | None = None,
) -> ShortSentenceDiagnosticCase:
    from pykokoro.audio_generator import (
        _join_timestamps,
        _record_short_sentence_timing_alignment,
        populate_short_sentence_boundary_metadata,
    )
    from pykokoro.short_sentence_handler import (
        PhraseResolveMode,
        ShortSentenceConfig,
        apply_short_sentence_mode,
        cut_short_sentence_phrase_audio,
    )
    from pykokoro.types import Trace

    settings = case.settings
    result = ShortSentenceDiagnosticCase(case.case_id, case.axis, case.value, settings.as_dict())
    started = time.perf_counter()
    phrase_language = segment.lang
    context_result = context_phonemizer(
        case.phrase_text.replace("{segment}", segment.text), phrase_language
    )
    result.context_tokens = [_token_dict(token) for token in getattr(context_result, "tokens", ())]
    result.metadata.update(
        {
            "context_result_id_count": len(getattr(context_result, "ids", ()) or ()),
            "context_result_ids": list(getattr(context_result, "ids", ()) or ()),
            "parameter_stage": case.parameter_stage,
            "active_silence_threshold": False,
        }
    )
    mode = PhraseResolveMode(
        phrase_selection=settings.phrase_selection,
        neutral_phrase=settings.neutral_phrase,
        end_phrase=settings.end_phrase,
        frame_duration_ms=settings.frame_duration_ms,
        energy_threshold=settings.energy_threshold,
        silence_threshold=settings.silence_threshold,
        min_silence_seconds=settings.min_silence_seconds,
        cutter=settings.cutter,
        search_radius_ms=settings.search_radius_ms,
        context_guard_ms=settings.context_guard_ms,
        analysis_window_ms=settings.analysis_window_ms,
    )
    config = ShortSentenceConfig(
        min_phoneme_length=settings.min_phoneme_length,
        phoneme_pretext=settings.phoneme_pretext,
        enabled=True,
        resolve_mode="phrase",
        resolve_modes={"phrase": mode},
        phrase_fallback_tries=0,
    )
    tokenizer = backend.tokenizer
    app = apply_short_sentence_mode(
        segment,
        segment.phonemes,
        list(segment.tokens),
        config,
        tokenizer.tokenize,
        rng=random.Random(settings.random_seed),
        context_phonemizer=context_phonemizer,
    )
    result.phrase_text = case.phrase_text.replace("{segment}", segment.text)
    result.phonemes = app.phonemes
    result.token_ids = list(app.tokens)
    result.metadata.update(app.metadata or {})
    result.timing_tokens = list(result.metadata.get("timing_tokens", []))
    result.metadata["raw_context_tokens"] = result.context_tokens
    result.metadata["context_geometry_preserved"] = bool(
        result.context_tokens
        and len(result.context_tokens) == len(result.timing_tokens)
        and all(
            context.get("model_token_count") == timing.get("model_token_count")
            and context.get("model_span_token_count") == timing.get("model_span_token_count")
            for context, timing in zip(result.context_tokens, result.timing_tokens, strict=True)
        )
    )
    acoustic_ids = list(tokenizer.tokenize(result.phonemes))
    result.metadata.update(
        {
            "acoustic_tokenizer_id_count": len(acoustic_ids),
            "acoustic_tokenizer_ids": acoustic_ids,
            "ids_equal": result.token_ids == acoustic_ids,
            "context_result_ids_equal": list(result.metadata.get("context_result_ids", []))
            == acoustic_ids,
            "first_id_difference_index": _ids_difference(result.token_ids, acoustic_ids),
            "phoneme_char_count": len(result.phonemes),
        }
    )
    trace = Trace()
    audio_generator = backend._audio_generator
    voice_style = backend.resolve_voice_style(voice or segment.voice_name or "af_sarah")
    audio, pred_dur = audio_generator._run_onnx(
        result.phonemes, voice_style, settings.speed, trace=trace, tokens=result.token_ids
    )
    result.audio_samples = int(np.asarray(audio).size)
    result.pred_dur = (
        None if pred_dur is None else [float(value) for value in np.asarray(pred_dur).reshape(-1)]
    )
    inference = trace.inference[-1] if trace.inference else {}
    result.runtime_ms = float(inference.get("runtime_ms", (time.perf_counter() - started) * 1000.0))
    result.cache_hit = bool(inference.get("cache_hit", False))
    result.metadata.update(
        {
            "pred_dur_present": pred_dur is not None,
            "pred_dur_dtype": None if pred_dur is None else str(np.asarray(pred_dur).dtype),
            "pred_dur_shape": None if pred_dur is None else list(np.asarray(pred_dur).shape),
            "pred_dur_values": result.pred_dur,
            "audio_samples": result.audio_samples,
            "runtime_ms": result.runtime_ms,
            "cache_hit": result.cache_hit,
        }
    )
    _record_short_sentence_timing_alignment(
        result.metadata,
        result.timing_tokens,
        pred_duration_count=None if pred_dur is None else len(result.pred_dur or []),
    )
    duration_valid = (
        pred_dur is not None
        and result.metadata.get("timing_alignment_complete") is True
        and result.metadata.get("pred_duration_count")
        == result.metadata.get("expected_pred_duration_count")
    )
    if pred_dur is None:
        result.metadata.update(
            {
                "timing_failure_detail": "missing-duration-output",
                "timing_failure_reason": "missing-duration-output",
                "failure_stage": "timing-token-build",
            }
        )
    elif result.metadata.get("pred_duration_count") != result.metadata.get(
        "expected_pred_duration_count"
    ):
        result.metadata.update(
            {
                "timing_failure_detail": "duration-position-count-mismatch",
                "timing_failure_reason": "timing-model-position-mismatch",
                "failure_stage": "timing-alignment",
            }
        )
    if duration_valid:
        timestamped = _join_timestamps(
            result.timing_tokens, np.asarray(pred_dur), strict=True, metadata=result.metadata
        )
        result.timestamped_tokens = timestamped
        result.metadata["join_attempted"] = True
        result.metadata["join_complete"] = bool(timestamped)
        populate_short_sentence_boundary_metadata(result.metadata, timestamped)
        result.metadata["target_timestamp_count"] = sum(
            1 for token in timestamped if token.get("is_target") and "start_ts" in token
        )
        if not timestamped:
            result.metadata.update(
                {
                    "timing_failure_reason": "timestamp-join-incomplete",
                    "failure_stage": "timestamp-join",
                }
            )
        elif result.metadata.get("target_timestamp_count", 0) == 0:
            result.metadata.update(
                {
                    "timing_failure_reason": "missing-target-timestamps",
                    "failure_stage": "target-boundary",
                }
            )
        if timestamped and result.metadata.get("target_timestamp_count", 0) > 0:
            cut = cut_short_sentence_phrase_audio(audio, result.metadata)
            result.metadata["cutter_success"] = cut is not None
            result.metadata["output_sample_count"] = 0 if cut is None else int(np.asarray(cut).size)
    else:
        result.metadata["join_attempted"] = False
        result.metadata["join_complete"] = False
    result.invariants = _invariants(result, acoustic_ids=acoustic_ids)
    result.metadata["parameter_relevant"] = _parameter_is_relevant(
        case.parameter_stage, result.metadata
    )
    result.succeeded = bool(result.metadata.get("cutter_success"))
    result.failure_stage, result.failure_reason = _classify_failure(
        result.metadata, succeeded=result.succeeded
    )
    result.metadata["failure_stage"] = result.failure_stage
    result.metadata["failure_reason"] = result.failure_reason
    return result


def _parameter_is_relevant(parameter_stage: str, metadata: dict[str, object]) -> bool:
    failure_stage = metadata.get("failure_stage")
    if not isinstance(failure_stage, str):
        return True
    order = {
        "timing-token-build": 0,
        "timing-alignment": 1,
        "timestamp-join": 2,
        "target-boundary": 3,
        "cut-window": 4,
        "cut-search": 5,
        "cut-validation": 6,
    }
    stage_order = {"context-g2p": 0, "onnx-duration": 1, "cut-search": 5}
    return order.get(failure_stage, 0) >= stage_order.get(parameter_stage, 0)


def _format_seconds(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "-"
    return f"{float(value):.6f}"


def format_case_report(result: ShortSentenceDiagnosticCase) -> str:
    metadata = result.metadata
    rows = compute_geometry_rows(result.timing_tokens)
    lines = [
        "=" * 80,
        f"CASE {result.case_id}  axis={result.axis} value={result.value}",
        "=" * 80,
        "",
        "RESULT",
        f"  success: {result.succeeded}",
        f"  failure_stage: {result.failure_stage or '-'}",
        f"  failure_reason: {result.failure_reason or '-'}",
        f"  cutter_reached: {metadata.get('cutter_reached', False)}",
        f"  parameter_stage: {PARAMETER_STAGE.get(result.axis, 'context-g2p')}",
        f"  parameter_relevant: {metadata.get('parameter_relevant', True)}",
        "",
        "SETTINGS",
        *_settings_lines(result.settings),
        "",
        "PHRASE",
        f"  template: {result.settings.get('neutral_phrase')!r}",
        f"  text: {result.phrase_text!r}",
        f"  phonemes: {result.phonemes!r}",
        f"  phoneme_char_count: {len(result.phonemes)}",
        f"  generated_token_count: {len(result.token_ids)}",
        f"  token_ids: {result.token_ids}",
        "",
        "RAW CONTEXT G2P TOKENS",
        "  idx text phonemes whitespace char_start char_end model_token_count model_span_token_count",
    ]
    for index, token in enumerate(result.context_tokens):
        lines.append(
            f"  {index:>3} {token.get('text', '')!r} {token.get('phonemes', '')!r} "
            f"{token.get('whitespace', '')!r} {token.get('char_start', '-')} {token.get('char_end', '-')} "
            f"{token.get('model_token_count', None)} {token.get('model_span_token_count', None)}"
        )
    lines.extend(
        [
            "",
            "TIMING TOKENS",
            "  idx target text phonemes whitespace model_count span_count cum_start cum_speech_end cum_span_end",
        ]
    )
    for row in rows:
        lines.append(
            f"  {row['index']:>3} {str(row['is_target']):<6} {row['text']!r} {row['phonemes']!r} "
            f"{row['whitespace']!r} {row['model_token_count']} {row['model_span_token_count']} "
            f"{row['cumulative_start']} {row['cumulative_speech_end']} {row['cumulative_span_end']}"
        )
    lines.extend(
        [
            "",
            "MODEL DURATION OUTPUT",
            f"  present: {metadata.get('pred_dur_present', False)}",
            f"  dtype: {metadata.get('pred_dur_dtype')}",
            f"  shape: {metadata.get('pred_dur_shape')}",
            f"  count: {metadata.get('pred_duration_count', 0)}",
            f"  expected_count: {metadata.get('expected_pred_duration_count')}",
            f"  count_delta: {metadata.get('pred_duration_count_delta')}",
            f"  values: {result.pred_dur}",
            "",
            "ALIGNMENT",
            f"  timing_token_count: {metadata.get('timing_token_count', 0)}",
            f"  timing_model_position_count: {metadata.get('timing_model_position_count', 0)}",
            f"  generated_token_count: {metadata.get('generated_token_count', len(result.token_ids))}",
            f"  timing_model_position_delta: {metadata.get('timing_model_position_delta')}",
            f"  timing_alignment_complete: {metadata.get('timing_alignment_complete', False)}",
            f"  context_geometry_preserved: {metadata.get('context_geometry_preserved', False)}",
            f"  context_result_id_count: {metadata.get('context_result_id_count')}",
            f"  acoustic_tokenizer_id_count: {metadata.get('acoustic_tokenizer_id_count')}",
            f"  ids_equal: {metadata.get('ids_equal')}",
            "",
            "TIMESTAMPS",
            f"  join_attempted: {metadata.get('join_attempted', False)}",
            f"  join_complete: {metadata.get('join_complete', False)}",
            f"  target_timestamp_count: {metadata.get('target_timestamp_count', 0)}",
            f"  target_start_ts: {_format_seconds(metadata.get('target_start_ts'))}",
            f"  target_end_ts: {_format_seconds(metadata.get('target_end_ts'))}",
            f"  previous_token_end_ts: {_format_seconds(metadata.get('previous_token_end_ts'))}",
            f"  next_token_start_ts: {_format_seconds(metadata.get('next_token_start_ts'))}",
        ]
    )
    for index, token in enumerate(result.timestamped_tokens):
        lines.append(
            f"  {index:>3} target={token.get('is_target', False)} text={token.get('text', '')!r} "
            f"start_s={_format_seconds(token.get('start_ts'))} "
            f"speech_end_s={_format_seconds(token.get('speech_end_ts', token.get('end_ts')))} "
            f"end_s={_format_seconds(token.get('end_ts'))}"
        )
    lines.extend(
        [
            "",
            "CUT",
            f"  invoked: {metadata.get('cutter_invoked', False)}",
            f"  reached: {metadata.get('cutter_reached', False)}",
            f"  configured_cutter: {metadata.get('cutter', result.settings.get('cutter'))}",
            f"  actual_strategy: {metadata.get('cut_strategy', '-')}",
            f"  cut_left: {metadata.get('cut_left', '-')}",
            f"  cut_right: {metadata.get('cut_right', '-')}",
            f"  output_samples: {metadata.get('output_sample_count', '-')}",
            "",
            "INVARIANTS",
        ]
    )
    lines.extend(
        f"  {name}: {'PASS' if value else 'FAIL'}" for name, value in result.invariants.items()
    )
    lines.extend(
        [
            "",
            "INFERENCE",
            f"  audio_samples: {result.audio_samples}",
            f"  runtime_ms: {result.runtime_ms:.3f}",
            f"  cache_hit: {result.cache_hit}",
        ]
    )
    return "\n".join(lines)


def format_summary(results: list[ShortSentenceDiagnosticCase]) -> str:
    lines = [
        "SUMMARY",
        "ID   Axis                 Value                 Result Stage              Geometry Durations Timestamps Cutter Cache",
    ]
    for result in results:
        metadata = result.metadata
        geometry = f"{metadata.get('timing_model_position_count', 0)}/{metadata.get('generated_token_count', len(result.token_ids))}"
        durations = f"{metadata.get('pred_duration_count', 0)}/{metadata.get('expected_pred_duration_count', 0)}"
        cutter = metadata.get("cut_strategy", "not-reached") if result.succeeded else "not-reached"
        lines.append(
            f"{result.case_id:<4} {result.axis:<20} {str(result.value):<21} "
            f"{'PASS' if result.succeeded else 'FAIL':<7} {str(result.failure_stage or '-'): <18} "
            f"{geometry:<8} {durations:<9} {len(result.timestamped_tokens):<10} {str(cutter):<20} "
            f"{'hit' if result.cache_hit else 'miss'}"
        )
    stages = Counter(result.failure_stage for result in results if result.failure_stage)
    reasons = Counter(result.failure_reason for result in results if result.failure_reason)
    valid_geometry = sum(result.invariants.get("C_phrase_span_sum", False) for result in results)
    valid_duration = sum(
        result.invariants.get("D_duration_vector_length", False) for result in results
    )
    lines.extend(
        [
            "",
            "AGGREGATE",
            f"  cases_total: {len(results)}",
            f"  cases_success: {sum(result.succeeded for result in results)}",
            f"  cases_failed: {sum(not result.succeeded for result in results)}",
            f"  failures_by_stage: {dict(stages)}",
            f"  failures_by_reason: {dict(reasons)}",
            f"  cases_with_exact_geometry: {valid_geometry}",
            f"  cases_with_valid_pred_dur_count: {valid_duration}",
            f"  cases_with_timestamp_join: {sum(bool(result.timestamped_tokens) for result in results)}",
            f"  cases_with_target_timestamps: {sum(result.invariants.get('G_target_timestamps_exist', False) for result in results)}",
            f"  cases_reaching_cutter: {sum(bool(result.metadata.get('cutter_reached')) for result in results)}",
            "",
            "DIAGNOSIS",
            f"  frontend_geometry_successes: {valid_geometry}/{len(results)}",
            f"  duration_output_present: {sum(result.pred_dur is not None for result in results)}/{len(results)}",
            f"  duration_count_match: {valid_duration}/{len(results)}",
            f"  cutter_reached: {sum(bool(result.metadata.get('cutter_reached')) for result in results)}/{len(results)}",
        ]
    )
    if stages:
        lines.extend(["", f"  Primary failing stage: {stages.most_common(1)[0][0]}"])
        if stages.most_common(1)[0][0] in {"timing-alignment", "timestamp-join", "target-boundary"}:
            lines.append("  Cutter tuning is currently not actionable.")
    return "\n".join(lines)


def _print_dry_run(args: argparse.Namespace, cases: list[DiagnosticCase]) -> None:
    print(format_header(args))
    print("\nDRY RUN CASES")
    for case in cases:
        print(f"CASE {case.case_id} axis={case.axis} value={case.value}")
        print(f"  parameter_stage: {case.parameter_stage}")
        print("\n".join(_settings_lines(case.settings.as_dict())))


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--text", default="who")
    parser.add_argument("--voice", default="af_sarah")
    parser.add_argument("--lang", default="en-us")
    parser.add_argument("--model-source", choices=("github", "huggingface"), default="huggingface")
    parser.add_argument("--model-variant", default="v1.0")
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--voices-path", type=Path)
    parser.add_argument("--provider")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument(
        "--matrix", choices=("default", "alignment", "cutter", "all"), default="all"
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--case")
    parser.add_argument("--axis")
    parser.add_argument("--stop-after-first-success", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument("--dry-run", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    _add_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.WARNING))
    try:
        cases = build_diagnostic_cases(
            text=args.text, matrix=args.matrix, speed=args.speed, random_seed=args.random_seed
        )
        if args.case is not None:
            cases = [case for case in cases if case.case_id == args.case]
            if not cases:
                parser.error(f"unknown case {args.case!r}")
        if args.axis is not None:
            cases = [case for case in cases if case.axis == args.axis]
            if not cases:
                parser.error(f"unknown or unavailable axis {args.axis!r}")
    except ValueError as exc:
        parser.error(str(exc))
    if args.dry_run:
        _print_dry_run(args, cases)
        return 0
    pipeline = prepared = None
    results: list[ShortSentenceDiagnosticCase] = []
    try:
        pipeline, prepared, segment, backend = _runtime(args)

        def context_phonemizer(text: str, language: str) -> Any:
            return pipeline.g2p.phonemize_context(text, language, pipeline.config)

        session_metadata = _session_metadata(backend)
        print(format_header(args, output_metadata=session_metadata))
        print("\nRUNTIME")
        print(json.dumps(session_metadata, indent=2, ensure_ascii=False))
        for case in cases:
            result = run_phrase_probe(
                case,
                segment=segment,
                backend=backend,
                context_phonemizer=context_phonemizer,
                voice=args.voice,
            )
            results.append(result)
            print(format_case_report(result))
            if args.stop_after_first_success and result.succeeded:
                break
        print(format_summary(results))
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(
                json.dumps(
                    {
                        "schema": SCHEMA,
                        "header": vars(args),
                        "runtime": session_metadata,
                        "cases": [result.to_dict() for result in results],
                    },
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n",
                encoding="utf-8",
            )
    finally:
        if prepared is not None:
            prepared.close()
        if pipeline is not None:
            pipeline.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
