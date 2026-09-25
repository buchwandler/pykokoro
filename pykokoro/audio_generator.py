"""Kokoro inference, short-sentence handling, and waveform diagnostics."""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import random
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from audiosig import trim as trim_audio

from ._onnxvoice import (
    KokoroTimingPayload,
    classify_timing,
    summarize_inference,
    timing_payload_from_result,
)
from .constants import MAX_PHONEME_LENGTH, SAMPLE_RATE
from .exceptions import ConfigurationError
from .runtime_protocol import KokoroInferenceRuntime
from .short_sentence_handler import (
    SHORT_SENTENCE_META_KEY,
    apply_short_sentence_mode,
    build_short_sentence_phrase_retry,
    cut_short_sentence_phrase_audio,
)
from .tokenizer import Tokenizer
from .types import G2PAlignmentToken, PhonemeSegment, WordTiming, _exact_timing_geometry
from .voice_level import VoiceLevelConfig, apply_voice_level_calibration
from .voice_manager import normalize_voice_style

if TYPE_CHECKING:
    from .short_sentence_handler import ShortSentenceConfig
    from .types import Trace

logger = logging.getLogger(__name__)


def _waveform_metrics(audio: np.ndarray) -> dict[str, float | int | bool]:
    """Return compact, JSON-friendly metrics for a segment waveform."""
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    finite = np.isfinite(values)
    finite_values = values[finite]
    differences = np.diff(finite_values) if finite_values.size else np.array([], dtype=np.float64)
    peak = float(np.max(np.abs(finite_values))) if finite_values.size else 0.0
    return {
        "samples": int(values.size),
        "finite": bool(finite.all()),
        "min": float(np.min(finite_values)) if finite_values.size else 0.0,
        "max": float(np.max(finite_values)) if finite_values.size else 0.0,
        "peak": peak,
        "mean": float(np.mean(finite_values)) if finite_values.size else 0.0,
        "dc_mean": float(np.mean(finite_values)) if finite_values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(finite_values)))) if finite_values.size else 0.0,
        "std": float(np.std(finite_values)) if finite_values.size else 0.0,
        "max_adjacent_jump": (float(np.max(np.abs(differences))) if differences.size else 0.0),
        "non_finite_samples": int((~finite).sum()),
        "clipped_samples": int(np.count_nonzero(finite & (np.abs(values) >= 1.0))),
    }


def _stationary_noise_metrics(audio: np.ndarray) -> dict[str, float]:
    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    frame_size = round(SAMPLE_RATE * 0.05)
    hop_size = round(SAMPLE_RATE * 0.025)
    if values.size < frame_size:
        return {
            "zcr": 0.0,
            "centroid": 0.0,
            "centroid_cv": 0.0,
            "high_band": 0.0,
            "flux": 0.0,
            "frame_rms_cv": 0.0,
        }
    frames = np.asarray(
        [
            values[start : start + frame_size]
            for start in range(0, values.size - frame_size + 1, hop_size)
        ]
    )
    spectra = np.square(np.abs(np.fft.rfft(frames * np.hanning(frame_size), axis=1)))
    spectra[:, 0] = 0.0
    frequencies = np.fft.rfftfreq(frame_size, 1.0 / SAMPLE_RATE)
    totals = np.maximum(np.sum(spectra, axis=1), np.finfo(np.float64).eps)
    normalized = spectra / totals[:, None]
    centroid = np.sum(normalized * frequencies[None, :], axis=1)
    flux = 0.5 * np.linalg.norm(np.diff(normalized, axis=0), axis=1)
    return {
        "zcr": float(np.mean(np.mean(np.diff(np.signbit(frames), axis=1), axis=1))),
        "centroid": float(np.mean(centroid)),
        "centroid_cv": float(np.std(centroid) / max(float(np.mean(centroid)), 1e-12)),
        "high_band": float(np.sum(spectra[:, frequencies >= 4000]) / np.sum(spectra)),
        "flux": float(np.mean(flux)),
        "frame_rms_cv": float(
            np.std(np.sqrt(np.mean(np.square(frames), axis=1)))
            / max(float(np.mean(np.sqrt(np.mean(np.square(frames), axis=1)))), 1e-12)
        ),
    }


def _is_stationary_broadband_noise(audio: np.ndarray) -> tuple[bool, dict[str, float]]:
    metrics = _stationary_noise_metrics(audio)
    is_noise = np.asarray(audio).size / SAMPLE_RATE >= 1.0 and all(
        (
            metrics["zcr"] > 0.45,
            metrics["centroid"] > 0.23 * SAMPLE_RATE,
            metrics["centroid_cv"] < 0.05,
            metrics["high_band"] > 0.65,
            metrics["frame_rms_cv"] < 0.08,
            metrics["flux"] < 0.05,
        )
    )
    return is_noise, metrics


# Model source type
ModelSource = Literal["huggingface", "github"]


class AudioGenerator:
    """Generate audio using a producer-neutral Kokoro inference runtime.

    Token preparation, voice/style selection, short-sentence handling, timing
    interpretation, and inference caching remain owned by PyKokoro.
    """

    def __init__(
        self,
        runtime: KokoroInferenceRuntime | Any | None = None,
        tokenizer: Tokenizer | None = None,
        model_source: ModelSource = "huggingface",
        short_sentence_config: ShortSentenceConfig | None = None,
        waveform_validation: Literal["off", "warn", "strict"] = "off",
        inference_audio_diagnostics: bool = False,
        inference_cache_enabled: bool = True,
        inference_cache_max_bytes: int = 128 * 1024 * 1024,
        *,
        session: Any | None = None,
    ):
        """Initialize the audio generator with an OnnxVoice-compatible runtime."""
        if runtime is None:
            runtime = session
        if runtime is None:
            raise ValueError("runtime is required")
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if inference_cache_max_bytes < 0:
            raise ValueError("inference_cache_max_bytes must be non-negative")
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._model_source = model_source
        self._short_sentence_config = short_sentence_config
        self._context_phonemizer: Callable[[str, str], Any] | None = None
        if waveform_validation not in {"off", "warn", "strict"}:
            raise ValueError(f"Unsupported waveform validation mode: {waveform_validation!r}")
        self._waveform_validation = waveform_validation
        self._inference_audio_diagnostics = inference_audio_diagnostics
        self._inference_cache_enabled = inference_cache_enabled and inference_cache_max_bytes > 0
        self._inference_cache_max_bytes = inference_cache_max_bytes
        self._inference_cache: OrderedDict[
            bytes, tuple[np.ndarray, KokoroTimingPayload | None, int]
        ] = OrderedDict()
        self._inference_cache_bytes = 0
        self._inference_call_number = 0
        runtime_support = getattr(runtime, "supports_timings", None)
        self._timestamp_support_declared = isinstance(runtime_support, bool)
        self._timestamp_support: bool | None = (
            runtime_support if self._timestamp_support_declared else None
        )
        self._timestamp_support_observed = False
        self._reported_missing_timestamp_output = False

    def _tokenize_phonemes(self, phonemes: str) -> list[int]:
        trimmed = phonemes[:MAX_PHONEME_LENGTH]
        return self._tokenizer.tokenize(trimmed)

    @staticmethod
    def _voice_style_index(voicepack_length: int, phoneme_count: int) -> int:
        """Return the Kokoro voicepack row for an effective phoneme length."""
        return min(
            max(phoneme_count - 1, 0),
            MAX_PHONEME_LENGTH - 1,
            max(voicepack_length - 1, 0),
        )

    def _select_voice_style(self, voice_style: np.ndarray, phoneme_count: int) -> np.ndarray:
        voice_style = normalize_voice_style(voice_style, expected_length=None)
        style_idx = self._voice_style_index(voice_style.shape[0], phoneme_count)
        voice_style_indexed = voice_style[style_idx]
        if voice_style_indexed.ndim == 1:
            voice_style_indexed = voice_style_indexed[None, :]
        return voice_style_indexed

    @staticmethod
    def _inference_cache_key(
        inputs: dict[str, np.ndarray | list[int]],
        runtime_identity: object = None,
    ) -> bytes:
        """Build a digest from producer inputs and runtime identity."""
        digest = hashlib.blake2b(digest_size=20)
        digest.update(repr(runtime_identity).encode("utf-8"))
        digest.update(b"\0")
        for name in sorted(inputs):
            array = np.ascontiguousarray(np.asarray(inputs[name]))
            digest.update(name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(b"\0")
            digest.update(memoryview(array).cast("B"))
        return digest.digest()

    def _get_cached_inference(
        self, key: bytes
    ) -> tuple[np.ndarray, KokoroTimingPayload | None] | None:
        if not self._inference_cache_enabled:
            return None
        cached = self._inference_cache.get(key)
        if cached is None:
            return None
        audio, timing, _ = cached
        self._inference_cache.move_to_end(key)
        return audio.copy(), timing

    def _put_cached_inference(
        self, key: bytes, audio: np.ndarray, timing: KokoroTimingPayload | None
    ) -> None:
        if not self._inference_cache_enabled:
            return
        cached_audio = np.array(audio, copy=True)
        cached_timing = timing
        entry_bytes = cached_audio.nbytes + (
            cached_timing.values.nbytes if cached_timing is not None else 0
        )
        if entry_bytes > self._inference_cache_max_bytes:
            return
        previous = self._inference_cache.pop(key, None)
        if previous is not None:
            self._inference_cache_bytes -= previous[2]
        self._inference_cache[key] = (cached_audio, cached_timing, entry_bytes)
        self._inference_cache_bytes += entry_bytes
        while self._inference_cache_bytes > self._inference_cache_max_bytes:
            _, (_, _, removed_bytes) = self._inference_cache.popitem(last=False)
            self._inference_cache_bytes -= removed_bytes

    def clear_inference_cache(self) -> None:
        """Release all cached raw inference outputs."""
        self._inference_cache.clear()
        self._inference_cache_bytes = 0

    def close(self) -> None:
        """Release generator-owned inference cache state."""
        self.clear_inference_cache()

    def _record_inference(
        self,
        trace: Trace | None,
        *,
        effective_phonemes: str,
        tokens: list[int],
        inputs: dict[str, np.ndarray | list[int]],
        audio: np.ndarray,
        runtime_s: float,
        cache_hit: bool,
        cache_key: bytes,
        attempt_kind: str,
    ) -> None:
        self._inference_call_number += 1
        if trace is None:
            return
        audio_samples = int(np.asarray(audio).size)
        audio_seconds = audio_samples / SAMPLE_RATE if audio_samples else 0.0
        trace.inference.append(
            {
                "call_number": self._inference_call_number,
                "cache_hit": cache_hit,
                "cache_key_short": cache_key.hex()[:12],
                "runtime_ms": runtime_s * 1000.0,
                "audio_samples": audio_samples,
                "audio_seconds": audio_seconds,
                "rtf": runtime_s / audio_seconds if audio_seconds else None,
                "effective_phonemes": effective_phonemes,
                "phoneme_count": len(effective_phonemes),
                "token_ids": list(tokens),
                "token_count": len(tokens),
                "attempt_kind": attempt_kind,
                "inputs": {
                    name: {
                        "dtype": str(np.asarray(value).dtype),
                        "shape": list(np.asarray(value).shape),
                    }
                    for name, value in inputs.items()
                },
            }
        )
        counters = trace.counters
        counters["logical_phoneme_segments"] = counters.get("logical_phoneme_segments", 0)
        if attempt_kind == "initial":
            counters["initial_onnx_calls"] = counters.get("initial_onnx_calls", 0) + 1
        elif attempt_kind == "retry":
            counters["short_sentence_retry_calls"] = (
                counters.get("short_sentence_retry_calls", 0) + 1
            )
        elif attempt_kind == "fallback":
            counters["fallback_onnx_calls"] = counters.get("fallback_onnx_calls", 0) + 1

    def _run_onnx(
        self,
        phonemes: str,
        voice_style: np.ndarray,
        speed: float,
        trace: Trace | None = None,
        *,
        attempt_kind: str = "initial",
        tokens: list[int] | None = None,
        seed: int | None = None,
    ) -> tuple[np.ndarray, KokoroTimingPayload | None]:
        effective_phonemes = phonemes[:MAX_PHONEME_LENGTH]
        effective_tokens = (
            list(tokens)
            if tokens is not None and len(phonemes) <= MAX_PHONEME_LENGTH
            else self._tokenizer.tokenize(effective_phonemes)
        )
        normalized_voice_style = normalize_voice_style(voice_style, expected_length=None)
        style_idx = self._voice_style_index(
            normalized_voice_style.shape[0], len(effective_phonemes)
        )
        voice_style_indexed = normalized_voice_style[style_idx]
        if voice_style_indexed.ndim == 1:
            voice_style_indexed = voice_style_indexed[None, :]
        inputs: dict[str, np.ndarray | list[int]] = {
            "token_ids": effective_tokens,
            "style": voice_style_indexed,
            "speed": np.asarray([speed], dtype=np.float32),
        }
        runtime_identity = getattr(self._runtime, "cache_identity", None)
        cache_key = self._inference_cache_key(inputs, runtime_identity)
        cached = self._get_cached_inference(cache_key)
        result: Any | None = None
        if cached is not None:
            audio, timing = cached
            runtime_s = 0.0
            cache_hit = True
        else:
            started = time.perf_counter()
            result = self._runtime.infer(
                effective_tokens,
                style=voice_style_indexed,
                speed=speed,
                seed=seed,
            )
            runtime_s = time.perf_counter() - started
            audio = np.asarray(getattr(result, "audio", result), dtype=np.float32).reshape(-1)
            timing = timing_payload_from_result(
                result,
                len(effective_tokens),
                declared_layout=getattr(self._runtime, "timing_layout", None),
            )
            self._put_cached_inference(cache_key, audio, timing)
            cache_hit = False
        if timing is not None:
            self._timestamp_support = True
            self._timestamp_support_observed = True
        elif self._timestamp_support is None:
            self._timestamp_support = False
            self._timestamp_support_observed = True
        self._record_inference(
            trace,
            effective_phonemes=effective_phonemes,
            tokens=effective_tokens,
            inputs=inputs,
            audio=audio,
            runtime_s=runtime_s,
            cache_hit=cache_hit,
            cache_key=cache_key,
            attempt_kind=attempt_kind,
        )
        timing_summary = None
        output_summary: dict[str, Any] = {}
        if result is not None:
            timing_summary, output_summary = summarize_inference(
                result, timing, input_token_count=len(effective_tokens)
            )
        elif timing is not None:
            timing_summary = {
                "shape": [int(size) for size in timing.values.shape],
                "dtype": str(timing.values.dtype),
                "count": timing.raw_count,
                "layout": timing.layout,
            }
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "inference.finish cache_hit=%s attempt=%s phonemes=%d tokens=%d "
                "samples=%d runtime_ms=%.3f timing_layout=%s timing_values=%d "
                "expected_timing_count=%s timestamp_support=%s outputs=%s",
                cache_hit,
                attempt_kind,
                len(effective_phonemes),
                len(effective_tokens),
                int(np.asarray(audio).size),
                runtime_s * 1000.0,
                timing.layout if timing is not None else "missing",
                timing.raw_count if timing is not None else 0,
                timing.expected_raw_count if timing is not None else None,
                (
                    "observed"
                    if self._timestamp_support_observed
                    else "declared"
                    if self._timestamp_support_declared
                    else "unknown"
                ),
                output_summary,
            )
        if trace is not None:
            style_values = np.asarray(voice_style_indexed, dtype=np.float32)
            diagnostic = trace.inference[-1]
            diagnostic.update(
                {
                    "style_row": style_idx,
                    "style": {
                        "shape": list(voice_style_indexed.shape),
                        "min": float(np.min(style_values)) if style_values.size else 0.0,
                        "max": float(np.max(style_values)) if style_values.size else 0.0,
                        "mean": float(np.mean(style_values)) if style_values.size else 0.0,
                        "std": float(np.std(style_values)) if style_values.size else 0.0,
                    },
                    "audio": {"samples": int(np.asarray(audio).size)},
                    "timings": timing_summary,
                    "outputs": output_summary,
                    "speed": {
                        "value": float(speed),
                        "dtype": str(np.asarray(inputs["speed"]).dtype),
                    },
                }
            )
            if self._inference_audio_diagnostics:
                noise_detected, noise_metrics = _is_stationary_broadband_noise(audio)
                audio_metrics = _waveform_metrics(audio)
                audio_metrics.update(
                    {
                        "stationary_broadband_noise": noise_detected,
                        "stationary_broadband_noise_metrics": noise_metrics,
                    }
                )
                diagnostic["audio"] = audio_metrics
        return audio, timing

    def _resolve_short_sentence_config(
        self, enable_short_sentence_override: bool | None
    ) -> ShortSentenceConfig | None:
        from .short_sentence_handler import ShortSentenceConfig, WrapResolveMode

        explicit_config = self._short_sentence_config is not None
        effective_config = self._short_sentence_config

        if enable_short_sentence_override is not None:
            if enable_short_sentence_override:
                if effective_config is None:
                    effective_config = ShortSentenceConfig(enabled=True)
                else:
                    effective_config = dataclasses.replace(effective_config, enabled=True)
            else:
                if effective_config is not None:
                    effective_config = dataclasses.replace(effective_config, enabled=False)

        if (
            effective_config is not None
            and effective_config.enabled
            and self._timestamp_support is False
            and self._uses_phrase_short_sentence_mode(effective_config)
        ):
            message = (
                "Loaded ONNX model has no timestamp output; phrase-based short "
                "sentence modes require timestamps. Falling back to wrap mode "
                "for this run."
            )
            if explicit_config:
                if not self._reported_missing_timestamp_output:
                    logger.warning(message)
                    self._reported_missing_timestamp_output = True
            else:
                logger.debug(
                    "Loaded ONNX model has no timestamp output; "
                    "using wrap short-sentence mode for the implicit default."
                )
            resolve_modes = dict(effective_config.resolve_modes)
            resolve_modes["wrap"] = resolve_modes.get("wrap", WrapResolveMode())
            effective_config = dataclasses.replace(
                effective_config,
                resolve_modes=resolve_modes,
                resolve_mode="wrap",
            )

        return effective_config

    @staticmethod
    def _uses_phrase_short_sentence_mode(config: ShortSentenceConfig) -> bool:
        if config.resolve_mode is False:
            return False
        mode = config.resolve_modes.get(config.resolve_mode)
        return mode is not None and mode.kind in {"phrase", "randomized-phrase"}

    def _preprocess_segments(
        self,
        segments: list[PhonemeSegment],
        enable_short_sentence_override: bool | None,
        random_seed: int | None = None,
        context_phonemizer: Callable[[str, str], Any] | None = None,
    ) -> list[PhonemeSegment]:
        from .short_sentence_handler import is_segment_empty, is_segment_short

        effective_config = self._resolve_short_sentence_config(enable_short_sentence_override)
        self._context_phonemizer = context_phonemizer
        phrase_rng = random.Random(random_seed) if random_seed is not None else None
        processed: list[PhonemeSegment] = []

        for segment in segments:
            phonemes = segment.phonemes or ""
            tokens = (
                list(segment.tokens)
                if segment.tokens
                else (self._tokenizer.tokenize(phonemes) if phonemes.strip() else [])
            )
            skip_audio = False

            if effective_config and is_segment_empty(segment, effective_config):
                logger.debug(f"Skipping phoneme segment: '{segment.text[:50]}'")
                skip_audio = True

            if skip_audio or not phonemes.strip():
                processed.append(
                    dataclasses.replace(
                        segment,
                        phonemes="",
                        tokens=[],
                        raw_audio=None,
                        processed_audio=None,
                    )
                )
                continue

            if effective_config:
                detection_segment = dataclasses.replace(segment, tokens=tokens)
                if is_segment_short(detection_segment, effective_config):
                    short_sentence = apply_short_sentence_mode(
                        segment,
                        phonemes,
                        tokens,
                        effective_config,
                        self._tokenizer.tokenize,
                        rng=phrase_rng,
                        context_phonemizer=context_phonemizer,
                    )
                    phonemes = short_sentence.phonemes
                    tokens = short_sentence.tokens
                    if short_sentence.metadata is not None:
                        metadata = dict(segment.engine_metadata or {})
                        metadata[SHORT_SENTENCE_META_KEY] = short_sentence.metadata
                        segment = dataclasses.replace(segment, engine_metadata=metadata)
                        if short_sentence.metadata.get("kind") in {
                            "phrase",
                            "randomized-phrase",
                        }:
                            # Phrase tokens describe synthetic context; use target metadata instead.
                            segment = dataclasses.replace(segment, alignment_tokens=[])
                        elif (
                            short_sentence.metadata.get("kind") == "wrap"
                            and segment.alignment_tokens
                        ):
                            pretext = effective_config.phoneme_pretext
                            pre_count = len(self._tokenizer.tokenize(pretext))
                            synthetic = G2PAlignmentToken(
                                text="",
                                phonemes=pretext,
                                model_token_count=pre_count,
                            )
                            segment = dataclasses.replace(
                                segment,
                                alignment_tokens=[
                                    synthetic,
                                    *segment.alignment_tokens,
                                    synthetic,
                                ],
                            )

            if len(tokens) > MAX_PHONEME_LENGTH:
                batches = [
                    tokens[i : i + MAX_PHONEME_LENGTH]
                    for i in range(0, len(tokens), MAX_PHONEME_LENGTH)
                ]
                for idx, batch_tokens in enumerate(batches):
                    batch_phonemes = self._tokenizer.detokenize(batch_tokens)
                    processed.append(
                        dataclasses.replace(
                            segment,
                            id=f"{segment.id}_ph{idx}",
                            phoneme_id=idx,
                            phonemes=batch_phonemes,
                            tokens=list(batch_tokens),
                            raw_audio=None,
                            processed_audio=None,
                        )
                    )
            else:
                processed.append(
                    dataclasses.replace(
                        segment,
                        phonemes=phonemes,
                        tokens=tokens,
                        raw_audio=None,
                        processed_audio=None,
                    )
                )

        return processed

    def _generate_raw_audio_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        trace: Trace | None = None,
    ) -> list[PhonemeSegment]:
        noise_flags: list[bool] = []
        for segment in segments:
            short_sentence_metadata = (segment.engine_metadata or {}).get(SHORT_SENTENCE_META_KEY)
            if trace is not None and isinstance(short_sentence_metadata, dict):
                trace.increment_counter("short_sentence_detected")
                if short_sentence_metadata.get("kind") in {"phrase", "randomized-phrase"}:
                    trace.increment_counter("short_sentence_phrase_initial")
            if not segment.phonemes.strip():
                segment.raw_audio = None
                continue

            if trace is not None:
                trace.counters["logical_phoneme_segments"] = (
                    trace.counters.get("logical_phoneme_segments", 0) + 1
                )
            segment_voice_style = voice_style
            if trace is None:
                audio, pred_dur = self._run_onnx(
                    segment.phonemes, segment_voice_style, speed, tokens=segment.tokens or None
                )
            else:
                audio, pred_dur = self._run_onnx(
                    segment.phonemes,
                    segment_voice_style,
                    speed,
                    trace=trace,
                    attempt_kind="initial",
                    tokens=segment.tokens or None,
                )
            if trace is not None:
                trace.inference[-1].update(
                    {
                        "segment_id": segment.id,
                        "effective_voice": segment.voice_name,
                    }
                )
            segment.word_timings = self._map_pred_dur_to_word_timings(segment, pred_dur, len(audio))
            self._log_short_sentence_timestamps(segment, pred_dur)
            segment.raw_audio = self._prepare_short_sentence_phrase_audio(
                segment,
                audio,
                segment_voice_style,
                speed,
                trace=trace,
            )

            if self._waveform_validation != "off":
                noise_detected, _ = _is_stationary_broadband_noise(audio)
                noise_flags.append(noise_detected)
        if noise_flags and all(noise_flags) and self._waveform_validation != "off":
            message = (
                "All raw ONNX segments resemble stationary broadband noise; "
                "model output is likely invalid"
            )
            if trace is not None:
                trace.warnings.append(message)
            if self._waveform_validation == "strict":
                raise ConfigurationError(message)
            logger.warning(message)
        return segments

    @staticmethod
    def _coerce_timing_payload(
        timing: KokoroTimingPayload | np.ndarray | None,
        generated_position_count: int,
    ) -> KokoroTimingPayload | None:
        if timing is None:
            return None
        if isinstance(timing, KokoroTimingPayload):
            return timing
        # Private mapping helpers historically accepted padded vectors; keep that
        # compatibility explicit rather than rediscovering layout from arbitrary counts.
        return classify_timing(timing, generated_position_count, declared_layout="special-padded")

    def _map_pred_dur_to_word_timings(
        self,
        segment: PhonemeSegment,
        pred_dur: KokoroTimingPayload | np.ndarray | None,
        audio_length: int,
    ) -> list[WordTiming]:
        metadata = (segment.engine_metadata or {}).get(SHORT_SENTENCE_META_KEY)
        if isinstance(metadata, dict) and isinstance(metadata.get("timing_tokens"), list):
            raw_tokens = [
                dict(token) for token in metadata["timing_tokens"] if isinstance(token, dict)
            ]
            return self._map_timing_tokens_to_word_timings(
                segment, raw_tokens, pred_dur, audio_length, local_target_offsets=True
            )
        if segment.alignment_tokens:
            return self._map_timing_tokens_to_word_timings(
                segment,
                [token.to_dict() for token in segment.alignment_tokens],
                pred_dur,
                audio_length,
            )
        return []

    def _map_timing_tokens_to_word_timings(
        self,
        segment: PhonemeSegment,
        raw_tokens: list[dict[str, object]],
        pred_dur: KokoroTimingPayload | np.ndarray | None,
        audio_length: int,
        *,
        local_target_offsets: bool = False,
    ) -> list[WordTiming]:
        if pred_dur is None or audio_length <= 0:
            return []
        generated_count = (
            len(segment.tokens)
            if segment.tokens
            else sum(
                geometry[1]
                for token in raw_tokens
                if (geometry := _exact_timing_geometry(token)) is not None
            )
        )
        timing = self._coerce_timing_payload(pred_dur, generated_count)
        if timing is None or not timing.is_valid:
            return []
        valid_tokens: list[dict[str, object]] = []
        for token in raw_tokens:
            if token.get("is_target") and local_target_offsets:
                char_start = token.get("char_start")
                char_end = token.get("char_end")
                if not isinstance(char_start, int) or not isinstance(char_end, int):
                    continue
                if not 0 <= char_start <= char_end <= len(segment.text):
                    continue
                token["char_start"] = segment.char_start + char_start
                token["char_end"] = segment.char_start + char_end
            valid_tokens.append(token)
        timestamped = _join_timestamps(
            cast(list[object], valid_tokens),
            timing,
            strict=True,
            require_final_cursor=False,
        )
        if not timestamped:
            return []
        grouped: dict[tuple[int, int], WordTiming] = {}
        for token in timestamped:
            if "is_target" in token and not token.get("is_target"):
                continue
            text = str(token.get("text") or "")
            char_start = token.get("char_start")
            char_end = token.get("char_end")
            start_ts = token.get("start_ts")
            end_ts = token.get("speech_end_ts", token.get("end_ts"))
            if (
                not any(char.isalnum() for char in text)
                or not isinstance(char_start, int)
                or not isinstance(char_end, int)
                or not isinstance(start_ts, (int, float))
                or not isinstance(end_ts, (int, float))
            ):
                continue
            start = max(0, min(audio_length, round(float(start_ts) * SAMPLE_RATE)))
            end = max(start, min(audio_length, round(float(end_ts) * SAMPLE_RATE)))
            if end <= start:
                continue
            key = (char_start, char_end)
            current = grouped.get(key)
            if current is None:
                grouped[key] = WordTiming(
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                    start_sample=start,
                    end_sample=end,
                    segment_id=segment.id,
                )
            else:
                grouped[key] = dataclasses.replace(
                    current,
                    start_sample=min(current.start_sample, start),
                    end_sample=max(current.end_sample, end),
                )
        return sorted(grouped.values(), key=lambda item: (item.start_sample, item.end_sample))

    def _log_short_sentence_timestamps(
        self,
        segment: PhonemeSegment,
        pred_dur: KokoroTimingPayload | np.ndarray | None,
    ) -> None:
        short_sentence_metadata = (segment.engine_metadata or {}).get(SHORT_SENTENCE_META_KEY)
        if not isinstance(short_sentence_metadata, dict):
            return
        timing_tokens = short_sentence_metadata.get("timing_tokens")
        if not isinstance(timing_tokens, list):
            return
        timing = (
            pred_dur
            if isinstance(pred_dur, KokoroTimingPayload)
            else self._coerce_timing_payload(
                pred_dur,
                int(short_sentence_metadata.get("generated_token_count", 0)),
            )
            if pred_dur is not None
            else None
        )
        short_sentence_metadata["pred_duration_count"] = 0 if timing is None else timing.raw_count
        _record_short_sentence_timing_alignment(
            short_sentence_metadata,
            timing_tokens,
            timing=timing,
            pred_duration_count=short_sentence_metadata["pred_duration_count"],
        )
        if timing is None:
            if self._timestamp_support is False and not self._reported_missing_timestamp_output:
                logger.warning(
                    "ONNX inference returned no timing output; phrase-based short-sentence "
                    "extraction requires timings. Falling back to wrap mode for this segment."
                )
                self._reported_missing_timestamp_output = True
            short_sentence_metadata.setdefault("timing_failure_reason", "missing-duration-output")
            short_sentence_metadata.setdefault("failure_stage", "timing-token-build")
            short_sentence_metadata.setdefault("cut_failure_reason", "missing-duration-output")
            return
        if not short_sentence_metadata.get("timing_alignment_complete", False):
            short_sentence_metadata["timestamp_join_complete"] = False
            return
        expected_duration_count = short_sentence_metadata.get("expected_pred_duration_count")
        actual_duration_count = short_sentence_metadata.get("pred_duration_count")
        if (
            isinstance(expected_duration_count, int)
            and isinstance(actual_duration_count, int)
            and actual_duration_count != expected_duration_count
        ):
            short_sentence_metadata["timing_failure_detail"] = "duration-position-count-mismatch"
            short_sentence_metadata["timing_failure_reason"] = "timing-model-position-mismatch"
            short_sentence_metadata["failure_stage"] = "timing-alignment"
            short_sentence_metadata["cut_failure_reason"] = "timing-model-position-mismatch"
            short_sentence_metadata["timestamp_join_complete"] = False
            short_sentence_metadata["cutter_reached"] = False
            return
        strict_join = "generated_token_count" in short_sentence_metadata
        timestamped = _join_timestamps(
            timing_tokens,
            timing,
            strict=strict_join,
            metadata=short_sentence_metadata,
        )
        short_sentence_metadata["timestamp_join_complete"] = bool(timestamped)
        if not timestamped:
            short_sentence_metadata.setdefault("timing_failure_reason", "timestamp-join-incomplete")
            short_sentence_metadata.setdefault("failure_stage", "timestamp-join")
            short_sentence_metadata.setdefault("cut_failure_reason", "timestamp-join-incomplete")
        populate_short_sentence_boundary_metadata(
            short_sentence_metadata,
            timestamped,
        )
        target_timestamp_count = sum(
            1
            for token in timestamped
            if token.get("is_target")
            and isinstance(token.get("start_ts"), (int, float))
            and isinstance(token.get("end_ts"), (int, float))
        )
        short_sentence_metadata["target_timestamp_count"] = target_timestamp_count
        if timestamped and target_timestamp_count == 0:
            short_sentence_metadata.setdefault("timing_failure_reason", "missing-target-timestamps")
            short_sentence_metadata.setdefault("failure_stage", "target-boundary")
            short_sentence_metadata.setdefault("cut_failure_reason", "missing-target-timestamps")
        for token in timestamped:
            if not token.get("is_target"):
                continue
            start_ts = token.get("start_ts")
            end_ts = token.get("speech_end_ts", token.get("end_ts"))
            if not isinstance(start_ts, (int, float)) or not isinstance(end_ts, (int, float)):
                continue
            logger.debug(
                "Short sentence target timestamp: segment='%s' token='%s' start=%.4f end=%.4f",
                segment.text[:50],
                str(token.get("text") or ""),
                float(start_ts),
                float(end_ts),
            )

    def _prepare_short_sentence_phrase_audio(
        self,
        segment: PhonemeSegment,
        audio: np.ndarray,
        voice_style: np.ndarray,
        speed: float,
        *,
        trace: Trace | None = None,
    ) -> np.ndarray:
        """Accept confident phrase cuts or regenerate a wrap fallback."""
        short_sentence_metadata = (segment.engine_metadata or {}).get(SHORT_SENTENCE_META_KEY)
        if not isinstance(short_sentence_metadata, dict):
            return audio
        if short_sentence_metadata.get("kind") not in {"phrase", "randomized-phrase"}:
            return audio

        cut_audio = cut_short_sentence_phrase_audio(audio, short_sentence_metadata)
        if cut_audio is not None:
            _clear_short_sentence_failure(short_sentence_metadata)
            _record_short_sentence_attempt(
                short_sentence_metadata,
                attempt=0,
                kind="initial",
                succeeded=True,
                audio_samples=len(audio),
            )
            _record_short_sentence_cut(short_sentence_metadata, len(audio), trace)
            _log_short_sentence_cut_success(segment, short_sentence_metadata)
            left_cut = short_sentence_metadata.get("cut_left")
            right_cut = short_sentence_metadata.get("cut_right")
            if isinstance(left_cut, int) and isinstance(right_cut, int):
                segment.word_timings = _crop_word_timings(segment.word_timings, left_cut, right_cut)
            short_sentence_metadata["cut_applied"] = True
            return cut_audio
        _log_short_sentence_cut_failure(segment, short_sentence_metadata, len(audio))
        _record_short_sentence_attempt(
            short_sentence_metadata,
            attempt=0,
            kind="initial",
            succeeded=False,
            audio_samples=len(audio),
        )
        _record_short_sentence_cut_failure(short_sentence_metadata, trace)

        retry_audio = self._try_short_sentence_phrase_fallbacks(
            segment,
            short_sentence_metadata,
            voice_style,
            speed,
            trace=trace,
            context_phonemizer=self._context_phonemizer,
        )
        if retry_audio is not None:
            return retry_audio

        fallback_phonemes = short_sentence_metadata.get("fallback_phonemes")
        if not isinstance(fallback_phonemes, str) or not fallback_phonemes.strip():
            logger.warning(
                "Short sentence phrase cut for '%s' lacked confident boundaries; "
                "no wrap fallback was available.",
                segment.text[:50],
            )
            return audio

        short_sentence_metadata["cut_strategy"] = "wrap"
        if trace is not None:
            trace.increment_counter("short_sentence_wrap_fallback")
        history = short_sentence_metadata.get("short_sentence_attempts")
        failures = (
            [item for item in history if isinstance(item, dict) and item.get("succeeded") is False]
            if isinstance(history, list)
            else []
        )
        reasons: dict[str, int] = {}
        for item in failures:
            reason = item.get("failure_reason")
            if isinstance(reason, str) and reason:
                reasons[reason] = reasons.get(reason, 0) + 1
        dominant_reason = max(reasons, key=reasons.get) if reasons else "unknown"
        dominant_count = reasons.get(dominant_reason, 0)
        logger.warning(
            "Short sentence phrase extraction failed after %d phrase attempts; "
            "falling back to wrap mode. Dominant failure: %s (%d/%d).",
            len(history) if isinstance(history, list) else 1,
            dominant_reason,
            dominant_count,
            len(failures) or 1,
        )
        fallback_tokens = short_sentence_metadata.get("fallback_tokens")
        prepared_fallback_tokens = (
            fallback_tokens
            if isinstance(fallback_tokens, list)
            and all(isinstance(token, int) for token in fallback_tokens)
            else None
        )
        if trace is None:
            fallback_audio, _ = self._run_onnx(
                fallback_phonemes,
                voice_style,
                speed,
                attempt_kind="fallback",
                tokens=prepared_fallback_tokens,
            )
        else:
            fallback_audio, _ = self._run_onnx(
                fallback_phonemes,
                voice_style,
                speed,
                trace=trace,
                attempt_kind="fallback",
                tokens=prepared_fallback_tokens,
            )
        short_sentence_metadata["cut_applied"] = True
        short_sentence_metadata["fallback_used"] = "wrap"
        short_sentence_metadata["cut_left"] = 0
        short_sentence_metadata["cut_right"] = len(fallback_audio)
        segment.phonemes = fallback_phonemes
        if isinstance(prepared_fallback_tokens, list):
            segment.tokens = prepared_fallback_tokens
        segment.word_timings = []
        return fallback_audio

    def _try_short_sentence_phrase_fallbacks(
        self,
        segment: PhonemeSegment,
        short_sentence_metadata: dict[str, object],
        voice_style: np.ndarray,
        speed: float,
        *,
        trace: Trace | None = None,
        context_phonemizer: Callable[[str, str], Any] | None = None,
    ) -> np.ndarray | None:
        templates = short_sentence_metadata.get("phrase_fallback_templates")
        if not isinstance(templates, list):
            return None

        non_retryable_details = {
            "invalid-timing-contract",
            "missing-duration-output",
            "duration-position-count-mismatch",
            "unresolved-model-span",
            "alignment-position-count-mismatch",
        }
        if short_sentence_metadata.get("timing_failure_detail") in non_retryable_details:
            short_sentence_metadata["retry_attempts"] = 0
            short_sentence_metadata["retry_skipped_reason"] = "non-retryable-timing-failure"
            return None
        max_attempts = _short_sentence_phrase_fallback_limit(
            short_sentence_metadata,
            default=len(templates),
        )
        if max_attempts == 0:
            short_sentence_metadata["retry_attempts"] = 0
            return None

        used_templates = {
            template
            for template in [short_sentence_metadata.get("phrase_template")]
            if isinstance(template, str)
        }
        retry_attempts = 0
        failed_template = short_sentence_metadata.get("phrase_template")
        for template in templates:
            if not isinstance(template, str) or not template.strip():
                continue
            if template in used_templates:
                continue
            used_templates.add(template)
            if retry_attempts >= max_attempts:
                break
            retry_attempts += 1
            if trace is not None:
                trace.increment_counter("short_sentence_phrase_retry")
            logger.info(
                "Short sentence phrase cut for '%s' lacked confident boundaries; "
                "trying another phrase %d/%d. Failed with: '%s'",
                segment.text[:50],
                retry_attempts,
                max_attempts,
                failed_template if isinstance(failed_template, str) else "",
            )

            retry = build_short_sentence_phrase_retry(
                segment,
                template,
                short_sentence_metadata,
                context_phonemizer=context_phonemizer,
                tokenize=self._tokenizer.tokenize,
            )
            if retry is None or retry.metadata is None:
                failed_retry_metadata = dict(short_sentence_metadata)
                failed_retry_metadata.update(
                    {
                        "phrase_template": template,
                        "failure_stage": "context-g2p",
                        "cut_failure_reason": "context-g2p-failed",
                    }
                )
                _record_short_sentence_attempt(
                    failed_retry_metadata,
                    attempt=retry_attempts,
                    kind="retry",
                    succeeded=False,
                )
                history = failed_retry_metadata.get("short_sentence_attempts")
                if isinstance(history, list):
                    short_sentence_metadata["short_sentence_attempts"] = history
                failed_template = template

            if trace is None:
                retry_audio, pred_dur = self._run_onnx(
                    retry.phonemes,
                    voice_style,
                    speed,
                    attempt_kind="retry",
                    tokens=retry.tokens,
                )
            else:
                retry_audio, pred_dur = self._run_onnx(
                    retry.phonemes,
                    voice_style,
                    speed,
                    trace=trace,
                    attempt_kind="retry",
                    tokens=retry.tokens,
                )
            timing_tokens = retry.metadata.get("timing_tokens")
            retry_timings: list[WordTiming] = []
            if pred_dur is not None and isinstance(timing_tokens, list):
                retry_timings = self._map_timing_tokens_to_word_timings(
                    segment,
                    [dict(token) for token in timing_tokens if isinstance(token, dict)],
                    pred_dur,
                    len(retry_audio),
                    local_target_offsets=True,
                )
                _record_short_sentence_timing_alignment(
                    retry.metadata,
                    timing_tokens,
                    timing=pred_dur if isinstance(pred_dur, KokoroTimingPayload) else None,
                    pred_duration_count=(
                        pred_dur.raw_count
                        if isinstance(pred_dur, KokoroTimingPayload)
                        else int(np.asarray(pred_dur).reshape(-1).size)
                    ),
                )
                expected_duration_count = retry.metadata.get("expected_pred_duration_count")
                actual_duration_count = retry.metadata.get("pred_duration_count")
                if not retry.metadata.get("timing_alignment_complete", False) or (
                    isinstance(expected_duration_count, int)
                    and actual_duration_count != expected_duration_count
                ):
                    timestamped = []
                    if retry.metadata.get("timing_alignment_complete", False):
                        retry.metadata["timing_failure_detail"] = "duration-position-count-mismatch"
                    retry.metadata["timestamp_join_complete"] = False
                else:
                    timestamped = _join_timestamps(
                        timing_tokens,
                        pred_dur,
                        strict=True,
                        metadata=retry.metadata,
                    )
                populate_short_sentence_boundary_metadata(retry.metadata, timestamped)
            cut_audio = cut_short_sentence_phrase_audio(retry_audio, retry.metadata)
            if cut_audio is None:
                _log_short_sentence_cut_failure(segment, retry.metadata, len(retry_audio))
                _record_short_sentence_attempt(
                    retry.metadata,
                    attempt=retry_attempts,
                    kind="retry",
                    succeeded=False,
                    audio_samples=len(retry_audio),
                )
                _record_short_sentence_cut_failure(retry.metadata, trace)
                failed_template = template
                continue
            _clear_short_sentence_failure(retry.metadata)
            _record_short_sentence_attempt(
                retry.metadata,
                attempt=retry_attempts,
                kind="retry",
                succeeded=True,
                audio_samples=len(retry_audio),
            )
            _record_short_sentence_cut(retry.metadata, len(retry_audio), trace)

            _log_short_sentence_cut_success(segment, retry.metadata)
            logger.info(
                "Short sentence phrase cut for '%s' succeeded using fallback phrase '%s' (%d/%d).",
                segment.text[:50],
                template,
                retry_attempts,
                max_attempts,
            )
            left_cut = retry.metadata.get("cut_left")
            right_cut = retry.metadata.get("cut_right")
            if isinstance(left_cut, int) and isinstance(right_cut, int):
                retry_timings = _crop_word_timings(retry_timings, left_cut, right_cut)
            else:
                retry_timings = []
            short_sentence_metadata.clear()
            short_sentence_metadata.update(retry.metadata)
            short_sentence_metadata["cut_applied"] = True
            short_sentence_metadata["fallback_used"] = "phrase"
            short_sentence_metadata["retry_attempts"] = retry_attempts
            segment.phonemes = retry.phonemes
            segment.tokens = retry.tokens
            segment.word_timings = retry_timings
            return cut_audio

        short_sentence_metadata["retry_attempts"] = retry_attempts
        return None

    def _postprocess_audio_segments(
        self,
        segments: list[PhonemeSegment],
        trim_silence: bool,
        voice_level_config: VoiceLevelConfig | None = None,
        trace: Trace | None = None,
    ) -> list[PhonemeSegment]:
        config = voice_level_config or VoiceLevelConfig()
        for segment in segments:
            if segment.raw_audio is None:
                segment.processed_audio = None
                continue

            audio = segment.raw_audio
            short_sentence_metadata = (segment.engine_metadata or {}).get(SHORT_SENTENCE_META_KEY)
            if isinstance(short_sentence_metadata, dict) and not short_sentence_metadata.get(
                "cut_applied"
            ):
                cut_audio = cut_short_sentence_phrase_audio(audio, short_sentence_metadata)
                if cut_audio is not None:
                    left_cut = short_sentence_metadata.get("cut_left")
                    right_cut = short_sentence_metadata.get("cut_right")
                    if isinstance(left_cut, int) and isinstance(right_cut, int):
                        segment.word_timings = _crop_word_timings(
                            segment.word_timings, left_cut, right_cut
                        )
                    audio = cut_audio

            if trim_silence:
                trim_result: Any = trim_audio(audio)
                if isinstance(trim_result, tuple) and len(trim_result) == 2:
                    audio, trim_bounds = trim_result
                    segment.word_timings = _crop_word_timings(
                        segment.word_timings, int(trim_bounds[0]), int(trim_bounds[1])
                    )
                else:
                    audio = trim_result

            segment.processed_audio = apply_voice_level_calibration(
                audio,
                config,
                segment.render_voice_key,
                trace=trace,
                segment_id=segment.id,
            )

        return segments


def _crop_word_timings(
    timings: list[WordTiming], start_sample: int, end_sample: int
) -> list[WordTiming]:
    if end_sample <= start_sample:
        return []
    cropped: list[WordTiming] = []
    for timing in timings:
        start = max(timing.start_sample, start_sample)
        end = min(timing.end_sample, end_sample)
        if end <= start:
            continue
        cropped.append(
            dataclasses.replace(
                timing,
                start_sample=start - start_sample,
                end_sample=end - start_sample,
            )
        )
    return cropped


def _join_timestamps(
    tokens: list[object],
    pred_dur: KokoroTimingPayload | np.ndarray,
    *,
    strict: bool = False,
    require_final_cursor: bool = True,
    metadata: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Map explicit model-position durations to G2P tokens."""
    if not tokens:
        return []
    if isinstance(pred_dur, KokoroTimingPayload):
        timing = pred_dur
    else:
        raw_values = np.asarray(pred_dur).reshape(-1)
        timing = classify_timing(
            raw_values, max(int(raw_values.size) - 2, 0), declared_layout="special-padded"
        )
    if not timing.is_valid:
        return []
    position_values = timing.model_position_values
    if position_values.size == 0:
        return []
    timestamped: list[dict[str, object]] = []
    divisor = 80
    leading_duration = timing.leading_duration
    left = right = 2 * max(0.0, leading_duration - 3) if leading_duration is not None else 0.0
    cursor = 0
    end_cursor = int(position_values.size)
    complete = True
    for raw_token in tokens:
        token = dict(raw_token) if isinstance(raw_token, dict) else {}
        phonemes = str(token.get("phonemes") or "")
        whitespace = str(token.get("whitespace") or "")
        geometry = _exact_timing_geometry(token)
        if geometry is None:
            if strict:
                complete = False
                break
            speech_count = len(phonemes)
            span_count = speech_count + (1 if whitespace else 0)
        else:
            speech_count, span_count = geometry
        if speech_count < 0 or span_count < speech_count:
            complete = False
            break
        token_end = cursor + span_count
        if token_end > end_cursor:
            complete = False
            break
        speech_end_cursor = cursor + speech_count
        gap_dur = float(position_values[speech_end_cursor:token_end].sum().item())
        if speech_count:
            token["start_ts"] = left / divisor
            token_dur = float(position_values[cursor:speech_end_cursor].sum().item())
            speech_end = right + (2 * token_dur)
            token["speech_end_ts"] = speech_end / divisor
            left = speech_end + gap_dur
            token["end_ts"] = left / divisor
            right = left + gap_dur
        elif span_count:
            left = right + gap_dur
            right = left + gap_dur
        cursor = token_end
        timestamped.append(token)
    if metadata is not None:
        metadata["timing_layout"] = timing.layout
        metadata["timing_raw_count"] = timing.raw_count
        metadata["timing_model_position_count"] = timing.model_position_count
        metadata["timing_final_duration_cursor"] = cursor
        metadata["timing_expected_final_duration_cursor"] = end_cursor
        metadata["timing_duration_cursor_delta"] = cursor - end_cursor
    if strict and (not complete or (require_final_cursor and cursor != end_cursor)):
        return []
    return timestamped


def _record_short_sentence_timing_alignment(
    metadata: dict[str, object],
    timing_tokens: list[object],
    timing: KokoroTimingPayload | None = None,
    pred_duration_count: int | None = None,
    *,
    timestamp_join_complete: bool | None = None,
    target_timestamp_count: int | None = None,
) -> None:
    """Record exact model-position accounting for a phrase timing payload."""
    model_position_count = 0
    complete = True
    target_token_count = 0
    first_unresolved: tuple[int, dict[str, object]] | None = None
    for index, raw_token in enumerate(timing_tokens):
        token = dict(raw_token) if isinstance(raw_token, dict) else {}
        if token.get("is_target"):
            target_token_count += 1
        geometry = _exact_timing_geometry(token)
        if geometry is None:
            complete = False
            if first_unresolved is None:
                first_unresolved = (index, token)
            continue
        _, span_count = geometry
        model_position_count += span_count
    generated_token_count = metadata.get("generated_token_count")
    metadata["timing_token_count"] = len(timing_tokens)
    metadata["timing_target_token_count"] = target_token_count
    metadata["timing_model_position_count"] = model_position_count
    metadata["timing_model_position_delta"] = (
        model_position_count - generated_token_count
        if isinstance(generated_token_count, int) and not isinstance(generated_token_count, bool)
        else None
    )
    timing_layout = timing.layout if timing is not None else "special-padded"
    metadata["timing_layout"] = timing_layout
    if timing is not None:
        metadata["timing_raw_count"] = timing.raw_count
        metadata["timing_model_position_values_count"] = timing.model_position_count
        metadata["timing_contract_error"] = timing.error
    expected_pred_duration_count = (
        generated_token_count + 2
        if timing_layout == "special-padded"
        and isinstance(generated_token_count, int)
        and not isinstance(generated_token_count, bool)
        else generated_token_count
        if timing_layout == "model-positions"
        and isinstance(generated_token_count, int)
        and not isinstance(generated_token_count, bool)
        else None
    )
    metadata["expected_pred_duration_count"] = expected_pred_duration_count
    if pred_duration_count is not None:
        metadata["pred_duration_count"] = pred_duration_count
        metadata["pred_duration_count_delta"] = (
            pred_duration_count - expected_pred_duration_count
            if expected_pred_duration_count is not None
            else None
        )
    if first_unresolved is not None:
        index, token = first_unresolved
        metadata["timing_first_unresolved_token_index"] = index
        metadata["timing_first_unresolved_token_text"] = token.get("text")
        metadata["timing_first_unresolved_token_phonemes"] = token.get("phonemes")
        metadata["timing_first_unresolved_token_whitespace"] = token.get("whitespace")
        metadata["timing_first_unresolved_token_model_token_count"] = token.get("model_token_count")
        metadata["timing_first_unresolved_token_model_span_token_count"] = token.get(
            "model_span_token_count"
        )
    metadata["timing_alignment_complete"] = (
        complete
        and isinstance(generated_token_count, int)
        and not isinstance(generated_token_count, bool)
        and model_position_count == generated_token_count
    )
    if timestamp_join_complete is not None:
        metadata["timestamp_join_complete"] = timestamp_join_complete
    if target_timestamp_count is not None:
        metadata["target_timestamp_count"] = target_timestamp_count
    if timing is not None and not timing.is_valid:
        metadata["timing_alignment_complete"] = False
        metadata["timing_failure_detail"] = "invalid-timing-contract"
        metadata.setdefault("timing_failure_reason", "timing-model-position-mismatch")
        metadata.setdefault("failure_stage", "timing-alignment")
        metadata.setdefault("cut_failure_reason", "timing-model-position-mismatch")
    if not metadata["timing_alignment_complete"]:
        metadata["timing_failure_detail"] = (
            "unresolved-model-span"
            if first_unresolved is not None
            else "alignment-position-count-mismatch"
        )
        metadata["cutter_reached"] = False
        metadata.setdefault("timing_failure_reason", "timing-model-position-mismatch")
        metadata.setdefault("failure_stage", "timing-alignment")
        metadata.setdefault("cut_failure_reason", "timing-model-position-mismatch")


def populate_short_sentence_boundary_metadata(
    metadata: dict[str, object],
    timestamped: list[dict[str, object]],
) -> None:
    """Populate production phrase-cut metadata from timestamped G2P tokens."""
    target_indices = [
        index
        for index, token in enumerate(timestamped)
        if token.get("is_target")
        and isinstance(token.get("start_ts"), (int, float))
        and isinstance(token.get("end_ts"), (int, float))
    ]
    if not target_indices:
        return

    target_tokens = [timestamped[index] for index in target_indices]
    target_boundary_tokens = [
        token for token in target_tokens if _is_spoken_token(token)
    ] or target_tokens
    metadata["target_start_ts"] = min(
        float(cast(Any, token["start_ts"])) for token in target_boundary_tokens
    )
    metadata["target_end_ts"] = max(
        float(cast(Any, token.get("speech_end_ts", token["end_ts"])))
        for token in target_boundary_tokens
    )

    first_target = min(target_indices)
    last_target = max(target_indices)
    previous_tokens = [
        token
        for token in timestamped[:first_target]
        if _is_spoken_token(token)
        and isinstance(token.get("speech_end_ts", token.get("end_ts")), (int, float))
    ]
    next_tokens = [
        token
        for token in timestamped[last_target + 1 :]
        if _is_spoken_token(token) and isinstance(token.get("start_ts"), (int, float))
    ]
    metadata["has_left_context"] = bool(previous_tokens)
    metadata["has_right_context"] = bool(next_tokens)
    if previous_tokens:
        previous_end = previous_tokens[-1].get(
            "speech_end_ts",
            previous_tokens[-1]["end_ts"],
        )
        metadata["previous_token_end_ts"] = float(cast(Any, previous_end))
    if next_tokens:
        metadata["next_token_start_ts"] = float(cast(Any, next_tokens[0]["start_ts"]))


def _short_sentence_phrase_fallback_limit(
    metadata: dict[str, object],
    *,
    default: int,
) -> int:
    value = metadata.get("phrase_fallback_tries", default)
    try:
        return max(0, int(cast(Any, value)))
    except (TypeError, ValueError):
        return max(0, int(default))


def _record_short_sentence_cut(
    metadata: dict[str, object],
    audio_length: int,
    trace: Trace | None,
) -> None:
    """Record cut strategy, geometry, and aggregate success counters."""
    strategy = str(metadata.get("cut_strategy", "energy-valley"))
    if strategy == "energy-valley":
        counter = "short_sentence_cut_strict_success"
    elif strategy.startswith("timestamp-"):
        counter = "short_sentence_cut_adaptive_success"
    else:
        counter = "short_sentence_cut_failure"
    if trace is not None:
        trace.increment_counter(counter)

    left_cut = metadata.get("cut_left")
    right_cut = metadata.get("cut_right")
    if not isinstance(left_cut, int) or not isinstance(right_cut, int):
        return
    metadata["cut_left"] = max(0, min(audio_length, left_cut))
    metadata["cut_right"] = max(0, min(audio_length, right_cut))
    target_start = _timestamp_to_sample(metadata.get("target_start_ts"))
    target_end = _timestamp_to_sample(metadata.get("target_end_ts"))
    if target_start is not None:
        metadata["left_anchor_distance_samples"] = abs(left_cut - target_start)
        metadata["left_anchor_distance_ms"] = abs(left_cut - target_start) * 1000.0 / SAMPLE_RATE
    if target_end is not None:
        metadata["right_anchor_distance_samples"] = abs(right_cut - target_end)
        metadata["right_anchor_distance_ms"] = abs(right_cut - target_end) * 1000.0 / SAMPLE_RATE
    guard_durations: dict[str, float] = {}
    if target_start is not None:
        guard_durations["left"] = abs(left_cut - target_start) * 1000.0 / SAMPLE_RATE
    if target_end is not None:
        guard_durations["right"] = abs(right_cut - target_end) * 1000.0 / SAMPLE_RATE
    if guard_durations:
        metadata["retained_guard_duration_ms"] = guard_durations


def _log_short_sentence_cut_failure(
    segment: PhonemeSegment,
    metadata: dict[str, object],
    audio_length: int,
) -> None:
    """Log an attempt failure without implying that the cutter was reached."""
    reason = str(metadata.setdefault("cut_failure_reason", "unknown-cutter"))
    stage = str(metadata.setdefault("failure_stage", "cut-search"))
    attempt_history = metadata.get("short_sentence_attempts")
    attempt_number = len(attempt_history) + 1 if isinstance(attempt_history, list) else 1
    max_attempts = int(metadata.get("phrase_fallback_tries", 0)) + 1
    logger.info(
        "short_sentence.phrase_attempt.failure segment=%r attempt=%d/%d stage=%s "
        "reason=%s configured_cutter=%s cutter_reached=%s timing_layout=%s "
        "generated_positions=%s timing_tokens=%s timing_positions=%s pred_durations=%s "
        "expected_pred_durations=%s model_position_delta=%s duration_count_delta=%s "
        "timing_detail=%s runtime_ref=%s distribution=%s retry_skipped_reason=%s "
        "template=%r audio_samples=%d",
        segment.text,
        attempt_number,
        max_attempts,
        stage,
        reason,
        metadata.get("cutter"),
        metadata.get("cutter_reached", stage not in {"timing-alignment", "timestamp-join"}),
        metadata.get("timing_layout", "unknown"),
        metadata.get("generated_token_count"),
        metadata.get(
            "timing_token_count",
            len(metadata.get("timing_tokens", []))
            if isinstance(metadata.get("timing_tokens"), list)
            else 0,
        ),
        metadata.get("timing_model_position_count"),
        metadata.get("pred_duration_count", 0),
        metadata.get("expected_pred_duration_count"),
        metadata.get("timing_model_position_delta"),
        metadata.get("pred_duration_count_delta"),
        metadata.get("timing_failure_detail"),
        metadata.get("runtime_ref", "-"),
        metadata.get("distribution", "-"),
        metadata.get("retry_skipped_reason", "-"),
        metadata.get("phrase_template"),
        audio_length,
    )


def _log_short_sentence_cut_success(
    segment: PhonemeSegment,
    metadata: dict[str, object],
) -> None:
    logger.debug(
        "short_sentence.phrase_attempt.success segment=%r strategy=%s left_cut=%s "
        "right_cut=%s left_distance_ms=%s right_distance_ms=%s phrase_language=%s",
        segment.text,
        metadata.get("cut_strategy"),
        metadata.get("cut_left"),
        metadata.get("cut_right"),
        metadata.get("left_anchor_distance_ms"),
        metadata.get("right_anchor_distance_ms"),
        metadata.get("phrase_language"),
    )


def _record_short_sentence_cut_failure(
    metadata: dict[str, object],
    trace: Trace | None,
) -> None:
    """Record an unsuccessful phrase attempt with a stage-specific counter."""
    reason = metadata.setdefault("cut_failure_reason", "unknown-cutter")
    stage = metadata.setdefault("failure_stage", "cut-search")
    if trace is not None:
        trace.increment_counter("short_sentence_cut_failure")
        trace.increment_counter("short_sentence_phrase_attempt_failure")
        counter_by_stage = {
            "timing-alignment": "short_sentence_timing_alignment_failure",
            "timestamp-join": "short_sentence_timestamp_join_failure",
            "target-boundary": "short_sentence_target_boundary_failure",
            "cut-search": "short_sentence_cutter_failure",
            "cut-window": "short_sentence_cutter_failure",
            "cut-validation": "short_sentence_cutter_failure",
        }
        counter = counter_by_stage.get(str(stage))
        if counter is not None:
            trace.increment_counter(counter)
    _ = reason


def _clear_short_sentence_failure(metadata: dict[str, object]) -> None:
    """Remove outcome fields from an attempt that ultimately succeeded."""
    for key in ("cut_failure_reason", "timing_failure_reason", "failure_stage"):
        metadata.pop(key, None)


def _record_short_sentence_attempt(
    metadata: dict[str, object],
    *,
    attempt: int,
    kind: str,
    succeeded: bool,
    audio_samples: int | None = None,
) -> None:
    """Append a self-contained attempt snapshot for benchmark consumers."""
    history = metadata.setdefault("short_sentence_attempts", [])
    if not isinstance(history, list):
        return
    entry: dict[str, object] = {
        "attempt": attempt,
        "ordinal": attempt + 1,
        "kind": kind,
        "phrase_template": metadata.get("phrase_template"),
        "phrase_language": metadata.get("phrase_language"),
        "terminal_form": metadata.get("phrase_terminal_form"),
        "configured_cutter": metadata.get("cutter"),
        "cutter": metadata.get("cutter"),
        "actual_cut_strategy": metadata.get("cut_strategy") if succeeded else None,
        "cut_strategy": metadata.get("cut_strategy") if succeeded else None,
        "succeeded": succeeded,
        "failure_stage": None if succeeded else metadata.get("failure_stage"),
        "failure_reason": (
            None
            if succeeded
            else metadata.get("timing_failure_reason", metadata.get("cut_failure_reason"))
        ),
        "audio_samples": audio_samples,
        "generated_token_count": metadata.get("generated_token_count"),
        "timing_model_position_count": metadata.get("timing_model_position_count"),
        "timing_model_position_delta": metadata.get("timing_model_position_delta"),
        "pred_duration_count": metadata.get("pred_duration_count"),
        "target_timestamp_count": metadata.get("target_timestamp_count", 0),
        "target_start_sample": _timestamp_to_sample(metadata.get("target_start_ts")),
        "target_end_sample": _timestamp_to_sample(metadata.get("target_end_ts")),
    }
    if succeeded:
        entry["cut_left"] = metadata.get("cut_left")
        entry["cut_right"] = metadata.get("cut_right")
    history.append(entry)


def _timestamp_to_sample(value: object) -> int | None:
    if not isinstance(value, (int, float)):
        return None
    return round(float(value) * SAMPLE_RATE)


def _is_spoken_token(token: dict[str, object]) -> bool:
    """Return whether a token corresponds to spoken lexical content."""
    text = str(token.get("text") or "")
    return any(char.isalnum() for char in text)
