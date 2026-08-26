"""Audio generation for PyKokoro."""

from __future__ import annotations

import dataclasses
import logging
import random
import re
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from audiosig import apply_gain_db, resample, resample_speed
from audiosig import trim as trim_audio

from .constants import MAX_PHONEME_LENGTH, SAMPLE_RATE
from .exceptions import ConfigurationError
from .prosody import apply_prosody, parse_pitch, parse_rate, parse_volume
from .short_sentence_handler import (
    SHORT_SENTENCE_META_KEY,
    apply_short_sentence_mode,
    build_short_sentence_phrase_retry,
    cut_short_sentence_phrase_audio,
)
from .tokenizer import Tokenizer
from .types import G2PAlignmentToken, PhonemeSegment, WordTiming, _model_span_token_count
from .utils import generate_silence
from .voice_manager import normalize_voice_style

if TYPE_CHECKING:
    import onnxruntime as rt

    from .prosody_config import ProsodyConfig
    from .short_sentence_handler import ShortSentenceConfig
    from .types import Trace

logger = logging.getLogger(__name__)


def _waveform_metrics(audio: np.ndarray) -> dict[str, float | int | bool]:
    """Return compact, JSON-friendly metrics for a segment waveform."""

    values = np.asarray(audio, dtype=np.float64).reshape(-1)
    differences = np.diff(values)
    return {
        "samples": int(values.size),
        "finite": bool(np.isfinite(values).all()),
        "peak": float(np.max(np.abs(values))) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "max_adjacent_jump": (float(np.max(np.abs(differences))) if differences.size else 0.0),
    }


def _has_prosody_metadata(segment: PhonemeSegment) -> bool:
    metadata = segment.ssmd_metadata or {}
    return bool(
        metadata.get("prosody_volume")
        or metadata.get("prosody_pitch")
        or metadata.get("prosody_rate")
    )


def _should_condition_boundary(
    left: PhonemeSegment,
    right: PhonemeSegment,
    config: ProsodyConfig | None,
) -> bool:
    return bool(
        config is not None
        and config.boundary_blend_ms > 0.0
        and (_has_prosody_metadata(left) or _has_prosody_metadata(right))
    )


def _boundary_jump(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return 0.0
    return abs(float(left[-1]) - float(right[0]))


def _condition_boundary(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sample_rate: int,
    blend_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a small boundary window without changing either segment length."""

    window = min(
        round(sample_rate * blend_ms / 1000.0),
        left.size,
        right.size,
    )
    if window < 2:
        return left, right

    positions = np.linspace(0.0, window - 1.0, 2 * window)
    source_positions = np.arange(window, dtype=np.float64)
    left_curve = np.interp(positions, source_positions, left[-window:])
    right_curve = np.interp(positions, source_positions, right[:window])
    mix = np.linspace(0.0, 1.0, 2 * window)
    transition = left_curve * (1.0 - mix) + right_curve * mix

    conditioned_left = np.array(left, copy=True)
    conditioned_right = np.array(right, copy=True)
    conditioned_left[-window:] = transition[:window]
    conditioned_right[:window] = transition[window:]
    return conditioned_left, conditioned_right


def _record_boundary_diagnostic(
    trace: Trace | None,
    segment: PhonemeSegment,
    before: float,
    after: float,
    conditioned_samples: int,
) -> None:
    if trace is None:
        return
    trace.prosody.append(
        {
            "kind": "boundary",
            "segment_id": segment.id,
            "boundary_jump_before": before,
            "boundary_jump_after": after,
            "conditioned_samples": conditioned_samples,
        }
    )


def resolve_audio_annotation(
    metadata: dict[str, Any],
    resolver: Any,
    *,
    sample_rate: int = SAMPLE_RATE,
    max_bytes: int = 20_000_000,
    max_duration_s: float = 120.0,
) -> np.ndarray:
    """Resolve and deterministically transform an SSMD audio annotation."""

    source = metadata.get("audio_src")
    if not isinstance(source, str) or not source:
        raise ValueError("audio_src must be a non-empty string")
    callback = getattr(resolver, "resolve", resolver)
    if not callable(callback):
        raise TypeError("audio_source_resolver must be callable or expose resolve()")
    result = callback(source)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("audio resolver must return (numpy_audio, sample_rate)")
    audio, source_rate = result
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    source_rate = int(source_rate)
    if source_rate <= 0 or audio.nbytes > max_bytes:
        raise ValueError("resolved audio exceeds configured source limits")

    def seconds(value: object) -> float | None:
        if value in (None, ""):
            return None
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s)\s*", str(value), re.I)
        if match is None:
            raise ValueError(f"invalid audio duration {value!r}")
        number = float(match.group(1))
        return number / 1000.0 if match.group(2).lower() == "ms" else number

    begin = seconds(metadata.get("audio_clip_begin"))
    end = seconds(metadata.get("audio_clip_end"))
    start_index = max(0, int((begin or 0.0) * source_rate))
    end_index = len(audio) if end is None else min(len(audio), int(end * source_rate))
    audio = audio[start_index : max(start_index, end_index)]

    speed = metadata.get("audio_speed")
    if speed:
        match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)%\s*", str(speed))
        if match is None or float(match.group(1)) <= 0:
            raise ValueError(f"invalid audio speed {speed!r}")
        factor = float(match.group(1)) / 100.0
        audio = resample_speed(audio, factor)

    repeat = metadata.get("audio_repeat_count")
    if repeat:
        audio = np.tile(audio, int(repeat))
    repeat_duration = seconds(metadata.get("audio_repeat_dur"))
    if repeat_duration is not None and audio.size:
        target = max(0, round(repeat_duration * source_rate))
        audio = np.resize(audio, target)

    level = metadata.get("audio_sound_level")
    if level:
        match = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*dB\s*", str(level), re.I)
        if match is None:
            raise ValueError(f"invalid audio level {level!r}")
        audio = apply_gain_db(audio, float(match.group(1)))

    if source_rate != sample_rate and audio.size:
        audio = resample(
            audio,
            source_rate=source_rate,
            target_rate=sample_rate,
        )
    if len(audio) > round(max_duration_s * sample_rate):
        raise ValueError("resolved audio exceeds configured duration limit")
    return audio.astype(np.float32, copy=False)


# Model source type
ModelSource = Literal["huggingface", "github"]


class AudioGenerator:
    """Generates audio from phonemes, tokens, and segments using ONNX inference.

    This class handles:
    - ONNX inference for single phoneme batches
    - Phoneme splitting for long inputs
    - Batch generation from phoneme lists
    - Segment-based generation with pause support
    - Token-to-audio generation
    - Short sentence handling via configured context modes

    Args:
        session: ONNX Runtime inference session
        tokenizer: Tokenizer for phoneme<->token conversion
        model_source: Model source ('huggingface' or 'github')
        short_sentence_config: Configuration for short sentence handling
    """

    def __init__(
        self,
        session: rt.InferenceSession,
        tokenizer: Tokenizer,
        model_source: ModelSource = "huggingface",
        short_sentence_config: ShortSentenceConfig | None = None,
    ):
        """Initialize the audio generator."""
        self._session = session
        self._tokenizer = tokenizer
        self._model_source = model_source
        self._short_sentence_config = short_sentence_config
        self._input_metas = {
            str(input_meta.name): input_meta for input_meta in session.get_inputs()
        }
        self._uses_input_ids = "input_ids" in self._input_metas
        get_outputs = getattr(session, "get_outputs", None)
        outputs = get_outputs() if callable(get_outputs) else []
        self._timestamp_output_index: int | None = None
        try:
            for index, output in enumerate(outputs):
                name = str(getattr(output, "name", "")).lower()
                if name in {"pred_dur", "pred_duration", "durations"}:
                    self._timestamp_output_index = index
                    break
            self._has_timestamp_output = self._timestamp_output_index is not None
        except TypeError:
            # Some lightweight test doubles expose get_outputs() as a bare Mock.
            self._has_timestamp_output = False
        self._reported_missing_timestamp_output = False

    def _tokenize_phonemes(self, phonemes: str) -> list[int]:
        trimmed = phonemes[:MAX_PHONEME_LENGTH]
        return self._tokenizer.tokenize(trimmed)

    def _select_voice_style(self, voice_style: np.ndarray, token_count: int) -> np.ndarray:
        voice_style = normalize_voice_style(voice_style, expected_length=None)
        max_style_idx = voice_style.shape[0] - 1 if len(voice_style.shape) > 0 else 0
        style_idx = min(token_count, MAX_PHONEME_LENGTH - 1, max_style_idx)
        voice_style_indexed = voice_style[style_idx]
        if voice_style_indexed.ndim == 1:
            voice_style_indexed = voice_style_indexed[None, :]
        return voice_style_indexed

    @staticmethod
    def _pad_tokens(tokens: list[int]) -> list[list[int]]:
        return [[0, *tokens, 0]]

    def _float_speed_input(self, speed: float) -> np.ndarray:
        return np.ones(1, dtype=np.float32) * speed

    def _int_speed_input(self, speed: float) -> np.ndarray:
        speed_int = max(1, round(speed))
        return np.array([speed_int], dtype=np.int32)

    def _input_dtype(self, name: str, default: np.dtype[Any]) -> np.dtype[Any]:
        meta = self._input_metas.get(name)
        type_name = str(getattr(meta, "type", "")).lower() if meta is not None else ""
        return {
            "tensor(float)": np.dtype(np.float32),
            "tensor(float16)": np.dtype(np.float16),
            "tensor(double)": np.dtype(np.float64),
            "tensor(int64)": np.dtype(np.int64),
            "tensor(int32)": np.dtype(np.int32),
            "tensor(int16)": np.dtype(np.int16),
        }.get(type_name, default)

    def _build_onnx_inputs(
        self,
        tokens_padded: list[list[int]],
        voice_style: np.ndarray,
        speed: float,
    ) -> dict[str, np.ndarray | list[list[int]]]:
        token_name = "input_ids" if self._uses_input_ids else "tokens"
        token_dtype = self._input_dtype(token_name, np.dtype(np.int64))
        style_name = "ref_s" if "ref_s" in self._input_metas else "style"
        style_dtype = self._input_dtype(style_name, np.dtype(np.float32))
        speed_dtype = self._input_dtype("speed", np.dtype(np.float32))
        if np.issubdtype(speed_dtype, np.integer):
            speed_input = np.array([max(1, round(speed))], dtype=speed_dtype)
        else:
            speed_input = np.ones(1, dtype=speed_dtype) * speed
        return {
            token_name: np.asarray(tokens_padded, dtype=token_dtype),
            style_name: np.asarray(voice_style, dtype=style_dtype),
            "speed": speed_input,
        }

    def _run_onnx(
        self,
        phonemes: str,
        voice_style: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        tokens = self._tokenize_phonemes(phonemes)
        voice_style_indexed = self._select_voice_style(voice_style, len(tokens))
        tokens_padded = self._pad_tokens(tokens)
        inputs = self._build_onnx_inputs(tokens_padded, voice_style_indexed, speed)
        results = self._session.run(None, inputs)
        audio = np.asarray(results[0]).T
        audio = np.squeeze(audio)
        timestamp_index = self._timestamp_output_index
        pred_dur = (
            np.asarray(results[timestamp_index]).squeeze()
            if timestamp_index is not None and timestamp_index < len(results)
            else None
        )
        return audio, pred_dur

    def generate_from_phonemes(
        self,
        phonemes: str,
        voice_style: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from a single phoneme batch.

        Core ONNX inference for a single phoneme batch.

        Args:
            phonemes: Phoneme string (will be truncated if > MAX_PHONEME_LENGTH)
            voice_style: Voice style vector
            speed: Speech speed multiplier

        Returns:
            Tuple of (audio samples, sample rate)
        """
        audio, _ = self._run_onnx(phonemes, voice_style, speed)
        return audio, SAMPLE_RATE

    def split_phonemes(self, phonemes: str) -> list[str]:  # noqa: C901
        """Split phonemes into batches at sentence-ending punctuation marks.

        Args:
            phonemes: Full phoneme string

        Returns:
            List of phoneme batches, each <= MAX_PHONEME_LENGTH
        """

        batches: list[str] = []
        current = ""
        current_tokens = 0

        def token_len(text: str) -> int:
            if not text:
                return 0
            return len(self._tokenizer.tokenize(text))

        def append_batch(text: str) -> None:
            if text:
                batches.append(text.strip())

        def split_long_sentence(sentence: str) -> bool:
            nonlocal current, current_tokens
            if current:
                append_batch(current)
                current = ""
                current_tokens = 0
            words = re.split(r"([.,;:!?\s])", sentence)
            if len(words) == 1:
                word_tokens = self._tokenizer.tokenize(words[0]) if words[0] else []
                if len(word_tokens) > MAX_PHONEME_LENGTH:
                    for i in range(0, len(word_tokens), MAX_PHONEME_LENGTH):
                        chunk_tokens = word_tokens[i : i + MAX_PHONEME_LENGTH]
                        batches.append(self._tokenizer.detokenize(chunk_tokens))
                    return True
            for word in words:
                if not word or word.isspace():
                    if current:
                        current += " "
                        current_tokens = token_len(current)
                    continue
                word_tokens = self._tokenizer.tokenize(word)
                if len(word_tokens) > MAX_PHONEME_LENGTH:
                    if current:
                        append_batch(current)
                        current = ""
                        current_tokens = 0
                    for i in range(0, len(word_tokens), MAX_PHONEME_LENGTH):
                        chunk_tokens = word_tokens[i : i + MAX_PHONEME_LENGTH]
                        batches.append(self._tokenizer.detokenize(chunk_tokens))
                    continue
                if current_tokens + len(word_tokens) > MAX_PHONEME_LENGTH:
                    if current:
                        append_batch(current)
                    current = word
                    current_tokens = token_len(current)
                else:
                    if current and not current.endswith((".", "!", "?", ",", ";", ":")):
                        current += " "
                    current += word
                    current_tokens = token_len(current)
            return False

        # Split on sentence-ending punctuation (., !, ?) while keeping them
        # Use lookbehind to split AFTER the punctuation
        sentences = re.split(r"(?<=[.!?])\s*", phonemes)

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = token_len(sentence)

            # If adding sentence would exceed limit, save current batch, start new
            if current and current_tokens + sentence_tokens > MAX_PHONEME_LENGTH:
                append_batch(current)
                current = sentence
                current_tokens = sentence_tokens
            # If the sentence itself is too long, we need to split it further
            elif sentence_tokens > MAX_PHONEME_LENGTH:
                if split_long_sentence(sentence):
                    continue
            else:
                # Add sentence to current batch
                if current:
                    current += " "
                current += sentence
                current_tokens = token_len(current)

        if current:
            append_batch(current)

        return batches if batches else [phonemes]

    def generate_from_phoneme_batches(
        self,
        batches: list[str],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
    ) -> np.ndarray:
        """Generate and concatenate audio from phoneme batches.

        Args:
            batches: List of phoneme strings (each <= MAX_PHONEME_LENGTH)
            voice_style: Voice style vector
            speed: Speech speed
            trim_silence: Whether to trim silence from each batch

        Returns:
            Concatenated audio array
        """
        audio_parts = []

        for batch in batches:
            audio, _ = self.generate_from_phonemes(batch, voice_style, speed)
            if trim_silence:
                audio, _ = trim_audio(audio)
            audio_parts.append(audio)

        return np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32)

    def _resolve_segment_voice(
        self,
        segment: PhonemeSegment,
        default_voice_style: np.ndarray,
        voice_resolver: Callable[[str], np.ndarray] | None,
    ) -> np.ndarray:
        """Resolve voice style for a segment, checking SSMD voice metadata.

        Args:
            segment: Phoneme segment to process
            default_voice_style: Default voice style if no metadata present
            voice_resolver: Optional callback to resolve voice names

        Returns:
            Voice style array for this segment
        """
        # Use default voice by default
        segment_voice_style = default_voice_style

        # Check for SSMD voice metadata override
        if voice_resolver and segment.ssmd_metadata:
            voice_name = segment.ssmd_metadata.get("voice_name")
            if not voice_name:
                voice_name = segment.ssmd_metadata.get("voice")
            if voice_name:
                try:
                    segment_voice_style = voice_resolver(voice_name)
                except (KeyError, RuntimeError, OSError, ValueError) as exc:
                    if segment.ssmd_metadata.get("missing_voice_policy") == "error":
                        raise ConfigurationError(
                            f"Unable to resolve SSMD voice target '{voice_name}'"
                        ) from exc
                    logger.warning(
                        "Failed to resolve voice '%s' for segment; using default voice: %s",
                        voice_name,
                        exc,
                    )

        return segment_voice_style

    def _resolve_short_sentence_config(
        self, enable_short_sentence_override: bool | None
    ) -> ShortSentenceConfig | None:
        from .short_sentence_handler import ShortSentenceConfig, WrapResolveMode

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
        elif effective_config is None:
            effective_config = ShortSentenceConfig()

        if (
            effective_config is not None
            and effective_config.enabled
            and not self._has_timestamp_output
            and self._uses_phrase_short_sentence_mode(effective_config)
        ):
            if not self._reported_missing_timestamp_output:
                message = (
                    "Loaded ONNX model has no timestamp output; phrase-based short "
                    "sentence modes require timestamps. Falling back to wrap mode "
                    "for this run."
                )
                print(message)
                self._reported_missing_timestamp_output = True
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
    ) -> list[PhonemeSegment]:
        from .short_sentence_handler import is_segment_empty, is_segment_short

        effective_config = self._resolve_short_sentence_config(enable_short_sentence_override)
        phrase_rng = random.Random(random_seed) if random_seed is not None else None
        processed: list[PhonemeSegment] = []

        for segment in segments:
            phonemes = segment.phonemes or ""
            tokens = self._tokenizer.tokenize(phonemes) if phonemes.strip() else []
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
                    )
                    phonemes = short_sentence.phonemes
                    tokens = short_sentence.tokens
                    if short_sentence.metadata is not None:
                        metadata = dict(segment.ssmd_metadata or {})
                        metadata[SHORT_SENTENCE_META_KEY] = short_sentence.metadata
                        segment = dataclasses.replace(segment, ssmd_metadata=metadata)
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
                total_batches = len(batches)
                for idx, batch_tokens in enumerate(batches):
                    batch_phonemes = self._tokenizer.detokenize(batch_tokens)
                    processed.append(
                        dataclasses.replace(
                            segment,
                            id=f"{segment.id}_ph{idx}",
                            phoneme_id=idx,
                            phonemes=batch_phonemes,
                            tokens=list(batch_tokens),
                            pause_before=segment.pause_before if idx == 0 else 0.0,
                            pause_after=(segment.pause_after if idx == total_batches - 1 else 0.0),
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
                        pause_before=segment.pause_before,
                        pause_after=segment.pause_after,
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
        voice_resolver: Callable[[str], np.ndarray] | None,
    ) -> list[PhonemeSegment]:
        for segment in segments:
            if not segment.phonemes.strip():
                segment.raw_audio = None
                continue

            segment_voice_style = self._resolve_segment_voice(segment, voice_style, voice_resolver)
            audio, pred_dur = self._run_onnx(segment.phonemes, segment_voice_style, speed)
            segment.word_timings = self._map_pred_dur_to_word_timings(segment, pred_dur, len(audio))
            self._log_short_sentence_timestamps(segment, pred_dur)
            segment.raw_audio = self._prepare_short_sentence_phrase_audio(
                segment,
                audio,
                segment_voice_style,
                speed,
            )

        return segments

    def _map_pred_dur_to_word_timings(
        self, segment: PhonemeSegment, pred_dur: np.ndarray | None, audio_length: int
    ) -> list[WordTiming]:
        metadata = (segment.ssmd_metadata or {}).get(SHORT_SENTENCE_META_KEY)
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
        pred_dur: np.ndarray | None,
        audio_length: int,
        *,
        local_target_offsets: bool = False,
    ) -> list[WordTiming]:
        if pred_dur is None or audio_length <= 0:
            return []
        durations = np.asarray(pred_dur).reshape(-1)
        if durations.size < 3 or not np.isfinite(durations).all() or np.any(durations < 0):
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
        timestamped = _join_timestamps(cast(list[object], valid_tokens), durations, strict=True)
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
        pred_dur: np.ndarray | None,
    ) -> None:
        if pred_dur is None:
            return
        short_sentence_metadata = (segment.ssmd_metadata or {}).get(SHORT_SENTENCE_META_KEY)
        if not isinstance(short_sentence_metadata, dict):
            return
        timing_tokens = short_sentence_metadata.get("timing_tokens")
        if not isinstance(timing_tokens, list):
            return
        timestamped = _join_timestamps(timing_tokens, pred_dur)
        populate_short_sentence_boundary_metadata(
            short_sentence_metadata,
            timestamped,
        )
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
    ) -> np.ndarray:
        """Accept confident phrase cuts or regenerate a wrap fallback."""
        short_sentence_metadata = (segment.ssmd_metadata or {}).get(SHORT_SENTENCE_META_KEY)
        if not isinstance(short_sentence_metadata, dict):
            return audio
        if short_sentence_metadata.get("kind") not in {"phrase", "randomized-phrase"}:
            return audio

        cut_audio = cut_short_sentence_phrase_audio(audio, short_sentence_metadata)
        if cut_audio is not None:
            left_cut = short_sentence_metadata.get("cut_left")
            right_cut = short_sentence_metadata.get("cut_right")
            if isinstance(left_cut, int) and isinstance(right_cut, int):
                segment.word_timings = _crop_word_timings(segment.word_timings, left_cut, right_cut)
            short_sentence_metadata["cut_applied"] = True
            return cut_audio

        retry_audio = self._try_short_sentence_phrase_fallbacks(
            segment,
            short_sentence_metadata,
            voice_style,
            speed,
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

        logger.warning(
            "Short sentence phrase cut for '%s' lacked confident boundaries; "
            "falling back to wrap mode.",
            segment.text[:50],
        )
        fallback_audio, _ = self._run_onnx(fallback_phonemes, voice_style, speed)
        short_sentence_metadata["cut_applied"] = True
        short_sentence_metadata["fallback_used"] = "wrap"
        segment.phonemes = fallback_phonemes
        fallback_tokens = short_sentence_metadata.get("fallback_tokens")
        if isinstance(fallback_tokens, list) and all(
            isinstance(token, int) for token in fallback_tokens
        ):
            segment.tokens = fallback_tokens
        segment.word_timings = []
        return fallback_audio

    def _try_short_sentence_phrase_fallbacks(
        self,
        segment: PhonemeSegment,
        short_sentence_metadata: dict[str, object],
        voice_style: np.ndarray,
        speed: float,
    ) -> np.ndarray | None:
        templates = short_sentence_metadata.get("phrase_fallback_templates")
        if not isinstance(templates, list):
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
            )
            if retry is None or retry.metadata is None:
                continue

            retry_audio, pred_dur = self._run_onnx(retry.phonemes, voice_style, speed)
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
                timestamped = _join_timestamps(timing_tokens, pred_dur)
                populate_short_sentence_boundary_metadata(retry.metadata, timestamped)
            cut_audio = cut_short_sentence_phrase_audio(retry_audio, retry.metadata)
            if cut_audio is None:
                failed_template = template
                continue

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
        prosody_config: ProsodyConfig | None = None,
        trace: Trace | None = None,
    ) -> list[PhonemeSegment]:
        for segment in segments:
            if segment.raw_audio is None:
                segment.processed_audio = None
                continue

            if not trim_silence and not segment.ssmd_metadata:
                segment.processed_audio = segment.raw_audio
                continue

            audio = segment.raw_audio
            short_sentence_metadata = (segment.ssmd_metadata or {}).get(SHORT_SENTENCE_META_KEY)
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
            if (
                trim_silence
                or (segment.ssmd_metadata or {}).get("deterministic_pause_boundary") == "true"
            ):
                trim_result: Any = trim_audio(audio)
                if isinstance(trim_result, tuple) and len(trim_result) == 2:
                    audio, trim_bounds = trim_result
                    segment.word_timings = _crop_word_timings(
                        segment.word_timings, int(trim_bounds[0]), int(trim_bounds[1])
                    )
                else:
                    audio = trim_result
            old_length = len(audio)
            processed_audio = self._apply_segment_prosody(
                audio,
                segment,
                prosody_config,
                trace,
            )
            if old_length and len(processed_audio) != old_length:
                segment.word_timings = _scale_word_timings(
                    segment.word_timings, old_length, len(processed_audio)
                )
            segment.processed_audio = processed_audio

        return segments

    def _concatenate_audio_segments(
        self,
        segments: list[PhonemeSegment],
        prosody_config: ProsodyConfig | None = None,
        trace: Trace | None = None,
    ) -> np.ndarray:
        audio_parts: list[np.ndarray] = []
        cursor_samples = 0
        previous_index: int | None = None
        previous_segment: PhonemeSegment | None = None

        for segment in segments:
            if segment.pause_before > 0:
                pause = generate_silence(segment.pause_before, SAMPLE_RATE)
                audio_parts.append(pause)
                cursor_samples += len(pause)
                previous_index = None
                previous_segment = None

            if segment.processed_audio is not None:
                segment.word_timings = _translate_word_timings(segment.word_timings, cursor_samples)
                current = np.asarray(segment.processed_audio)
                if (
                    previous_index is not None
                    and previous_segment is not None
                    and _should_condition_boundary(previous_segment, segment, prosody_config)
                ):
                    left = audio_parts[previous_index]
                    boundary_before = _boundary_jump(left, current)
                    conditioned_left, conditioned_right = _condition_boundary(
                        left,
                        current,
                        sample_rate=SAMPLE_RATE,
                        blend_ms=(
                            prosody_config.boundary_blend_ms if prosody_config is not None else 0.0
                        ),
                    )
                    audio_parts[previous_index] = conditioned_left
                    current = conditioned_right
                    _record_boundary_diagnostic(
                        trace,
                        segment,
                        boundary_before,
                        _boundary_jump(conditioned_left, conditioned_right),
                        min(len(left), len(current)),
                    )
                audio_parts.append(current)
                previous_index = len(audio_parts) - 1
                previous_segment = segment
                cursor_samples += len(current)
            else:
                previous_index = None
                previous_segment = None

            if segment.pause_after > 0:
                pause = generate_silence(segment.pause_after, SAMPLE_RATE)
                audio_parts.append(pause)
                cursor_samples += len(pause)
                previous_index = None
                previous_segment = None

        return np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32)

    def generate_from_segments(
        self,
        segments: list[PhonemeSegment],
        voice_style: np.ndarray,
        speed: float,
        trim_silence: bool,
        voice_resolver: Callable[[str], np.ndarray] | None = None,
        enable_short_sentence_override: bool | None = None,
        random_seed: int | None = None,
        prosody_config: ProsodyConfig | None = None,
    ) -> np.ndarray:
        """Generate audio from list of PhonemeSegment instances.

        Unified audio generation method that handles:
        - Segments with phonemes (generate speech)
        - Empty segments (skip, only use pause_after)
        - Pause insertion based on pause_before and pause_after fields
        - Per-segment voice switching via SSMD voice metadata
        - Optional silence trimming
        - Per-call short sentence handling override

        Args:
            segments: List of PhonemeSegment instances
            voice_style: Default voice style vector (used when no voice metadata)
            speed: Speech speed multiplier
            trim_silence: Whether to trim silence from segment boundaries
            voice_resolver: Optional callback to resolve voice names to style vectors.
                Takes voice name (str) and returns voice style array.
                If provided and segment has voice metadata, uses per-segment voice.
            enable_short_sentence_override: Override short sentence handling.
                None (default): Use config setting
                True: Force enable short sentence handling
                False: Force disable short sentence handling
            random_seed: Optional seed for reproducible randomized short-sentence
                phrase selection.

        Returns:
            Concatenated audio array
        """
        preprocessed = self._preprocess_segments(
            segments, enable_short_sentence_override, random_seed
        )
        generated = self._generate_raw_audio_segments(
            preprocessed, voice_style, speed, voice_resolver
        )
        processed = self._postprocess_audio_segments(generated, trim_silence, prosody_config)
        return self._concatenate_audio_segments(processed, prosody_config)

    def _apply_segment_prosody(
        self,
        audio: np.ndarray,
        segment: PhonemeSegment,
        prosody_config: ProsodyConfig | None = None,
        trace: Trace | None = None,
    ) -> np.ndarray:
        """Apply prosody modifications from segment metadata to audio.

        Args:
            audio: Input audio array
            segment: PhonemeSegment with potential prosody metadata

        Returns:
            Audio with prosody modifications applied
        """
        if not segment.ssmd_metadata:
            return audio

        volume = segment.ssmd_metadata.get("prosody_volume")
        pitch = segment.ssmd_metadata.get("prosody_pitch")
        rate = segment.ssmd_metadata.get("prosody_rate")

        # Apply prosody if any prosody metadata is present
        if volume or pitch or rate:
            source_metrics = _waveform_metrics(audio)
            resolved_config = prosody_config or ProsodyConfig()
            parsed: dict[str, float | None] = {}
            for name, value, parser in (
                ("volume_db", volume, parse_volume),
                ("pitch_semitones", pitch, parse_pitch),
                ("rate_multiplier", rate, parse_rate),
            ):
                if value is None:
                    parsed[name] = None
                    continue
                try:
                    parsed[name] = float(parser(value))
                except (TypeError, ValueError):
                    parsed[name] = None

            started = time.perf_counter()
            audio = apply_prosody(
                audio,
                SAMPLE_RATE,
                volume=volume,
                pitch=pitch,
                rate=rate,
                config=resolved_config,
            )
            if trace is not None:
                output_metrics = _waveform_metrics(audio)
                short_sentence = (segment.ssmd_metadata or {}).get(SHORT_SENTENCE_META_KEY)
                trace.prosody.append(
                    {
                        "segment_id": segment.id,
                        "text": segment.text,
                        "method": resolved_config.method,
                        "strict": resolved_config.strict,
                        "rate": rate,
                        "pitch": pitch,
                        "volume": volume,
                        "rate_multiplier": parsed["rate_multiplier"],
                        "pitch_semitones": parsed["pitch_semitones"],
                        "volume_db": parsed["volume_db"],
                        "short_sentence": dict(short_sentence)
                        if isinstance(short_sentence, dict)
                        else None,
                        "runtime_ms": (time.perf_counter() - started) * 1000.0,
                        "source": source_metrics,
                        "output": output_metrics,
                        "edge_jump_before": source_metrics["max_adjacent_jump"],
                        "edge_jump_after": output_metrics["max_adjacent_jump"],
                    }
                )

        return audio

    def generate_from_tokens(
        self,
        tokens: list[int],
        voice_style: np.ndarray,
        speed: float,
    ) -> tuple[np.ndarray, int]:
        """Generate audio from token IDs directly.

        This provides the lowest-level interface, useful for pre-tokenized
        content and maximum control.

        Args:
            tokens: List of token IDs
            voice_style: Voice style vector
            speed: Speech speed

        Returns:
            Tuple of (audio samples as numpy array, sample rate)
        """
        # Detokenize to phonemes and generate audio
        phonemes = self._tokenizer.detokenize(tokens)

        # Split phonemes into batches and generate audio
        batches = self.split_phonemes(phonemes)
        audio = self.generate_from_phoneme_batches(batches, voice_style, speed, trim_silence=False)

        return audio, SAMPLE_RATE


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


def _scale_word_timings(
    timings: list[WordTiming], old_length: int, new_length: int
) -> list[WordTiming]:
    if old_length <= 0 or new_length <= 0:
        return []
    return [
        dataclasses.replace(
            timing,
            start_sample=max(
                0, min(new_length, round(timing.start_sample * new_length / old_length))
            ),
            end_sample=max(0, min(new_length, round(timing.end_sample * new_length / old_length))),
        )
        for timing in timings
        if timing.start_sample < timing.end_sample
    ]


def _translate_word_timings(timings: list[WordTiming], sample_offset: int) -> list[WordTiming]:
    if not sample_offset:
        return list(timings)
    return [
        dataclasses.replace(
            timing,
            start_sample=timing.start_sample + sample_offset,
            end_sample=timing.end_sample + sample_offset,
        )
        for timing in timings
    ]


def _join_timestamps(
    tokens: list[object],
    pred_dur: np.ndarray,
    *,
    strict: bool = False,
) -> list[dict[str, object]]:
    """Map model durations to G2P tokens, optionally rejecting partial mappings."""
    durations = np.asarray(pred_dur).reshape(-1)
    if not tokens or len(durations) < 3:
        return []

    timestamped: list[dict[str, object]] = []
    divisor = 80
    left = right = 2 * max(0.0, float(durations[0].item()) - 3)
    i = 1
    complete = True

    for raw_token in tokens:
        if i >= len(durations) - 1:
            complete = False
            break
        token = dict(raw_token) if isinstance(raw_token, dict) else {}
        phonemes = str(token.get("phonemes") or "")
        whitespace = str(token.get("whitespace") or "")

        if not phonemes:
            if whitespace and i < len(durations):
                i += 1
                if i < len(durations):
                    left = right + float(durations[i].item())
                    right = left + float(durations[i].item())
                    i += 1
                else:
                    complete = False
            timestamped.append(token)
            continue

        token_count = token.get("model_token_count")
        if not isinstance(token_count, int) or isinstance(token_count, bool) or token_count <= 0:
            token_count = len(phonemes)
        model_span_count = _model_span_token_count(token)
        if model_span_count is None:
            model_span_count = token_count + (1 if whitespace else 0)
        speech_end_index = i + token_count
        if speech_end_index >= len(durations):
            complete = False
            break
        space_dur = 0.0
        if whitespace:
            if speech_end_index >= len(durations) - 1:
                complete = False
                break
            space_dur = float(durations[speech_end_index].item())
        token["start_ts"] = left / divisor
        token_dur = float(durations[i:speech_end_index].sum().item())
        speech_end = right + (2 * token_dur)
        token["speech_end_ts"] = speech_end / divisor
        left = speech_end + space_dur
        token["end_ts"] = left / divisor
        right = left + space_dur
        i += model_span_count
        timestamped.append(token)

    if strict and not complete:
        return []
    return timestamped


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


def _is_spoken_token(token: dict[str, object]) -> bool:
    """Return whether a token corresponds to spoken lexical content."""
    text = str(token.get("text") or "")
    return any(char.isalnum() for char in text)
