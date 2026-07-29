"""Audio generation for PyKokoro."""

from __future__ import annotations

import dataclasses
import logging
import random
import re
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

from .constants import MAX_PHONEME_LENGTH, SAMPLE_RATE
from .exceptions import ConfigurationError
from .prosody import apply_prosody
from .short_sentence_handler import (
    SHORT_SENTENCE_META_KEY,
    apply_short_sentence_mode,
    build_short_sentence_phrase_retry,
    cut_short_sentence_phrase_audio,
)
from .tokenizer import Tokenizer
from .trim import trim as trim_audio
from .types import PhonemeSegment
from .utils import generate_silence
from .voice_manager import normalize_voice_style

if TYPE_CHECKING:
    import onnxruntime as rt

    from .short_sentence_handler import ShortSentenceConfig

logger = logging.getLogger(__name__)


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
        target_length = max(1, round(len(audio) / factor))
        positions = np.linspace(0, max(0, len(audio) - 1), target_length)
        audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)

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
        audio = audio * (10.0 ** (float(match.group(1)) / 20.0))

    if source_rate != sample_rate and audio.size:
        target_length = max(1, round(len(audio) * sample_rate / source_rate))
        positions = np.linspace(0, max(0, len(audio) - 1), target_length)
        audio = np.interp(positions, np.arange(len(audio)), audio).astype(np.float32)
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
        self._uses_input_ids = any(
            input_meta.name == "input_ids" for input_meta in session.get_inputs()
        )
        get_outputs = getattr(session, "get_outputs", None)
        outputs = get_outputs() if callable(get_outputs) else []
        try:
            self._has_timestamp_output = len(outputs) > 1
        except TypeError:
            # Some lightweight test doubles expose get_outputs() as a bare Mock.
            # Treat an unsized result as no timestamp output.
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

    def _build_onnx_inputs(
        self,
        tokens_padded: list[list[int]],
        voice_style: np.ndarray,
        speed: float,
    ) -> dict[str, np.ndarray | list[list[int]]]:
        if self._uses_input_ids:
            if self._model_source == "github":
                return {
                    "input_ids": np.array(tokens_padded, dtype=np.int64),
                    "style": np.array(voice_style, dtype=np.float32),
                    "speed": self._int_speed_input(speed),
                }
            return {
                "input_ids": tokens_padded,
                "style": voice_style,
                "speed": self._float_speed_input(speed),
            }
        return {
            "tokens": tokens_padded,
            "style": voice_style,
            "speed": self._float_speed_input(speed),
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
        pred_dur = np.asarray(results[1]).squeeze() if len(results) > 1 else None
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
            self._log_short_sentence_timestamps(segment, pred_dur)
            segment.raw_audio = self._prepare_short_sentence_phrase_audio(
                segment,
                audio,
                segment_voice_style,
                speed,
            )

        return segments

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
            if pred_dur is not None and isinstance(timing_tokens, list):
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
            short_sentence_metadata.clear()
            short_sentence_metadata.update(retry.metadata)
            short_sentence_metadata["cut_applied"] = True
            short_sentence_metadata["fallback_used"] = "phrase"
            short_sentence_metadata["retry_attempts"] = retry_attempts
            segment.phonemes = retry.phonemes
            segment.tokens = retry.tokens
            return cut_audio

        short_sentence_metadata["retry_attempts"] = retry_attempts
        return None

    def _postprocess_audio_segments(
        self, segments: list[PhonemeSegment], trim_silence: bool
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
                    audio = cut_audio
            if (
                trim_silence
                or (segment.ssmd_metadata or {}).get("deterministic_pause_boundary") == "true"
            ):
                audio, _ = trim_audio(audio)
            segment.processed_audio = self._apply_segment_prosody(audio, segment)

        return segments

    def _concatenate_audio_segments(self, segments: list[PhonemeSegment]) -> np.ndarray:
        audio_parts: list[np.ndarray] = []

        for segment in segments:
            if segment.pause_before > 0:
                audio_parts.append(generate_silence(segment.pause_before, SAMPLE_RATE))
            if segment.processed_audio is not None:
                audio_parts.append(segment.processed_audio)
            if segment.pause_after > 0:
                audio_parts.append(generate_silence(segment.pause_after, SAMPLE_RATE))

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
        processed = self._postprocess_audio_segments(generated, trim_silence)
        return self._concatenate_audio_segments(processed)

    def _apply_segment_prosody(self, audio: np.ndarray, segment: PhonemeSegment) -> np.ndarray:
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
            audio = apply_prosody(audio, SAMPLE_RATE, volume=volume, pitch=pitch, rate=rate)

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


def _join_timestamps(
    tokens: list[object],
    pred_dur: np.ndarray,
) -> list[dict[str, object]]:
    """Map timestamped model durations to G2P token metadata."""
    durations = np.asarray(pred_dur).reshape(-1)
    if not tokens or len(durations) < 3:
        return []

    timestamped: list[dict[str, object]] = []
    divisor = 80
    left = right = 2 * max(0.0, float(durations[0].item()) - 3)
    i = 1

    for raw_token in tokens:
        if i >= len(durations) - 1:
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
            timestamped.append(token)
            continue

        j = i + len(phonemes)
        if j >= len(durations):
            break

        token["start_ts"] = left / divisor
        token_dur = float(durations[i:j].sum().item())
        space_dur = float(durations[j].item()) if whitespace else 0.0
        speech_end = right + (2 * token_dur)
        token["speech_end_ts"] = speech_end / divisor
        left = speech_end + space_dur
        token["end_ts"] = left / divisor
        right = left + space_dur
        i = j + (1 if whitespace else 0)
        timestamped.append(token)

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
