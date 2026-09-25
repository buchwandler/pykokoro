"""Request-local ONNX inference and postprocessing for the public synthesis engine."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .exceptions import (
    BackendError,
    ConfigurationError,
    InvalidVoiceError,
    SynthesisInputTooLongError,
)
from .model_profiles import get_model_profile
from .prepared_g2p import PreparedSynthesis
from .synthesis_config import SynthesisConfig, resolve_synthesis_config
from .synthesis_identity import build_synthesis_identity
from .synthesis_types import (
    RenderedSegment,
    SynthesisSegment,
)
from .types import PhonemeSegment, Trace, WordTiming
from .voice_level import resolve_voice_level_application


class _RequestG2PAdapter(Protocol):
    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> Any: ...

    def phonemize(
        self, segment: SynthesisSegment, config: SynthesisConfig
    ) -> PreparedSynthesis: ...


class OnnxRequestRenderer:
    """Render each prepared request through Kokoro's ONNX inference backend."""

    def __init__(
        self,
        g2p_adapter: _RequestG2PAdapter,
        backend_factory: Callable[[SynthesisConfig], Any] | None = None,
    ) -> None:
        self._g2p_adapter = g2p_adapter
        self._backend_factory = backend_factory or self._create_backend
        self._backend: Any | None = None
        self._backend_key: tuple[str, ...] | None = None

    def close(self) -> None:
        """Release the active model runtime and its cached inference resources."""
        if self._backend is not None:
            close = getattr(self._backend, "close", None)
            if callable(close):
                close()
        self._backend = None
        self._backend_key = None

    def render(
        self,
        prepared: PreparedSynthesis,
        request: SynthesisSegment,
        config: SynthesisConfig,
    ) -> RenderedSegment:
        resolved = resolve_synthesis_config(config, language=request.language, voice=request.voice)
        identity = build_synthesis_identity(
            resolved, language=request.language, voice=resolved.voice
        )
        profile = get_model_profile(
            resolved.model_variant or "v1.0", resolved.model_source or "github"
        )
        voice = resolved.voice
        voice_name = voice if isinstance(voice, str) else None
        trace = Trace() if resolved.return_trace else None
        if trace is not None:
            trace.model.update(
                {
                    "model_source": resolved.model_source,
                    "model_variant": resolved.model_variant,
                    "model_quality": resolved.model_quality,
                    "synthesis_identity": identity.to_dict(),
                    "voice": voice_name,
                }
            )
            trace.warnings.extend(prepared.diagnostics)

        if not prepared.token_ids:
            raise BackendError(f"frontend produced no model tokens for request {request.id!r}")
        if voice is None:
            raise InvalidVoiceError("a voice must resolve before ONNX rendering")

        phoneme_segments = self._build_phoneme_segments(
            request, prepared, resolved, profile.max_tokens, trace
        )
        backend = self._get_backend(resolved)
        try:
            voice_style = backend.resolve_voice_style(voice)
        except (KeyError, ValueError, FileNotFoundError, ConfigurationError) as exc:
            raise InvalidVoiceError(
                f"Voice {voice!r} is unavailable for the resolved model"
            ) from exc

        def context_phonemizer(text: str, language: str) -> Any:
            return self._g2p_adapter.phonemize_context(text, language, resolved)

        phoneme_segments = backend.preprocess_segments(
            phoneme_segments,
            resolved.generation.enable_short_sentence,
            random_seed=resolved.generation.random_seed,
            context_phonemizer=context_phonemizer,
        )
        oversized_segment = next(
            (segment for segment in phoneme_segments if len(segment.tokens) > profile.max_tokens),
            None,
        )
        if oversized_segment is not None:
            raise SynthesisInputTooLongError(
                text_length=len(oversized_segment.text),
                token_count=len(oversized_segment.tokens),
                max_tokens=profile.max_tokens,
                model_id=identity.model_id,
            )
        if not phoneme_segments:
            raise BackendError(f"request {request.id!r} produced no acoustic chunks")
        if trace is not None:
            trace.model["acoustic_chunk_count"] = len(phoneme_segments)
        raw_segments = backend.generate_raw_audio_segments(
            phoneme_segments,
            voice_style,
            resolved.generation.speed,
            default_voice_name=voice_name,
            trace=trace,
        )

        processed_segments = backend.postprocess_audio_segments(
            raw_segments,
            trim_silence=False,
            trace=trace,
            voice_level_config=resolved.voice_level,
        )
        if len(processed_segments) != len(phoneme_segments):
            raise BackendError(
                "postprocessing must preserve acoustic chunks; "
                f"received {len(processed_segments)} for {len(phoneme_segments)} input chunks"
            )
        audio_parts: list[np.ndarray] = []
        word_timings: list[WordTiming] = []
        sample_offset = 0
        for phoneme_segment in processed_segments:
            audio = phoneme_segment.processed_audio
            if audio is None:
                continue
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            audio_parts.append(audio)
            word_timings.extend(
                replace(
                    timing,
                    start_sample=timing.start_sample + sample_offset,
                    end_sample=timing.end_sample + sample_offset,
                    segment_id=request.id,
                )
                for timing in phoneme_segment.word_timings
            )
            sample_offset += len(audio)
        rendered_audio = (
            np.concatenate(audio_parts).astype(np.float32, copy=False)
            if audio_parts
            else np.zeros(0, dtype=np.float32)
        )
        if rendered_audio.size == 0:
            raise BackendError(f"backend produced no audio samples for request {request.id!r}")
        voice_level_applications = tuple(
            resolve_voice_level_application(resolved.voice_level, segment.render_voice_key)
            for segment in processed_segments
            if segment.processed_audio is not None
        )
        short_metadata = (processed_segments[0].engine_metadata or {}).get("__short_sentence")
        short_sentence_mode = None
        if isinstance(short_metadata, dict):
            short_kind = short_metadata.get("kind")
            if isinstance(short_kind, str):
                short_sentence_mode = short_kind
        identity = replace(identity, resolved_short_sentence_mode=short_sentence_mode)
        if trace is not None:
            trace.model["synthesis_identity"] = identity.to_dict()
        return RenderedSegment(
            id=request.id,
            audio=rendered_audio,
            sample_rate=profile.sample_rate,
            text=request.text,
            language=request.language,
            voice=voice_name,
            phonemes=prepared.phonemes,
            token_ids=prepared.token_ids,
            word_timings=tuple(word_timings),
            diagnostics=prepared.diagnostics,
            trace=trace,
            synthesis_identity=identity,
            voice_level_applications=voice_level_applications,
            short_sentence_mode=short_sentence_mode,
        )

    def _get_backend(self, config: SynthesisConfig) -> Any:
        key = self._backend_configuration_key(config)
        if self._backend is None or key != self._backend_key:
            self.close()
            self._backend = self._backend_factory(config)
            self._backend_key = key
        return self._backend

    @staticmethod
    def _backend_configuration_key(config: SynthesisConfig) -> tuple[str, ...]:
        fields = (
            "model_quality",
            "model_source",
            "model_variant",
            "model_path",
            "voices_path",
            "model_config_path",
            "release_manifest_path",
            "provider",
            "provider_options",
            "session_options",
            "tokenizer_config",
            "espeak_config",
            "short_sentence_config",
            "waveform_validation",
            "inference_audio_diagnostics",
            "inference_cache_enabled",
            "inference_cache_max_bytes",
            "asset_progress",
            "allow_experimental_frontend",
        )
        return tuple(repr(getattr(config, name)) for name in fields)

    @staticmethod
    def _create_backend(config: SynthesisConfig) -> Any:
        from .onnx_backend import Kokoro

        profile = get_model_profile(config.model_variant or "v1.0", config.model_source or "github")
        return Kokoro(
            model_path=Path(config.model_path) if config.model_path is not None else None,
            voices_path=Path(config.voices_path) if config.voices_path is not None else None,
            model_config_path=(
                Path(config.model_config_path) if config.model_config_path is not None else None
            ),
            provider=config.provider,
            session_options=config.session_options,
            provider_options=config.provider_options,
            vocab_version=profile.tokenizer_vocab_version,
            espeak_config=config.espeak_config,
            tokenizer_config=config.tokenizer_config,
            model_quality=config.model_quality,
            model_source=config.model_source or "github",
            model_variant=config.model_variant or "v1.0",
            short_sentence_config=config.short_sentence_config,
            waveform_validation=config.waveform_validation,
            inference_audio_diagnostics=config.inference_audio_diagnostics,
            inference_cache_enabled=config.inference_cache_enabled,
            inference_cache_max_bytes=config.inference_cache_max_bytes,
            asset_progress=config.asset_progress,
        )

    def _build_phoneme_segments(
        self,
        request: SynthesisSegment,
        prepared: PreparedSynthesis,
        config: SynthesisConfig,
        max_tokens: int,
        trace: Trace | None = None,
    ) -> list[PhonemeSegment]:
        if request.voice is None and config.voice is not None:
            request = replace(request, voice=config.voice)
        token_ids = list(prepared.token_ids)
        if not token_ids:
            raise BackendError(f"frontend produced no model tokens for request {request.id!r}")
        if len(token_ids) <= max_tokens:
            segments = [
                self._phoneme_segment_from_prepared(
                    request, prepared, start=0, end=len(request.text), phoneme_id=0
                )
            ]
        elif config.long_text_split == "sentence":
            if request.phonemes is not None:
                raise SynthesisInputTooLongError(
                    "whole-request phonemes cannot be split while preserving source alignment",
                    text_length=len(request.text),
                    token_count=len(token_ids),
                    max_tokens=max_tokens,
                    model_id=config.model_identity,
                )
            segments = self._split_oversized_request(request, config, max_tokens)
        else:
            model_id = config.model_identity or (
                f"{config.model_source}:{config.model_variant}:{config.model_quality}"
            )
            raise SynthesisInputTooLongError(
                text_length=len(request.text),
                token_count=len(token_ids),
                max_tokens=max_tokens,
                model_id=model_id,
            )

        if trace is not None:
            trace.model.update(
                {
                    "model_token_count": sum(len(segment.tokens) for segment in segments),
                    "model_max_tokens": max_tokens,
                    "model_chunk_token_counts": [len(segment.tokens) for segment in segments],
                    "acoustic_chunk_count": len(segments),
                }
            )
        return segments

    def _split_oversized_request(
        self, request: SynthesisSegment, config: SynthesisConfig, max_tokens: int
    ) -> list[PhonemeSegment]:
        prepared_spans: dict[tuple[int, int], PreparedSynthesis] = {}

        def phonemize_span(start: int, end: int) -> PreparedSynthesis:
            key = (start, end)
            if key not in prepared_spans:
                prepared_spans[key] = self._phonemize_source_span(
                    request, config, start=start, end=end
                )
            return prepared_spans[key]

        sentence_spans = self._split_source_spans(
            request, config, start=0, end=len(request.text), mode="sentence"
        )
        safe_spans: list[tuple[int, int]] = []
        for sentence_start, sentence_end in sentence_spans:
            sentence_prepared = phonemize_span(sentence_start, sentence_end)
            if len(sentence_prepared.token_ids) <= max_tokens:
                safe_spans.append((sentence_start, sentence_end))
                continue

            clause_spans = self._split_source_spans(
                request,
                config,
                start=sentence_start,
                end=sentence_end,
                mode="clause",
            )
            if len(clause_spans) == 1 and clause_spans[0] == (sentence_start, sentence_end):
                clause_spans = self._word_source_spans(
                    request, start=sentence_start, end=sentence_end
                )
            for clause_start, clause_end in clause_spans:
                clause_prepared = phonemize_span(clause_start, clause_end)
                if len(clause_prepared.token_ids) <= max_tokens:
                    safe_spans.append((clause_start, clause_end))
                    continue

                word_spans = self._word_source_spans(request, start=clause_start, end=clause_end)
                for word_start, word_end in word_spans:
                    word_prepared = phonemize_span(word_start, word_end)
                    if len(word_prepared.token_ids) > max_tokens:
                        model_id = config.model_identity or (
                            f"{config.model_source}:{config.model_variant}:{config.model_quality}"
                        )
                        raise SynthesisInputTooLongError(
                            "a single word exceeds the model token limit and cannot be split safely",
                            text_length=word_end - word_start,
                            token_count=len(word_prepared.token_ids),
                            max_tokens=max_tokens,
                            model_id=model_id,
                        )
                    safe_spans.append((word_start, word_end))

        if not safe_spans:
            raise BackendError(f"PhraseSplit produced no usable spans for request {request.id!r}")

        packed: list[tuple[int, int, PreparedSynthesis]] = []
        chunk_start, chunk_end = safe_spans[0]
        chunk_prepared = phonemize_span(chunk_start, chunk_end)
        for next_start, next_end in safe_spans[1:]:
            candidate = phonemize_span(chunk_start, next_end)
            if len(candidate.token_ids) <= max_tokens:
                chunk_end = next_end
                chunk_prepared = candidate
                continue
            packed.append((chunk_start, chunk_end, chunk_prepared))
            chunk_start, chunk_end = next_start, next_end
            chunk_prepared = phonemize_span(chunk_start, chunk_end)
        packed.append((chunk_start, chunk_end, chunk_prepared))

        return [
            self._phoneme_segment_from_prepared(
                request, prepared, start=start, end=end, phoneme_id=index
            )
            for index, (start, end, prepared) in enumerate(packed)
        ]

    def _phonemize_source_span(
        self,
        request: SynthesisSegment,
        config: SynthesisConfig,
        *,
        start: int,
        end: int,
    ) -> PreparedSynthesis:
        text = request.text[start:end]
        overrides = tuple(
            replace(
                override,
                start=max(start, override.start) - start,
                end=min(end, override.end) - start,
            )
            for override in request.pronunciation_overrides
            if override.start < end and override.end > start
        )
        tokens = tuple(
            replace(
                token,
                start=overlap_start - start,
                end=overlap_end - start,
                text=request.text[overlap_start:overlap_end] if token.text is not None else None,
            )
            for token in request.tokens
            if (overlap_start := max(start, token.start)) < (overlap_end := min(end, token.end))
        )
        child_request = SynthesisSegment(
            id=f"{request.id}:long:{start}-{end}",
            text=text,
            language=request.language,
            voice=request.voice,
            pronunciation_overrides=overrides,
            tokens=tokens,
        )
        return self._g2p_adapter.phonemize(child_request, config)

    def _split_source_spans(
        self,
        request: SynthesisSegment,
        config: SynthesisConfig,
        *,
        start: int,
        end: int,
        mode: str,
    ) -> list[tuple[int, int]]:
        from phrasplit import split_with_offsets

        text = request.text[start:end]
        split_segments = split_with_offsets(
            text,
            mode=mode,
            use_spacy=config.long_text_use_spacy,
            language=request.language,
        )
        spans: list[tuple[int, int]] = []
        cursor = 0
        for split_segment in split_segments:
            split_end = min(len(text), split_segment.char_end)
            if split_end <= cursor:
                continue
            spans.append((start + cursor, start + split_end))
            cursor = split_end
        if cursor < len(text):
            spans.append((start + cursor, end))
        if not spans:
            spans.append((start, end))
        return self._merge_override_boundaries(spans, request)

    @staticmethod
    def _word_source_spans(
        request: SynthesisSegment, *, start: int, end: int
    ) -> list[tuple[int, int]]:
        text = request.text[start:end]
        spans: list[tuple[int, int]] = []
        cursor = 0
        for match in re.finditer(r"\S+\s*", text):
            if match.end() > cursor:
                spans.append((start + cursor, start + match.end()))
                cursor = match.end()
        if cursor < len(text):
            spans.append((start + cursor, end))
        if not spans:
            spans.append((start, end))
        return OnnxRequestRenderer._merge_override_boundaries(spans, request)

    @staticmethod
    def _merge_override_boundaries(
        spans: list[tuple[int, int]], request: SynthesisSegment
    ) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged and any(
                override.phonemes is not None and override.start < merged[-1][1] < override.end
                for override in request.pronunciation_overrides
            ):
                merged[-1] = (merged[-1][0], end)
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _phoneme_segment_from_prepared(
        request: SynthesisSegment,
        prepared: PreparedSynthesis,
        *,
        start: int,
        end: int,
        phoneme_id: int,
    ) -> PhonemeSegment:
        voice_name = request.voice if isinstance(request.voice, str) else None
        alignment_tokens = [
            replace(
                token,
                char_start=token.char_start + start if token.char_start is not None else None,
                char_end=token.char_end + start if token.char_end is not None else None,
            )
            for token in prepared.alignment_tokens
        ]
        return PhonemeSegment(
            id=f"{request.id}:phoneme:{phoneme_id}",
            segment_id=request.id,
            phoneme_id=phoneme_id,
            text=request.text[start:end],
            phonemes=prepared.phonemes,
            tokens=list(prepared.token_ids),
            lang=request.language,
            char_start=start,
            char_end=end,
            voice_name=voice_name,
            alignment_tokens=alignment_tokens,
        )
