"""Request-local ONNX inference and postprocessing for the public synthesis engine."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from ._error_utils import backend_error_from_exception
from .exceptions import (
    BackendError,
    ConfigurationError,
    InvalidRequestError,
    InvalidVoiceError,
    PyKokoroError,
    SynthesisInputTooLongError,
)
from .frontend_projection import project_frontend_text
from .model_profiles import get_model_profile

if TYPE_CHECKING:
    from .prepared_g2p import PreparedSynthesis
    from .reference_audio import PreparedReferenceAudio
from .reference_voice import ReferenceVoice
from .synthesis_config import SynthesisConfig, resolve_synthesis_config
from .synthesis_identity import build_synthesis_identity
from .synthesis_types import RenderedSegment, SynthesisSegment
from .types import (
    PhonemeSegment,
    Trace,
    WordTiming,
    _exact_timing_geometry,
)
from .voice_enrollment import InnoEnrollmentOptions
from .voice_level import resolve_voice_level_application
from .voice_pack import KokoroVoicePack

_AKINVOX_NARRATION_CONTEXT_KEY = "__akinvox_narration_context"


def _call_backend_operation(
    function: Callable[[], Any],
    *,
    operation: str,
    stage: str,
    request_id: str,
) -> Any:
    try:
        return function()
    except PyKokoroError:
        raise
    except Exception as exc:
        raise backend_error_from_exception(
            f"Kokoro {operation} failed for request {request_id!r}",
            exc,
            stage=stage,
            details={"operation": operation, "request_id": request_id},
        ) from exc


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
            resolved,
            language=request.language,
            voice=resolved.voice,
            say_as=request.say_as,
        )
        profile = get_model_profile(
            resolved.model_variant or "v1.0", resolved.model_source or "github"
        )
        voice = resolved.voice
        voice_name = voice if isinstance(voice, str) else None
        reference_voice = voice if isinstance(voice, ReferenceVoice) else None
        voice_pack = voice if isinstance(voice, KokoroVoicePack) else None
        rendered_voice = voice_name
        if voice_pack is not None:
            rendered_voice = f"voicepack:{voice_pack.fingerprint}"
        if reference_voice is not None:
            rendered_voice = f"reference:{reference_voice.fingerprint}"
            if resolved.generation.speed != 1.0:
                raise ConfigurationError("reference voice synthesis requires speed=1.0")
        trace = Trace() if resolved.return_trace else None
        if trace is not None:
            if request.say_as:
                projection = project_frontend_text(request.text, request.say_as)
                trace.frontend["say_as"] = [
                    {
                        "start": span.start,
                        "end": span.end,
                        "interpret_as": span.interpret_as,
                        "canonical": request.text[span.start : span.end],
                        "frontend": projection.frontend_text[
                            slice(*projection.map_canonical_span(span.start, span.end))
                        ],
                    }
                    for span in request.say_as
                ]
            trace.model.update(
                {
                    "model_source": resolved.model_source,
                    "model_variant": resolved.model_variant,
                    "model_quality": resolved.model_quality,
                    "synthesis_identity": identity.to_dict(),
                    "voice": rendered_voice,
                }
            )
            if reference_voice is not None:
                trace.model.update(
                    {
                        "voice_mode": "reference",
                        "reference_fingerprint": reference_voice.fingerprint,
                    }
                )
            if voice_pack is not None:
                trace.model.update(
                    {"voice_mode": "static", "voicepack_fingerprint": voice_pack.fingerprint}
                )
            trace.warnings.extend(prepared.diagnostics)
            pronunciation_alignment = [
                item.to_dict()
                for item in prepared.alignment_tokens
                if item.pronunciation_source is not None or item.heteronym_resolution is not None
            ]
            if pronunciation_alignment:
                trace.model["pronunciation_alignment"] = pronunciation_alignment

        if not prepared.token_ids:
            raise BackendError(
                f"frontend produced no model tokens for request {request.id!r}",
                stage="frontend",
                details={"request_id": request.id},
            )
        if voice is None:
            raise InvalidVoiceError("a voice must resolve before ONNX rendering")

        phoneme_segments = self._build_phoneme_segments(
            request, prepared, resolved, profile.max_tokens, trace
        )
        backend = self._get_backend(resolved)
        if voice_name is not None:
            validate = getattr(backend, "validate_voice_language", None)
            if callable(validate):
                validate(voice_name, request.language)
        if reference_voice is not None:
            voice_style = None
        elif voice_pack is not None:
            voice_style = voice_pack.data
        else:
            try:
                voice_style = backend.resolve_voice_style(voice)
            except ConfigurationError:
                raise
            except (KeyError, ValueError, FileNotFoundError) as exc:
                raise InvalidVoiceError(
                    f"Voice {voice!r} is unavailable for the resolved model"
                ) from exc
            except Exception as exc:
                raise backend_error_from_exception(
                    f"Kokoro voice preparation failed for request {request.id!r}",
                    exc,
                    stage="model_open",
                    details={"operation": "resolve_voice_style", "request_id": request.id},
                ) from exc

        def context_phonemizer(text: str, language: str) -> Any:
            return self._g2p_adapter.phonemize_context(text, language, resolved)

        phoneme_segments = _call_backend_operation(
            lambda: backend.preprocess_segments(
                phoneme_segments,
                resolved.generation.enable_short_sentence,
                random_seed=resolved.generation.random_seed,
                context_phonemizer=context_phonemizer,
            ),
            operation="preprocess_segments",
            stage="inference",
            request_id=request.id,
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
            raise BackendError(
                f"request {request.id!r} produced no acoustic chunks",
                stage="frontend",
                details={"request_id": request.id},
            )
        if trace is not None:
            trace.model["acoustic_chunk_count"] = len(phoneme_segments)
        if reference_voice is not None:
            raw_segments = _call_backend_operation(
                lambda: backend.generate_reference_audio_segments(
                    phoneme_segments,
                    reference_voice,
                    resolved.generation.speed,
                    seed=resolved.generation.random_seed,
                    trace=trace,
                ),
                operation="generate_reference_audio_segments",
                stage="inference",
                request_id=request.id,
            )
        else:
            raw_segments = _call_backend_operation(
                lambda: backend.generate_raw_audio_segments(
                    phoneme_segments,
                    voice_style,
                    resolved.generation.speed,
                    default_voice_name=voice_name,
                    trace=trace,
                ),
                operation="generate_raw_audio_segments",
                stage="inference",
                request_id=request.id,
            )

        processed_segments = _call_backend_operation(
            lambda: backend.postprocess_audio_segments(
                raw_segments,
                trim_silence=False,
                trace=trace,
                voice_level_config=resolved.voice_level,
            ),
            operation="postprocess_audio_segments",
            stage="postprocess",
            request_id=request.id,
        )
        if len(processed_segments) != len(phoneme_segments):
            raise BackendError(
                "postprocessing must preserve acoustic chunks; "
                f"received {len(processed_segments)} for {len(phoneme_segments)} input chunks",
                stage="postprocess",
                details={"request_id": request.id},
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
            raise BackendError(
                f"backend produced no audio samples for request {request.id!r}",
                stage="postprocess",
                details={"request_id": request.id},
            )
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
            voice=rendered_voice,
            phonemes=prepared.phonemes,
            token_ids=prepared.token_ids,
            word_timings=tuple(word_timings),
            diagnostics=prepared.diagnostics,
            trace=trace,
            synthesis_identity=identity,
            voice_level_applications=voice_level_applications,
            short_sentence_mode=short_sentence_mode,
        )

    def enroll_inno_voice(
        self,
        audio: PreparedReferenceAudio,
        config: SynthesisConfig,
        *,
        enroller: str,
        language: str,
        options: InnoEnrollmentOptions,
        name: str | None,
    ) -> KokoroVoicePack:
        resolved = resolve_synthesis_config(config, language=language)
        profile = get_model_profile(
            resolved.model_variant or "v1.0", resolved.model_source or "github"
        )
        if profile.voice_mode != "static":
            raise InvalidVoiceError("Inno enrollment requires a static Kokoro model")
        backend = self._get_backend(resolved)
        return backend.enroll_voice(
            audio.mono_audio,
            sample_rate=audio.original_sample_rate,
            enroller=enroller,
            options=options.to_runtime_options(),
            name=name,
        )

    def enroll_voice(
        self,
        prepared: PreparedSynthesis,
        audio: PreparedReferenceAudio,
        config: SynthesisConfig,
        *,
        name: str | None,
        reference_text_sha256: str,
    ) -> ReferenceVoice:
        config = resolve_synthesis_config(
            config, language=prepared.language, reference_enrollment=True
        )
        profile = get_model_profile(config.model_variant or "v1.0", config.model_source or "github")
        if profile.voice_mode != "reference":
            raise InvalidVoiceError("Reference enrollment requires a reference-only cloning model")
        if not prepared.token_ids:
            raise BackendError(
                "frontend produced no model tokens for the reference transcript",
                stage="frontend",
                details={"engine": "akinvox", "model_variant": config.model_variant},
            )
        backend = self._get_backend(config)
        return backend.prepare_reference_voice(
            prepared.token_ids,
            audio_24k=audio.audio_24k,
            audio_16k=audio.audio_16k,
            name=name,
            metadata={
                "reference_audio_sha256": audio.audio_sha256,
                "reference_text_sha256": reference_text_sha256,
            },
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
            "model_artifacts",
            "release_manifest_path",
            "provider",
            "provider_options",
            "session_options",
            "tokenizer_config",
            "cache_dir",
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
            model_artifacts=config.model_artifacts,
            provider=config.provider,
            session_options=config.session_options,
            provider_options=config.provider_options,
            vocab_version=profile.tokenizer_vocab_version,
            cache_dir=config.cache_dir,
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
        context_counts = self._akinvox_narration_context_counts(request, config)
        if context_counts is not None:
            return self._pack_contextual_chunks(
                request,
                safe_spans,
                phonemize_span,
                max_tokens,
                context_counts,
            )

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

    @staticmethod
    def _akinvox_narration_context_counts(
        request: SynthesisSegment, config: SynthesisConfig
    ) -> tuple[int, int] | None:
        if (
            config.model_source != "github"
            or config.model_variant != "akinvox-multispeaker-v1"
            or not isinstance(request.voice, str)
            or not request.voice.startswith("AV_")
        ):
            return None
        language = request.language.strip().lower().replace("_", "-").split("-", 1)[0]
        if language == "en":
            return 1, 2
        if language == "de":
            return 1, 1
        return None

    @staticmethod
    def _narration_context_window(
        text: str,
        start: int,
        end: int,
        before_words: int,
        after_words: int,
    ) -> dict[str, Any] | None:
        words = [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]
        body_indices = [
            index
            for index, (word_start, word_end) in enumerate(words)
            if word_start < end and word_end > start
        ]
        if not body_indices:
            return None
        if any(
            word_start < start < word_end or word_start < end < word_end
            for word_start, word_end in words
        ):
            return None
        first, last = body_indices[0], body_indices[-1]
        before_index = max(0, first - before_words)
        after_index = min(len(words) - 1, last + after_words)
        actual_before = first - before_index
        actual_after = after_index - last
        if not actual_before and not actual_after:
            return None
        context_start = words[before_index][0] if actual_before else start
        context_end = words[after_index][1] if actual_after else end
        return {
            "context_char_start": context_start,
            "context_char_end": context_end,
            "body_char_start": start,
            "body_char_end": end,
            "context_before_words": actual_before,
            "context_after_words": actual_after,
            "source_word_spans": [list(span) for span in words[before_index : after_index + 1]],
            "body_word_spans": [list(words[index]) for index in range(first, last + 1)],
        }

    @staticmethod
    def _has_exact_narration_alignment(
        request: SynthesisSegment,
        prepared: PreparedSynthesis,
        context: dict[str, Any],
    ) -> bool:
        context_start = context["context_char_start"]
        context_end = context["context_char_end"]
        body_start = context["body_char_start"]
        body_end = context["body_char_end"]
        if (
            prepared.text != request.text[context_start:context_end]
            or not prepared.alignment_tokens
        ):
            return False
        intervals: list[tuple[int, int]] = []
        for token in prepared.alignment_tokens:
            start = token.char_start
            end = token.char_end
            if (
                not isinstance(start, int)
                or isinstance(start, bool)
                or not isinstance(end, int)
                or isinstance(end, bool)
                or start < 0
                or end <= start
                or end > context_end - context_start
                or prepared.text[start:end] != token.text
                or _exact_timing_geometry(token) is None
            ):
                return False
            absolute_start = context_start + start
            absolute_end = context_start + end
            if (
                absolute_start < body_start < absolute_end
                or absolute_start < body_end < absolute_end
            ):
                return False
            intervals.append((absolute_start, absolute_end))

        for word_start, word_end in context["source_word_spans"]:
            cursor = word_start
            for start, end in sorted(
                (left, right) for left, right in intervals if left < word_end and right > word_start
            ):
                if start > cursor:
                    return False
                cursor = max(cursor, end)
            if cursor < word_end:
                return False
        return True

    def _pack_contextual_chunks(
        self,
        request: SynthesisSegment,
        safe_spans: list[tuple[int, int]],
        phonemize_span: Callable[[int, int], PreparedSynthesis],
        max_tokens: int,
        context_counts: tuple[int, int],
    ) -> list[PhonemeSegment]:
        before_words, after_words = context_counts
        segments: list[PhonemeSegment] = []
        position = 0
        phoneme_id = 0
        while position < len(safe_spans):
            body_start = safe_spans[position][0]
            best_body: tuple[int, int, PreparedSynthesis] | None = None
            best_context: (
                tuple[int, int, PreparedSynthesis, int, int, PreparedSynthesis, dict[str, Any]]
                | None
            ) = None
            context_budget_exhausted = False
            for stop in range(position + 1, len(safe_spans) + 1):
                body_end = safe_spans[stop - 1][1]
                body_prepared = phonemize_span(body_start, body_end)
                if len(body_prepared.token_ids) > max_tokens:
                    break
                best_body = (stop, body_end, body_prepared)
                if context_budget_exhausted:
                    continue
                context = self._narration_context_window(
                    request.text, body_start, body_end, before_words, after_words
                )
                if context is None:
                    continue
                context_start = context["context_char_start"]
                context_end = context["context_char_end"]
                try:
                    context_prepared = phonemize_span(context_start, context_end)
                except InvalidRequestError:
                    continue
                if len(context_prepared.token_ids) > max_tokens:
                    context_budget_exhausted = True
                    continue
                if self._has_exact_narration_alignment(request, context_prepared, context):
                    best_context = (
                        stop,
                        body_end,
                        body_prepared,
                        context_start,
                        context_end,
                        context_prepared,
                        context,
                    )
            if best_context is not None:
                (
                    stop,
                    body_end,
                    body_prepared,
                    context_start,
                    context_end,
                    context_prepared,
                    context,
                ) = best_context
                segment = self._phoneme_segment_from_prepared(
                    request,
                    context_prepared,
                    start=context_start,
                    end=context_end,
                    phoneme_id=phoneme_id,
                )
                body_segment = self._phoneme_segment_from_prepared(
                    request,
                    body_prepared,
                    start=body_start,
                    end=body_end,
                    phoneme_id=phoneme_id,
                )
                context["context_text"] = request.text[context_start:context_end]
                context["body"] = {
                    "text": body_segment.text,
                    "phonemes": body_segment.phonemes,
                    "tokens": list(body_segment.tokens),
                    "char_start": body_segment.char_start,
                    "char_end": body_segment.char_end,
                    "alignment_tokens": [
                        token.to_dict() for token in body_segment.alignment_tokens
                    ],
                }
                segment.engine_metadata = {_AKINVOX_NARRATION_CONTEXT_KEY: context}
            else:
                if best_body is None:
                    raise BackendError(
                        f"PhraseSplit produced no usable chunks for request {request.id!r}"
                    )
                stop, body_end, body_prepared = best_body
                segment = self._phoneme_segment_from_prepared(
                    request,
                    body_prepared,
                    start=body_start,
                    end=body_end,
                    phoneme_id=phoneme_id,
                )
            segments.append(segment)
            position = stop
            phoneme_id += 1
        return segments

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
        say_as = []
        for span in request.say_as:
            if span.start >= start and span.end <= end:
                say_as.append(replace(span, start=span.start - start, end=span.end - start))
            elif span.start < end and span.end > start:
                raise InvalidRequestError("long-text child span cuts through a say_as span")
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
            say_as=tuple(say_as),
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
        return self._merge_protected_boundaries(spans, request)

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
        return OnnxRequestRenderer._merge_protected_boundaries(spans, request)

    @staticmethod
    def _merge_protected_boundaries(
        spans: list[tuple[int, int]], request: SynthesisSegment
    ) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        for start, end in spans:
            if merged:
                boundary = merged[-1][1]
                protected = any(
                    override.phonemes is not None and override.start < boundary < override.end
                    for override in request.pronunciation_overrides
                ) or any(span.start < boundary < span.end for span in request.say_as)
                if protected:
                    merged[-1] = (merged[-1][0], end)
                    continue
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
