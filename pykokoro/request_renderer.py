"""Request-local ONNX inference and postprocessing for the public synthesis engine."""

from __future__ import annotations

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
from .synthesis_types import RenderedSegment, SynthesisSegment
from .types import PhonemeSegment, Trace, WordTiming
from .voice_level import resolve_voice_level_application


class _RequestG2PAdapter(Protocol):
    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> Any: ...


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
        postprocessed_token_count = sum(len(segment.tokens) for segment in phoneme_segments)
        if postprocessed_token_count > profile.max_tokens:
            raise SynthesisInputTooLongError(
                text_length=len(request.text),
                token_count=postprocessed_token_count,
                max_tokens=profile.max_tokens,
                model_id=identity.model_id,
            )
        if len(phoneme_segments) != 1:
            raise BackendError(
                f"short-sentence preprocessing must preserve one request segment; "
                f"received {len(phoneme_segments)}"
            )

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
        if len(processed_segments) != 1:
            raise BackendError(
                f"postprocessing must preserve one request segment; received {len(processed_segments)}"
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
        token_ids = list(prepared.token_ids)
        if not token_ids:
            raise BackendError(f"frontend produced no model tokens for request {request.id!r}")
        if len(token_ids) > max_tokens:
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
                    "model_token_count": len(token_ids),
                    "model_max_tokens": max_tokens,
                    "acoustic_chunk_count": 1,
                }
            )

        voice = config.voice
        voice_name = voice if isinstance(voice, str) else None
        return [
            PhonemeSegment(
                id=f"{request.id}:phoneme:0",
                segment_id=request.id,
                phoneme_id=0,
                text=request.text,
                phonemes=prepared.phonemes,
                tokens=token_ids,
                lang=request.language,
                char_start=0,
                char_end=len(request.text),
                voice_name=voice_name,
                alignment_tokens=list(prepared.alignment_tokens),
            )
        ]
