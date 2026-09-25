"""Request-local ONNX inference and postprocessing for the public synthesis engine."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np

from .model_profiles import get_model_profile
from .prepared_g2p import PreparedSynthesis
from .synthesis_config import SynthesisConfig, resolve_synthesis_config
from .synthesis_types import RenderedSegment, SynthesisSegment
from .types import G2PAlignmentToken, PhonemeSegment, Trace, WordTiming


class _RequestG2PAdapter(Protocol):
    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> Any: ...

    def ids_to_phonemes(self, token_ids: list[int] | tuple[int, ...], target_model: str) -> str: ...


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
                    "voice": voice_name,
                }
            )
            trace.warnings.extend(prepared.diagnostics)

        if not prepared.token_ids:
            return RenderedSegment(
                id=request.id,
                audio=np.zeros(0, dtype=np.float32),
                sample_rate=profile.sample_rate,
                text=request.text,
                language=request.language,
                voice=voice_name,
                phonemes=prepared.phonemes,
                token_ids=prepared.token_ids,
                diagnostics=prepared.diagnostics,
                trace=trace,
            )
        if voice is None:
            raise ValueError("A voice must resolve before ONNX rendering")

        backend = self._get_backend(resolved)
        voice_style = backend.resolve_voice_style(voice)
        phoneme_segments = self._build_phoneme_segments(
            request, prepared, resolved, profile.max_tokens
        )

        def context_phonemizer(text: str, language: str) -> Any:
            return self._g2p_adapter.phonemize_context(text, language, resolved)

        phoneme_segments = backend.preprocess_segments(
            phoneme_segments,
            resolved.generation.enable_short_sentence,
            random_seed=resolved.generation.random_seed,
            context_phonemizer=context_phonemizer,
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
    ) -> list[PhonemeSegment]:
        token_ids = list(prepared.token_ids)
        if not token_ids:
            return []
        if len(token_ids) <= max_tokens:
            groups = [(token_ids, list(prepared.alignment_tokens), prepared.phonemes)]
        else:
            aligned_groups = self._split_at_alignment_boundaries(
                token_ids, prepared.alignment_tokens, max_tokens
            )
            if aligned_groups is None:
                id_chunks = [
                    token_ids[index : index + max_tokens]
                    for index in range(0, len(token_ids), max_tokens)
                ]
                groups = [
                    (
                        chunk,
                        [],
                        self._g2p_adapter.ids_to_phonemes(chunk, self._target_model(config)),
                    )
                    for chunk in id_chunks
                ]
            else:
                groups = [
                    (
                        chunk_ids,
                        alignment,
                        self._g2p_adapter.ids_to_phonemes(chunk_ids, self._target_model(config)),
                    )
                    for chunk_ids, alignment in aligned_groups
                ]

        voice = config.voice
        voice_name = voice if isinstance(voice, str) else None
        result: list[PhonemeSegment] = []
        for index, (chunk_ids, alignment, phonemes) in enumerate(groups):
            char_offsets = [
                (item.char_start, item.char_end)
                for item in alignment
                if isinstance(item.char_start, int)
                and isinstance(item.char_end, int)
                and 0 <= item.char_start <= item.char_end <= len(request.text)
            ]
            if char_offsets:
                char_start = min(start for start, _ in char_offsets)
                char_end = max(end for _, end in char_offsets)
                text = request.text[char_start:char_end]
            elif len(groups) == 1:
                char_start, char_end, text = 0, len(request.text), request.text
            else:
                char_start, char_end, text = 0, len(request.text), request.text
            result.append(
                PhonemeSegment(
                    id=f"{request.id}:phoneme:{index}",
                    segment_id=request.id,
                    phoneme_id=index,
                    text=text,
                    phonemes=phonemes,
                    tokens=chunk_ids,
                    lang=request.language,
                    char_start=char_start,
                    char_end=char_end,
                    voice_name=voice_name,
                    alignment_tokens=alignment,
                )
            )
        return result

    @staticmethod
    def _split_at_alignment_boundaries(
        token_ids: Sequence[int],
        alignments: Sequence[G2PAlignmentToken],
        max_tokens: int,
    ) -> list[tuple[list[int], list[G2PAlignmentToken]]] | None:
        counts = [token.model_span_token_count for token in alignments]
        if not counts or any(
            not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts
        ):
            return None
        count_values = [cast(int, count) for count in counts]
        if sum(count_values) != len(token_ids):
            return None
        groups: list[tuple[list[int], list[G2PAlignmentToken]]] = []
        current: list[G2PAlignmentToken] = []
        current_count = 0
        token_offset = 0
        group_start = 0
        for token, count in zip(alignments, count_values, strict=True):
            if count > max_tokens:
                return None
            if current and current_count + count > max_tokens:
                groups.append((list(token_ids[group_start:token_offset]), current))
                group_start = token_offset
                current = []
                current_count = 0
            current.append(token)
            current_count += count
            token_offset += count
        if current:
            groups.append((list(token_ids[group_start:token_offset]), current))
        if token_offset != len(token_ids) or any(not chunk for chunk, _ in groups):
            return None
        return groups

    @staticmethod
    def _target_model(config: SynthesisConfig) -> str:
        profile = get_model_profile(config.model_variant or "v1.0", config.model_source or "github")
        return profile.tokenizer_vocab_version
