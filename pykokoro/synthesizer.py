"""Request-centric public synthesizer orchestration."""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .exceptions import (
    AlignmentError,
    BackendError,
    EmptyTextError,
    InvalidLanguageError,
    InvalidRequestError,
    PyKokoroError,
    SynthesisStateError,
)
from .model_profiles import normalize_language_code, resolve_voice_enroller

if TYPE_CHECKING:
    from .prepared_g2p import PreparedG2PAdapter, PreparedSynthesis
    from .reference_audio import PreparedReferenceAudio
from .reference_voice import ReferenceVoice
from .synthesis_config import SynthesisConfig, resolve_synthesis_config
from .synthesis_types import (
    RenderedSegment,
    SynthesisRequest,
    SynthesisSegment,
    VoiceConditioning,
)
from .voice_enrollment import InnoEnrollmentOptions, VoiceEnrollmentEngine
from .voice_pack import KokoroVoicePack


class _RequestRenderer(Protocol):
    def render(
        self,
        prepared: PreparedSynthesis,
        request: SynthesisSegment,
        config: SynthesisConfig,
    ) -> RenderedSegment: ...
    def enroll_voice(
        self,
        prepared: PreparedSynthesis,
        audio: PreparedReferenceAudio,
        config: SynthesisConfig,
        *,
        name: str | None,
        reference_text_sha256: str,
    ) -> ReferenceVoice: ...

    def enroll_inno_voice(
        self,
        audio: PreparedReferenceAudio,
        config: SynthesisConfig,
        *,
        enroller: str,
        language: str,
        options: InnoEnrollmentOptions,
        name: str | None,
    ) -> KokoroVoicePack: ...


class KokoroSynthesizer:
    """Phonemize and render prepared requests as independent Kokoro speech results."""

    def __init__(
        self,
        config: SynthesisConfig | None = None,
        *,
        g2p: PreparedG2PAdapter | None = None,
        renderer: _RequestRenderer | None = None,
    ) -> None:
        self.config = config or SynthesisConfig()
        if g2p is None:
            from .prepared_g2p import PreparedG2PAdapter

            g2p = PreparedG2PAdapter()
        self.g2p = g2p
        if renderer is None:
            from .request_renderer import OnnxRequestRenderer

            renderer = OnnxRequestRenderer(self.g2p)
        self._renderer = renderer
        self._closed = False

    def prepare(self, segment: SynthesisSegment) -> PreparedSynthesis:
        """Return model-ready Kokoro frontend data for one prepared request."""
        self._ensure_open()
        self._validate_request(segment)
        config = resolve_synthesis_config(
            self.config, language=segment.language, voice=segment.voice
        )
        try:
            return self.g2p.phonemize(segment, config)
        except PyKokoroError:
            raise
        except Exception as exc:
            raise BackendError(f"Kokoro frontend failed for request {segment.id!r}") from exc

    def synthesize(self, segment: SynthesisSegment) -> RenderedSegment:
        """Render exactly one independent prepared speech request."""
        self._ensure_open()
        self._validate_request(segment)
        config = resolve_synthesis_config(
            self.config, language=segment.language, voice=segment.voice
        )
        try:
            prepared = self.g2p.phonemize(segment, config)
            result = self._renderer.render(prepared, segment, config)
        except PyKokoroError:
            raise
        except Exception as exc:
            raise BackendError(f"Kokoro synthesis failed for request {segment.id!r}") from exc
        if not isinstance(result, RenderedSegment):
            raise BackendError("request renderer must return a RenderedSegment")
        if result.id != segment.id or result.text != segment.text:
            raise AlignmentError(
                "rendered result ID must match request, and result text must match request text"
            )
        return result

    def synthesize_text(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: VoiceConditioning = None,
    ) -> RenderedSegment:
        """Synthesize one already-prepared string without document parsing."""
        effective_language = language if language is not None else self.config.generation.lang
        if effective_language is None:
            raise InvalidLanguageError("a synthesis language is required")
        return self.synthesize(
            SynthesisSegment(
                id=f"request-{uuid.uuid4().hex}",
                text=text,
                language=effective_language,
                voice=voice,
            )
        )

    def enroll_voice(
        self,
        reference_audio: str | Path | np.ndarray,
        reference_text: str | None = None,
        *,
        engine: VoiceEnrollmentEngine = "inno",
        sample_rate: int | None = None,
        language: str = "en",
        name: str | None = None,
        options: InnoEnrollmentOptions | None = None,
    ) -> KokoroVoicePack | ReferenceVoice:
        """Enroll a static voice pack or an AkinVox reference-conditioned voice."""
        self._ensure_open()
        if engine not in ("inno", "akinvox"):
            raise InvalidRequestError("engine must be 'inno' or 'akinvox'")
        if not isinstance(language, str) or not language.strip():
            raise InvalidLanguageError("a supported enrollment language is required")
        normalized_language = normalize_language_code(language)
        from .reference_audio import prepare_reference_audio

        if engine == "inno":
            if reference_text is not None:
                raise InvalidRequestError("reference_text must be omitted for Inno enrollment")
            if options is not None and not isinstance(options, InnoEnrollmentOptions):
                raise InvalidRequestError("Inno options must be an InnoEnrollmentOptions instance")
            audio = prepare_reference_audio(reference_audio, sample_rate=sample_rate)
            model_variant, enroller = resolve_voice_enroller(self.config.model_variant, "inno-v0.2")
            if enroller.transcript_required:
                raise InvalidRequestError(
                    "the selected Inno enroller unexpectedly requires a transcript"
                )
            config = replace(self.config, model_variant=model_variant, voice=None)
            try:
                return self._renderer.enroll_inno_voice(
                    audio,
                    config,
                    language=normalized_language,
                    enroller=enroller.id,
                    options=options or InnoEnrollmentOptions(),
                    name=name,
                )
            except PyKokoroError:
                raise
            except Exception as exc:
                raise BackendError("Kokoro Inno voice enrollment failed") from exc

        if options is not None:
            raise InvalidRequestError("options are only supported for Inno enrollment")
        if not isinstance(reference_text, str) or not reference_text.strip():
            raise EmptyTextError("reference_text must be non-empty and match the reference speech")
        if normalized_language not in {"en", "en-us"}:
            raise InvalidLanguageError(
                "AkinVox reference enrollment currently supports English only"
            )

        audio = prepare_reference_audio(reference_audio, sample_rate=sample_rate)
        config = self.config
        if config.model_variant is None:
            config = replace(config, model_variant="en-akinvox-cloning-v1")
        config = replace(config, voice=None)
        request = SynthesisSegment(
            id=f"reference-{uuid.uuid4().hex}",
            text=reference_text,
            language=normalized_language,
        )
        try:
            prepared = self.g2p.phonemize(request, config)
            return self._renderer.enroll_voice(
                prepared,
                audio,
                config,
                name=name,
                reference_text_sha256=hashlib.sha256(reference_text.encode("utf-8")).hexdigest(),
            )
        except PyKokoroError:
            raise
        except Exception as exc:
            raise BackendError("Kokoro AkinVox reference enrollment failed") from exc

    def synthesize_segments(
        self, segments: Iterable[SynthesisSegment]
    ) -> Iterator[RenderedSegment]:
        """Render requests in input order, yielding independent results without joining audio."""
        for segment in segments:
            yield self.synthesize(segment)

    def close(self) -> None:
        """Release engine-owned renderer resources, if any."""
        if self._closed:
            return
        self._closed = True
        close = getattr(self._renderer, "close", None)
        if callable(close):
            close()

    @staticmethod
    def _validate_request(segment: SynthesisSegment) -> None:
        if not isinstance(segment, SynthesisRequest):
            raise InvalidRequestError("synthesize requires a SynthesisRequest")

    def _ensure_open(self) -> None:
        if self._closed:
            raise SynthesisStateError("KokoroSynthesizer is closed")

    def __enter__(self) -> KokoroSynthesizer:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
