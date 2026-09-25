"""Request-centric public synthesizer orchestration."""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Iterator
from typing import Protocol

from .prepared_g2p import PreparedG2PAdapter, PreparedSynthesis
from .synthesis_config import SynthesisConfig, resolve_synthesis_config
from .synthesis_types import RenderedSegment, SynthesisSegment
from .voice_manager import VoiceBlend


class _RequestRenderer(Protocol):
    def render(
        self,
        prepared: PreparedSynthesis,
        request: SynthesisSegment,
        config: SynthesisConfig,
    ) -> RenderedSegment: ...


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
        self.g2p = g2p or PreparedG2PAdapter()
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
        return self.g2p.phonemize(segment, config)

    def synthesize(self, segment: SynthesisSegment) -> RenderedSegment:
        """Render exactly one independent prepared speech request."""
        self._ensure_open()
        self._validate_request(segment)
        config = resolve_synthesis_config(
            self.config, language=segment.language, voice=segment.voice
        )
        prepared = self.g2p.phonemize(segment, config)
        result = self._renderer.render(prepared, segment, config)
        if not isinstance(result, RenderedSegment):
            raise TypeError("request renderer must return a RenderedSegment")
        if result.id != segment.id:
            raise ValueError("rendered result ID must match the synthesis request ID")
        return result

    def synthesize_text(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | VoiceBlend | None = None,
    ) -> RenderedSegment:
        """Synthesize one already-prepared string without document parsing."""
        effective_language = language if language is not None else self.config.generation.lang
        if effective_language is None:
            raise ValueError("a synthesis language is required")
        return self.synthesize(
            SynthesisSegment(
                id=f"request-{uuid.uuid4().hex}",
                text=text,
                language=effective_language,
                voice=voice,
            )
        )

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
        if not isinstance(segment, SynthesisSegment):
            raise TypeError("synthesize requires a SynthesisSegment")

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("KokoroSynthesizer is closed")

    def __enter__(self) -> KokoroSynthesizer:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
