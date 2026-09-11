from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    from ...onnx_backend import Kokoro
    from ...pipeline_config import PipelineConfig


class OnnxPhonemeProcessorAdapter:
    def __init__(
        self,
        kokoro: Kokoro,
        *,
        owns_kokoro: bool = False,
        context_phonemizer: Callable[[str, str, PipelineConfig], Any] | None = None,
    ) -> None:
        self._kokoro = kokoro
        self._owns_kokoro = owns_kokoro
        self._context_phonemizer = context_phonemizer

    def close(self) -> None:
        if self._owns_kokoro:
            self._kokoro.close()

    def process(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        _ = trace
        context_phonemizer = (
            (lambda text, language: self._context_phonemizer(text, language, cfg))
            if self._context_phonemizer is not None
            else None
        )
        if cfg.generation.random_seed is None:
            if context_phonemizer is None:
                return self._kokoro.preprocess_segments(
                    phoneme_segments, cfg.generation.enable_short_sentence
                )
            return self._kokoro.preprocess_segments(
                phoneme_segments,
                cfg.generation.enable_short_sentence,
                context_phonemizer=context_phonemizer,
            )
        if context_phonemizer is None:
            return self._kokoro.preprocess_segments(
                phoneme_segments,
                cfg.generation.enable_short_sentence,
                cfg.generation.random_seed,
            )
        return self._kokoro.preprocess_segments(
            phoneme_segments,
            cfg.generation.enable_short_sentence,
            cfg.generation.random_seed,
            context_phonemizer=context_phonemizer,
        )
