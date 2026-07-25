from __future__ import annotations

from typing import TYPE_CHECKING

from ...onnx_backend import Kokoro
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


class OnnxPhonemeProcessorAdapter:
    def __init__(self, kokoro: Kokoro, *, owns_kokoro: bool = False) -> None:
        self._kokoro = kokoro
        self._owns_kokoro = owns_kokoro

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
        if cfg.generation.random_seed is None:
            return self._kokoro.preprocess_segments(
                phoneme_segments,
                cfg.generation.enable_short_sentence,
            )
        return self._kokoro.preprocess_segments(
            phoneme_segments,
            cfg.generation.enable_short_sentence,
            cfg.generation.random_seed,
        )
