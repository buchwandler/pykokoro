from __future__ import annotations

from typing import TYPE_CHECKING

from ...onnx_backend import Kokoro
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    import numpy as np

    from ...pipeline_config import PipelineConfig


class OnnxAudioPostprocessingAdapter:
    def __init__(self, kokoro: Kokoro, *, owns_kokoro: bool = False) -> None:
        self._kokoro = kokoro
        self._owns_kokoro = owns_kokoro

    def close(self) -> None:
        if self._owns_kokoro:
            self._kokoro.close()

    def postprocess(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> np.ndarray:
        _ = trace
        trim_silence = cfg.generation.pause_mode in {"manual", "auto"}
        processed = self._kokoro.postprocess_audio_segments(
            phoneme_segments, trim_silence
        )
        return self._kokoro.concatenate_audio_segments(processed)
