from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...constants import SAMPLE_RATE
from ...types import PhonemeSegment, Trace
from ...utils import generate_silence

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


@dataclass
class NoopAudioPostprocessingAdapter:
    def postprocess(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> np.ndarray:
        _ = (cfg, trace)
        audio_parts: list[np.ndarray] = []

        for segment in phoneme_segments:
            if segment.raw_audio is None:
                segment.processed_audio = None
            else:
                segment.processed_audio = segment.raw_audio

            if segment.pause_before > 0:
                audio_parts.append(generate_silence(segment.pause_before, SAMPLE_RATE))
            if segment.processed_audio is not None:
                audio_parts.append(segment.processed_audio)
            if segment.pause_after > 0:
                audio_parts.append(generate_silence(segment.pause_after, SAMPLE_RATE))

        return np.concatenate(audio_parts) if audio_parts else np.array([], dtype=np.float32)
