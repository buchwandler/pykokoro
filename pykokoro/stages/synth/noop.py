from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ...constants import SAMPLE_RATE
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    from ...pipeline_config import PipelineConfig


@dataclass
class NoopSynthesizerAdapter:
    seconds_per_segment: float = 0.1

    def synthesize(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> np.ndarray:
        _ = (cfg, trace)
        total_samples = int(SAMPLE_RATE * self.seconds_per_segment) * len(phoneme_segments)
        return np.zeros(total_samples, dtype=np.float32)
