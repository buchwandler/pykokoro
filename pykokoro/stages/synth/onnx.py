from __future__ import annotations

import numpy as np

from ...onnx_backend import Kokoro
from ...pipeline_config import PipelineConfig
from ...types import PhonemeSegment, Trace


class OnnxSynthesizerAdapter:
    def __init__(self, kokoro: Kokoro | None = None) -> None:
        self._kokoro = kokoro

    def synthesize(
        self, phoneme_segments: list[PhonemeSegment], cfg: PipelineConfig, trace: Trace
    ) -> np.ndarray:
        kokoro = self._kokoro or Kokoro(
            model_quality=cfg.model_quality,
            model_source=cfg.model_source,
            model_variant=cfg.model_variant,
            provider=cfg.provider,
            provider_options=cfg.provider_options,
            session_options=cfg.session_options,
            tokenizer_config=cfg.tokenizer_config,
            espeak_config=cfg.espeak_config,
            short_sentence_config=cfg.short_sentence_config,
        )
        self._kokoro = kokoro

        generation = cfg.generation
        voice_style = kokoro._resolve_voice_style(cfg.voice)
        trim_silence = generation.pause_mode in {"manual", "auto"}

        return kokoro._generate_from_segments(
            phoneme_segments,
            voice_style,
            generation.speed,
            trim_silence,
            generation.enable_short_sentence,
        )
