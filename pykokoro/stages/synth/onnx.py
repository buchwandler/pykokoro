from __future__ import annotations

from pathlib import Path

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
        from ...pipeline_config import resolve_model_defaults

        cfg = resolve_model_defaults(cfg)
        assert cfg.model_source is not None
        assert cfg.model_variant is not None
        assert cfg.voice is not None
        kokoro = self._kokoro or Kokoro(
            model_path=Path(cfg.model_path) if cfg.model_path else None,
            voices_path=Path(cfg.voices_path) if cfg.voices_path else None,
            model_config_path=Path(cfg.model_config_path) if cfg.model_config_path else None,
            model_quality=cfg.model_quality,
            model_source=cfg.model_source,
            model_variant=cfg.model_variant,
            provider=cfg.provider,
            provider_options=cfg.provider_options,
            session_options=cfg.session_options,
            tokenizer_config=cfg.tokenizer_config,
            espeak_config=cfg.espeak_config,
            short_sentence_config=cfg.short_sentence_config,
            waveform_validation=cfg.waveform_validation,
        )
        self._kokoro = kokoro

        trace.model.update(getattr(kokoro, "runtime_metadata", {}))
        generation = cfg.generation
        voice_style = kokoro._resolve_voice_style(cfg.voice)
        trim_silence = generation.pause_mode in {"manual", "auto"}

        if generation.random_seed is None:
            return kokoro._generate_from_segments(
                phoneme_segments,
                voice_style,
                generation.speed,
                trim_silence,
                generation.enable_short_sentence,
                prosody_config=cfg.prosody,
            )

        return kokoro._generate_from_segments(
            phoneme_segments,
            voice_style,
            generation.speed,
            trim_silence,
            generation.enable_short_sentence,
            random_seed=generation.random_seed,
            prosody_config=cfg.prosody,
        )
