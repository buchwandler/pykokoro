from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from ...exceptions import ConfigurationError
from ...types import PhonemeSegment, Trace

if TYPE_CHECKING:
    import numpy as np

    from ...onnx_backend import Kokoro
    from ...pipeline_config import PipelineConfig


class OnnxAudioGenerationAdapter:
    def __init__(self, kokoro: Kokoro, *, owns_kokoro: bool = False) -> None:
        self._kokoro = kokoro
        self._owns_kokoro = owns_kokoro

    def close(self) -> None:
        if self._owns_kokoro:
            self._kokoro.close()

    def generate(
        self,
        phoneme_segments: list[PhonemeSegment],
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        from ...pipeline_config import resolve_model_defaults

        cfg = resolve_model_defaults(cfg)
        assert cfg.voice is not None
        for segment in phoneme_segments:
            metadata = segment.ssmd_metadata or {}
            voice_name = metadata.get("voice_name")
            if not isinstance(voice_name, str) or not voice_name:
                continue
            try:
                self._kokoro.get_voice_style(voice_name)
            except (KeyError, RuntimeError, OSError, ValueError) as exc:
                if cfg.ssmd.missing_voice == "error":
                    reference = metadata.get("voice_reference", voice_name)
                    raise ConfigurationError(
                        f"Unable to resolve SSMD voice reference '{reference}' to "
                        f"'{voice_name}' for provider '{cfg.ssmd.provider}'"
                    ) from exc
                metadata.pop("voice_name", None)
                metadata.pop("voice", None)
                trace.warnings.append(
                    f"ssmd.missing_voice: using default voice for unavailable target '{voice_name}'"
                )
        voice_style = self._kokoro.resolve_voice_style(cfg.voice)

        def voice_resolver(voice_name: str) -> np.ndarray:
            return self._kokoro.get_voice_style(voice_name)

        generate_raw = self._kokoro.generate_raw_audio_segments
        arguments: list[Any] = [
            phoneme_segments,
            voice_style,
            cfg.generation.speed,
            voice_resolver,
        ]
        kwargs: dict[str, Any] = {
            "default_voice_name": cfg.voice if isinstance(cfg.voice, str) else None
        }
        if "trace" in inspect.signature(generate_raw).parameters:
            kwargs["trace"] = trace
        return generate_raw(*arguments, **kwargs)
