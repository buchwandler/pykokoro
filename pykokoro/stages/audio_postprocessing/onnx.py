from __future__ import annotations

from typing import TYPE_CHECKING

from ...audio_generator import resolve_audio_annotation
from ...exceptions import CapabilityError
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
        for segment in phoneme_segments:
            metadata = segment.ssmd_metadata or {}
            emphasis = metadata.get("emphasis")
            if emphasis and cfg.ssmd.emphasis_mode == "plain":
                continue
            if emphasis and cfg.ssmd.emphasis_mode == "error":
                raise CapabilityError("SSMD emphasis is disabled by emphasis_mode='error'")
            if emphasis and cfg.ssmd.emphasis_mode == "warn":
                trace.warnings.append("ssmd.emphasis_unsupported: using unmodified speech")
            elif emphasis and cfg.ssmd.emphasis_mode == "approximate":
                if emphasis in {"strong", "moderate"}:
                    metadata.setdefault(
                        "prosody_volume", "loud" if emphasis == "strong" else "medium"
                    )
                    metadata.setdefault(
                        "prosody_rate", "fast" if emphasis == "strong" else "medium"
                    )
                elif emphasis == "reduced":
                    metadata.setdefault("prosody_volume", "soft")
                    metadata.setdefault("prosody_rate", "slow")
        trim_silence = cfg.generation.pause_mode in {"manual", "auto"} or any(
            (segment.ssmd_metadata or {}).get("deterministic_pause_boundary") == "true"
            for segment in phoneme_segments
        )
        processed = self._kokoro.postprocess_audio_segments(phoneme_segments, trim_silence)
        resolver = cfg.ssmd.audio_source_resolver
        for segment in processed:
            metadata = segment.ssmd_metadata or {}
            if not metadata.get("audio_src"):
                continue
            if resolver is None:
                trace.warnings.append(
                    "ssmd.audio_unresolved: speaking alt_text because no audio resolver was supplied"
                )
                continue
            try:
                segment.processed_audio = resolve_audio_annotation(
                    metadata,
                    resolver,
                    max_bytes=cfg.ssmd.audio_max_bytes,
                    max_duration_s=cfg.ssmd.audio_max_duration_s,
                )
            except (OSError, TypeError, ValueError) as exc:
                trace.warnings.append(f"ssmd.audio_fallback: {exc}")
        return self._kokoro.concatenate_audio_segments(processed)
