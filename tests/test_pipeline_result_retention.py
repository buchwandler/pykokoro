from __future__ import annotations

import numpy as np

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import BoundaryEvent, PhonemeSegment, Segment


class RetentionDocumentParser:
    def parse(self, text, cfg, trace):
        _ = cfg, trace
        segments = [
            Segment(id="seg0", text=text[:3], char_start=0, char_end=3),
            Segment(id="seg1", text=text[3:], char_start=3, char_end=len(text)),
        ]
        return DocumentResult(
            clean_text=text,
            segments=segments,
            boundary_events=[BoundaryEvent(pos=3, kind="marker", attrs={"marker": "middle"})],
            header={
                "title": "Retention test",
                "voice_bindings": {"default": "af"},
                "pause_defaults": {"sentence": 0.2},
            },
        )


class RetentionG2P:
    def phonemize(self, segments, doc, cfg, trace):
        _ = doc, trace
        return [
            PhonemeSegment(
                id=f"{segment.id}_ph0",
                segment_id=segment.id,
                phoneme_id=0,
                text=segment.text,
                phonemes="a",
                tokens=[1],
                lang=cfg.generation.lang,
                char_start=segment.char_start,
                char_end=segment.char_end,
                ssmd_metadata={"short_sentence": {"enabled": True}},
            )
            for segment in segments
        ]


class CountingKokoro:
    instances = 0

    def __init__(self, *args, **kwargs) -> None:
        _ = args, kwargs
        type(self).instances += 1
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1

    def preprocess_segments(self, phoneme_segments, enable_short_sentence, *args):
        _ = enable_short_sentence, args
        return phoneme_segments

    def resolve_voice_style(self, voice):
        _ = voice
        return np.zeros(1, dtype=np.float32)

    def generate_raw_audio_segments(
        self,
        phoneme_segments,
        voice_style,
        speed,
        voice_resolver,
        *,
        default_voice_name=None,
    ):
        _ = voice_style, speed, voice_resolver
        for index, segment in enumerate(phoneme_segments):
            segment.raw_audio = np.full(2 + index, index + 1, dtype=np.float32)
        return phoneme_segments

    def postprocess_audio_segments(self, phoneme_segments, trim_silence, prosody_config=None):
        _ = trim_silence, prosody_config
        for segment in phoneme_segments:
            assert segment.raw_audio is not None
            segment.processed_audio = segment.raw_audio + 0.5
        return phoneme_segments

    def concatenate_audio_segments(self, phoneme_segments):
        return np.concatenate(
            [
                segment.processed_audio
                for segment in phoneme_segments
                if segment.processed_audio is not None
            ]
        )


def _pipeline() -> KokoroPipeline:
    return KokoroPipeline(
        PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us"), return_trace=True),
        doc_parser=RetentionDocumentParser(),
        g2p=RetentionG2P(),
    )


def test_default_retention_keeps_segment_audio(monkeypatch) -> None:
    CountingKokoro.instances = 0
    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", CountingKokoro)
    pipeline = _pipeline()
    try:
        result = pipeline.run("abcdef")
    finally:
        pipeline.close()

    assert result.audio.size == 5
    assert all(segment.raw_audio is not None for segment in result.phoneme_segments)
    assert all(segment.processed_audio is not None for segment in result.phoneme_segments)
    assert result.trace is not None


def test_compact_mode_preserves_audio_markers_and_metadata(monkeypatch) -> None:
    CountingKokoro.instances = 0
    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", CountingKokoro)
    retained_pipeline = _pipeline()
    compact_pipeline = _pipeline()
    try:
        retained = retained_pipeline.run("abcdef", retain_segment_audio=True)
        compact = compact_pipeline.run("abcdef", retain_segment_audio=False)
    finally:
        retained_pipeline.close()
        compact_pipeline.close()

    np.testing.assert_array_equal(retained.audio, compact.audio)
    assert retained.markers == compact.markers
    assert compact.markers == [{"name": "middle", "char_offset": 3, "sample_offset": 2}]
    assert retained.document_metadata == compact.document_metadata
    assert len(retained.segments) == len(compact.segments) == 2
    assert len(retained.phoneme_segments) == len(compact.phoneme_segments) == 2
    assert compact.phoneme_segments[0].ssmd_metadata == {"short_sentence": {"enabled": True}}
    assert compact.trace is not None
    assert all(segment.raw_audio is None for segment in compact.phoneme_segments)
    assert all(segment.processed_audio is None for segment in compact.phoneme_segments)


def test_retention_override_reuses_backend(monkeypatch) -> None:
    CountingKokoro.instances = 0
    monkeypatch.setattr("pykokoro.onnx_backend.Kokoro", CountingKokoro)
    pipeline = _pipeline()
    try:
        retained = pipeline.run("abcdef", retain_segment_audio=True)
        compact = pipeline.run("abcdef", retain_segment_audio=False)

        assert CountingKokoro.instances == 1
        assert retained.audio.size == compact.audio.size == 5
        assert all(segment.raw_audio is None for segment in compact.phoneme_segments)
    finally:
        pipeline.close()
