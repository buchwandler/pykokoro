import numpy as np

from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.runtime.spans import slice_spans
from pykokoro.stages.g2p.kokorog2p import KokoroG2PAdapter
from pykokoro.types import PhonemeSegment


class DummyG2PAdapter:
    def __init__(self) -> None:
        self._span_helper = KokoroG2PAdapter()

    def phonemize(self, segments, doc, cfg, trace):
        _ = trace
        out = []
        for segment in segments:
            ssmd_metadata: dict[str, str] = {}
            for span in slice_spans(
                doc.annotation_spans,
                segment.char_start,
                segment.char_end,
                overlap_mode=cfg.overlap_mode,
            ):
                self._span_helper._apply_span_metadata(span.attrs, ssmd_metadata)
            out.append(
                PhonemeSegment(
                    id=f"{segment.id}_ph0",
                    segment_id=segment.id,
                    phoneme_id=0,
                    text=segment.text,
                    phonemes="test",
                    tokens=[1],
                    lang=cfg.generation.lang,
                    char_start=segment.char_start,
                    char_end=segment.char_end,
                    paragraph_idx=segment.paragraph_idx,
                    sentence_idx=segment.sentence_idx,
                    clause_idx=segment.clause_idx,
                    ssmd_metadata=ssmd_metadata or None,
                )
            )
        return out


class DummyPhonemeProcessor:
    def process(self, phoneme_segments, cfg, trace):
        _ = cfg
        _ = trace
        return phoneme_segments


class DummyAudioGenerator:
    def __init__(self, voice_resolver) -> None:
        self._voice_resolver = voice_resolver

    def generate(self, phoneme_segments, cfg, trace):
        _ = cfg
        _ = trace
        for segment in phoneme_segments:
            if not segment.ssmd_metadata:
                continue
            voice_name = segment.ssmd_metadata.get("voice_name")
            if not voice_name:
                voice_name = segment.ssmd_metadata.get("voice")
            if voice_name:
                self._voice_resolver(voice_name)
        return phoneme_segments


class DummyPostprocessor:
    def postprocess(self, phoneme_segments, cfg, trace):
        _ = phoneme_segments
        _ = cfg
        _ = trace
        return np.zeros(1, dtype=np.float32)


def test_pipeline_voice_switching_calls_resolver():
    calls = []

    def resolver(name: str) -> np.ndarray:
        calls.append(name)
        return np.zeros((1, 1), dtype=np.float32)

    pipeline = KokoroPipeline(
        PipelineConfig(voice="af", generation=GenerationConfig(lang="en-us")),
        g2p=DummyG2PAdapter(),
        phoneme_processing=DummyPhonemeProcessor(),
        audio_generation=DummyAudioGenerator(resolver),
        audio_postprocessing=DummyPostprocessor(),
    )
    text = '<div voice="af_sarah">Hello</div>\n\n<div voice="am_michael">World</div>'

    pipeline.run(text)

    assert calls == ["af_sarah", "am_michael"]
