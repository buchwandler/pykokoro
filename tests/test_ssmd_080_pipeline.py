import numpy as np

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import PhonemeSegment, Segment


class DummyG2P:
    def phonemize(self, segments, doc, cfg, trace):
        return [
            PhonemeSegment(
                id="segment-ph0",
                segment_id=segments[0].id,
                phoneme_id=0,
                text=segments[0].text,
                phonemes="a",
                tokens=[],
                char_start=segments[0].char_start,
                char_end=segments[0].char_end,
            )
        ]


class DummyDoc:
    def parse(self, text, cfg, trace):
        return DocumentResult(
            clean_text=text,
            segments=[Segment("segment", text, 0, len(text))],
            header={"title": "Metadata only", "pause_defaults": {"enabled": True}},
            body=text,
        )


class DummyStage:
    def process(self, segments, cfg, trace):
        return segments

    def generate(self, segments, cfg, trace):
        return segments

    def postprocess(self, segments, cfg, trace):
        return np.zeros(4, dtype=np.float32)


def test_custom_stages_remain_compatible_and_metadata_reaches_result():
    pipe = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        doc_parser=DummyDoc(),
        g2p=DummyG2P(),
        phoneme_processing=DummyStage(),
        audio_generation=DummyStage(),
        audio_postprocessing=DummyStage(),
    )
    result = pipe.run("Hello")
    assert result.document_metadata["title"] == "Metadata only"
    assert result.segments[0].text == "Hello"
