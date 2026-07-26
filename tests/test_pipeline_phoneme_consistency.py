import pytest

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter

CASES = [
    (
        "I'm a cat. Don’t you think it’s fine?",
        "I'm a cat. Don’t you think it’s fine?",
    ),
    (
        "I don't ...s think you'll understand. It's a test.",
        "I don't think you'll understand. It's a test.",
    ),
    (
        "A ...s cat won't mind. I'm a friend.",
        "A cat won't mind. I'm a friend.",
    ),
]


def _build_pipeline(doc_parser, cfg: PipelineConfig) -> KokoroPipeline:
    return KokoroPipeline(
        cfg,
        doc_parser=doc_parser,
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(seconds_per_segment=0.01),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )


def _normalize_phonemes(segments) -> str:
    phonemes = " ".join(segment.phonemes for segment in segments if segment.phonemes)
    return " ".join(phonemes.split())


@pytest.mark.parametrize("ssmd_text, plain_text", CASES)
def test_ssmd_and_plain_phonemes_match(ssmd_text, plain_text):
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    ssmd_pipeline = _build_pipeline(SsmdDocumentParser(), cfg)
    plain_pipeline = _build_pipeline(PlainTextDocumentParser(), cfg)

    ssmd_res = ssmd_pipeline.run(ssmd_text)
    plain_res = plain_pipeline.run(plain_text)

    assert ssmd_res.phoneme_segments
    assert plain_res.phoneme_segments
    assert _normalize_phonemes(ssmd_res.phoneme_segments) == _normalize_phonemes(
        plain_res.phoneme_segments
    )
