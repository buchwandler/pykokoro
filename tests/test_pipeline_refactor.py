import os

import numpy as np
import pytest

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline_config import PipelineConfig as PipelineConfigType
from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.protocols import DocumentResult
from pykokoro.types import PhonemeSegment, Segment, Trace


class DummyPhonemeProcessor:
    def process(self, phoneme_segments, cfg, trace):
        _ = (cfg, trace)
        return phoneme_segments


class DummyAudioGeneration:
    def generate(self, phoneme_segments, cfg, trace):
        _ = (cfg, trace)
        return phoneme_segments


class DummyAudioPostprocessing:
    def postprocess(self, phoneme_segments, cfg, trace):
        _ = (phoneme_segments, cfg, trace)
        return np.zeros(240, dtype=np.float32)


class DummyDocParser:
    def parse(self, text, cfg, trace):
        _ = (cfg, trace)
        return DocumentResult(
            clean_text=text,
            segments=[
                Segment(
                    id="p0_s0_c0_seg0",
                    text=text,
                    char_start=0,
                    char_end=len(text),
                    paragraph_idx=0,
                    sentence_idx=0,
                    clause_idx=0,
                )
            ],
        )


class DummyG2P:
    def __init__(self) -> None:
        self.last_lang = None

    def phonemize(self, segments, doc, cfg, trace):
        self.last_lang = cfg.generation.lang
        return [
            PhonemeSegment(
                id=f"{segments[0].id}_ph0",
                segment_id=segments[0].id,
                phoneme_id=0,
                text=doc.clean_text,
                phonemes="a",
                tokens=[],
                lang=cfg.generation.lang,
                char_start=segments[0].char_start,
                char_end=segments[0].char_end,
                paragraph_idx=segments[0].paragraph_idx,
                sentence_idx=segments[0].sentence_idx,
                clause_idx=segments[0].clause_idx,
            )
        ]


def test_pipeline_imports():
    assert KokoroPipeline is not None
    assert PipelineConfig is not None


def test_modular_ssmd_parser_spans_and_breaks():
    parser = SsmdDocumentParser()
    cfg = PipelineConfigType()
    trace = Trace()
    doc = parser.parse("[Bonjour]{lang='fr'} le monde.", cfg, trace)
    assert doc.clean_text == "Bonjour le monde."
    assert any(span.attrs.get("lang") == "fr" for span in doc.annotation_spans)

    doc_break = parser.parse("Hello ...500ms world", cfg, trace)
    assert any(boundary.duration_s == 0.5 for boundary in doc_break.boundary_events)


def test_phrasplit_offsets_match_slices():
    pytest.importorskip("phrasplit")
    parser = PlainTextDocumentParser()
    cfg = PipelineConfigType()
    trace = Trace()
    doc = parser.parse("Hello world. Second sentence.", cfg, trace)
    for segment in doc.segments:
        assert segment.text == doc.clean_text[segment.char_start : segment.char_end]


def test_phrasplit_policy_does_not_preconstruct_a_model():
    from pykokoro.spacy_models import make_spacy_model_request

    request = make_spacy_model_request()
    assert request.model is None
    assert request.size is None


def test_pipeline_run_overrides_lang():
    cfg = PipelineConfig()
    g2p = DummyG2P()
    pipe = KokoroPipeline(
        cfg,
        doc_parser=DummyDocParser(),
        g2p=g2p,
        phoneme_processing=DummyPhonemeProcessor(),
        audio_generation=DummyAudioGeneration(),
        audio_postprocessing=DummyAudioPostprocessing(),
    )
    res = pipe.run("Hallo", lang="de")
    assert g2p.last_lang == "de"
    assert res.segments[0].text == "Hallo"
    assert res.phoneme_segments

    res = pipe.run("Salut", generation=GenerationConfig(lang="fr"), lang="it")
    assert g2p.last_lang == "it"
    assert res.segments[0].text == "Salut"
    assert res.phoneme_segments


def test_pipeline_requires_explicit_language() -> None:
    cfg = PipelineConfig(voice="bf_lily")
    g2p = DummyG2P()
    pipe = KokoroPipeline(
        cfg,
        doc_parser=DummyDocParser(),
        g2p=g2p,
        phoneme_processing=DummyPhonemeProcessor(),
        audio_generation=DummyAudioGeneration(),
        audio_postprocessing=DummyAudioPostprocessing(),
    )
    with pytest.raises(ValueError, match="document language is required"):
        pipe.run("Hello")


@pytest.mark.skipif(
    os.getenv("PYKOKORO_ONNX_SMOKE") != "1",
    reason="Enable with PYKOKORO_ONNX_SMOKE=1",
)
def test_onnx_smoke():
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    res = KokoroPipeline(cfg).run("Hello")
    assert res.audio.size > 0


def test_pipeline_stage_order_is_explicit() -> None:
    calls: list[str] = []

    class Parser:
        def parse(self, text, cfg, trace):
            calls.append("doc_parser")
            return DocumentResult(clean_text=text)

    class Preparer:
        def prepare(self, doc, cfg, trace):
            calls.append("text_preparer")
            return doc

    class Segmenter:
        def split(self, doc, cfg, trace):
            calls.append("sentence_segmenter")
            return [Segment("seg", doc.clean_text, 0, len(doc.clean_text), {}, 0, 0, 0)]

    class G2P:
        def phonemize(self, segments, doc, cfg, trace):
            calls.append("g2p")
            return [
                PhonemeSegment(
                    "ph",
                    segments[0].id,
                    0,
                    segments[0].text,
                    "a",
                    [],
                    char_start=0,
                    char_end=len(doc.clean_text),
                    paragraph_idx=0,
                    sentence_idx=0,
                )
            ]

    class Processor:
        def process(self, segments, cfg, trace):
            calls.append("phoneme_processing")
            return segments

    class Generator:
        def generate(self, segments, cfg, trace):
            calls.append("audio_generation")
            return segments

    class Postprocessor:
        def postprocess(self, segments, cfg, trace):
            calls.append("audio_postprocessing")
            return np.zeros(4, dtype=np.float32)

    pipe = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        doc_parser=Parser(),
        text_preparer=Preparer(),
        sentence_segmenter=Segmenter(),
        g2p=G2P(),
        phoneme_processing=Processor(),
        audio_generation=Generator(),
        audio_postprocessing=Postprocessor(),
    )
    pipe.run("Hello.")
    assert calls == [
        "doc_parser",
        "text_preparer",
        "sentence_segmenter",
        "g2p",
        "phoneme_processing",
        "audio_generation",
        "audio_postprocessing",
    ]


def test_pipeline_trace_aggregates_top_level_stage_timings() -> None:
    config = PipelineConfigType(
        generation=GenerationConfig(lang="en-us"),
        return_trace=True,
    )
    pipeline = KokoroPipeline(
        config,
        doc_parser=DummyDocParser(),
        g2p=DummyG2P(),
        phoneme_processing=DummyPhonemeProcessor(),
        audio_generation=DummyAudioGeneration(),
        audio_postprocessing=DummyAudioPostprocessing(),
    )
    result = pipeline.run("Hello")
    assert result.trace is not None

    summary = result.trace.event_summary()
    required = (
        ("doc", "parse"),
        ("language_plan", "source"),
        ("linguistics", "pass_a"),
        ("text_preparation", "prepare"),
        ("language_plan", "prepared"),
        ("linguistics", "pass_b"),
        ("segmentation", "split"),
        ("g2p", "phonemize"),
        ("runtime", "resolve_stages"),
        ("phoneme_processing", "preprocess"),
        ("audio_generation", "generate"),
        ("audio_postprocessing", "postprocess"),
    )
    assert all(summary[key] > 0.0 for key in required)
    assert not any(
        event.stage == "linguistics" and event.ms == 0.0 for event in result.trace.events
    )


def test_reusable_frontend_renders_multiple_lexicons_without_repreparation() -> None:
    from pykokoro.tokenizer import TokenizerConfig

    class CountingParser(DummyDocParser):
        calls = 0

        def parse(self, text, cfg, trace):
            self.calls += 1
            return super().parse(text, cfg, trace)

    class CountingG2P(DummyG2P):
        lexicons: list[tuple[str, ...] | None] = []

        def phonemize(self, segments, doc, cfg, trace):
            self.lexicons.append(
                cfg.tokenizer_config.lexicons if cfg.tokenizer_config is not None else None
            )
            return super().phonemize(segments, doc, cfg, trace)

    parser = CountingParser()
    g2p = CountingG2P()
    pipeline = KokoroPipeline(
        PipelineConfigType(
            generation=GenerationConfig(lang="en-us"),
            return_trace=True,
        ),
        doc_parser=parser,
        g2p=g2p,
        phoneme_processing=DummyPhonemeProcessor(),
        audio_generation=DummyAudioGeneration(),
        audio_postprocessing=DummyAudioPostprocessing(),
    )
    frontend = pipeline.prepare_frontend("Hello")
    try:
        gold = pipeline.render_frontend(
            frontend, tokenizer_config=TokenizerConfig(lexicons=("gold",))
        )
        crane = pipeline.render_frontend(
            frontend, tokenizer_config=TokenizerConfig(lexicons=("crane",))
        )
        assert parser.calls == 1
        assert g2p.lexicons == [("gold",), ("crane",)]
        assert frontend.segments
        assert gold.trace is not None
        assert crane.trace is not None
    finally:
        frontend.close()
    with pytest.raises(RuntimeError, match="PreparedFrontend is closed"):
        _ = frontend.text
