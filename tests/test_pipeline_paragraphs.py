import pytest

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.runtime.spans import slice_boundaries
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.doc_parsers.ssmd import SsmdDocumentParser
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter
from pykokoro.types import PhonemeSegment, Trace

TEXT = "This is paragraph1. Sentence 2.\n\nThis is paragrph2. Sentence2."
TEXT_EXPLICIT_BREAK = "Hello ...500ms world"


class DummyG2P:
    def phonemize(
        self,
        segments,
        doc,
        cfg: PipelineConfig,
        trace: Trace,
    ) -> list[PhonemeSegment]:
        _ = trace
        out = []
        for segment in segments:
            boundaries = slice_boundaries(
                doc.boundary_events,
                segment.char_start,
                segment.char_end,
                doc_end=len(doc.clean_text),
            )
            pause_before, pause_after = _resolve_pauses(boundaries, cfg.generation)
            out.append(
                PhonemeSegment(
                    id=f"{segment.id}_ph0",
                    segment_id=segment.id,
                    phoneme_id=0,
                    text=segment.text,
                    phonemes="a",
                    tokens=[],
                    lang=cfg.generation.lang,
                    char_start=segment.char_start,
                    char_end=segment.char_end,
                    paragraph_idx=segment.paragraph_idx,
                    sentence_idx=segment.sentence_idx,
                    clause_idx=segment.clause_idx,
                    pause_before=pause_before,
                    pause_after=pause_after,
                )
            )
        return out


def _resolve_pauses(boundaries, generation):
    pause_before = 0.0
    pause_after = 0.0
    for boundary in boundaries:
        if boundary.kind != "pause":
            continue
        duration = boundary.duration_s
        if duration is None:
            strength = boundary.attrs.get("strength")
            if strength == "c":
                duration = generation.pause_clause
            elif strength == "s":
                duration = generation.pause_sentence
            elif strength == "p":
                duration = generation.pause_paragraph
            elif strength == "w":
                duration = 0.15
            elif strength == "n":
                duration = 0.0
        if duration is None:
            continue
        if boundary.pos == 0:
            pause_before = max(pause_before, duration)
        else:
            pause_after = max(pause_after, duration)
    return pause_before, pause_after


def _build_pipeline(cfg: PipelineConfig) -> KokoroPipeline:
    return KokoroPipeline(
        cfg,
        doc_parser=SsmdDocumentParser(),
        g2p=DummyG2P(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(seconds_per_segment=0.01),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )


def test_pipeline_paragraph_indices():
    cfg = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    pipeline = _build_pipeline(cfg)
    res = pipeline.run(TEXT)

    paragraph_ids = [segment.paragraph_idx for segment in res.phoneme_segments]
    assert len(res.phoneme_segments) == 4
    assert paragraph_ids == [0, 0, 1, 1]


def test_pipeline_manual_paragraph_pause():
    generation = GenerationConfig(lang="en-us", pause_mode="manual", pause_paragraph=1.25)
    cfg = PipelineConfig(generation=generation)
    pipeline = _build_pipeline(cfg)
    res = pipeline.run(TEXT)

    assert len(res.phoneme_segments) == 4
    paragraph_zero = [segment for segment in res.phoneme_segments if segment.paragraph_idx == 0]
    assert paragraph_zero
    assert paragraph_zero[-1].pause_after == pytest.approx(generation.pause_paragraph)


def test_pipeline_explicit_break_pause():
    generation = GenerationConfig(lang="en-us", pause_mode="manual")
    cfg = PipelineConfig(generation=generation)
    pipeline = _build_pipeline(cfg)
    res = pipeline.run(TEXT_EXPLICIT_BREAK)

    assert res.phoneme_segments
    assert any(
        segment.pause_after == pytest.approx(0.5) or segment.pause_before == pytest.approx(0.5)
        for segment in res.phoneme_segments
    )
