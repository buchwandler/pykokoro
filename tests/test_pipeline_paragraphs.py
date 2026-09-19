import pytest

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
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
            plan_pauses = doc.metadata.get("utterplan_pauses", {})
            pause_before, pause_after = plan_pauses.get(segment.id, (0.0, 0.0))
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


def _build_pipeline(cfg: PipelineConfig) -> KokoroPipeline:
    return KokoroPipeline(
        cfg,
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
