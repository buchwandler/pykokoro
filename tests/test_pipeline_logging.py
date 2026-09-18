from __future__ import annotations

import logging

from pykokoro.stages.doc_parsers.plain import PlainTextDocumentParser

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.g2p.noop import NoopG2PAdapter
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter


def test_pipeline_logs_stage_and_unit_completion(caplog) -> None:
    pipeline = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        doc_parser=PlainTextDocumentParser(),
        g2p=NoopG2PAdapter(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(seconds_per_segment=0.001),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )
    caplog.set_level(logging.DEBUG, logger="pykokoro")

    result = pipeline.run("A sentence.")

    messages = [record.message for record in caplog.records]
    assert any("stage.start stage=doc name=parse" in message for message in messages)
    assert any("stage.finish stage=g2p name=phonemize" in message for message in messages)
    assert any("unit.finish index=0 unit_kind=paragraph" in message for message in messages)
    assert "samples=" in next(message for message in messages if "unit.finish" in message)
    assert result.audio.size > 0
