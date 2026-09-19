from __future__ import annotations

from dataclasses import replace

from utterplan import UtterancePlanner

from pykokoro import KokoroPipeline, PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.planning import planner_config_from_pipeline
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.g2p.noop import NoopG2PAdapter
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter


def _pipeline() -> KokoroPipeline:
    return KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us"), return_trace=True),
        g2p=NoopG2PAdapter(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(seconds_per_segment=0.0),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )


def test_pipeline_uses_current_planner_and_renderer_stages() -> None:
    pipeline = _pipeline()
    try:
        result = pipeline.run("Hello world.")
    finally:
        pipeline.close()

    assert result.clean_text == "Hello world."
    assert result.segments
    assert result.trace is not None
    assert any(event.stage == "planning" for event in result.trace.events)


def test_pipeline_run_overrides_lang() -> None:
    pipeline = _pipeline()
    try:
        result = pipeline.run("Hallo.", lang="de")
    finally:
        pipeline.close()

    assert result.phoneme_segments
    assert result.phoneme_segments[0].lang == "de"


def test_pipeline_renders_an_explicit_plan() -> None:
    config = PipelineConfig(generation=GenerationConfig(lang="en-us"))
    planner = UtterancePlanner(
        replace(
            planner_config_from_pipeline(config, unit="paragraph"),
            text_preparation="identity",
        )
    )
    try:
        plan = planner.plan("Hello world.")
    finally:
        planner.close()

    pipeline = _pipeline()
    try:
        result = pipeline.run_plan(plan)
    finally:
        pipeline.close()

    assert result.clean_text == "Hello world."
    assert result.document_metadata["utterplan"]["plan_id"] == plan.plan_id


def test_pipeline_trace_contains_renderer_stage_timings() -> None:
    pipeline = _pipeline()
    try:
        result = pipeline.run("Hello.")
    finally:
        pipeline.close()

    assert result.trace is not None
    assert any(event.stage == "g2p" for event in result.trace.events)
    assert any(event.stage == "audio_generation" for event in result.trace.events)
