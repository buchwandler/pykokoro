from __future__ import annotations

from dataclasses import replace

import pytest
from utterplan import PlannerConfig, UtterancePlan, UtterancePlanner

from pykokoro.exceptions import PlanConfigurationConflict, PlanConsumptionError
from pykokoro.generation_config import GenerationConfig
from pykokoro.pipeline import KokoroPipeline
from pykokoro.pipeline_config import PipelineConfig
from pykokoro.stages.audio_generation.noop import NoopAudioGenerationAdapter
from pykokoro.stages.audio_postprocessing.noop import NoopAudioPostprocessingAdapter
from pykokoro.stages.g2p.noop import NoopG2PAdapter
from pykokoro.stages.phoneme_processing.noop import NoopPhonemeProcessorAdapter


@pytest.fixture
def pipeline() -> KokoroPipeline:
    return KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        g2p=NoopG2PAdapter(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )


def test_run_plan_does_not_mutate_plan(pipeline: KokoroPipeline) -> None:
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Hello world.")
    before = plan.to_json(indent=None)
    plan_id = plan.plan_id

    result = pipeline.run_plan(plan)

    assert result.clean_text == "Hello world."
    assert plan.to_json(indent=None) == before
    assert plan.plan_id == plan_id
    assert result.trace is None


def test_run_plan_round_trips_saved_plan(pipeline: KokoroPipeline, tmp_path) -> None:
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Round trip.")
    path = tmp_path / "case.utterplan.json"
    plan.save(path)

    loaded = UtterancePlan.load(path)
    left = pipeline.run_plan(plan)
    right = pipeline.run_plan(loaded)

    assert left.clean_text == right.clean_text
    assert left.sample_rate == right.sample_rate
    assert left.audio.shape == right.audio.shape


def test_run_plan_does_not_call_planner(monkeypatch, pipeline: KokoroPipeline) -> None:
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Already planned.")

    def fail(*args, **kwargs):
        raise AssertionError("run_plan called the planner")

    monkeypatch.setattr("pykokoro.pipeline.UtterancePlanner.plan", fail)
    assert pipeline.run_plan(plan).clean_text == "Already planned."


def test_run_plan_rejects_planning_override(pipeline: KokoroPipeline) -> None:
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Frozen.")

    with pytest.raises(PlanConfigurationConflict, match="planning override 'lang'"):
        pipeline.run_plan(plan, lang="de-de")


def test_run_plan_supports_empty_plan_without_g2p() -> None:
    class FailingG2P:
        def phonemize(self, *args, **kwargs):
            raise AssertionError("empty plan called G2P")

    pipeline = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        g2p=FailingG2P(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("")

    result = pipeline.run_plan(plan)

    assert result.audio.size == 0
    assert result.clean_text == ""


def test_invalid_plan_unit_is_rejected(pipeline: KokoroPipeline) -> None:
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Invalid unit.")
    invalid = replace(plan, units=(replace(plan.units[0], segment_ids=("missing",)),))

    with pytest.raises(PlanConsumptionError):
        pipeline.run_plan(invalid)


def test_run_plan_preserves_plan_when_g2p_fails() -> None:
    class FailingG2P:
        def phonemize(self, *args, **kwargs):
            raise RuntimeError("g2p failed")

    pipeline = KokoroPipeline(
        PipelineConfig(generation=GenerationConfig(lang="en-us")),
        g2p=FailingG2P(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Failure path.")
    before = plan.to_json(indent=None)

    with pytest.raises(RuntimeError, match="g2p failed"):
        pipeline.run_plan(plan)

    assert plan.to_json(indent=None) == before


def test_text_and_plan_paths_share_rendering(pipeline: KokoroPipeline) -> None:
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Parity.")

    text_result = pipeline.run("Parity.")
    plan_result = pipeline.run_plan(plan)

    assert text_result.clean_text == plan_result.clean_text
    assert text_result.audio.shape == plan_result.audio.shape
    assert [segment.text for segment in text_result.segments] == [
        segment.text for segment in plan_result.segments
    ]


def test_plan_trace_contains_identity() -> None:
    pipeline = KokoroPipeline(
        PipelineConfig(
            generation=GenerationConfig(lang="en-us"),
            return_trace=True,
        ),
        g2p=NoopG2PAdapter(),
        phoneme_processing=NoopPhonemeProcessorAdapter(),
        audio_generation=NoopAudioGenerationAdapter(),
        audio_postprocessing=NoopAudioPostprocessingAdapter(),
    )
    plan = UtterancePlanner(PlannerConfig(language="en-us")).plan("Trace.")

    result = pipeline.run_plan(plan)

    assert result.trace is not None
    consume = [
        event
        for event in result.trace.events
        if event.stage == "planning" and event.name == "consume"
    ]
    assert consume and consume[0].details["plan_id"] == plan.plan_id
