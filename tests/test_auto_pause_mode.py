from __future__ import annotations

from utterplan import UtterancePlanner

from pykokoro import PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.planning import planner_config_from_pipeline


def _plan(text: str):
    config = PipelineConfig(generation=GenerationConfig(lang="en-us", pause_mode="auto"))
    planner = UtterancePlanner(planner_config_from_pipeline(config, unit="paragraph"))
    try:
        return planner.plan(text)
    finally:
        planner.close()


def test_plain_text_auto_sentence_boundaries() -> None:
    plan = _plan("First sentence. Second sentence.")
    assert len(plan.segments) == 2
    assert any(boundary.kind == "sentence" for boundary in plan.boundaries)


def test_ssmd_auto_sentence_boundaries() -> None:
    plan = _plan("First sentence.\n\nSecond sentence.")
    assert len(plan.segments) == 2
    assert plan.segments[0].spoken_end <= plan.segments[1].spoken_start


def test_auto_pause_plan_is_renderer_independent() -> None:
    plan = _plan("A sentence.")
    assert plan.diagnostics
    assert plan.diagnostics[-1].code == "planning.complete"
