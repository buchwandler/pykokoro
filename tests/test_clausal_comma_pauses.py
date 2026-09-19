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


def test_clausal_comma_plan_keeps_clean_offsets() -> None:
    plan = _plan("When the sun rises, the birds sing.")
    assert plan.texts.spoken == "When the sun rises, the birds sing."
    assert all(
        segment.text == plan.texts.spoken[segment.spoken_start : segment.spoken_end]
        for segment in plan.segments
    )


def test_non_auto_planning_still_produces_renderer_segments() -> None:
    config = PipelineConfig(generation=GenerationConfig(lang="en-us", pause_mode="manual"))
    planner = UtterancePlanner(planner_config_from_pipeline(config, unit="paragraph"))
    try:
        plan = planner.plan("A list, with commas, remains valid.")
    finally:
        planner.close()
    assert plan.segments


def test_clausal_planning_has_no_overlapping_segments() -> None:
    plan = _plan("The cat sat, and the dog slept.")
    assert all(
        left.spoken_end <= right.spoken_start
        for left, right in zip(plan.segments, plan.segments[1:])
    )
