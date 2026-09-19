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


def test_parenthetical_text_remains_in_clean_text() -> None:
    plan = _plan("The result (as expected) was correct.")
    assert "as expected" in plan.texts.spoken
    assert plan.segments


def test_final_parenthetical_plan_has_stable_offsets() -> None:
    plan = _plan("The result was correct (and useful).")
    assert all(segment.spoken_start < segment.spoken_end for segment in plan.segments)


def test_parenthetical_planning_is_safe_without_renderer() -> None:
    plan = _plan("A sentence (with an aside).")
    assert plan.diagnostics
    assert plan.plan_id.startswith("sha256:")
