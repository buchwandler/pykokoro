from __future__ import annotations

from utterplan import UtterancePlanner

from pykokoro import PipelineConfig
from pykokoro.generation_config import GenerationConfig
from pykokoro.planning import planner_config_from_pipeline


def test_german_abbreviations_and_ordinals_keep_monotonic_offsets() -> None:
    config = PipelineConfig(generation=GenerationConfig(lang="de"))
    planner = UtterancePlanner(planner_config_from_pipeline(config, unit="paragraph"))
    try:
        plan = planner.plan("Dr. Schmidt kommt am 5. März. Das ist gut.")
    finally:
        planner.close()

    assert plan.segments
    assert all(
        left.spoken_end <= right.spoken_start
        for left, right in zip(plan.segments, plan.segments[1:], strict=False)
    )
