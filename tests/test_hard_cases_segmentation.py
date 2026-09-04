from __future__ import annotations

from benchmarks.hard_cases.frontend import NoOnnxFrontend
from benchmarks.hard_cases.segmentation import build_segment_plan, validate_offsets


def test_segment_plan_uses_clean_text_offsets() -> None:
    with NoOnnxFrontend("en-US") as frontend:
        result = frontend.run("Wait—now. Are you ready?")
    plan = build_segment_plan(result)
    assert plan
    assert validate_offsets(result.clean_text, plan) == ()
    assert all(result.clean_text[item.char_start : item.char_end] == item.text for item in plan)


def test_ssmd_frontend_retains_prepared_plan() -> None:
    with NoOnnxFrontend("de-DE", ssmd=True) as frontend:
        result = frontend.run('<speak>Achtung<break time="300ms"/> jetzt.</speak>')
    assert result.clean_text
    assert build_segment_plan(result)
