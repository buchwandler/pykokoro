from __future__ import annotations

from benchmarks.hard_cases.data import load_cases
from benchmarks.hard_cases.frontend import NoOnnxFrontend
from benchmarks.hard_cases.phonemes import evaluate_case


def test_english_frontend_runs_without_onnx() -> None:
    case = load_cases(locale="en-US", case_id="en_shared_001")[0]
    with NoOnnxFrontend("en-US") as frontend:
        result = frontend.run(case.text)
        evaluation = evaluate_case(case, frontend)
    assert result.clean_text == "The value is twelve dollars and fifty cents."
    assert result.phoneme_segments
    assert not evaluation.failed


def test_german_frontend_runs_without_onnx() -> None:
    case = load_cases(locale="de-DE", case_id="de_shared_001")[0]
    with NoOnnxFrontend("de-DE") as frontend:
        result = frontend.run(case.text)
    assert result.clean_text
    assert result.phoneme_segments
