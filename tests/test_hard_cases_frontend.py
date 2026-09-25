from __future__ import annotations

import pytest

pytest.importorskip("benchmarks.hard_cases.frontend")

from benchmarks.hard_cases.frontend import NoOnnxFrontend


def test_english_frontend_prepares_one_plain_request_without_onnx() -> None:
    text = "The value is twelve dollars and fifty cents."
    with NoOnnxFrontend("en-US") as frontend:
        result = frontend.run(text)

    assert result.clean_text == text
    assert result.source_text == text
    assert len(result.segments) == 1
    assert len(result.phoneme_segments) == 1
    assert result.phoneme_segments[0].text == text
    assert result.phoneme_segments[0].tokens


def test_german_frontend_prepares_one_plain_request_without_onnx() -> None:
    text = "Der Wert beträgt zwölf Euro."
    with NoOnnxFrontend("de-DE") as frontend:
        result = frontend.run(text)

    assert result.clean_text == text
    assert result.phoneme_segments
