from __future__ import annotations

from typing import Any, cast

import pytest

from pykokoro.exceptions import InvalidRequestError, UnsupportedFeatureError
from pykokoro.frontend_projection import project_frontend_text
from pykokoro.synthesis_types import SayAsSpan


def _span(text: str, surface: str) -> SayAsSpan:
    start = text.index(surface)
    return SayAsSpan(start, start + len(surface), "characters")


def test_projection_handles_compact_spaced_and_digit_sequences() -> None:
    compact = project_frontend_text("ART", (SayAsSpan(0, 3, "characters"),))
    spaced = project_frontend_text("A R T", (SayAsSpan(0, 5, "characters"),))
    digits = project_frontend_text("R2D2", (SayAsSpan(0, 4, "characters"),))

    assert compact.frontend_text == "A- R- T"
    assert spaced.frontend_text == "A- R- T"
    assert digits.frontend_text == "R- 2- D- 2"
    assert compact.map_canonical_span(0, 3) == (0, 7)
    assert spaced.map_canonical_span(0, 5) == (0, 7)
    assert compact.map_canonical_span(1, 2) == (3, 4)
    assert compact.map_frontend_span(0, 7) == (0, 3)


def test_projection_preserves_possessive_suffixes_and_unmarked_text() -> None:
    straight = "A R T's house is nice."
    curly = "A R T’s house is nice."

    projected_straight = project_frontend_text(straight, (_span(straight, "A R T"),))
    projected_curly = project_frontend_text(curly, (_span(curly, "A R T"),))

    assert projected_straight.frontend_text == "A- R- T's house is nice."
    assert projected_curly.frontend_text == "A- R- T’s house is nice."
    assert "T-'" not in projected_straight.frontend_text
    assert "T-’" not in projected_curly.frontend_text
    assert projected_straight.canonical_text == straight


def test_projection_composes_multiple_spans_and_maps_cumulative_offsets() -> None:
    text = "A R T met U S A."
    first = _span(text, "A R T")
    second = _span(text, "U S A")
    projection = project_frontend_text(text, (first, second))

    assert projection.frontend_text == "A- R- T met U- S- A."
    assert projection.map_canonical_span(first.start, first.end) == (0, 7)
    second_frontend = projection.map_canonical_span(second.start, second.end)
    assert projection.frontend_text[slice(*second_frontend)] == "U- S- A"
    assert projection.map_frontend_span(*second_frontend) == (second.start, second.end)


@pytest.mark.parametrize(
    "surface",
    ["A&B", "A / B", "A-B", "A, B", "one two", "A R word", "A"],
)
def test_projection_rejects_unsupported_character_surfaces(surface: str) -> None:
    projection_span = SayAsSpan(0, len(surface), "characters")

    with pytest.raises(InvalidRequestError, match="unsupported surface|at least two"):
        project_frontend_text(surface, (projection_span,))


def test_projection_rejects_unsupported_interpretations_and_invalid_boundaries() -> None:
    with pytest.raises(UnsupportedFeatureError, match="not supported"):
        project_frontend_text("ART", (SayAsSpan(0, 3, "date"),))

    projection = project_frontend_text("A R T", (SayAsSpan(0, 5, "characters"),))
    with pytest.raises(InvalidRequestError, match="frontend span"):
        projection.map_frontend_span(0, 8)
    with pytest.raises(InvalidRequestError, match="boundary bias"):
        projection.map_canonical_span(0, 1, start_bias=cast(Any, "middle"))
