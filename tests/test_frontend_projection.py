from __future__ import annotations

from typing import Any, cast

import pytest

from pykokoro.exceptions import InvalidRequestError, UnsupportedFeatureError
from pykokoro.frontend_projection import project_frontend_text
from pykokoro.synthesis_types import SayAsSpan


def _span(text: str, surface: str) -> SayAsSpan:
    start = text.index(surface)
    return SayAsSpan(start, start + len(surface), "characters")


def test_projection_inserts_spaces_between_compact_characters_with_boundary_bias() -> None:
    projection = project_frontend_text("ART", (SayAsSpan(0, 3, "characters"),))

    assert projection.frontend_text == "A R T"
    assert projection.canonical_text == "ART"
    assert projection.map_canonical_span(0, 3) == (0, 5)
    assert projection.map_frontend_span(0, 5) == (0, 3)
    assert projection.map_canonical_span(1, 1, start_bias="left", end_bias="left") == (1, 1)
    assert projection.map_canonical_span(1, 1, start_bias="right", end_bias="right") == (2, 2)
    assert projection.map_canonical_span(2, 2, start_bias="left", end_bias="left") == (3, 3)
    assert projection.map_canonical_span(2, 2, start_bias="right", end_bias="right") == (4, 4)
    assert projection.map_canonical_span(3, 3) == (5, 5)


def test_projection_separates_mixed_ascii_letters_and_digits() -> None:
    projection = project_frontend_text("R2D2", (SayAsSpan(0, 4, "characters"),))

    assert projection.frontend_text == "R 2 D 2"
    assert projection.map_canonical_span(0, 4) == (0, 7)
    assert projection.map_frontend_span(0, 7) == (0, 4)


def test_projection_preserves_already_spaced_core_and_identity_boundaries() -> None:
    text = "A R T"
    projection = project_frontend_text(text, (SayAsSpan(0, len(text), "characters"),))

    assert projection.frontend_text == text
    assert projection.canonical_text == text
    assert projection.map_canonical_span(0, len(text)) == (0, len(text))
    assert projection.map_frontend_span(0, len(text)) == (0, len(text))
    for boundary in range(len(text) + 1):
        assert projection.map_canonical_span(boundary, boundary) == (boundary, boundary)
        assert projection.map_frontend_span(boundary, boundary) == (boundary, boundary)


def test_projection_preserves_outer_parentheses_around_character_span() -> None:
    text = "(A R T)"
    span = SayAsSpan(0, len(text), "characters")
    projection = project_frontend_text(text, (span,))

    assert projection.frontend_text == "(A R T)"
    assert projection.canonical_text == text
    assert (span.start, span.end) == (0, len(text))
    assert projection.map_canonical_span(0, len(text)) == (0, len(text))
    assert projection.map_frontend_span(0, len(text)) == (0, len(text))
    inner_start = text.index("A")
    inner_end = text.rindex("T") + 1
    assert projection.map_canonical_span(inner_start, inner_end) == (inner_start, inner_end)


@pytest.mark.parametrize(
    ("surface", "expected"),
    [
        ("[ART]", "[A R T]"),
        ('"R2D2"', '"R 2 D 2"'),
        ("(A R T).", "(A R T)."),
    ],
)
def test_projection_preserves_outer_wrapper_text(surface: str, expected: str) -> None:
    projection = project_frontend_text(surface, (SayAsSpan(0, len(surface), "characters"),))

    assert projection.frontend_text == expected
    assert projection.canonical_text == surface


@pytest.mark.parametrize(
    ("text", "surface", "expected"),
    [
        ("ART", "ART", "A R T"),
        ("A R T", "A R T", "A R T"),
        ("R2D2", "R2D2", "R 2 D 2"),
        ("(A R T)", "(A R T)", "(A R T)"),
        ('"R2D2"', '"R2D2"', '"R 2 D 2"'),
    ],
)
def test_projection_maps_every_boundary_and_complete_marked_span(
    text: str, surface: str, expected: str
) -> None:
    span = _span(text, surface)
    projection = project_frontend_text(text, (span,))
    frontend_span = projection.map_canonical_span(span.start, span.end)

    assert projection.frontend_text[frontend_span[0] : frontend_span[1]] == expected
    assert projection.map_frontend_span(*frontend_span) == (span.start, span.end)
    for boundary in range(len(text) + 1):
        projection.map_canonical_span(boundary, boundary)
    for boundary in range(len(projection.frontend_text) + 1):
        projection.map_frontend_span(boundary, boundary)


def test_projection_maps_post_span_ranges_after_inserted_spaces() -> None:
    text = "ART now"
    projection = project_frontend_text(text, (_span(text, "ART"),))

    assert projection.frontend_text == "A R T now"
    frontend_now = projection.map_canonical_span(4, 7)
    assert frontend_now == (6, 9)
    assert projection.frontend_text[frontend_now[0] : frontend_now[1]] == "now"
    assert projection.map_frontend_span(*frontend_now) == (4, 7)


@pytest.mark.parametrize(
    ("text", "surface", "expected"),
    [
        ("A R T's house is nice.", "A R T", "A R T's house is nice."),
        ("A R T’s house is nice.", "A R T", "A R T’s house is nice."),
        ("A R T 's house is nice.", "A R T", "A R T's house is nice."),
        ("A R T ’s house is nice.", "A R T", "A R T’s house is nice."),
        ("ART 's house is nice.", "ART", "A R T's house is nice."),
        ("A R T 'sample' is quoted.", "A R T", "A R T 'sample' is quoted."),
        ("A R T\n's house is split.", "A R T", "A R T\n's house is split."),
    ],
)
def test_projection_joins_only_adjacent_detached_possessive_after_marked_characters(
    text: str, surface: str, expected: str
) -> None:
    projection = project_frontend_text(text, (_span(text, surface),))

    assert projection.frontend_text == expected
    assert projection.canonical_text == text


def test_projection_does_not_join_unmarked_possessive() -> None:
    text = "A R T 's house"
    projection = project_frontend_text(text)

    assert projection.frontend_text == text
    assert projection.canonical_text == text


def test_deleted_possessive_separator_has_zero_width_source_mapping() -> None:
    text = "A R T 's house"
    projection = project_frontend_text(text, (_span(text, "A R T"),))

    assert projection.frontend_text == "A R T's house"
    assert projection.canonical_text == text
    assert projection.map_canonical_span(5, 6) == (5, 5)
    assert projection.map_canonical_span(9, 14) == (8, 13)
    assert projection.map_frontend_span(0, 7) == (0, 8)
    assert projection.map_frontend_span(8, 13) == (9, 14)
    assert len(projection._canonical_left) == len(text) + 1
    assert len(projection._canonical_right) == len(text) + 1
    assert len(projection._frontend_left) == len(projection.frontend_text) + 1
    assert len(projection._frontend_right) == len(projection.frontend_text) + 1
    for mapping in (
        projection._canonical_left,
        projection._canonical_right,
        projection._frontend_left,
        projection._frontend_right,
    ):
        assert tuple(mapping) == tuple(sorted(mapping))


def test_projection_composes_multiple_compact_spans_and_maps_cumulative_offsets() -> None:
    text = "ART met USA."
    first = _span(text, "ART")
    second = _span(text, "USA")
    projection = project_frontend_text(text, (first, second))

    assert projection.frontend_text == "A R T met U S A."
    assert projection.map_canonical_span(first.start, first.end) == (0, 5)
    second_frontend = projection.map_canonical_span(second.start, second.end)
    assert projection.frontend_text[slice(*second_frontend)] == "U S A"
    assert projection.map_frontend_span(*second_frontend) == (second.start, second.end)
    for boundary in range(len(text) + 1):
        projection.map_canonical_span(boundary, boundary)
    for boundary in range(len(projection.frontend_text) + 1):
        projection.map_frontend_span(boundary, boundary)


@pytest.mark.parametrize(
    "surface",
    ["A&B", "A / B", "A-B", "A, B", "one two", "A R word", "A", "(A-B)"],
)
def test_projection_rejects_unsupported_character_surfaces(surface: str) -> None:
    projection_span = SayAsSpan(0, len(surface), "characters")

    with pytest.raises(InvalidRequestError, match="unsupported character payload|at least two"):
        project_frontend_text(surface, (projection_span,))


def test_projection_rejects_unsupported_interpretations_and_invalid_boundaries() -> None:
    with pytest.raises(UnsupportedFeatureError, match="not supported"):
        project_frontend_text("ART", (SayAsSpan(0, 3, "date"),))

    projection = project_frontend_text("A R T", (SayAsSpan(0, 5, "characters"),))
    with pytest.raises(InvalidRequestError, match="frontend span"):
        projection.map_frontend_span(0, 8)
    with pytest.raises(InvalidRequestError, match="boundary bias"):
        projection.map_canonical_span(0, 1, start_bias=cast(Any, "middle"))
