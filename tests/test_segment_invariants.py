from __future__ import annotations

from pykokoro.debug.segment_invariants import check_segment_invariants
from pykokoro.types import Segment


def segment(start: int, end: int, text: str = "") -> Segment:
    return Segment(id=f"segment-{start}-{end}", text=text, char_start=start, char_end=end)


def test_out_of_bounds_ranges_return_diagnostics_without_raising() -> None:
    result = check_segment_invariants([segment(0, 999)], "abc", report_fn=None)

    assert not result.ok
    assert any("exceeds document length" in error for error in result.errors)


def test_negative_ranges_return_diagnostics_without_raising() -> None:
    result = check_segment_invariants([segment(-10, 2)], "abc", report_fn=None)

    assert not result.ok
    assert any("invalid offsets" in error for error in result.errors)


def test_reversed_ranges_return_diagnostics_without_raising() -> None:
    result = check_segment_invariants([segment(2, 1)], "abc", report_fn=None)

    assert not result.ok
    assert any("invalid offsets" in error for error in result.errors)


def test_huge_range_returns_diagnostics_without_raising() -> None:
    result = check_segment_invariants([segment(10, 10_000_000)], "abc", report_fn=None)

    assert not result.ok
    assert any("exceeds document length" in error for error in result.errors)


def test_empty_document_with_empty_segment_is_valid() -> None:
    result = check_segment_invariants([segment(0, 0)], "", report_fn=None)

    assert result.ok


def test_valid_segments_and_whitespace_gap_are_valid() -> None:
    result = check_segment_invariants(
        [segment(0, 3, "abc"), segment(4, 7, "def")],
        "abc def",
        report_fn=None,
    )

    assert result.ok


def test_valid_overlap_is_reported() -> None:
    result = check_segment_invariants(
        [segment(0, 3, "abc"), segment(2, 5, "cde")],
        "abcde",
        report_fn=None,
    )

    assert not result.ok
    assert any("overlap" in error.lower() for error in result.errors)
