from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ..types import Segment


@dataclass
class SegmentInvariantResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def check_segment_invariants(
    segments: Sequence[Segment],
    clean_text: str,
    *,
    allow_whitespace_gaps: bool = True,
    report_fn: Callable[[str], None] | None = print,
) -> SegmentInvariantResult:
    errors: list[str] = []
    warnings: list[str] = []

    if not segments:
        errors.append("No segments provided for invariant check.")
        return _emit_report(errors, warnings, report_fn)

    for seg in segments:
        if seg.char_start < 0 or seg.char_end < seg.char_start:
            errors.append(
                f"Segment has invalid offsets: {_format_segment(seg)} (len={len(clean_text)})."
            )
        if seg.char_end > len(clean_text):
            errors.append(
                f"Segment exceeds document length: {_format_segment(seg)} (len={len(clean_text)})."
            )

    for prev, current in _pairwise(segments):
        if current.char_start < prev.char_start:
            errors.append(
                "Segments are not sorted by char_start: "
                f"{_format_segment(prev)} before {_format_segment(current)}."
            )
        if prev.char_end > current.char_start:
            errors.append(_format_overlap(prev, current, clean_text))
        elif prev.char_end < current.char_start:
            gap_text = clean_text[prev.char_end : current.char_start]
            if gap_text.strip():
                message = (
                    "Gap between segments contains non-whitespace: "
                    f"{_format_segment(prev)} -> {_format_segment(current)} "
                    f"gap={gap_text!r}."
                )
                if allow_whitespace_gaps:
                    warnings.append(message)
                else:
                    errors.append(message)

    coverage_errors = _check_coverage(segments, clean_text, allow_whitespace_gaps)
    errors.extend(coverage_errors)

    return _emit_report(errors, warnings, report_fn)


def _check_coverage(
    segments: Sequence[Segment],
    clean_text: str,
    allow_whitespace_gaps: bool,
) -> list[str]:
    if not clean_text:
        return []

    coverage = [0] * len(clean_text)
    for seg in segments:
        for idx in range(seg.char_start, seg.char_end):
            coverage[idx] += 1

    errors: list[str] = []
    overlap_positions = [idx for idx, count in enumerate(coverage) if count > 1]
    if overlap_positions:
        errors.append(
            f"Coverage overlap detected at positions: {_format_positions(overlap_positions)}."
        )

    if allow_whitespace_gaps:
        gap_positions = [
            idx
            for idx, count in enumerate(coverage)
            if count == 0 and not clean_text[idx].isspace()
        ]
        if gap_positions:
            errors.append(
                "Coverage gap detected at non-whitespace positions: "
                f"{_format_positions(gap_positions)}."
            )
    else:
        if any(count == 0 for count in coverage):
            errors.append("Coverage gap detected (missing characters in segments).")

    return errors


def _format_overlap(prev: Segment, current: Segment, clean_text: str) -> str:
    overlap_start = max(prev.char_start, current.char_start)
    overlap_end = min(prev.char_end, current.char_end)
    overlap_text = clean_text[overlap_start:overlap_end]
    context_start = max(0, overlap_start - 25)
    context_end = min(len(clean_text), overlap_end + 25)
    context_text = clean_text[context_start:context_end]
    return (
        "Segment overlap detected:\n"
        f"  prev: {_format_segment(prev)}\n"
        f"  curr: {_format_segment(current)}\n"
        f"  overlap: {overlap_text!r} @ {overlap_start}:{overlap_end}\n"
        f"  context: {context_text!r}"
    )


def _format_positions(positions: Sequence[int], limit: int = 6) -> str:
    if len(positions) <= limit:
        return ", ".join(str(idx) for idx in positions)
    head = ", ".join(str(idx) for idx in positions[:limit])
    return f"{head}, ... (+{len(positions) - limit} more)"


def _format_segment(seg: Segment) -> str:
    return f"{seg.id} {seg.char_start}:{seg.char_end} {seg.text!r}"


def _pairwise(items: Sequence[Segment]) -> Iterable[tuple[Segment, Segment]]:
    for idx in range(1, len(items)):
        yield items[idx - 1], items[idx]


def _emit_report(
    errors: list[str],
    warnings: list[str],
    report_fn: Callable[[str], None] | None,
) -> SegmentInvariantResult:
    ok = not errors
    if report_fn is not None:
        for message in errors:
            report_fn(f"[segment-invariants][error] {message}")
        for message in warnings:
            report_fn(f"[segment-invariants][warn] {message}")
    return SegmentInvariantResult(ok=ok, errors=errors, warnings=warnings)
