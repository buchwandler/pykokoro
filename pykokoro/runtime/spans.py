from __future__ import annotations

from typing import Literal

from ..types import AnnotationSpan, BoundaryEvent


def slice_spans(
    spans: list[AnnotationSpan],
    seg_start: int,
    seg_end: int,
    overlap_mode: Literal["snap", "strict"] = "snap",
    warnings: list[str] | None = None,
) -> list[AnnotationSpan]:
    sliced: list[AnnotationSpan] = []
    for span in spans:
        if span.char_end <= seg_start or span.char_start >= seg_end:
            continue
        if overlap_mode == "strict":
            if span.char_start < seg_start or span.char_end > seg_end:
                if warnings is not None:
                    warnings.append(
                        f"Dropped partial annotation span at {span.char_start}:{span.char_end}"
                    )
                continue
            start = span.char_start - seg_start
            end = span.char_end - seg_start
        else:
            start = max(span.char_start, seg_start) - seg_start
            end = min(span.char_end, seg_end) - seg_start
        sliced.append(
            AnnotationSpan(
                char_start=start,
                char_end=end,
                attrs=dict(span.attrs),
            )
        )
    return sliced


def slice_boundaries(
    boundaries: list[BoundaryEvent],
    seg_start: int,
    seg_end: int,
    doc_end: int | None = None,
) -> list[BoundaryEvent]:
    sliced: list[BoundaryEvent] = []
    for boundary in boundaries:
        if boundary.pos < seg_start or boundary.pos > seg_end:
            continue
        if (
            doc_end is not None
            and boundary.pos == seg_end
            and boundary.pos != doc_end
            and boundary.attrs.get("anchor", "before") != "after"
        ):
            continue
        if seg_start <= boundary.pos <= seg_end:
            sliced.append(
                BoundaryEvent(
                    pos=boundary.pos - seg_start,
                    kind=boundary.kind,
                    duration_s=boundary.duration_s,
                    attrs=dict(boundary.attrs),
                )
            )
    return sliced
