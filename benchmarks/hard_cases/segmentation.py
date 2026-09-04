from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .schema import HardCase


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    id: str
    text: str
    char_start: int
    char_end: int
    sentence_idx: int | None
    clause_idx: int | None
    pause_before: float
    pause_after: float
    sentence_boundary: bool
    clause_boundary: bool


@dataclass(frozen=True, slots=True)
class PlanEvaluation:
    case_id: str
    clean_text: str
    segments: tuple[SegmentPlan, ...]
    passed: bool
    errors: tuple[str, ...] = ()
    likely_owner: str = "none"

    @property
    def failed(self) -> bool:
        return not self.passed


def build_segment_plan(result: Any) -> tuple[SegmentPlan, ...]:
    phonemes = tuple(result.phoneme_segments)
    output: list[SegmentPlan] = []
    for index, segment in enumerate(result.segments):
        related = [item for item in phonemes if item.segment_id == segment.id]
        first, last = (related[0], related[-1]) if related else (None, None)
        output.append(
            SegmentPlan(
                id=segment.id,
                text=segment.text,
                char_start=segment.char_start,
                char_end=segment.char_end,
                sentence_idx=segment.sentence_idx,
                clause_idx=segment.clause_idx,
                pause_before=float(first.pause_before) if first else 0.0,
                pause_after=float(last.pause_after) if last else 0.0,
                sentence_boundary=index + 1 < len(result.segments)
                and segment.sentence_idx != result.segments[index + 1].sentence_idx,
                clause_boundary=index + 1 < len(result.segments)
                and segment.clause_idx != result.segments[index + 1].clause_idx,
            )
        )
    return tuple(output)


def validate_offsets(clean_text: str, segments: tuple[SegmentPlan, ...]) -> tuple[str, ...]:
    errors: list[str] = []
    previous_end = 0
    for segment in segments:
        if not 0 <= segment.char_start <= segment.char_end <= len(clean_text):
            errors.append(f"invalid offsets for {segment.id}")
        elif clean_text[segment.char_start : segment.char_end] != segment.text:
            errors.append(f"offset slice mismatch for {segment.id}")
        if segment.char_start < previous_end:
            errors.append(f"overlapping segments at {segment.id}")
        previous_end = max(previous_end, segment.char_end)
    return tuple(errors)


def evaluate_plan(case: HardCase, frontend: Any) -> PlanEvaluation:
    try:
        result = frontend.run(case.text)
        plan = build_segment_plan(result)
        errors = list(validate_offsets(result.clean_text, plan))
        expected = case.expect.segment_expectations
        for index, requirement in enumerate(expected):
            if index >= len(plan):
                errors.append(f"missing expected segment {index}")
                continue
            actual = plan[index]
            if requirement.text is not None and actual.text != requirement.text:
                errors.append(f"segment {index} text mismatch")
            if requirement.char_start is not None and actual.char_start != requirement.char_start:
                errors.append(f"segment {index} start mismatch")
            if requirement.char_end is not None and actual.char_end != requirement.char_end:
                errors.append(f"segment {index} end mismatch")
            if (
                requirement.sentence_boundary is not None
                and actual.sentence_boundary != requirement.sentence_boundary
            ):
                errors.append(f"segment {index} sentence boundary mismatch")
            if (
                requirement.clause_boundary is not None
                and actual.clause_boundary != requirement.clause_boundary
            ):
                errors.append(f"segment {index} clause boundary mismatch")
        for _index, requirement in enumerate(case.expect.pause_expectations):
            candidates = [
                item
                for item in plan
                if requirement.position is None or item.char_end == requirement.position
            ]
            if not candidates:
                errors.append(f"pause at {requirement.position!r} not found")
                continue
            pause = candidates[0].pause_after
            if requirement.min_duration_s is not None and pause < requirement.min_duration_s:
                errors.append(f"pause below minimum at {requirement.position!r}")
            if requirement.max_duration_s is not None and pause > requirement.max_duration_s:
                errors.append(f"pause above maximum at {requirement.position!r}")
        owner = "phrasplit" if errors else "none"
        return PlanEvaluation(case.id, result.clean_text, plan, not errors, tuple(errors), owner)
    except Exception as exc:
        return PlanEvaluation(
            case.id, "", (), False, (f"{type(exc).__name__}: {exc}",), "pykokoro_pipeline"
        )


__all__ = [
    "PlanEvaluation",
    "SegmentPlan",
    "build_segment_plan",
    "evaluate_plan",
    "validate_offsets",
]
