"""Kokoro-only projection from canonical request text to frontend text."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

from .exceptions import InvalidRequestError, UnsupportedFeatureError
from .synthesis_types import SayAsSpan

BoundaryBias = Literal["left", "right"]
_CHARACTER = re.compile(r"[A-Za-z0-9]")
_COMPACT_CHARACTERS = re.compile(r"[A-Za-z0-9]{2,}")
_SPACED_CHARACTERS = re.compile(r"[A-Za-z0-9](?:\s+[A-Za-z0-9])+")


@dataclass(frozen=True, slots=True)
class FrontendTextProjection:
    """Canonical and Kokoro frontend text plus deterministic span mappings."""

    canonical_text: str
    frontend_text: str
    _canonical_left: tuple[int, ...]
    _canonical_right: tuple[int, ...]
    _frontend_left: tuple[int, ...]
    _frontend_right: tuple[int, ...]

    def map_canonical_span(
        self,
        start: int,
        end: int,
        *,
        start_bias: BoundaryBias = "right",
        end_bias: BoundaryBias = "left",
    ) -> tuple[int, int]:
        """Map canonical boundaries, choosing explicitly how to include insertions."""
        _validate_span(start, end, len(self.canonical_text), "canonical")
        _validate_bias(start_bias)
        _validate_bias(end_bias)
        frontend_start = _boundary_value(
            self._canonical_left, self._canonical_right, start, start_bias
        )
        frontend_end = _boundary_value(self._canonical_left, self._canonical_right, end, end_bias)
        return frontend_start, max(frontend_start, frontend_end)

    def map_frontend_span(
        self,
        start: int,
        end: int,
        *,
        start_bias: BoundaryBias = "left",
        end_bias: BoundaryBias = "right",
    ) -> tuple[int, int]:
        """Map frontend boundaries back to canonical text with explicit boundary bias."""
        _validate_span(start, end, len(self.frontend_text), "frontend")
        _validate_bias(start_bias)
        _validate_bias(end_bias)
        canonical_start = _boundary_value(
            self._frontend_left, self._frontend_right, start, start_bias
        )
        canonical_end = _boundary_value(self._frontend_left, self._frontend_right, end, end_bias)
        return canonical_start, max(canonical_start, canonical_end)


def project_frontend_text(
    canonical_text: str,
    say_as: Sequence[object] = (),
) -> FrontendTextProjection:
    """Project marked character spans into the frontend syntax understood by Kokoro."""
    if not isinstance(canonical_text, str):
        raise InvalidRequestError("canonical text must be a string")
    if not isinstance(say_as, (tuple, list)):
        raise InvalidRequestError("say_as must be a tuple or list of SayAsSpan values")

    raw_spans = tuple(say_as)
    spans: list[SayAsSpan] = []
    previous: SayAsSpan | None = None
    for index, candidate in enumerate(raw_spans):
        if not isinstance(candidate, SayAsSpan):
            raise InvalidRequestError("say_as must contain SayAsSpan values")
        span = cast(SayAsSpan, candidate)
        _validate_span(span.start, span.end, len(canonical_text), f"say_as[{index}]")
        if previous is not None:
            if (span.start, span.end) < (previous.start, previous.end):
                raise InvalidRequestError("say_as spans must be sorted by (start, end)")
            if span.start < previous.end:
                raise InvalidRequestError("overlapping say_as spans are ambiguous")
        if span.interpret_as != "characters":
            raise UnsupportedFeatureError(
                f"say_as interpretation {span.interpret_as!r} is not supported"
            )
        spans.append(span)
        previous = span

    frontend_parts: list[str] = []
    canonical_left: list[int | None] = [None] * (len(canonical_text) + 1)
    canonical_right: list[int | None] = [None] * (len(canonical_text) + 1)
    frontend_left: list[int] = []
    frontend_right: list[int] = []
    canonical_cursor = 0
    frontend_cursor = 0

    for span in spans:
        if canonical_cursor < span.start:
            piece = canonical_text[canonical_cursor : span.start]
            local_frontier = list(range(len(piece) + 1))
            _append_piece(
                piece,
                canonical_cursor,
                frontend_cursor,
                local_frontier,
                local_frontier,
                frontend_parts,
                canonical_left,
                canonical_right,
                frontend_left,
                frontend_right,
            )
            frontend_cursor += len(piece)

        surface = canonical_text[span.start : span.end]
        frontend, local_canonical_left, local_canonical_right, local_frontend = (
            _project_character_surface(surface, span.start)
        )
        _append_piece(
            frontend,
            span.start,
            frontend_cursor,
            local_canonical_left,
            local_canonical_right,
            frontend_parts,
            canonical_left,
            canonical_right,
            frontend_left,
            frontend_right,
            frontend_to_canonical=local_frontend,
        )
        frontend_cursor += len(frontend)
        canonical_cursor = span.end

    if canonical_cursor < len(canonical_text):
        piece = canonical_text[canonical_cursor:]
        local_frontier = list(range(len(piece) + 1))
        _append_piece(
            piece,
            canonical_cursor,
            frontend_cursor,
            local_frontier,
            local_frontier,
            frontend_parts,
            canonical_left,
            canonical_right,
            frontend_left,
            frontend_right,
        )
        frontend_cursor += len(piece)
    elif not spans:
        local_frontier = list(range(len(canonical_text) + 1))
        _append_piece(
            canonical_text,
            0,
            0,
            local_frontier,
            local_frontier,
            frontend_parts,
            canonical_left,
            canonical_right,
            frontend_left,
            frontend_right,
        )
        frontend_cursor = len(canonical_text)

    if any(value is None for value in (*canonical_left, *canonical_right)):
        raise RuntimeError("canonical-to-frontend projection left an unmapped boundary")
    if len(frontend_left) != frontend_cursor + 1 or len(frontend_right) != frontend_cursor + 1:
        raise RuntimeError("frontend-to-canonical projection left an unmapped boundary")

    return FrontendTextProjection(
        canonical_text=canonical_text,
        frontend_text="".join(frontend_parts),
        _canonical_left=tuple(value for value in canonical_left if value is not None),
        _canonical_right=tuple(value for value in canonical_right if value is not None),
        _frontend_left=tuple(frontend_left),
        _frontend_right=tuple(frontend_right),
    )


def _append_piece(
    frontend: str,
    canonical_start: int,
    frontend_start: int,
    canonical_to_frontend_left: list[int],
    canonical_to_frontend_right: list[int],
    frontend_parts: list[str],
    canonical_left: list[int | None],
    canonical_right: list[int | None],
    frontend_left: list[int],
    frontend_right: list[int],
    *,
    frontend_to_canonical: list[int] | None = None,
) -> None:
    frontend_parts.append(frontend)
    for local, (left, right) in enumerate(
        zip(canonical_to_frontend_left, canonical_to_frontend_right, strict=True)
    ):
        index = canonical_start + local
        canonical_left[index] = frontend_start + left
        canonical_right[index] = frontend_start + right

    if frontend_to_canonical is None:
        frontend_to_canonical = list(range(canonical_start, canonical_start + len(frontend) + 1))
    if len(frontend_to_canonical) != len(frontend) + 1:
        raise RuntimeError("projection piece has invalid frontend boundary mapping")
    piece_left = frontend_to_canonical
    piece_right = frontend_to_canonical
    if not frontend_left:
        frontend_left.extend(piece_left)
        frontend_right.extend(piece_right)
        return
    if frontend_left[-1] != piece_left[0] or frontend_right[-1] != piece_right[0]:
        raise RuntimeError("projection pieces disagree at their shared boundary")
    frontend_left.extend(piece_left[1:])
    frontend_right.extend(piece_right[1:])


def _project_character_surface(
    surface: str,
    canonical_start: int,
) -> tuple[str, list[int], list[int], list[int]]:
    prefix, core, suffix = _split_character_surface(surface)
    core_start = canonical_start + len(prefix)

    frontend_parts: list[str] = []
    canonical_left: list[int | None] = [None] * (len(surface) + 1)
    canonical_right: list[int | None] = [None] * (len(surface) + 1)
    frontend_left: list[int] = []
    frontend_right: list[int] = []
    frontend_cursor = 0

    prefix_frontier = list(range(len(prefix) + 1))
    _append_piece(
        prefix,
        0,
        frontend_cursor,
        prefix_frontier,
        prefix_frontier,
        frontend_parts,
        canonical_left,
        canonical_right,
        frontend_left,
        frontend_right,
        frontend_to_canonical=list(range(canonical_start, canonical_start + len(prefix) + 1)),
    )
    frontend_cursor += len(prefix)

    core_frontend, core_canonical_left, core_canonical_right, core_frontend_to_canonical = (
        _project_character_core(core, core_start)
    )
    _append_piece(
        core_frontend,
        len(prefix),
        frontend_cursor,
        core_canonical_left,
        core_canonical_right,
        frontend_parts,
        canonical_left,
        canonical_right,
        frontend_left,
        frontend_right,
        frontend_to_canonical=core_frontend_to_canonical,
    )
    frontend_cursor += len(core_frontend)

    suffix_start = core_start + len(core)
    suffix_frontier = list(range(len(suffix) + 1))
    _append_piece(
        suffix,
        len(prefix) + len(core),
        frontend_cursor,
        suffix_frontier,
        suffix_frontier,
        frontend_parts,
        canonical_left,
        canonical_right,
        frontend_left,
        frontend_right,
        frontend_to_canonical=list(range(suffix_start, suffix_start + len(suffix) + 1)),
    )
    frontend_cursor += len(suffix)

    if any(value is None for value in (*canonical_left, *canonical_right)):
        raise RuntimeError("character surface projection left a canonical boundary unmapped")
    if len(frontend_left) != frontend_cursor + 1 or len(frontend_right) != frontend_cursor + 1:
        raise RuntimeError("character surface projection left a frontend boundary unmapped")
    return (
        "".join(frontend_parts),
        [value for value in canonical_left if value is not None],
        [value for value in canonical_right if value is not None],
        frontend_left,
    )


def _split_character_surface(surface: str) -> tuple[str, str, str]:
    matches = tuple(_CHARACTER.finditer(surface))
    if len(matches) < 2:
        raise InvalidRequestError("say_as characters spans must contain at least two characters")

    core_start = matches[0].start()
    core_end = matches[-1].end()
    prefix = surface[:core_start]
    core = surface[core_start:core_end]
    suffix = surface[core_end:]
    if not (_COMPACT_CHARACTERS.fullmatch(core) or _SPACED_CHARACTERS.fullmatch(core)):
        raise InvalidRequestError(
            f"say_as characters span has unsupported character payload {core!r}; "
            "use at least two ASCII letters or digits, compact or whitespace-separated; "
            "outer punctuation is allowed"
        )
    return prefix, core, suffix


def _project_character_core(
    core: str,
    canonical_start: int,
) -> tuple[str, list[int], list[int], list[int]]:
    matches = tuple(_CHARACTER.finditer(core))
    characters = [match.group() for match in matches]
    frontend = "- ".join(characters)
    canonical_left: list[int | None] = [None] * (len(core) + 1)
    canonical_right: list[int | None] = [None] * (len(core) + 1)
    frontend_to_canonical: list[int | None] = [None] * (len(frontend) + 1)
    output_cursor = 0

    for index, match in enumerate(matches):
        source_start = match.start()
        source_end = match.end()
        canonical_boundary_start = canonical_start + source_start
        canonical_boundary_end = canonical_start + source_end
        if canonical_left[source_start] is None:
            canonical_left[source_start] = output_cursor
        canonical_right[source_start] = output_cursor
        frontend_to_canonical[output_cursor] = canonical_boundary_start
        output_cursor += 1
        canonical_left[source_end] = output_cursor
        canonical_right[source_end] = output_cursor
        frontend_to_canonical[output_cursor] = canonical_boundary_end

        if index + 1 < len(matches):
            next_match = matches[index + 1]
            next_start = next_match.start()
            separator_start = output_cursor
            if source_end == next_start:
                canonical_left[source_end] = separator_start
                canonical_right[source_end] = separator_start + 2
            else:
                for boundary in range(source_end, next_start):
                    canonical_left[boundary] = separator_start
                    canonical_right[boundary] = separator_start + 2
                canonical_left[next_start] = separator_start + 2
                canonical_right[next_start] = separator_start + 2
            frontend_to_canonical[separator_start] = canonical_boundary_end
            frontend_to_canonical[separator_start + 1] = canonical_boundary_end
            frontend_to_canonical[separator_start + 2] = canonical_start + next_start
            output_cursor += 2

    canonical_left[0] = 0
    canonical_right[0] = 0
    canonical_left[-1] = len(frontend)
    canonical_right[-1] = len(frontend)
    if any(value is None for value in (*canonical_left, *canonical_right)):
        raise RuntimeError("character projection left a canonical boundary unmapped")
    if any(value is None for value in frontend_to_canonical):
        raise RuntimeError("character projection left a frontend boundary unmapped")
    return (
        frontend,
        [value for value in canonical_left if value is not None],
        [value for value in canonical_right if value is not None],
        [value for value in frontend_to_canonical if value is not None],
    )


def _validate_span(start: int, end: int, length: int, label: str) -> None:
    if isinstance(start, bool) or not isinstance(start, int):
        raise InvalidRequestError(f"{label} start must be an integer")
    if isinstance(end, bool) or not isinstance(end, int):
        raise InvalidRequestError(f"{label} end must be an integer")
    if not 0 <= start <= end <= length:
        raise InvalidRequestError(
            f"{label} span must satisfy 0 <= start <= end <= {length}, got [{start}, {end})"
        )


def _validate_bias(bias: BoundaryBias) -> None:
    if bias not in ("left", "right"):
        raise InvalidRequestError("span boundary bias must be 'left' or 'right'")


def _boundary_value(
    left_values: tuple[int, ...],
    right_values: tuple[int, ...],
    boundary: int,
    bias: BoundaryBias,
) -> int:
    return left_values[boundary] if bias == "left" else right_values[boundary]
