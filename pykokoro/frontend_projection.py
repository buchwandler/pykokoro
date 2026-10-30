"""Kokoro-only projection from canonical request text to frontend text."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

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

    spans: list[SayAsSpan] = []
    previous: SayAsSpan | None = None
    for index, candidate in enumerate(say_as):
        if not isinstance(candidate, SayAsSpan):
            raise InvalidRequestError("say_as must contain SayAsSpan values")
        span = candidate
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

    builder = _ProjectionBuilder(canonical_text)
    for index, span in enumerate(spans):
        builder.copy_until(span.start)
        surface = canonical_text[span.start : span.end]
        prefix, core, _ = _split_character_surface(surface)
        core_start = span.start + len(prefix)
        builder.copy_until(core_start)

        if _SPACED_CHARACTERS.fullmatch(core):
            builder.copy_until(core_start + len(core))
        else:
            characters = tuple(_CHARACTER.finditer(core))
            for character_index, match in enumerate(characters):
                builder.copy_until(core_start + match.end())
                if character_index + 1 < len(characters):
                    builder.insert(" ")

        builder.copy_until(span.end)
        next_span_start = spans[index + 1].start if index + 1 < len(spans) else len(canonical_text)
        separator_end = _detached_possessive_separator_end(
            canonical_text, span.end, next_span_start
        )
        if separator_end is not None:
            builder.delete_until(separator_end)

    builder.copy_until(len(canonical_text))
    return builder.finish()


class _ProjectionBuilder:
    """Build projection text and both boundary maps from source-aligned edits."""

    def __init__(self, canonical_text: str) -> None:
        self._canonical_text = canonical_text
        self._canonical_cursor = 0
        self._frontend_parts: list[str] = []
        self._canonical_left: list[int | None] = [None] * (len(canonical_text) + 1)
        self._canonical_right: list[int | None] = [None] * (len(canonical_text) + 1)
        self._frontend_left = [0]
        self._frontend_right = [0]
        self._set_canonical_boundary(0, 0, 0)

    def copy_until(self, end: int) -> None:
        if not self._canonical_cursor <= end <= len(self._canonical_text):
            raise RuntimeError("projection copy range is not monotonic")
        while self._canonical_cursor < end:
            self._copy_character()

    def insert(self, text: str) -> None:
        if not text:
            return
        frontend_start = len(self._frontend_left) - 1
        frontend_end = frontend_start + len(text)
        self._set_canonical_boundary(self._canonical_cursor, frontend_start, frontend_end)
        self._frontend_parts.append(text)
        for _ in text:
            self._frontend_left.append(self._canonical_cursor)
            self._frontend_right.append(self._canonical_cursor)

    def delete_until(self, end: int) -> None:
        if not self._canonical_cursor <= end <= len(self._canonical_text):
            raise RuntimeError("projection deletion range is not monotonic")
        if end == self._canonical_cursor:
            return
        frontend_boundary = len(self._frontend_left) - 1
        start = self._canonical_cursor
        for boundary in range(start, end + 1):
            self._set_canonical_boundary(boundary, frontend_boundary, frontend_boundary)
        self._frontend_left[frontend_boundary] = min(self._frontend_left[frontend_boundary], start)
        self._frontend_right[frontend_boundary] = max(self._frontend_right[frontend_boundary], end)
        self._canonical_cursor = end

    def finish(self) -> FrontendTextProjection:
        if self._canonical_cursor != len(self._canonical_text):
            raise RuntimeError("projection builder did not consume all canonical text")
        if any(value is None for value in (*self._canonical_left, *self._canonical_right)):
            raise RuntimeError("canonical-to-frontend projection left an unmapped boundary")
        frontend_text = "".join(self._frontend_parts)
        if (
            len(self._frontend_left) != len(frontend_text) + 1
            or len(self._frontend_right) != len(frontend_text) + 1
        ):
            raise RuntimeError("frontend-to-canonical projection left an unmapped boundary")
        mappings = tuple(
            tuple(value for value in mapping if value is not None)
            for mapping in (
                self._canonical_left,
                self._canonical_right,
                self._frontend_left,
                self._frontend_right,
            )
        )
        if any(mapping != tuple(sorted(mapping)) for mapping in mappings):
            raise RuntimeError("projection boundary mappings are not monotonic")
        return FrontendTextProjection(
            canonical_text=self._canonical_text,
            frontend_text=frontend_text,
            _canonical_left=tuple(value for value in self._canonical_left if value is not None),
            _canonical_right=tuple(value for value in self._canonical_right if value is not None),
            _frontend_left=tuple(self._frontend_left),
            _frontend_right=tuple(self._frontend_right),
        )

    def _copy_character(self) -> None:
        canonical_index = self._canonical_cursor
        frontend_boundary = len(self._frontend_left) - 1
        self._set_canonical_boundary(canonical_index, frontend_boundary, frontend_boundary)
        self._frontend_left[frontend_boundary] = min(
            self._frontend_left[frontend_boundary], canonical_index
        )
        self._frontend_right[frontend_boundary] = max(
            self._frontend_right[frontend_boundary], canonical_index
        )
        self._frontend_parts.append(self._canonical_text[canonical_index])
        self._canonical_cursor += 1
        next_frontend_boundary = frontend_boundary + 1
        self._frontend_left.append(self._canonical_cursor)
        self._frontend_right.append(self._canonical_cursor)
        self._set_canonical_boundary(
            self._canonical_cursor, next_frontend_boundary, next_frontend_boundary
        )

    def _set_canonical_boundary(self, boundary: int, left: int, right: int) -> None:
        existing_left = self._canonical_left[boundary]
        existing_right = self._canonical_right[boundary]
        self._canonical_left[boundary] = left if existing_left is None else min(existing_left, left)
        self._canonical_right[boundary] = (
            right if existing_right is None else max(existing_right, right)
        )


def _detached_possessive_separator_end(text: str, start: int, limit: int) -> int | None:
    """Return the start of an adjacent possessive suffix's horizontal separator."""
    cursor = start
    while cursor < limit and (text[cursor] == "\t" or unicodedata.category(text[cursor]) == "Zs"):
        cursor += 1
    if cursor == start or cursor + 2 > limit or text[cursor : cursor + 2] not in ("'s", "’s"):
        return None
    suffix_end = cursor + 2
    if suffix_end < len(text) and re.match(r"\w", text[suffix_end]) is not None:
        return None
    return cursor


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
