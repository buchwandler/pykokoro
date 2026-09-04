from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Deterministic Level 1 output and its provenance."""

    source_text: str
    spoken_text: str
    replacements: tuple[dict[str, Any], ...] = ()
    warnings: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    offset_map: dict[str, Any] | None = None

    @property
    def changed(self) -> bool:
        return self.source_text != self.spoken_text


def from_pipeline_result(result: Any) -> NormalizationResult:
    preparation = getattr(result, "document_metadata", {}).get("text_preparation", {})
    source = result.source_text if result.source_text is not None else result.clean_text
    return NormalizationResult(
        source_text=source,
        spoken_text=result.clean_text,
        replacements=tuple(preparation.get("replacements", ()))
        if isinstance(preparation.get("replacements", ()), (list, tuple))
        else (),
        warnings=tuple(getattr(result.trace, "warnings", ())),
        languages=tuple(preparation.get("languages", ())),
        offset_map=preparation.get("offset_map"),
    )


def compare_spoken_text(
    actual: str, expected: str | None, alternatives: tuple[str, ...] = ()
) -> bool | None:
    """Compare a prepared string when a case supplies an explicit contract."""
    if expected is None:
        return None
    return actual in (expected, *alternatives)


__all__ = ["NormalizationResult", "compare_spoken_text", "from_pipeline_result"]
