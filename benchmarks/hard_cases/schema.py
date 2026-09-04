from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
SUPPORTED_LOCALES = ("en-US", "en-GB", "de-DE")
LOCALE_LANGUAGE = {"en-US": "en", "en-GB": "en", "de-DE": "de"}
SUPPORTED_LANGUAGES = ("en", "de")
CATEGORIES = (
    "normalization",
    "abbreviations",
    "numbers_dates_units",
    "acronyms_initialisms",
    "names_foreign_words",
    "homographs",
    "heteronyms",
    "morphology",
    "compounds",
    "prefix_stress",
    "denglisch",
    "codeswitch",
    "punctuation_prosody",
    "questions_focus",
    "dirty_text",
    "ssmd",
    "long_form",
)
_CASE_ID = re.compile(r"^[a-z][a-z0-9_]+$")


class HardCaseError(ValueError):
    """Raised when a hard-case row violates the benchmark schema."""


@dataclass(frozen=True, slots=True)
class Provenance:
    kind: str = "pykokoro"
    source: str = "first-party"
    license: str = "project-license"


@dataclass(frozen=True, slots=True)
class CriticalPronunciation:
    source: str
    sense: str | None = None
    accepted: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CriticalPronunciation:
        source = _required_str(value, "source")
        sense = _optional_str(value.get("sense"), "sense")
        accepted = _string_tuple(value.get("accepted", ()), "accepted")
        if not accepted:
            raise HardCaseError("critical pronunciation accepted must not be empty")
        return cls(source, sense, accepted)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "sense": self.sense, "accepted": list(self.accepted)}


@dataclass(frozen=True, slots=True)
class SegmentExpectation:
    text: str | None = None
    char_start: int | None = None
    char_end: int | None = None
    pause_before: float | None = None
    pause_after: float | None = None
    sentence_boundary: bool | None = None
    clause_boundary: bool | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SegmentExpectation:
        return cls(
            _optional_str(value.get("text"), "text"),
            _optional_int(value.get("char_start"), "char_start"),
            _optional_int(value.get("char_end"), "char_end"),
            _optional_float(value.get("pause_before"), "pause_before"),
            _optional_float(value.get("pause_after"), "pause_after"),
            _optional_bool(value.get("sentence_boundary"), "sentence_boundary"),
            _optional_bool(value.get("clause_boundary"), "clause_boundary"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "text": self.text,
                "char_start": self.char_start,
                "char_end": self.char_end,
                "pause_before": self.pause_before,
                "pause_after": self.pause_after,
                "sentence_boundary": self.sentence_boundary,
                "clause_boundary": self.clause_boundary,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class PauseExpectation:
    position: int | None = None
    min_duration_s: float | None = None
    max_duration_s: float | None = None
    strength: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PauseExpectation:
        result = cls(
            _optional_int(value.get("position"), "position"),
            _optional_float(value.get("min_duration_s"), "min_duration_s"),
            _optional_float(value.get("max_duration_s"), "max_duration_s"),
            _optional_str(value.get("strength"), "strength"),
        )
        if (
            result.min_duration_s is not None
            and result.max_duration_s is not None
            and result.min_duration_s > result.max_duration_s
        ):
            raise HardCaseError("pause min_duration_s cannot exceed max_duration_s")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "position": self.position,
                "min_duration_s": self.min_duration_s,
                "max_duration_s": self.max_duration_s,
                "strength": self.strength,
            }.items()
            if value is not None
        }


@dataclass(frozen=True, slots=True)
class AcousticConstraints:
    must_be_finite: bool = True
    min_duration_s: float | None = None
    max_duration_s: float | None = None
    max_peak: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AcousticConstraints:
        result = cls(
            _optional_bool(value.get("must_be_finite"), "must_be_finite")
            if "must_be_finite" in value
            else True,
            _optional_float(value.get("min_duration_s"), "min_duration_s"),
            _optional_float(value.get("max_duration_s"), "max_duration_s"),
            _optional_float(value.get("max_peak"), "max_peak"),
        )
        if (
            result.min_duration_s is not None
            and result.max_duration_s is not None
            and result.min_duration_s > result.max_duration_s
        ):
            raise HardCaseError("acoustic min_duration_s cannot exceed max_duration_s")
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "must_be_finite": self.must_be_finite,
            **({"min_duration_s": self.min_duration_s} if self.min_duration_s is not None else {}),
            **({"max_duration_s": self.max_duration_s} if self.max_duration_s is not None else {}),
            **({"max_peak": self.max_peak} if self.max_peak is not None else {}),
        }


@dataclass(frozen=True, slots=True)
class Expectations:
    spoken_text: str | None = None
    spoken_text_alternatives: tuple[str, ...] = ()
    equivalent_to: str | None = None
    critical_pronunciations: tuple[CriticalPronunciation, ...] = ()
    segment_expectations: tuple[SegmentExpectation, ...] = ()
    pause_expectations: tuple[PauseExpectation, ...] = ()
    acoustic_constraints: AcousticConstraints = field(default_factory=AcousticConstraints)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Expectations:
        spoken = _optional_str(value.get("spoken_text"), "spoken_text")
        alternatives = _string_tuple(
            value.get("spoken_text_alternatives", ()), "spoken_text_alternatives"
        )
        critical = tuple(
            CriticalPronunciation.from_dict(item)
            for item in _mapping_tuple(
                value.get("critical_pronunciations", ()), "critical_pronunciations"
            )
        )
        segments = tuple(
            SegmentExpectation.from_dict(item)
            for item in _mapping_tuple(
                value.get("segment_expectations", ()), "segment_expectations"
            )
        )
        pauses = tuple(
            PauseExpectation.from_dict(item)
            for item in _mapping_tuple(value.get("pause_expectations", ()), "pause_expectations")
        )
        constraints = value.get("acoustic_constraints", {})
        if not isinstance(constraints, Mapping):
            raise HardCaseError("acoustic_constraints must be an object")
        return cls(
            spoken,
            alternatives,
            _optional_str(value.get("equivalent_to"), "equivalent_to"),
            critical,
            segments,
            pauses,
            AcousticConstraints.from_dict(constraints),
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"acoustic_constraints": self.acoustic_constraints.to_dict()}
        if self.spoken_text is not None:
            result["spoken_text"] = self.spoken_text
        if self.spoken_text_alternatives:
            result["spoken_text_alternatives"] = list(self.spoken_text_alternatives)
        if self.equivalent_to is not None:
            result["equivalent_to"] = self.equivalent_to
        if self.critical_pronunciations:
            result["critical_pronunciations"] = [
                item.to_dict() for item in self.critical_pronunciations
            ]
        if self.segment_expectations:
            result["segment_expectations"] = [item.to_dict() for item in self.segment_expectations]
        if self.pause_expectations:
            result["pause_expectations"] = [item.to_dict() for item in self.pause_expectations]
        return result


@dataclass(frozen=True, slots=True)
class HardCase:
    schema_version: int
    id: str
    language: str
    locale: str | None
    category: str
    subcategory: str | None
    text: str
    provenance: Provenance
    tags: tuple[str, ...]
    expect: Expectations
    evaluation: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HardCase:
        forbidden = {
            "human_reference",
            "reference_audio_dir",
            "reference_audio",
            "human_speaker_id",
        }
        present = forbidden.intersection(value)
        if present:
            raise HardCaseError(f"human-reference fields are not supported: {sorted(present)}")
        version = value.get("schema_version")
        if version != SCHEMA_VERSION:
            raise HardCaseError(f"unsupported schema_version: {version!r}")
        case_id = _required_str(value, "id")
        if not _CASE_ID.fullmatch(case_id):
            raise HardCaseError(f"invalid case id: {case_id!r}")
        language = _required_str(value, "language")
        if language not in SUPPORTED_LANGUAGES:
            raise HardCaseError(f"unsupported language: {language!r}")
        locale = value.get("locale")
        if locale is not None:
            locale = _required_str(value, "locale")
            if locale not in SUPPORTED_LOCALES:
                raise HardCaseError(f"unsupported locale: {locale!r}")
            if LOCALE_LANGUAGE[locale] != language:
                raise HardCaseError(f"locale {locale!r} does not belong to language {language!r}")
        category = _required_str(value, "category")
        if category not in CATEGORIES:
            raise HardCaseError(f"unsupported category: {category!r}")
        text = _required_str(value, "text")
        if not text.strip():
            raise HardCaseError("text must not be empty")
        provenance_value = value.get("provenance", {})
        if not isinstance(provenance_value, Mapping):
            raise HardCaseError("provenance must be an object")
        provenance = Provenance(
            _optional_str(provenance_value.get("kind"), "provenance.kind") or "pykokoro",
            _optional_str(provenance_value.get("source"), "provenance.source") or "first-party",
            _optional_str(provenance_value.get("license"), "provenance.license")
            or "project-license",
        )
        tags = _string_tuple(value.get("tags", ()), "tags")
        evaluation = _string_tuple(value.get("evaluation", ()), "evaluation")
        expect_value = value.get("expect", {})
        if not isinstance(expect_value, Mapping):
            raise HardCaseError("expect must be an object")
        return cls(
            schema_version=SCHEMA_VERSION,
            id=case_id,
            language=language,
            locale=locale,
            category=category,
            subcategory=_optional_str(value.get("subcategory"), "subcategory"),
            text=text,
            provenance=provenance,
            tags=tags,
            expect=Expectations.from_dict(expect_value),
            evaluation=evaluation,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "id": self.id,
            "language": self.language,
            "locale": self.locale,
            "category": self.category,
            "subcategory": self.subcategory,
            "text": self.text,
            "provenance": {
                "kind": self.provenance.kind,
                "source": self.provenance.source,
                "license": self.provenance.license,
            },
            "evaluation": list(self.evaluation),
            "tags": list(self.tags),
            "expect": self.expect.to_dict(),
        }


def parse_case(value: Mapping[str, Any], *, source: str = "case") -> HardCase:
    try:
        return HardCase.from_dict(value)
    except HardCaseError as exc:
        raise HardCaseError(f"{source}: {exc}") from exc


def load_jsonl(path: str | Path) -> list[HardCase]:
    path = Path(path)
    cases: list[HardCase] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise HardCaseError(f"{path}:{line_number}: invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise HardCaseError(f"{path}:{line_number}: expected an object")
        case = parse_case(value, source=f"{path}:{line_number}")
        if case.id in seen:
            raise HardCaseError(f"{path}:{line_number}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    return cases


def validate_cases(cases: Iterable[HardCase]) -> tuple[HardCase, ...]:
    result = tuple(cases)
    ids = [case.id for case in result]
    if len(ids) != len(set(ids)):
        raise HardCaseError("case IDs must be unique")
    return result


def _required_str(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise HardCaseError(f"{key} must be a non-empty string")
    return item


def _optional_str(value: Any, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HardCaseError(f"{key} must be a string or null")
    return value


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise HardCaseError(f"{key} must be an integer or null")
    return value


def _optional_float(value: Any, key: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise HardCaseError(f"{key} must be a finite number or null")
    if value < 0 and key.endswith("duration_s"):
        raise HardCaseError(f"{key} must not be negative")
    return float(value)


def _optional_bool(value: Any, key: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise HardCaseError(f"{key} must be a boolean or null")
    return value


def _string_tuple(value: Any, key: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise HardCaseError(f"{key} must be an array")
    if not all(isinstance(item, str) and item for item in value):
        raise HardCaseError(f"{key} must contain non-empty strings")
    return tuple(value)


def _mapping_tuple(value: Any, key: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, (list, tuple)):
        raise HardCaseError(f"{key} must be an array")
    if not all(isinstance(item, Mapping) for item in value):
        raise HardCaseError(f"{key} must contain objects")
    return tuple(value)


__all__ = [
    "AcousticConstraints",
    "CATEGORIES",
    "CriticalPronunciation",
    "Expectations",
    "HardCase",
    "HardCaseError",
    "LOCALE_LANGUAGE",
    "PauseExpectation",
    "Provenance",
    "SCHEMA_VERSION",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_LOCALES",
    "SegmentExpectation",
    "load_jsonl",
    "parse_case",
    "validate_cases",
]
