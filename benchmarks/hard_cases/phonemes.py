from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .normalization import NormalizationResult, compare_spoken_text, from_pipeline_result
from .schema import HardCase

_PRESENTATION = frozenset(".,!?;:\"'“”‘’()[]{}-–—…")


@dataclass(frozen=True, slots=True)
class PhonemeObservation:
    phonemes: str
    tokens: tuple[int, ...]
    segment_count: int
    warnings: tuple[str, ...] = ()
    segments: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    locale: str | None
    language: str
    category: str
    level: str
    normalization: NormalizationResult | None
    observation: PhonemeObservation | None
    expected_observation: PhonemeObservation | None
    spoken_text_pass: bool | None
    raw_phoneme_exact: bool | None
    semantic_phoneme_exact: bool | None
    token_exact: bool | None
    phoneme_edit_distance: int
    token_edit_distance: int
    critical_pronunciation_pass: bool | None
    likely_owner: str
    error: str | None = None
    expectation_errors: tuple[str, ...] = ()
    quarantined: bool = False
    quarantine_reason: str | None = None

    @property
    def failed(self) -> bool:
        return (
            bool(
                self.error
                or any(
                    item is False
                    for item in (
                        self.spoken_text_pass,
                        self.semantic_phoneme_exact,
                        self.token_exact,
                        self.critical_pronunciation_pass,
                    )
                )
            )
            and not self.quarantined
        )

    @property
    def passed(self) -> bool:
        return not self.failed and not self.error

    @property
    def warning_count(self) -> int:
        return len(self.observation.warnings) if self.observation else 0


def semantic_phoneme_key(value: str) -> str:
    value = unicodedata.normalize("NFC", value)
    return "".join(char for char in value if char not in _PRESENTATION and not char.isspace())


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, 1):
        current = [row]
        for col, right_item in enumerate(right, 1):
            current.append(
                min(
                    previous[col] + 1,
                    current[col - 1] + 1,
                    previous[col - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


def observation_from_result(result: Any) -> PhonemeObservation:
    segments = tuple(result.phoneme_segments)
    return PhonemeObservation(
        phonemes=" ".join(segment.phonemes for segment in segments if segment.phonemes).strip(),
        tokens=tuple(token for segment in segments for token in segment.tokens),
        segment_count=len(segments),
        warnings=tuple(getattr(result.trace, "warnings", ())),
        segments=segments,
    )


def evaluate_case(
    case: HardCase,
    frontend: Any,
    *,
    level: str = "frontend",
    quarantine: dict[str, str] | None = None,
    expected_frontend: Any | None = None,
) -> CaseEvaluation:
    normalization = None
    observation = None
    expected_observation = None
    errors: list[str] = []
    try:
        result = frontend.run(case.text)
        normalization = from_pipeline_result(result)
        observation = observation_from_result(result)
    except Exception as exc:  # benchmark output must identify, not hide, failures
        errors.append(f"{type(exc).__name__}: {exc}")

    expected_text = case.expect.spoken_text
    alternatives = case.expect.spoken_text_alternatives
    spoken_pass = compare_spoken_text(
        normalization.spoken_text if normalization else "", expected_text, alternatives
    )
    if expected_text is not None and expected_frontend is None:
        expected_frontend = frontend
    if observation is not None and (expected_text is not None or alternatives):
        target = expected_text or alternatives[0]
        try:
            expected_observation = observation_from_result(expected_frontend.run(target))
        except Exception as exc:
            errors.append(f"expected {type(exc).__name__}: {exc}")
    if expected_observation is None and observation is not None:
        expected_observation = observation

    raw_exact = semantic_exact = token_exact = None
    phoneme_edit = token_edit = 0
    if observation is not None and expected_observation is not None:
        raw_exact = observation.phonemes == expected_observation.phonemes
        semantic_exact = semantic_phoneme_key(observation.phonemes) == semantic_phoneme_key(
            expected_observation.phonemes
        )
        token_exact = observation.tokens == expected_observation.tokens
        phoneme_edit = edit_distance(
            tuple(observation.phonemes), tuple(expected_observation.phonemes)
        )
        token_edit = edit_distance(observation.tokens, expected_observation.tokens)

    critical_pass = _critical_match(case, observation)
    owner = _owner(errors, spoken_pass, semantic_exact, token_exact, critical_pass)
    expectation_errors = tuple(errors)
    if spoken_pass is False:
        expectation_errors += ("prepared spoken text differs from the accepted expectation",)
    quarantined = case.id in (quarantine or {})
    return CaseEvaluation(
        case.id,
        case.locale,
        case.language,
        case.category,
        level,
        normalization,
        observation,
        expected_observation,
        spoken_pass,
        raw_exact,
        semantic_exact,
        token_exact,
        phoneme_edit,
        token_edit,
        critical_pass,
        owner,
        "; ".join(errors) or None,
        expectation_errors,
        quarantined,
        (quarantine or {}).get(case.id),
    )


def _critical_match(case: HardCase, observation: PhonemeObservation | None) -> bool | None:
    if not case.expect.critical_pronunciations:
        return None
    if observation is None:
        return False
    for item in case.expect.critical_pronunciations:
        candidates = [
            segment.phonemes
            for segment in observation.segments
            if item.source.lower() in segment.text.lower()
        ]
        if not candidates or not any(
            accepted == candidate
            or semantic_phoneme_key(accepted) == semantic_phoneme_key(candidate)
            for candidate in candidates
            for accepted in item.accepted
        ):
            return False
    return True


def _owner(
    errors: list[str],
    spoken: bool | None,
    semantic: bool | None,
    tokens: bool | None,
    critical: bool | None,
) -> str:
    if errors:
        return "pykokoro_pipeline"
    if spoken is False:
        return "spokenform"
    if critical is False or semantic is False or tokens is False:
        return "kokorog2p_or_spokenform"
    return "none"


__all__ = [
    "CaseEvaluation",
    "PhonemeObservation",
    "edit_distance",
    "evaluate_case",
    "observation_from_result",
    "semantic_phoneme_key",
]
