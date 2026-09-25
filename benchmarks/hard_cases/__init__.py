"""Deterministic first-party hard-case benchmark for PyKokoro."""

from .data import available_languages, available_locales, case_counts, load_all_cases, load_cases
from .frontend import FrontendResult, FrontendVariant, NoOnnxFrontend, PyKokoroFrontend
from .phonemes import CaseEvaluation, PhonemeObservation, evaluate_case, semantic_phoneme_key
from .schema import (
    CATEGORIES,
    LOCALE_LANGUAGE,
    SCHEMA_VERSION,
    SUPPORTED_LANGUAGES,
    SUPPORTED_LOCALES,
    AcousticConstraints,
    CriticalPronunciation,
    Expectations,
    HardCase,
    HardCaseError,
    PauseExpectation,
    Provenance,
    SegmentExpectation,
)

__all__ = [
    "AcousticConstraints",
    "CATEGORIES",
    "CaseEvaluation",
    "CriticalPronunciation",
    "Expectations",
    "FrontendResult",
    "FrontendVariant",
    "HardCase",
    "HardCaseError",
    "LOCALE_LANGUAGE",
    "NoOnnxFrontend",
    "PauseExpectation",
    "PhonemeObservation",
    "Provenance",
    "PyKokoroFrontend",
    "SCHEMA_VERSION",
    "SegmentExpectation",
    "SUPPORTED_LANGUAGES",
    "SUPPORTED_LOCALES",
    "available_languages",
    "available_locales",
    "case_counts",
    "evaluate_case",
    "load_all_cases",
    "load_cases",
    "semantic_phoneme_key",
]
