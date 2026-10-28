"""Dependency-light declaration of the public request API contract."""

from __future__ import annotations

from dataclasses import dataclass

REQUEST_API_VERSION = 2


@dataclass(frozen=True, slots=True)
class RequestApiContract:
    """Stable capability declaration for request-oriented integrations."""

    version: int
    request_type: str
    result_type: str
    synthesizer_type: str
    prepare_method: str
    synthesize_method: str
    supports_linguistic_tokens: bool
    supports_pronunciation_overrides: bool
    supports_whole_request_phonemes: bool
    supports_word_timings: bool
    supports_voice_level: bool
    caller_owns_text_boundaries: bool

    measurement_method: str | None = None
    voice_enrollment_method: str | None = None
    supports_request_measurement: bool = False
    supports_reference_voice: bool = False
    supports_voice_enrollment: bool = False
    supports_say_as: bool = False
    say_as_interpretations: tuple[str, ...] = ()


def request_api_contract() -> RequestApiContract:
    """Return the stable request API capabilities without initializing runtime code."""
    return RequestApiContract(
        version=REQUEST_API_VERSION,
        request_type="SynthesisRequest",
        result_type="RenderedSegment",
        synthesizer_type="KokoroSynthesizer",
        prepare_method="prepare",
        synthesize_method="synthesize",
        supports_linguistic_tokens=True,
        supports_pronunciation_overrides=True,
        supports_whole_request_phonemes=True,
        supports_word_timings=True,
        supports_voice_level=True,
        caller_owns_text_boundaries=True,
        measurement_method="measure_request",
        voice_enrollment_method="enroll_voice",
        supports_request_measurement=True,
        supports_reference_voice=True,
        supports_voice_enrollment=True,
        supports_say_as=True,
        say_as_interpretations=("characters",),
    )
