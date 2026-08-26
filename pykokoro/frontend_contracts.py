"""Explicit frontend contracts for release profiles.

The fallback phonemes are diagnostic espeak outputs only. They are not golden
pronunciation evidence for profiles whose training frontend is external.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config_types import ModelVariant
from .model_profiles import get_model_profile


@dataclass(frozen=True, slots=True)
class FrontendFixture:
    variant: ModelVariant
    language: str
    text: str
    diagnostic_backend: str
    diagnostic_phonemes: str
    release_ready: bool = False


FRONTEND_FIXTURES: dict[ModelVariant, FrontendFixture] = {
    "vi-contextbox": FrontendFixture("vi-contextbox", "vi", "Xin chào", "espeak", "sˈi1n tʃˈaː2w"),
    "vi-anphunl": FrontendFixture("vi-anphunl", "vi", "Xin chào", "espeak", "sˈi1n tʃˈaː2w"),
    "ar-nabra": FrontendFixture("ar-nabra", "ar", "مَرْحَبًا", "espeak", "mˈarħabˌan"),
    "de-crane": FrontendFixture("de-crane", "de", "Hallo", "espeak", "hˈaloː"),
    "he-hebrew-nc": FrontendFixture("he-hebrew-nc", "he", "שלום", "espeak", "ʃalˈom"),
}


def require_frontend(variant: ModelVariant, *, allow_experimental: bool) -> str:
    """Return the required frontend, rejecting unsupported silent fallbacks."""
    profile = get_model_profile(variant, "github")
    if profile.frontend_experimental and not allow_experimental:
        raise ValueError(
            f"Model profile {variant!r} requires {profile.frontend}; "
            "pass allow_experimental_frontend=True only for an experimental smoke test."
        )
    return profile.frontend
