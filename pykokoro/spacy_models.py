from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpacyModelSize = Literal["sm", "md", "lg", "trf"]

_WEB_MODEL_LANGS = {"en", "zh"}
_LANGUAGE_ALIASES = {
    "cmn": "zh",
}


def normalize_spacy_language(lang: str | None) -> str:
    """Normalize a language code to a spaCy package language prefix."""
    lang_code = (lang or "en").strip().lower().replace("_", "-")
    if not lang_code:
        return "en"
    lang_code = lang_code.split("-", 1)[0]
    return _LANGUAGE_ALIASES.get(lang_code, lang_code)


@dataclass(frozen=True)
class SpacyModelRequest:
    """Validated spaCy selection settings forwarded to lower-level libraries."""

    model: str | None = None
    size: SpacyModelSize | None = None

    @property
    def mode(self) -> Literal["explicit", "size", "highest_available"]:
        if self.model is not None:
            return "explicit"
        if self.size is not None:
            return "size"
        return "highest_available"


def normalize_spacy_model(model: str | None) -> str | None:
    """Normalize the compatibility ``auto`` model sentinel to unset."""

    if model is None:
        return None
    if not isinstance(model, str):
        raise TypeError(f"spacy_model must be str | None, got {type(model)!r}")
    normalized = model.strip()
    if not normalized or normalized.lower() == "auto":
        return None
    return normalized


def validate_spacy_model_size(size: SpacyModelSize | str | None) -> SpacyModelSize | None:
    """Validate and normalize an optional exact spaCy model tier."""

    if size is None:
        return None
    if size not in {"sm", "md", "lg", "trf"}:
        raise ValueError("spacy_model_size must be one of 'sm', 'md', 'lg', or 'trf'")
    return size  # type: ignore[return-value]


def make_spacy_model_request(
    *, model: str | None = None, size: SpacyModelSize | str | None = None
) -> SpacyModelRequest:
    """Build one normalized request without discovering installed models."""

    return SpacyModelRequest(
        model=normalize_spacy_model(model),
        size=validate_spacy_model_size(size),
    )


def spacy_selection_metadata(
    *,
    language: str | None,
    request: SpacyModelRequest,
    selected_model: str | None,
    selected_size: SpacyModelSize | None = None,
) -> dict[str, object]:
    """Create JSON-friendly diagnostics for one lower-library selection."""

    return {
        "language": normalize_spacy_language(language),
        "requested_model": request.model,
        "requested_size": request.size,
        "selected_model": selected_model,
        "selected_model_size": selected_size,
        "selection_mode": request.mode,
    }


def resolve_spacy_model(lang: str | None, *, size: SpacyModelSize) -> str:
    """Construct a spaCy package name for an explicit compatibility tier.

    Examples:
        - en-us + md -> en_core_web_md
        - de + sm -> de_core_news_sm
        - zh-cn + lg -> zh_core_web_lg
    """
    validate_spacy_model_size(size)
    lang_code = normalize_spacy_language(lang)
    package_type = "web" if lang_code in _WEB_MODEL_LANGS else "news"
    return f"{lang_code}_core_{package_type}_{size}"


def resolve_configured_spacy_model(
    *,
    spacy_model: str | None,
    lang: str | None,
    size: SpacyModelSize | None = None,
) -> str | None:
    """Return an explicit package or exact-size package for compatibility.

    Normal execution should pass ``model`` and ``size`` directly to phrasplit or
    kokorog2p. This helper remains for callers that require a package string.
    Unset settings intentionally return ``None`` rather than selecting medium.
    """

    request = make_spacy_model_request(model=spacy_model, size=size)
    if request.model is not None:
        return request.model
    if request.size is not None:
        return resolve_spacy_model(lang, size=request.size)
    return None
