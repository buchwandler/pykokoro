from __future__ import annotations

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


def resolve_spacy_model(lang: str | None, *, size: SpacyModelSize = "md") -> str:
    """Resolve a spaCy package name from language and size.

    Examples:
        - en-us + md -> en_core_web_md
        - de + sm -> de_core_news_sm
        - zh-cn + lg -> zh_core_web_lg
    """
    lang_code = normalize_spacy_language(lang)
    package_type = "web" if lang_code in _WEB_MODEL_LANGS else "news"
    return f"{lang_code}_core_{package_type}_{size}"


def resolve_configured_spacy_model(
    *,
    spacy_model: str,
    lang: str | None,
    size: SpacyModelSize = "md",
) -> str:
    """Resolve configured spaCy model, supporting the "auto" sentinel."""
    configured = spacy_model.strip()
    if configured and configured.lower() != "auto":
        return configured
    return resolve_spacy_model(lang, size=size)
