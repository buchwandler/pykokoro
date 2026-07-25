"""Mixed-language phonemization support for pykokoro.

This module handles automatic language detection and mixed-language text-to-phoneme
conversion using kokorog2p's preprocess_multilang capability.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import kokorog2p as _kokorog2p
from kokorog2p.multilang import preprocess_multilang

from .constants import SUPPORTED_LANGUAGES

if TYPE_CHECKING:
    from kokorog2p.types import OverrideSpan

    from .tokenizer import TokenizerConfig

logger = logging.getLogger(__name__)
ANNOTATION_REGEX = getattr(
    _kokorog2p, "ANNOTATION_REGEX", re.compile(r"\[[^\]]+\]\{[^}]+\}")
)


class MixedLanguageHandler:
    """Handles mixed-language G2P configuration and preprocessing.

    This class manages:
    - Mixed-language configuration validation
    - Text preprocessing with language detection
    """

    def __init__(self, config: TokenizerConfig, kokorog2p_model: str | None = None):
        """Initialize mixed-language handler.

        Args:
            config: TokenizerConfig instance with mixed-language settings
            kokorog2p_model: Optional kokorog2p model version (e.g., 'v0.1', 'v1.0')
        """
        self.config = config
        self._kokorog2p_model = kokorog2p_model

    def validate_config(self) -> None:
        """Validate mixed-language configuration.

        Raises:
            ValueError: If mixed-language is enabled but configuration is invalid
        """
        if not self.config.use_mixed_language:
            return

        # Require allowed_languages to be explicitly set
        if not self.config.mixed_language_allowed:
            raise ValueError(
                "use_mixed_language is enabled but mixed_language_allowed is not set. "
                "You must explicitly specify which languages to detect, e.g., "
                "mixed_language_allowed=['de', 'en-us', 'fr']"
            )

        # Validate all allowed languages are supported
        for lang in self.config.mixed_language_allowed:
            # Map to kokorog2p format for validation
            kokorog2p_lang = SUPPORTED_LANGUAGES.get(lang, lang)
            if kokorog2p_lang not in SUPPORTED_LANGUAGES.values():
                supported = sorted(set(SUPPORTED_LANGUAGES.keys()))
                raise ValueError(
                    f"Language '{lang}' in mixed_language_allowed is not supported. "
                    f"Supported languages: {supported}"
                )

        # Validate primary language if set
        if self.config.mixed_language_primary:
            primary = self.config.mixed_language_primary
            kokorog2p_primary = SUPPORTED_LANGUAGES.get(primary, primary)
            if kokorog2p_primary not in SUPPORTED_LANGUAGES.values():
                supported = sorted(set(SUPPORTED_LANGUAGES.keys()))
                raise ValueError(
                    f"Primary language '{primary}' is not supported. "
                    f"Supported languages: {supported}"
                )

            # Primary MUST be in allowed languages
            if primary not in self.config.mixed_language_allowed:
                raise ValueError(
                    f"Primary language '{primary}' must be in allowed_languages. "
                    f"Got primary='{primary}' but "
                    f"allowed={self.config.mixed_language_allowed}"
                )

        # Validate confidence threshold
        if not 0.0 <= self.config.mixed_language_confidence <= 1.0:
            raise ValueError(
                f"mixed_language_confidence must be between 0.0 and 1.0, "
                f"got {self.config.mixed_language_confidence}"
            )

    def preprocess_text(self, text: str, default_language: str) -> str:
        """Preprocess text for mixed-language phonemization.

        Uses kokorog2p's preprocess_multilang to add language annotations.
        Respects existing annotations and only tags unannotated spans.

        Args:
            text: Input text to preprocess
            default_language: Default language for unannotated words

        Returns:
            Text with language annotations in SSMD format

        Raises:
            ValueError: If mixed-language config is invalid
        """
        if not self.config.use_mixed_language:
            return text

        # Validate configuration first
        self.validate_config()

        # Map primary language to kokorog2p format
        primary_lang = self.config.mixed_language_primary or default_language
        kokorog2p_primary = SUPPORTED_LANGUAGES.get(primary_lang, primary_lang)

        # Map allowed languages to kokorog2p format
        mixed_allowed = self.config.mixed_language_allowed or []
        allowed_langs = [
            SUPPORTED_LANGUAGES.get(lang_code, lang_code) for lang_code in mixed_allowed
        ]

        try:
            if ANNOTATION_REGEX.search(text):
                return self._preprocess_unannotated(
                    text,
                    kokorog2p_primary,
                    allowed_langs,
                )
            return self._run_preprocess_multilang(
                text, kokorog2p_primary, allowed_langs
            )
        except ImportError:
            logger.warning(
                "Mixed-language mode requested but lingua-language-detector is "
                "not available. Install lingua-language-detector to enable detection."
            )
            return text

    def _run_preprocess_multilang(
        self,
        text: str,
        kokorog2p_primary: str,
        allowed_langs: list[str],
    ) -> str:
        if not text.strip():
            return text
        preprocess_func = cast(Any, preprocess_multilang)
        supports_markdown = False
        try:
            import inspect

            params = inspect.signature(preprocess_multilang).parameters
            if "markdown_syntax" in params:
                supports_markdown = True
        except (TypeError, ValueError):
            pass

        kwargs: dict[str, Any] = {
            "text": text,
            "default_language": kokorog2p_primary,
            "allowed_languages": allowed_langs,
            "confidence_threshold": self.config.mixed_language_confidence,
        }
        if supports_markdown:
            kwargs["markdown_syntax"] = "ssmd"

        overrides = preprocess_func(**kwargs)

        typed_overrides = cast("str | list[OverrideSpan]", overrides)
        if isinstance(typed_overrides, str):
            return typed_overrides
        return self._apply_overrides(text, typed_overrides)

    def _preprocess_unannotated(
        self,
        text: str,
        kokorog2p_primary: str,
        allowed_langs: list[str],
    ) -> str:
        out: list[str] = []
        cursor = 0
        processed_any = False

        for match in ANNOTATION_REGEX.finditer(text):
            start, end = match.span()
            if start > cursor:
                chunk = text[cursor:start]
                processed_chunk = self._run_preprocess_multilang(
                    chunk,
                    kokorog2p_primary,
                    allowed_langs,
                )
                if processed_chunk != chunk:
                    processed_any = True
                out.append(processed_chunk)
            out.append(match.group(0))
            cursor = end

        if cursor < len(text):
            chunk = text[cursor:]
            processed_chunk = self._run_preprocess_multilang(
                chunk,
                kokorog2p_primary,
                allowed_langs,
            )
            if processed_chunk != chunk:
                processed_any = True
            out.append(processed_chunk)

        if not processed_any:
            return text
        return "".join(out)

    def _apply_overrides(self, text: str, overrides: Sequence[object]) -> str:
        if not overrides:
            return text

        try:
            from kokorog2p.types import OverrideSpan
        except ImportError:
            return text

        spans = [s for s in overrides if isinstance(s, OverrideSpan)]
        spans.sort(key=lambda s: s.char_start)
        if not spans:
            return text

        out: list[str] = []
        cursor = 0
        for span in spans:
            if span.char_start < cursor:
                continue
            out.append(text[cursor : span.char_start])
            chunk = text[span.char_start : span.char_end]
            lang = span.attrs.get("lang")
            if lang:
                out.append(f'[{chunk}]{{lang="{lang}"}}')
            else:
                out.append(chunk)
            cursor = span.char_end
        out.append(text[cursor:])
        return "".join(out)
