"""Request-centric prepared-text integration with KokoroG2P."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from kokorog2p import OverrideSpan, TokenAnnotation

from .constants import SUPPORTED_LANGUAGES
from .frontend_contracts import require_frontend
from .lexicon_data import create_g2p_with_lexphon_retry
from .model_profiles import get_model_profile
from .synthesis_config import SynthesisConfig
from .synthesis_types import LinguisticToken, PronunciationOverride, SynthesisSegment
from .tokenizer import (
    TokenizerConfig,
    _default_lexicons_for_language,
    _effective_lexicons,
    _fallback_kwargs,
)
from .types import G2PAlignmentToken


@dataclass(frozen=True, slots=True)
class PreparedSynthesis:
    """Kokoro frontend output for one independent synthesis request."""

    request_id: str
    text: str
    language: str
    voice: str | None
    phonemes: str
    token_ids: tuple[int, ...]
    alignment_tokens: tuple[G2PAlignmentToken, ...] = ()
    diagnostics: tuple[str, ...] = ()


class PreparedG2PAdapter:
    """Call KokoroG2P prepared-text APIs while reusing language frontend instances."""

    def __init__(self, g2p_module: Any | None = None) -> None:
        self._g2p_module = g2p_module
        self._instances: dict[tuple[tuple[str, str], ...], Any] = {}
        self._lock = threading.RLock()

    def phonemize(self, segment: SynthesisSegment, config: SynthesisConfig) -> PreparedSynthesis:
        g2p_module = self._load()
        target_model, profile = self._model_target(config)
        tokenizer_config = config.tokenizer_config or TokenizerConfig()
        if profile is not None:
            require_frontend(profile.variant, allow_experimental=config.allow_experimental_frontend)
        language = segment.language
        g2p = self._get_g2p_instance(language, config, tokenizer_config, profile)
        overrides = tuple(self._to_override(item) for item in segment.pronunciation_overrides)
        annotations = tuple(self._to_annotation(item) for item in segment.annotations)
        routing = config.language_routing
        language_routing = (
            None
            if routing is None or routing.mode == "off"
            else {"mode": routing.mode, "languages": routing.languages}
        )

        if segment.phonemes is not None:
            phonemes = segment.phonemes
            token_ids = tuple(g2p_module.phonemes_to_ids(phonemes, model=target_model))
            raw_tokens: tuple[Any, ...] = ()
            warnings: tuple[str, ...] = ()
        else:
            result = g2p_module.phonemize_prepared(
                segment.text,
                language=language,
                overrides=overrides or None,
                annotations=annotations or None,
                return_ids=True,
                return_phonemes=True,
                language_routing=language_routing,
                target_model=target_model,
                alignment="span",
                overlap="snap",
                g2p=g2p,
                g2p_resolver=lambda candidate: self._get_g2p_instance(
                    candidate, config, tokenizer_config, profile
                ),
            )
            phonemes = str(getattr(result, "phonemes", "") or "")
            raw_ids = getattr(result, "token_ids", None)
            if raw_ids is None:
                raw_ids = getattr(result, "ids", ())
            token_ids = tuple(int(value) for value in raw_ids)
            raw_tokens = tuple(getattr(result, "tokens", ()) or ())
            warnings = tuple(str(item) for item in getattr(result, "warnings", ()) or ())

        alignments = tuple(
            self._alignment_token(token, g2p_module, target_model) for token in raw_tokens
        )
        return PreparedSynthesis(
            request_id=segment.id,
            text=segment.text,
            language=language,
            voice=segment.voice if isinstance(segment.voice, str) else None,
            phonemes=phonemes,
            token_ids=token_ids,
            alignment_tokens=alignments,
            diagnostics=warnings,
        )

    def phonemize_context(self, text: str, language: str, config: SynthesisConfig) -> Any:
        """Use the cached prepared-text frontend for engine-generated short-sentence context."""
        g2p_module = self._load()
        target_model, profile = self._model_target(config)
        tokenizer_config = config.tokenizer_config or TokenizerConfig()
        if profile is not None:
            require_frontend(profile.variant, allow_experimental=config.allow_experimental_frontend)
        g2p = self._get_g2p_instance(language, config, tokenizer_config, profile)
        routing = config.language_routing
        language_routing = (
            None
            if routing is None or routing.mode == "off"
            else {"mode": routing.mode, "languages": routing.languages}
        )
        return g2p_module.phonemize_prepared(
            text,
            language=language,
            overrides=None,
            annotations=None,
            return_ids=True,
            return_phonemes=True,
            alignment="span",
            language_routing=language_routing,
            target_model=target_model,
            g2p=g2p,
            g2p_resolver=lambda candidate: self._get_g2p_instance(
                candidate, config, tokenizer_config, profile
            ),
        )

    def ids_to_phonemes(self, token_ids: list[int] | tuple[int, ...], target_model: str) -> str:
        """Decode a bounded model-token chunk using KokoroG2P's active vocabulary."""
        return str(self._load().ids_to_phonemes(list(token_ids), model=target_model))

    @staticmethod
    def _to_override(value: PronunciationOverride) -> OverrideSpan:
        attrs: dict[str, str] = {}
        if value.phonemes is not None:
            attrs["ph"] = value.phonemes
        if value.language is not None:
            attrs["lang"] = value.language
        return OverrideSpan(char_start=value.start, char_end=value.end, attrs=attrs)

    @staticmethod
    def _to_annotation(value: LinguisticToken) -> TokenAnnotation:
        return TokenAnnotation(
            start=value.start,
            end=value.end,
            text=value.text,
            pos=value.pos,
            tag=value.tag,
            lemma=value.lemma,
            language=value.language,
        )

    @staticmethod
    def _alignment_token(value: Any, g2p_module: Any, target_model: str) -> G2PAlignmentToken:
        def get(name: str, default: Any = None) -> Any:
            if isinstance(value, dict):
                return value.get(name, default)
            return getattr(value, name, default)

        text = get("text", "")
        phonemes = get("phonemes", "")
        whitespace = get("whitespace", "") or ""
        model_token_count = get("model_token_count")
        if (
            not isinstance(model_token_count, int)
            or isinstance(model_token_count, bool)
            or model_token_count < 0
        ):
            try:
                model_token_count = len(g2p_module.phonemes_to_ids(phonemes, model=target_model))
            except (AttributeError, TypeError, ValueError, RuntimeError):
                model_token_count = None
        model_span_token_count = get("model_span_token_count")
        if (
            not isinstance(model_span_token_count, int)
            or isinstance(model_span_token_count, bool)
            or model_span_token_count < 0
        ):
            model_span_token_count = (
                model_token_count + (1 if whitespace else 0)
                if isinstance(model_token_count, int)
                else None
            )
        return G2PAlignmentToken(
            text=text if isinstance(text, str) else "",
            phonemes=phonemes if isinstance(phonemes, str) else "",
            whitespace=whitespace if isinstance(whitespace, str) else "",
            char_start=get("char_start"),
            char_end=get("char_end"),
            model_token_count=model_token_count,
            model_span_token_count=model_span_token_count,
            pronunciation_source=get("pronunciation_source"),
            pronunciation_provider=get("pronunciation_provider"),
            pronunciation_lexicon_id=get("pronunciation_lexicon_id"),
            pronunciation_requested_language=get("pronunciation_requested_language"),
            pronunciation_source_ipa=get("pronunciation_source_ipa"),
            pronunciation_language_markers=get("pronunciation_language_markers"),
        )

    def _load(self) -> Any:
        if self._g2p_module is None:
            import kokorog2p

            self._g2p_module = kokorog2p
        return self._g2p_module

    @staticmethod
    def _model_target(config: SynthesisConfig) -> tuple[str, Any | None]:
        source = config.model_source or "github"
        variant = config.model_variant or "v1.0"
        profile = get_model_profile(variant, source)
        return profile.tokenizer_vocab_version, profile

    def _get_g2p_instance(
        self,
        language: str,
        config: SynthesisConfig,
        tokenizer_config: TokenizerConfig,
        profile: Any | None,
    ) -> Any:
        from kokorog2p.language_codes import normalize_language_code

        kokorog2p_language = SUPPORTED_LANGUAGES.get(language, language)
        backend = (
            profile.g2p_backend if profile and profile.g2p_backend else tokenizer_config.backend
        )
        lexicons = _effective_lexicons(tokenizer_config)
        normalized_language = normalize_language_code(kokorog2p_language)
        if lexicons is None and backend == "kokorog2p":
            lexicons = _default_lexicons_for_language(normalized_language)
        kwargs: dict[str, Any] = {
            "language": kokorog2p_language,
            "version": self._model_target(config)[0],
            "phoneme_quotes": "curly",
            **_fallback_kwargs(tokenizer_config.fallback),
            "use_spacy": tokenizer_config.use_spacy,
            "spacy_model": tokenizer_config.spacy_model,
            "spacy_model_size": tokenizer_config.spacy_model_size,
            "backend": backend,
            "lexicons": lexicons,
        }
        if profile is not None and profile.variant == "ar-nabra":
            kwargs["model_profile"] = "nabra-82m-v0.1"
        key = tuple(sorted((name, repr(value)) for name, value in kwargs.items())) + (
            ("lexicon_data_policy", tokenizer_config.lexicon_data_policy),
        )
        with self._lock:
            instance = self._instances.get(key)
            if instance is not None:
                return instance
            module = self._load()
            instance = create_g2p_with_lexphon_retry(
                module,
                language=kokorog2p_language,
                config=tokenizer_config,
                kwargs=kwargs,
            )
            self._instances[key] = instance
            return instance
