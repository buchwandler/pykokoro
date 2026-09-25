"""Kokoro vocabulary encoding and G2P frontend configuration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import kokorog2p as _kokorog2p

from .constants import MAX_PHONEME_LENGTH
from .spacy_models import SpacyModelSize, make_spacy_model_request

N_TOKENS = _kokorog2p.N_TOKENS
BackendType: TypeAlias = str
FallbackMode: TypeAlias = Literal["none", "espeak", "goruut"]
LexiconDataPolicy: TypeAlias = Literal["auto", "installed-only"]


def _fallback_kwargs(mode: FallbackMode) -> dict[str, bool]:
    if mode == "none":
        return {"use_espeak_fallback": False, "use_goruut_fallback": False}
    if mode == "espeak":
        return {"use_espeak_fallback": True, "use_goruut_fallback": False}
    return {"use_espeak_fallback": False, "use_goruut_fallback": True}


def _normalize_kokorog2p_version(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "1.0": "1.0",
        "v1.0": "1.0",
        "1.1": "1.1",
        "v1.1": "1.1",
        "1.1-zh": "1.1",
        "v1.1-zh": "1.1",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported KokoroG2P vocabulary/version identifier: {value!r}") from exc


@dataclass
class TokenizerConfig:
    """Settings forwarded to the KokoroG2P prepared-text frontend."""

    fallback: FallbackMode = "espeak"
    use_spacy: bool | None = None
    spacy_model: str | None = None
    spacy_model_size: SpacyModelSize | None = None
    backend: BackendType = "kokorog2p"
    lexicons: str | Sequence[str] | None = None
    lexicon_data_policy: LexiconDataPolicy = "auto"

    def __post_init__(self) -> None:
        request = make_spacy_model_request(model=self.spacy_model, size=self.spacy_model_size)
        self.spacy_model = request.model
        self.spacy_model_size = request.size
        if self.fallback not in {"none", "espeak", "goruut"}:
            raise ValueError("fallback must be 'none', 'espeak', or 'goruut'")
        if self.lexicon_data_policy not in {"auto", "installed-only"}:
            raise ValueError("lexicon_data_policy must be 'auto' or 'installed-only'")
        if self.lexicons is not None:
            values = (self.lexicons,) if isinstance(self.lexicons, str) else tuple(self.lexicons)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError("lexicons must contain non-empty names")
            self.lexicons = tuple(value.strip() for value in values)


def _default_lexicons_for_language(language: str) -> tuple[str, ...] | None:
    normalized = language.lower().replace("_", "-")
    if normalized in {"de", "de-de", "de-at", "de-ch", "deu", "german"}:
        return ("espeak",)
    return None


def _effective_lexicons(config: TokenizerConfig) -> tuple[str, ...] | None:
    if config.lexicons is None:
        return None
    if isinstance(config.lexicons, str):
        return (config.lexicons,)
    return tuple(config.lexicons)


@dataclass
class EspeakConfig:
    """Optional eSpeak asset paths managed by KokoroG2P."""

    lib_path: str | None = None
    data_path: str | None = None


class Tokenizer:
    """Encode and decode phonemes using one Kokoro model vocabulary."""

    def __init__(
        self,
        espeak_config: EspeakConfig | None = None,
        vocab_version: str = "v1.0",
        vocab: dict[str, int] | None = None,
        config: TokenizerConfig | None = None,
    ) -> None:
        del espeak_config
        self.vocab_version = vocab_version
        self.config = config or TokenizerConfig()
        self._kokorog2p_model = _normalize_kokorog2p_version(vocab_version)
        self.vocab = (
            vocab if vocab is not None else _kokorog2p.get_kokoro_vocab(model=self._kokorog2p_model)
        )
        self._reverse_vocab: dict[int, str] | None = None

    @staticmethod
    def normalize_text(text: str) -> str:
        """Trim surrounding whitespace without interpreting markup."""
        return text.strip()

    @property
    def reverse_vocab(self) -> dict[int, str]:
        if self._reverse_vocab is None:
            self._reverse_vocab = {value: key for key, value in self.vocab.items()}
        return self._reverse_vocab

    def tokenize(self, phonemes: str) -> list[int]:
        if len(phonemes) > MAX_PHONEME_LENGTH:
            raise ValueError(
                f"Phoneme string too long ({len(phonemes)} chars); maximum is {MAX_PHONEME_LENGTH}"
            )
        return [
            self.vocab[char] for char in phonemes if char in self.vocab and self.vocab[char] != 0
        ]

    def detokenize(self, tokens: list[int]) -> str:
        return "".join(
            self.reverse_vocab[token_id]
            for token_id in tokens
            if token_id != 0 and token_id in self.reverse_vocab
        )

    def get_vocab_info(self) -> dict[str, int | str]:
        return {
            "version": self.vocab_version,
            "num_tokens": len(self.vocab),
            "max_token_id": max(self.vocab.values()) if self.vocab else 0,
            "max_phoneme_length": MAX_PHONEME_LENGTH,
            "n_tokens": N_TOKENS,
        }

    def validate_phonemes(self, phonemes: str) -> tuple[bool, list[str]]:
        invalid = [char for char in phonemes if char not in self.vocab]
        return not invalid, invalid
