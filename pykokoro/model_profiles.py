"""Local runtime metadata for supported model frontends.

Published artifact inventory is intentionally resolved by :mod:`release_catalog`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .config_types import ModelSource, ModelVariant

VocabularySource = Literal["builtin-v1.0", "downloaded-config", "downloaded-release"]


@dataclass(frozen=True, slots=True)
class RuntimeProfile:
    """Runtime behavior owned by pykokoro, independent of release inventory."""

    source: ModelSource
    variant: ModelVariant
    language_codes: tuple[str, ...]
    default_voice: str | None
    vocabulary_source: VocabularySource
    tokenizer_vocab_version: str
    frontend: str
    frontend_experimental: bool
    onnx_inputs: Mapping[str, str] = field(default_factory=dict)
    sample_rate: int = 24000
    max_tokens: int = 510
    quality_files: Mapping[str, str] = field(default_factory=dict)
    voice_names: tuple[str, ...] = ()

    @property
    def available_qualities(self) -> tuple[str, ...]:
        return tuple(self.quality_files)


ModelProfile = RuntimeProfile


def _github_profile(
    variant: str,
    language_codes: tuple[str, ...],
    frontend: str,
    onnx_inputs: Mapping[str, str],
    *,
    vocabulary_source: VocabularySource = "builtin-v1.0",
    tokenizer_vocab_version: str = "1.0",
    frontend_experimental: bool = True,
    max_tokens: int = 510,
) -> RuntimeProfile:
    default_voices = {
        "v1.2-de-martin": "martin",
        "vi-contextbox": "diem_trinh",
        "vi-anphunl": "diem_trinh",
        "ar-nabra": "af_msa",
        "de-crane": "df_kerstin",
        "he-hebrew-nc": "he_shaul",
    }
    return RuntimeProfile(
        source="github",
        variant=variant,  # type: ignore[arg-type]
        language_codes=language_codes,
        default_voice=default_voices.get(variant),
        vocabulary_source=vocabulary_source,
        tokenizer_vocab_version=tokenizer_vocab_version,
        frontend=frontend,
        frontend_experimental=frontend_experimental,
        onnx_inputs=onnx_inputs,
        max_tokens=max_tokens,
    )


MODEL_PROFILES: dict[tuple[ModelSource, ModelVariant], RuntimeProfile] = {
    ("github", "v1.2-de-martin"): _github_profile(
        "v1.2-de-martin",
        ("de", "de-de", "de-at", "de-ch"),
        "german-ipa-v1",
        {"tokens": "int64", "style": "float32", "speed": "float32"},
        frontend_experimental=False,
    ),
    ("github", "vi-contextbox"): _github_profile(
        "vi-contextbox",
        ("vi",),
        "vig2p-v1",
        {"tokens": "int64", "style": "float32", "speed": "float32"},
    ),
    ("github", "vi-anphunl"): _github_profile(
        "vi-anphunl",
        ("vi",),
        "vig2p-v1",
        {"tokens": "int64", "style": "float32", "speed": "float32"},
    ),
    ("github", "ar-nabra"): _github_profile(
        "ar-nabra",
        ("ar",),
        "nabra-arabic-v1",
        {"input_ids": "int64", "ref_s": "float32", "speed": "float32"},
        vocabulary_source="downloaded-release",
        tokenizer_vocab_version="nabra-82m-v0.1",
        frontend_experimental=False,
    ),
    ("github", "de-crane"): _github_profile(
        "de-crane",
        ("de",),
        "german-ipa-v1",
        {"input_ids": "int64", "style": "float32", "speed": "float32"},
    ),
    ("github", "he-hebrew-nc"): _github_profile(
        "he-hebrew-nc",
        ("he",),
        "hebrew-g2p-v1",
        {"tokens": "int64", "style": "float32", "speed": "float32"},
        frontend_experimental=True,
    ),
}

GERMAN_MARTIN_V1_2 = MODEL_PROFILES[("github", "v1.2-de-martin")]


def _legacy_profile(variant: ModelVariant, source: ModelSource) -> RuntimeProfile:
    from .asset_constants import (
        MODEL_QUALITY_CACHE_FILES_HF_V1_0,
        MODEL_QUALITY_FILES_GITHUB_V1_0,
        MODEL_QUALITY_FILES_GITHUB_V1_1_ZH,
        MODEL_QUALITY_FILES_HF,
    )

    if variant not in {"v1.0", "v1.1-zh"}:
        raise ValueError(f"Unknown model profile: {source}/{variant}")
    if source == "github":
        qualities = (
            MODEL_QUALITY_FILES_GITHUB_V1_0
            if variant == "v1.0"
            else MODEL_QUALITY_FILES_GITHUB_V1_1_ZH
        )
        return RuntimeProfile(
            source=source,
            variant=variant,
            language_codes=("en", "es", "fr", "hi", "it", "ja", "pt", "zh")
            if variant == "v1.0"
            else ("zh",),
            default_voice="af_heart" if variant == "v1.0" else "af_maple",
            vocabulary_source="downloaded-release",
            tokenizer_vocab_version="1.1" if variant == "v1.1-zh" else "1.0",
            frontend="pykokoro-native-v1",
            frontend_experimental=False,
            quality_files=qualities,
        )
    qualities = (
        MODEL_QUALITY_CACHE_FILES_HF_V1_0
        if variant == "v1.0"
        else {key: value for key, value in MODEL_QUALITY_FILES_HF.items() if key not in {"q8f16", "uint8f16"}}
    )
    if variant == "v1.1-zh":
        qualities.update({"int8": "model_int8.onnx", "bnb4": "model_bnb4.onnx"})
    return RuntimeProfile(
        source=source,
        variant=variant,
        language_codes=(),
        default_voice="af_heart" if variant == "v1.0" else "af_maple",
        vocabulary_source="downloaded-config",
        tokenizer_vocab_version="1.1" if variant == "v1.1-zh" else "1.0",
        frontend="pykokoro-native-v1",
        frontend_experimental=False,
        quality_files=qualities,
    )


def get_model_profile(variant: ModelVariant, source: ModelSource = "github") -> RuntimeProfile:
    """Return local runtime metadata, not remote release inventory."""
    profile = MODEL_PROFILES.get((source, variant))
    if profile is not None:
        return profile
    if variant in {"v1.0", "v1.1-zh"}:
        return _legacy_profile(variant, source)
    raise ValueError(f"Unknown or incompatible model profile: {source}/{variant}")


def normalize_language_code(lang: str) -> str:
    return lang.strip().lower().replace("_", "-")


def profile_for_language(lang: str) -> RuntimeProfile | None:
    normalized = normalize_language_code(lang)
    for profile in MODEL_PROFILES.values():
        if normalized in profile.language_codes:
            return profile
    return None


DEFAULT_PROFILE_BY_VOICE = {"martin": "v1.2-de-martin"}


def profile_for_voice(voice: str) -> RuntimeProfile | None:
    variant = DEFAULT_PROFILE_BY_VOICE.get(voice)
    if variant is not None:
        return get_model_profile(variant, "github")
    matches = [profile for profile in MODEL_PROFILES.values() if voice in profile.voice_names]
    return matches[0] if len(matches) == 1 else None
