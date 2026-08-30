"""Local runtime metadata for supported model frontends.

Published artifact inventory is intentionally resolved by :mod:`release_catalog`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .config_types import ModelSource, ModelVariant

VocabularySource = Literal["builtin-v1.0", "downloaded-config", "downloaded-release"]

G2PBackend = Literal["kokorog2p", "espeak", "goruut"]


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

    layout: str = "single-onnx-v1"
    runtime_available: bool = True
    redistribution_allowed: bool = True
    support_status: str = "ready"

    g2p_backend: G2PBackend | None = None

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
    g2p_backend: G2PBackend | None = None,
) -> RuntimeProfile:
    default_voices = {
        "v1.2-de-martin": "martin",
        "vi-contextbox": "default",
        "vi-anphunl": "default",
        "ar-nabra": "default",
        "de-crane": "default",
        "he-hebrew-nc": "default",
    }
    return RuntimeProfile(
        source="github",
        variant=variant,
        language_codes=language_codes,
        default_voice=default_voices.get(variant),
        vocabulary_source=vocabulary_source,
        tokenizer_vocab_version=tokenizer_vocab_version,
        frontend=frontend,
        frontend_experimental=frontend_experimental,
        onnx_inputs=onnx_inputs,
        max_tokens=max_tokens,
        g2p_backend=g2p_backend,
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
        g2p_backend="espeak",
    ),
    ("github", "vi-anphunl"): _github_profile(
        "vi-anphunl",
        ("vi",),
        "vig2p-v1",
        {"tokens": "int64", "style": "float32", "speed": "float32"},
        g2p_backend="espeak",
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
        g2p_backend="kokorog2p",
    ),
    ("github", "he-hebrew-nc"): _github_profile(
        "he-hebrew-nc",
        ("he",),
        "hebrew-g2p-v1",
        {"tokens": "int64", "style": "float32", "speed": "float32"},
        frontend_experimental=True,
        g2p_backend="espeak",
    ),
}

MODEL_PROFILES[("github", "vi-anphunl")] = replace(
    MODEL_PROFILES[("github", "vi-anphunl")],
    runtime_available=False,
    support_status="registry-unavailable",
)

MODEL_PROFILES.update(
    {
        ("github", "sv-joakim"): RuntimeProfile(
            "github",
            "sv-joakim",
            ("sv",),
            "Alice",
            "builtin-v1.0",
            "1.0",
            "kokorog2p-sv-v1",
            False,
            voice_names=(
                "Alice",
                "Anton",
                "Björn",
                "Ebba",
                "Elsa",
                "Greta",
                "Lars",
                "Nils",
                "Oskar",
                "Stina",
            ),
        ),
        ("github", "de-thorsten"): RuntimeProfile(
            "github",
            "de-thorsten",
            ("de",),
            "thorsten",
            "downloaded-config",
            "1.0",
            "kokorog2p-de-thorsten-v1",
            False,
            voice_names=("thorsten",),
            g2p_backend="kokorog2p",
        ),
        ("github", "kk-anuarsv"): RuntimeProfile(
            "github",
            "kk-anuarsv",
            ("kk",),
            "km_m1",
            "downloaded-config",
            "1.0",
            "kokorog2p-kk-v1",
            False,
            voice_names=("km_m1",),
        ),
        ("github", "th-wayu"): RuntimeProfile(
            "github",
            "th-wayu",
            ("th",),
            "f_young_clear",
            "downloaded-config",
            "1.0",
            "kokorog2p-th-wayu-v1",
            False,
            layout="split-onnx-v1",
            voice_names=(
                "f_teen_bright",
                "f_young_bright",
                "f_young_clear",
                "f_young_warm",
                "f_mid_clear",
                "f_mid_warm",
                "f_elderly_soft",
                "f_elderly_low",
                "m_teen_bright",
                "m_young_clear",
                "m_mid_warm",
                "m_elderly_deep",
            ),
        ),
        ("huggingface", "ru-zaakirio-base"): RuntimeProfile(
            "huggingface",
            "ru-zaakirio-base",
            ("ru",),
            "sveta",
            "downloaded-config",
            "1.0",
            "kokorog2p-ru-v1",
            False,
            voice_names=("sveta", "masha"),
        ),
        ("huggingface", "ru-zaakirio-dima"): RuntimeProfile(
            "huggingface",
            "ru-zaakirio-dima",
            ("ru",),
            "dima",
            "downloaded-config",
            "1.0",
            "kokorog2p-ru-v1",
            False,
            voice_names=("dima",),
        ),
    }
)

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
        else {
            key: value
            for key, value in MODEL_QUALITY_FILES_HF.items()
            if key not in {"q8f16", "uint8f16"}
        }
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


def model_id_for_voice(voice: str) -> str | None:
    """Return a registry model ID for voices that select their own checkpoint."""
    return {
        "sveta": "ru-zaakirio-base",
        "masha": "ru-zaakirio-base",
        "dima": "ru-zaakirio-dima",
    }.get(voice)


VOICE_ALIASES: dict[tuple[str, str], str] = {
    ("de-crane", "df_kerstin"): "default",
    ("de-thorsten", "thorsten"): "thorsten",
}

IMPLEMENTED_FRONTENDS = {
    "pykokoro-native-v1",
    "german-ipa-v1",
    "vig2p-v1",
    "nabra-arabic-v1",
    "kokorog2p-sv-v1",
    "kokorog2p-de-thorsten-v1",
    "kokorog2p-kk-v1",
    "kokorog2p-ru-v1",
    "kokorog2p-th-wayu-v1",
}
IMPLEMENTED_LAYOUTS = {"single-onnx-v1", "split-onnx-v1"}


def canonical_voice_name(model_id: str, voice: str) -> str:
    """Return a user alias's canonical registry voice name."""
    return VOICE_ALIASES.get((model_id, voice), voice)


def registry_support_status(model: Any) -> str:
    """Classify registry metadata against PyKokoro implementation capabilities."""
    if not model.runtime_available:
        return "registry-unavailable"
    if model.layout not in IMPLEMENTED_LAYOUTS:
        return "unsupported-layout"
    if model.frontend not in IMPLEMENTED_FRONTENDS:
        return "unsupported-frontend"
    if not model.redistribution_allowed:
        return "restricted"
    return "ready"


def get_registry_model_profile(
    model_id: str,
    *,
    preference: Literal["auto", "github", "huggingface", "upstream"] = "auto",
    offline: bool = False,
    registry: Any | None = None,
) -> RuntimeProfile:
    """Build a profile from canonical registry metadata and local capabilities."""
    from .model_registry import ModelRegistryError, RegistryClient

    if registry is None:
        registry = RegistryClient().load(offline=offline)
    model = registry.model(model_id)
    distribution = model.distribution(preference) if model.runtime_available else None
    source: ModelSource = (
        "github"
        if distribution is not None and distribution.provider == "github-release"
        else "huggingface"
    )
    local = MODEL_PROFILES.get((source, model_id))
    qualities = (
        {
            artifact.quality: artifact.local_name
            for artifact in distribution.artifacts
            if artifact.role == "model" and artifact.quality is not None
        }
        if distribution is not None
        else {}
    )
    if not model.runtime_available:
        raise ModelRegistryError(f"Model profile {model_id!r} has no runtime-ready distribution")
    assert distribution is not None
    vocabulary_source: VocabularySource = (
        "downloaded-release"
        if any(artifact.role == "vocab" for artifact in distribution.artifacts)
        else "downloaded-config"
        if any(artifact.role == "config" for artifact in distribution.artifacts)
        else "builtin-v1.0"
    )
    tokenizer_version = (
        local.tokenizer_vocab_version
        if local is not None
        else str(model.data.get("model_version", "1.0"))
    )
    onnx_inputs = local.onnx_inputs if local is not None else model.onnx_contract.get("inputs", {})
    return RuntimeProfile(
        source=source,
        variant=model_id,
        language_codes=model.language_codes,
        default_voice=model.default_voice,
        vocabulary_source=vocabulary_source,
        tokenizer_vocab_version=tokenizer_version,
        frontend=model.frontend,
        frontend_experimental=local.frontend_experimental if local is not None else False,
        g2p_backend=local.g2p_backend if local is not None else None,
        onnx_inputs=onnx_inputs if isinstance(onnx_inputs, Mapping) else {},
        sample_rate=model.sample_rate,
        max_tokens=model.max_tokens,
        quality_files=qualities,
        voice_names=model.voices,
        layout=model.layout,
        runtime_available=model.runtime_available,
        redistribution_allowed=model.redistribution_allowed,
        support_status=registry_support_status(model),
    )
