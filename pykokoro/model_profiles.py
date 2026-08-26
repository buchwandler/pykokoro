"""Authoritative, dependency-light metadata for supported model profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from .config_types import ModelSource, ModelVariant

VocabularySource = Literal["builtin-v1.0", "downloaded-config", "downloaded-release"]


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Metadata that describes one source/variant asset and tokenizer profile."""

    source: ModelSource
    variant: ModelVariant
    language_codes: tuple[str, ...]
    quality_files: Mapping[str, str]
    voices_filename: str
    voice_names: tuple[str, ...]
    default_voice: str
    vocabulary_source: VocabularySource
    tokenizer_vocab_version: Literal["1.0", "1.1", "nabra-82m-v0.1"]
    release_repository: str | None = None
    release_tag: str | None = None
    release_commit: str | None = None
    model_sha256: Mapping[str, str] | None = None
    voices_sha256: str | None = None
    model_sizes: Mapping[str, int] | None = None
    voices_size: int | None = None
    suggested_speed: float | None = None
    frontend: str = "pykokoro-native"
    frontend_experimental: bool = False
    onnx_inputs: Mapping[str, str] = field(default_factory=dict)
    sample_rate: int = 24000
    vocabulary_filename: str | None = None
    max_tokens: int = 510
    publication_enabled: bool = True

    @property
    def available_qualities(self) -> tuple[str, ...]:
        return tuple(self.quality_files)

    @property
    def release_revision(self) -> str | None:
        """Compatibility alias for the associated release/tag commit."""
        return self.release_commit

    @property
    def recommended_speed(self) -> float | None:
        """Compatibility alias; the speed is advisory and never implicit."""
        return self.suggested_speed


GERMAN_MARTIN_V1_2 = ModelProfile(
    source="github",
    variant="v1.2-de-martin",
    language_codes=("de", "de-de", "de-at", "de-ch"),
    quality_files={"fp32": "kokoro-german-martin-v1.2.onnx"},
    voices_filename="voices-german-martin-v1.2.bin",
    voice_names=("martin",),
    default_voice="martin",
    vocabulary_source="builtin-v1.0",
    tokenizer_vocab_version="1.0",
    release_repository="buchwandler/kokoro-onnx-models",
    release_tag="model-files-german-martin-v1.2",
    release_commit=None,
    model_sha256={
        "kokoro-german-martin-v1.2.onnx": (
            "c302f1d8bc7adf40a842cb550e18c39a5026bdb1afdd29dbb700b501cb49276b"
        ),
    },
    voices_sha256=("5b9c8553398d7abf67498ce500c186cefaa7b68fed3e3d415da5380670105acd"),
    model_sizes={"kokoro-german-martin-v1.2.onnx": 325_512_630},
    voices_size=522_506,
    suggested_speed=1.125,
)


def _release_profile(
    variant: ModelVariant,
    language: str,
    tag: str,
    model_filename: str,
    voices_filename: str,
    voice_names: tuple[str, ...],
    frontend: str,
    onnx_inputs: Mapping[str, str],
    *,
    publication_enabled: bool = True,
    suggested_speed: float | None = None,
    frontend_experimental: bool = True,
    vocabulary_source: VocabularySource = "downloaded-config",
    tokenizer_vocab_version: Literal["1.0", "1.1", "nabra-82m-v0.1"] = "1.0",
    vocabulary_filename: str | None = None,
    max_tokens: int = 510,
) -> ModelProfile:
    return ModelProfile(
        source="github",
        variant=variant,
        language_codes=(language,),
        quality_files={"fp32": model_filename},
        voices_filename=voices_filename,
        voice_names=voice_names,
        default_voice=voice_names[0],
        vocabulary_source=vocabulary_source,
        tokenizer_vocab_version=tokenizer_vocab_version,
        release_repository="buchwandler/kokoro-onnx-models",
        release_tag=tag,
        frontend=frontend,
        frontend_experimental=frontend_experimental,
        onnx_inputs=onnx_inputs,
        sample_rate=24000,
        vocabulary_filename=vocabulary_filename,
        max_tokens=max_tokens,
        publication_enabled=publication_enabled,
        suggested_speed=suggested_speed,
    )


VI_CONTEXTBOX = _release_profile(
    "vi-contextbox", "vi", "model-files-vietnamese-v1.0",
    "kokoro-vietnamese-v1.0.onnx",
    "voices-vietnamese-v1.0.npz",
    ("diem_trinh", "hung_thinh", "mai_linh", "mai_loan", "manh_dung", "my_yen",
     "ngoc_huyen", "phat_tai", "thanh_dat", "thuc_trinh", "tuan_ngoc", "storyvert",
     "duc_an", "duc_duy"),
    "vig2p", {"tokens": "int64", "style": "float32", "speed": "float32"},
 )
VI_ANPHUNL = _release_profile(
    "vi-anphunl", "vi", "model-files-vietnamese-anphunl-v1.0",
    "kokoro-vietnamese-anphunl-v1.0.onnx",
    "voices-vietnamese-anphunl-v1.0.npz", VI_CONTEXTBOX.voice_names,
    "vig2p", {"tokens": "int64", "style": "float32", "speed": "float32"},
 )
AR_NABRA = _release_profile(
    "ar-nabra", "ar", "model-files-arabic-nabra-v0.1",
    "kokoro-arabic-nabra-v0.1.onnx",
    "voices-arabic-nabra-v0.1.npz", ("af_msa",),
    "Arabic diacritizer + espeak-ng + Nabra cleanup",
    {"input_ids": "int64", "ref_s": "float32", "speed": "float32"},
    frontend_experimental=False,
    vocabulary_source="downloaded-release",
    tokenizer_vocab_version="nabra-82m-v0.1",
    vocabulary_filename="vocab-arabic-nabra-v0.1.json",
    max_tokens=510,
)
DE_CRANE = _release_profile(
    "de-crane", "de", "model-files-german-kerstin-v1.0",
    "kokoro-german-kerstin-v1.0.onnx",
    "voices-german-kerstin-v1.0.npz", ("df_kerstin",),
    "German IPA", {"input_ids": "int64", "style": "float32", "speed": "float32"},
 )
HE_HEBREW_NC = _release_profile(
    "he-hebrew-nc", "he", "model-files-hebrew-nc-v1.0",
    "kokoro-hebrew-nc-v1.0.onnx",
    "voices-hebrew-nc-v1.0.npz", ("he_shaul",),
    "Hebrew-specific G2P", {"tokens": "int64", "style": "float32", "speed": "float32"},
    publication_enabled=False,
 )
MODEL_PROFILES: dict[tuple[ModelSource, ModelVariant], ModelProfile] = {
    ("github", "v1.2-de-martin"): GERMAN_MARTIN_V1_2,
    ("github", "vi-contextbox"): VI_CONTEXTBOX,
    ("github", "vi-anphunl"): VI_ANPHUNL,
    ("github", "ar-nabra"): AR_NABRA,
    ("github", "de-crane"): DE_CRANE,
    ("github", "he-hebrew-nc"): HE_HEBREW_NC,
}


def get_model_profile(
    variant: ModelVariant,
    source: ModelSource = "github",
) -> ModelProfile:
    """Return profile metadata, including compatibility profiles for old variants."""
    profile = MODEL_PROFILES.get((source, variant))
    if profile is not None:
        return profile

    from .asset_constants import (
        GITHUB_VOICES_FILENAME_V1_0,
        GITHUB_VOICES_FILENAME_V1_1_ZH,
        MODEL_QUALITY_CACHE_FILES_HF_V1_0,
        MODEL_QUALITY_FILES_GITHUB_V1_0,
        MODEL_QUALITY_FILES_GITHUB_V1_1_ZH,
        MODEL_QUALITY_FILES_HF,
    )

    if source == "github":
        data: dict[ModelVariant, tuple[Mapping[str, str], str, tuple[str, ...]]] = {
            "v1.0": (MODEL_QUALITY_FILES_GITHUB_V1_0, GITHUB_VOICES_FILENAME_V1_0, ("af",)),
            "v1.1-zh": (
                MODEL_QUALITY_FILES_GITHUB_V1_1_ZH,
                GITHUB_VOICES_FILENAME_V1_1_ZH,
                ("af_maple",),
            ),
        }
        try:
            quality_files, voices_filename, voice_names = data[variant]
        except KeyError as exc:
            raise ValueError(f"Unknown model profile: {source}/{variant}") from exc
        return ModelProfile(
            source=source,
            variant=variant,
            language_codes=(),
            quality_files=quality_files,
            voices_filename=voices_filename,
            voice_names=voice_names,
            default_voice=voice_names[0],
            vocabulary_source="downloaded-config" if variant == "v1.1-zh" else "builtin-v1.0",
            tokenizer_vocab_version="1.1" if variant == "v1.1-zh" else "1.0",
            release_repository="thewh1teagle/kokoro-onnx",
            release_tag={
                "v1.0": "model-files-v1.0",
                "v1.1-zh": "model-files-v1.1",
            }[variant],
            release_commit={
                "v1.0": "6843c53fc280ab130b7a8d206ebd3407e094efdc",
                "v1.1-zh": "b85309f90fd2660ea3309cf0f2581360e4327555",
            }[variant],
        )
    if source == "huggingface" and variant in {"v1.0", "v1.1-zh"}:
        return ModelProfile(
            source=source,
            variant=variant,
            language_codes=(),
            quality_files=(
                MODEL_QUALITY_CACHE_FILES_HF_V1_0 if variant == "v1.0" else MODEL_QUALITY_FILES_HF
            ),
            voices_filename="voices.bin.npz",
            voice_names=(),
            default_voice="af",
            vocabulary_source="downloaded-config",
            tokenizer_vocab_version="1.1" if variant == "v1.1-zh" else "1.0",
        )
    raise ValueError(f"Unknown or incompatible model profile: {source}/{variant}")


def normalize_language_code(lang: str) -> str:
    """Normalize language tags for profile matching."""
    return lang.strip().lower().replace("_", "-")


def profile_for_language(lang: str) -> ModelProfile | None:
    """Return the automatic profile for a language tag, if one exists."""
    normalized = normalize_language_code(lang)
    for profile in MODEL_PROFILES.values():
        if normalized in profile.language_codes:
            return profile
    return None


def profile_for_voice(voice: str) -> ModelProfile | None:
    """Return a profile when ``voice`` uniquely identifies one profile."""
    matches = [profile for profile in MODEL_PROFILES.values() if voice in profile.voice_names]
    return matches[0] if len(matches) == 1 else None
