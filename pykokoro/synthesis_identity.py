"""Stable output-affecting identity for one resolved synthesis request."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from .synthesis_config import SynthesisConfig
from .voice_level import default_voice_calibration
from .voice_manager import VoiceBlend


def _identity_value(value: Any) -> Any:
    if is_dataclass(value):
        return _identity_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _identity_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if not callable(item)
        }
    if isinstance(value, (list, tuple)):
        return [_identity_value(item) for item in value]
    if isinstance(value, Path):
        return {"configured_path": True}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if callable(value):
        return None
    raise TypeError(f"unsupported value in synthesis identity: {type(value).__name__}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _identity_value(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SynthesisIdentity:
    """JSON-safe identity fields that can form a stable output cache key."""

    package_version: str
    model_id: str
    voice: str
    language: str
    speed: float
    random_seed: int | None
    short_sentence: str
    resolved_short_sentence_mode: str | None
    frontend: str
    provider: str | None
    provider_options_sha256: str | None
    voice_level_mode: str
    voice_level_gain_db: float | None
    calibration_dataset: str | None
    calibration_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe representation without transient filesystem paths."""
        return asdict(self)

    @property
    def cache_key(self) -> str:
        """Return a stable digest of all identity fields."""
        return _digest(self.to_dict())


def build_synthesis_identity(
    config: SynthesisConfig,
    *,
    language: str,
    voice: str | VoiceBlend | None,
) -> SynthesisIdentity:
    """Build a stable identity from resolved settings that can affect output."""
    from . import __version__
    from .short_sentence_handler import ShortSentenceConfig

    if isinstance(voice, VoiceBlend):
        voice_identity = _canonical_json(
            {"voices": voice.voices, "interpolation": voice.interpolation}
        )
    else:
        voice_identity = voice or ""

    short_sentence_config = config.short_sentence_config
    short_sentence_override = config.generation.enable_short_sentence
    if short_sentence_override is False:
        short_sentence_identity: Any = {"enabled": False}
    elif short_sentence_config is not None:
        short_sentence_identity = asdict(short_sentence_config)
        if short_sentence_override is True:
            short_sentence_identity["enabled"] = True
    elif short_sentence_override is True:
        short_sentence_identity = asdict(ShortSentenceConfig())
    else:
        short_sentence_identity = {"enabled": False}

    frontend_identity = {
        "tokenizer": config.tokenizer_config,
        "espeak_configured": config.espeak_config is not None,
        "language_routing": config.language_routing,
        "allow_experimental_frontend": config.allow_experimental_frontend,
    }
    model_id = config.model_identity or ":".join(
        str(value) for value in (config.model_source, config.model_variant, config.model_quality)
    )

    calibration_dataset = None
    calibration_sha256 = None
    if config.voice_level.mode == "calibrated" and config.voice_level.gain_db is None:
        catalog = default_voice_calibration()
        calibration_dataset = catalog.corpus
        calibration_sha256 = _digest(
            {
                "schema": catalog.schema,
                "method": catalog.method,
                "corpus": catalog.corpus,
                "reference_lufs": catalog.reference_lufs,
                "generated_with": catalog.generated_with,
                "voices": {str(key): value for key, value in catalog.voices.items()},
            }
        )

    return SynthesisIdentity(
        package_version=__version__,
        model_id=model_id,
        voice=voice_identity,
        language=language,
        speed=float(config.generation.speed),
        random_seed=config.generation.random_seed,
        short_sentence=_canonical_json(short_sentence_identity),
        resolved_short_sentence_mode=None,
        frontend=_canonical_json(frontend_identity),
        provider=config.provider,
        provider_options_sha256=(
            None if config.provider_options is None else _digest(config.provider_options)
        ),
        voice_level_mode=config.voice_level.mode,
        voice_level_gain_db=(
            None if config.voice_level.gain_db is None else float(config.voice_level.gain_db)
        ),
        calibration_dataset=calibration_dataset,
        calibration_sha256=calibration_sha256,
    )
