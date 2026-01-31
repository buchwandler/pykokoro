"""Transcript schema and validation utilities for audio-ready output."""

from __future__ import annotations

import json
from typing import Any

TRANSCRIPT_VERSION = "1.0"


def export_transcript(transcript: dict[str, Any], *, indent: int = 2) -> str:
    """Serialize a transcript dictionary to JSON."""
    return json.dumps(transcript, ensure_ascii=False, indent=indent)


def load_transcript(transcript_or_json: dict[str, Any] | str) -> dict[str, Any]:
    """Load a transcript from a dict or JSON string."""
    if isinstance(transcript_or_json, str):
        try:
            parsed = json.loads(transcript_or_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Transcript JSON is invalid") from exc
        if not isinstance(parsed, dict):
            raise ValueError("Transcript JSON must decode to an object")
        return parsed
    return transcript_or_json


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a dict")
    return value


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _require_number(value: Any, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string or null")
    return value


def _optional_number(value: Any, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number or null")
    return float(value)


def validate_transcript(transcript: dict[str, Any]) -> None:
    """Validate a transcript against the minimal schema.

    Raises:
        ValueError: If transcript is invalid.
    """
    if not isinstance(transcript, dict):
        raise ValueError("Transcript must be a dict")

    format_version = transcript.get("format_version")
    if format_version != TRANSCRIPT_VERSION:
        raise ValueError(f"Unsupported transcript format_version: {format_version!r}")

    defaults = _require_dict(transcript.get("defaults"), "defaults")
    _require_str(defaults.get("lang"), "defaults.lang")
    _require_number(defaults.get("speed"), "defaults.speed")
    pause_mode = defaults.get("pause_mode")
    if pause_mode not in {"tts", "manual", "auto"}:
        raise ValueError("defaults.pause_mode must be 'tts', 'manual', or 'auto'")

    pause = _require_dict(defaults.get("pause"), "defaults.pause")
    _require_number(pause.get("clause"), "defaults.pause.clause")
    _require_number(pause.get("sentence"), "defaults.pause.sentence")
    _require_number(pause.get("paragraph"), "defaults.pause.paragraph")
    _require_number(pause.get("variance"), "defaults.pause.variance")
    random_seed = pause.get("random_seed")
    if random_seed is not None and not isinstance(random_seed, int):
        raise ValueError("defaults.pause.random_seed must be an int or null")

    prosody = _require_dict(defaults.get("prosody"), "defaults.prosody")
    _optional_str(prosody.get("rate"), "defaults.prosody.rate")
    _optional_str(prosody.get("pitch"), "defaults.prosody.pitch")
    _optional_str(prosody.get("volume"), "defaults.prosody.volume")

    short_sentence = _require_dict(
        defaults.get("short_sentence"), "defaults.short_sentence"
    )
    _require_bool(short_sentence.get("enabled"), "defaults.short_sentence.enabled")
    _require_number(
        short_sentence.get("min_phoneme_length"),
        "defaults.short_sentence.min_phoneme_length",
    )
    _require_str(
        short_sentence.get("phoneme_pretext"),
        "defaults.short_sentence.phoneme_pretext",
    )

    voice = _require_dict(defaults.get("voice"), "defaults.voice")
    _optional_str(voice.get("name"), "defaults.voice.name")
    _optional_str(voice.get("variant"), "defaults.voice.variant")
    _optional_str(voice.get("language"), "defaults.voice.language")
    _optional_str(voice.get("gender"), "defaults.voice.gender")
    if "blend" in voice and voice["blend"] is not None:
        blend = _require_dict(voice.get("blend"), "defaults.voice.blend")
        voices = _require_list(blend.get("voices"), "defaults.voice.blend.voices")
        _require_str(blend.get("interpolation"), "defaults.voice.blend.interpolation")
        for idx, voice_item in enumerate(voices):
            if not isinstance(voice_item, dict):
                raise ValueError(f"defaults.voice.blend.voices[{idx}] must be a dict")
            _require_str(
                voice_item.get("name"),
                f"defaults.voice.blend.voices[{idx}].name",
            )
            _require_number(
                voice_item.get("weight"),
                f"defaults.voice.blend.voices[{idx}].weight",
            )

    include_tokens = defaults.get("include_tokens", False)
    if not isinstance(include_tokens, bool):
        raise ValueError("defaults.include_tokens must be a boolean")

    segments = _require_list(transcript.get("segments"), "segments")
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{index}] must be a dict")
        if segment.get("type") != "phoneme_segment":
            raise ValueError(f"segments[{index}].type must be 'phoneme_segment'")
        _require_str(segment.get("text"), f"segments[{index}].text")
        _require_str(segment.get("phonemes"), f"segments[{index}].phonemes")
        _require_str(segment.get("lang"), f"segments[{index}].lang")

        pause_seg = _require_dict(segment.get("pause"), f"segments[{index}].pause")
        _require_number(pause_seg.get("before"), f"segments[{index}].pause.before")
        _require_number(pause_seg.get("after"), f"segments[{index}].pause.after")

        if "voice" in segment:
            seg_voice = _require_dict(segment.get("voice"), f"segments[{index}].voice")
            _optional_str(seg_voice.get("name"), f"segments[{index}].voice.name")
            _optional_str(seg_voice.get("variant"), f"segments[{index}].voice.variant")
            _optional_str(
                seg_voice.get("language"), f"segments[{index}].voice.language"
            )
            _optional_str(seg_voice.get("gender"), f"segments[{index}].voice.gender")

        if "prosody" in segment:
            seg_prosody = _require_dict(
                segment.get("prosody"), f"segments[{index}].prosody"
            )
            _optional_str(seg_prosody.get("rate"), f"segments[{index}].prosody.rate")
            _optional_str(seg_prosody.get("pitch"), f"segments[{index}].prosody.pitch")
            _optional_str(
                seg_prosody.get("volume"), f"segments[{index}].prosody.volume"
            )

        if "flags" in segment:
            flags = _require_dict(segment.get("flags"), f"segments[{index}].flags")
            if "say_as_applied" in flags:
                _require_bool(
                    flags.get("say_as_applied"),
                    f"segments[{index}].flags.say_as_applied",
                )
            if "is_short_sentence" in flags:
                _require_bool(
                    flags.get("is_short_sentence"),
                    f"segments[{index}].flags.is_short_sentence",
                )
            if "use_repeat_and_cut" in flags:
                _require_bool(
                    flags.get("use_repeat_and_cut"),
                    f"segments[{index}].flags.use_repeat_and_cut",
                )

        if include_tokens:
            tokens = segment.get("tokens")
            if not isinstance(tokens, list) or not all(
                isinstance(token, int) for token in tokens
            ):
                raise ValueError(f"segments[{index}].tokens must be a list of integers")
