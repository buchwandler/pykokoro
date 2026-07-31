"""Tests for public prosody backend configuration."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pykokoro import PipelineConfig, ProsodyConfig
from pykokoro.prosody_config import canonical_prosody_method


def test_default_method_is_wsola() -> None:
    assert ProsodyConfig().method == "wsola"
    assert PipelineConfig().prosody == ProsodyConfig()


def test_psola_alias_is_accepted_and_canonicalized() -> None:
    config = ProsodyConfig(method="psola")

    assert config.method == "psola"
    assert canonical_prosody_method(config.method) == "td_psola"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"method": "unknown"}, "unsupported prosody method"),
        ({"fallback_methods": ("unknown",)}, "unsupported method"),
        ({"n_fft": 1}, "n_fft"),
        ({"hop_length": 0}, "hop_length"),
        ({"hop_length": 2049}, "hop_length"),
        ({"filter_width": 0}, "filter_width"),
        ({"rolloff": 0.0}, "rolloff"),
        ({"rolloff": 1.1}, "rolloff"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ProsodyConfig(**kwargs)  # type: ignore[arg-type]


def test_configuration_is_frozen() -> None:
    config = ProsodyConfig()

    with pytest.raises(FrozenInstanceError):
        config.method = "esola"  # type: ignore[misc]
