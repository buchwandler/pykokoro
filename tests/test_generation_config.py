from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pykokoro.generation_config import GenerationConfig


def test_generation_config_defaults_and_request_controls() -> None:
    config = GenerationConfig()
    assert config.speed == 1.0
    assert config.lang is None
    assert config.random_seed is None
    assert config.enable_short_sentence is None

    configured = GenerationConfig(
        speed=1.5,
        lang="en-gb",
        random_seed=42,
        enable_short_sentence=True,
    )
    assert configured.speed == 1.5
    assert configured.lang == "en-gb"
    assert configured.random_seed == 42
    assert configured.enable_short_sentence is True


def test_generation_config_rejects_invalid_speed() -> None:
    for speed in (0.0, -1.0, float("inf"), float("nan"), True):
        with pytest.raises(ValueError, match="speed"):
            GenerationConfig(speed=speed)  # type: ignore[arg-type]


def test_generation_config_validates_language_seed_and_short_sentence() -> None:
    with pytest.raises(ValueError, match="lang"):
        GenerationConfig(lang=" ")
    with pytest.raises(ValueError, match="random_seed"):
        GenerationConfig(random_seed=True)
    with pytest.raises(ValueError, match="enable_short_sentence"):
        GenerationConfig(enable_short_sentence=1)  # type: ignore[arg-type]


def test_generation_config_is_immutable() -> None:
    config = GenerationConfig()
    with pytest.raises(FrozenInstanceError):
        config.speed = 2.0  # type: ignore[misc]
