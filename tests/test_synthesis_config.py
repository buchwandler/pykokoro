from __future__ import annotations

from typing import get_args

import pytest

import pykokoro
from pykokoro import LongTextSplitMode, SynthesisConfig, SynthesisInputTooLongError
from pykokoro.exceptions import ConfigurationError, InvalidModelError
from pykokoro.generation_config import GenerationConfig
from pykokoro.synthesis_config import resolve_synthesis_config


def test_synthesis_config_never_splits_long_text() -> None:
    config = SynthesisConfig()

    assert config.long_text_split == "none"


@pytest.mark.parametrize("mode", ["sentence", "token"])
def test_synthesis_config_rejects_legacy_splitting_modes(mode: str) -> None:
    with pytest.raises(ConfigurationError, match="split text in the caller"):
        SynthesisConfig(long_text_split=mode)  # type: ignore[arg-type]


def test_synthesis_config_rejects_unknown_long_text_mode() -> None:
    with pytest.raises(ConfigurationError, match="long_text_split only accepts 'none'"):
        SynthesisConfig(long_text_split="unsupported")  # type: ignore[arg-type]


def test_long_text_mode_and_input_error_are_public_and_structured() -> None:
    error = SynthesisInputTooLongError(
        text_length=100,
        token_count=511,
        max_tokens=510,
        model_id="github:v1.0:fp32",
    )

    assert get_args(LongTextSplitMode) == ("none",)
    assert issubclass(SynthesisInputTooLongError, ValueError)
    assert error.text_length == 100
    assert error.token_count == 511
    assert error.max_tokens == 510
    assert error.model_id == "github:v1.0:fp32"
    assert "LongTextSplitMode" in pykokoro.__all__
    assert "SynthesisInputTooLongError" in pykokoro.__all__


def test_invalid_speed_is_a_typed_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="speed must"):
        GenerationConfig(speed=0)


def test_model_resolution_uses_typed_error() -> None:
    with pytest.raises(InvalidModelError, match="Invalid model"):
        resolve_synthesis_config(
            SynthesisConfig(model_variant="unknown"),  # type: ignore[arg-type]
            language="en-us",
        )
