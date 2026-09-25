from __future__ import annotations

from typing import get_args

import pytest

import pykokoro
from pykokoro import LongTextSplitMode, SynthesisConfig, SynthesisInputTooLongError


def test_synthesis_config_defaults_to_sentence_splitting() -> None:
    config = SynthesisConfig()

    assert config.long_text_split == "sentence"


def test_synthesis_config_accepts_supported_long_text_modes() -> None:
    for mode in ("sentence", "token", "none"):
        config = SynthesisConfig(long_text_split=mode)  # type: ignore[arg-type]
        assert config.long_text_split == mode


def test_synthesis_config_rejects_unknown_long_text_mode() -> None:
    with pytest.raises(ValueError, match="long_text_split must be"):
        SynthesisConfig(long_text_split="unsupported")  # type: ignore[arg-type]


def test_long_text_mode_and_input_error_are_public() -> None:
    assert get_args(LongTextSplitMode) == ("sentence", "token", "none")
    assert issubclass(SynthesisInputTooLongError, ValueError)
    assert "LongTextSplitMode" in pykokoro.__all__
    assert "SynthesisInputTooLongError" in pykokoro.__all__
