from __future__ import annotations

import math

import pytest

from pykokoro import LoudnessConfig, PipelineConfig
from pykokoro.pipeline import _coerce_pipeline_config


def test_loudness_defaults_are_disabled() -> None:
    config = LoudnessConfig()
    assert config.voice_leveling == "off"
    assert config.target_lufs is None
    assert PipelineConfig().loudness == config


@pytest.mark.parametrize("field", ["target_lufs", "true_peak_ceiling_dbtp", "voice_gain_db"])
def test_loudness_rejects_non_finite_values(field: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        LoudnessConfig(**{field: math.inf})


def test_loudness_mapping_is_coerced() -> None:
    config = _coerce_pipeline_config(
        {"loudness": {"voice_leveling": "calibrated", "voice_gain_db": -2.0}}
    )
    assert config.loudness == LoudnessConfig(voice_leveling="calibrated", voice_gain_db=-2.0)


def test_loudness_public_exports_are_importable() -> None:
    assert LoudnessConfig.__name__ == "LoudnessConfig"
