from __future__ import annotations

import numpy as np
import pytest

from pykokoro.model_assets import get_model_asset_paths
from pykokoro.onnx_backend import _validate_voice_archive


def test_martin_model_asset_paths_are_config_free(tmp_path, monkeypatch):
    monkeypatch.setattr("pykokoro.model_assets.get_user_cache_path", lambda: tmp_path)
    paths = get_model_asset_paths(source="github", variant="v1.2-de-martin", quality="fp32")
    assert paths.config is None
    assert paths.model.name == "kokoro-german-martin-v1.2.onnx"
    assert paths.voices.name == "voices-german-martin-v1.2.bin"


@pytest.mark.parametrize("quality", ["q8", "fp16", "q4"])
def test_martin_rejects_unsupported_quality(quality):
    with pytest.raises(ValueError, match="Available: fp32"):
        get_model_asset_paths(source="github", variant="v1.2-de-martin", quality=quality)


def test_small_martin_voice_archive_is_valid(tmp_path):
    path = tmp_path / "voices-german-martin-v1.2.bin"
    with path.open("wb") as archive:
        np.savez(archive, martin=np.zeros((512, 1, 256), dtype=np.float16))
    assert path.stat().st_size < 1_000_000
    _validate_voice_archive(path, expected_voice_names=("martin",))


def test_martin_voice_archive_requires_martin(tmp_path):
    path = tmp_path / "voices.bin"
    with path.open("wb") as archive:
        np.savez(archive, df_eva=np.zeros((512, 1, 256), dtype=np.float16))
    with pytest.raises(RuntimeError, match="missing expected voices: martin"):
        _validate_voice_archive(path, expected_voice_names=("martin",))
