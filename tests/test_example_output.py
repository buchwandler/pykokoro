from __future__ import annotations

from pathlib import Path

import pytest

from examples._output import ARTIFACT_DIR, artifact_dir, artifact_path


def test_artifact_dir_uses_repository_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PYKOKORO_EXAMPLE_OUTPUT_DIR", raising=False)

    assert artifact_dir() == ARTIFACT_DIR.resolve()


def test_artifact_dir_honors_runner_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYKOKORO_EXAMPLE_OUTPUT_DIR", str(tmp_path / "example"))

    result = artifact_dir()

    assert result == (tmp_path / "example").resolve()
    assert result.is_dir()


def test_artifact_path_rejects_paths_outside_artifact_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYKOKORO_EXAMPLE_OUTPUT_DIR", str(tmp_path / "example"))

    with pytest.raises(ValueError, match="escapes"):
        artifact_path("../outside.wav")


def test_artifact_path_creates_nested_parent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PYKOKORO_EXAMPLE_OUTPUT_DIR", str(tmp_path / "example"))

    result = artifact_path("nested/result.wav")

    assert result == (tmp_path / "example" / "nested" / "result.wav").resolve()
    assert result.parent.is_dir()
