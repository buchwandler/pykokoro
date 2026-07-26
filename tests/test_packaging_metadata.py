"""Release metadata and dependency policy checks."""

from __future__ import annotations

from pathlib import Path

import tomllib

ROOT = Path(__file__).parents[1]


def test_license_and_release_fallback_version_are_present() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "Apache License" in license_text
    assert pyproject["tool"]["setuptools_scm"]["fallback_version"] != "0.0.0"


def test_provider_extras_do_not_install_every_runtime_distribution() -> None:
    optional = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]

    provider_packages = {
        package.split(">=", 1)[0]
        for extra in ("cpu", "gpu", "openvino", "directml", "coreml")
        for package in optional[extra]
        if package.startswith("onnxruntime")
    }
    assert optional["all"]
    assert not any(package.startswith("onnxruntime") for package in optional["all"])
    assert provider_packages == {
        "onnxruntime",
        "onnxruntime-gpu",
        "onnxruntime-openvino",
        "onnxruntime-directml",
    }
