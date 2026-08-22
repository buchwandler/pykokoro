"""Release metadata and dependency policy checks."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).parents[1]


def test_license_and_release_fallback_version_are_present() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "Apache License" in license_text
    assert pyproject["tool"]["setuptools_scm"]["fallback_version"] == "0.8.3"


def test_companion_dependency_floors_match_the_080_spokenform_contract() -> None:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]

    assert "kokorog2p[espeak,en]>=0.8.1,<0.9" in dependencies
    assert "phrasplit>=0.3.4,<0.4" in dependencies


def test_test_requirements_keep_kokorog2p_in_supported_window() -> None:
    requirements = (ROOT / "requirements-test.txt").read_text(encoding="utf-8")
    assert "kokorog2p[all]>=0.8.0,<0.9" in requirements


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


def test_ssmd_dependency_targets_08_contract() -> None:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    assert "ssmd>=0.8.2,<0.9" in dependencies


def test_audiosig_is_the_only_declared_dsp_backend() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    optional = pyproject["project"]["optional-dependencies"]

    assert "audiosig>=0.1.1,<0.2" in dependencies
    assert "prosody" not in optional

    requirements = [requirement.lower() for requirement in dependencies]
    requirements.extend(
        requirement.lower() for values in optional.values() for requirement in values
    )
    forbidden = (
        "librosa",
        "scipy",
        "audiomentations",
        "signalsmith",
        "python-stretch",
        "resampy",
        "soxr",
        "torchaudio",
    )
    assert not any(name in requirement for requirement in requirements for name in forbidden)


def test_production_source_has_no_forbidden_dsp_imports() -> None:
    forbidden = (
        "import librosa",
        "from librosa",
        "import scipy",
        "from scipy",
        "import audiomentations",
        "from audiomentations",
        "import signalsmith_stretch",
        "import python_stretch",
        "import resampy",
        "import soxr",
        "import torchaudio",
    )
    source_files = [
        path for path in (ROOT / "pykokoro").rglob("*.py") if "__pycache__" not in path.parts
    ]
    matches = [
        str(path.relative_to(ROOT))
        for path in source_files
        if any(token in path.read_text(encoding="utf-8") for token in forbidden)
    ]
    assert matches == []


def test_playback_extra_is_optional_and_keeps_compatibility_alias() -> None:
    optional = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]

    assert optional["playback"] == ["sounddevice"]
    assert optional["sounddevice"] == ["sounddevice"]
    assert "sounddevice" not in dependencies
