"""Release metadata and dependency policy checks."""

from __future__ import annotations

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python 3.10
    import tomli as tomllib


def test_public_discovery_api_symbols_are_packaged() -> None:
    import importlib.metadata

    import pykokoro
    from pykokoro import ModelCapabilities, ModelDiscoveryResult, discover_models

    assert callable(discover_models)
    assert ModelCapabilities
    assert ModelDiscoveryResult
    assert pykokoro.__file__
    assert importlib.metadata.version("pykokoro")
    assert (Path(pykokoro.__file__).parent / "discovery.py").is_file()


ROOT = Path(__file__).parents[1]


def test_license_and_release_fallback_version_are_present() -> None:
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert "Apache License" in license_text
    assert pyproject["tool"]["setuptools_scm"]["fallback_version"] == "0.9.0"


def test_companion_dependency_floors_match_current_integration_contract() -> None:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]

    assert "kokorog2p[espeak,en]>=0.9.2,<1.0" in dependencies
    assert "lexphon>=0.1.0,<0.2" in dependencies
    assert "phrasplit>=0.3.7,<0.4" in dependencies


def test_test_requirements_keep_kokorog2p_in_supported_window() -> None:
    requirements = (ROOT / "requirements-test.txt").read_text(encoding="utf-8")
    assert "kokorog2p[all]>=0.9.2,<1.0" in requirements
    assert "lexphon>=0.1.0,<0.2" in requirements


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


def test_ssmd_dependency_targets_current_contract() -> None:
    dependencies = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    assert "ssmd>=0.8.6,<0.9" in dependencies
    assert "spokenform>=0.3.6,<0.4" in dependencies


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


def test_version_fallbacks_target_release() -> None:
    source = (ROOT / "pykokoro" / "__init__.py").read_text(encoding="utf-8")

    assert re.search(r'__version__ = "0\.9\.0"', source)
    assert re.search(r"__version_tuple__ = \(0, 9, 0\)", source)


def test_lower_bound_workflow_pins_match_project_floors() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]

    floors = {}
    for requirement in dependencies:
        match = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?>=([^,<;]+)", requirement)
        if match:
            floors[match.group(1).lower()] = match.group(2)

    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    pins = {
        name.lower(): version
        for name, version in re.findall(r'"([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\"]+)"', workflow)
        if version and version[0].isdigit()
    }
    expected_packages = {
        "kokorog2p",
        "lexphon",
        "phrasplit",
        "ssmd",
        "spokenform",
        "audiosig",
    }

    assert {package: pins[package] for package in expected_packages} == {
        package: floors[package] for package in expected_packages
    }
    assert "python -m pip install -e . --no-deps" not in workflow


def test_package_resource_workflow_covers_kokorog2p_window() -> None:
    workflow = (ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")

    assert 'kokorog2p-version: ["0.9.2"]' in workflow
    assert "working-directory: ${{ runner.temp }}" in workflow
    assert "pykokoro-*.whl" in workflow
