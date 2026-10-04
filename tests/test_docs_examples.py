"""Smoke checks for the maintained documentation and example surface."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pykokoro
from examples import run_all

ROOT = Path(__file__).parents[1]
MAINTAINED_EXAMPLES = tuple(
    path for path in sorted((ROOT / "examples").glob("*.py")) if path.name != "__init__.py"
)
INPUT_DRIVEN_EXAMPLES = {"reference_voice.py"}
PYTHON_FENCE = re.compile(r"(?ms)^```python[ \t]*\n(.*?)^```[ \t]*$")


def test_current_docs_python_fences_parse_and_use_public_names() -> None:
    public_names = set(pykokoro.__all__)
    sources = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for source in PYTHON_FENCE.findall(text):
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("pykokoro."), (path, node.module)
                if isinstance(node, ast.ImportFrom) and node.module == "pykokoro":
                    imported = {alias.name for alias in node.names}
                    assert imported <= public_names, (path, imported - public_names)
            assert not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "create"
                for node in ast.walk(tree)
            ), path
            assert not any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "Kokoro"
                for node in ast.walk(tree)
            ), path


def test_maintained_examples_compile_and_import_without_running_main() -> None:
    for path in MAINTAINED_EXAMPLES:
        _import_example(path)


def _import_example(path: Path) -> None:
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_name = f"pykokoro_smoke_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    assert module.__name__ == module_name


def test_readme_has_no_undefined_playback_helper() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "play_audio(res.audio, res.sample_rate)" not in readme


def test_api_reference_covers_every_root_export() -> None:
    reference = (ROOT / "docs" / "api_reference.md").read_text(encoding="utf-8")
    missing = sorted(name for name in pykokoro.__all__ if name not in reference)
    assert not missing
    assert "EspeakConfig" not in reference


def test_example_catalogs_cover_every_runnable_script() -> None:
    expected = {
        path.name
        for path in (ROOT / "examples").glob("*.py")
        if path.name not in {"__init__.py", "_output.py", "run_all.py"}
    }
    grouped = set().union(*run_all._GROUPS.values())
    assert grouped == expected - INPUT_DRIVEN_EXAMPLES

    catalogs = (ROOT / "docs" / "examples.md", ROOT / "examples" / "README.md")
    for catalog_path in catalogs:
        text = catalog_path.read_text(encoding="utf-8")
        assert "Assets, network, and cost" in text
        assert "Expected output" in text
        for name in sorted(expected):
            assert f"python examples/{name}" in text, (catalog_path, name)


def test_example_catalogs_document_runner_group_boundaries() -> None:
    for catalog_path in (ROOT / "docs" / "examples.md", ROOT / "examples" / "README.md"):
        text = catalog_path.read_text(encoding="utf-8")
        assert "--group core" in text
        assert "--group feature" in text
        assert "--group language-showcase" in text
        assert "--group optional-heavy" in text
        assert "--include-optional" in text
