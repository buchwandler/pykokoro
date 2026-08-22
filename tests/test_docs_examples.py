"""Smoke checks for the maintained documentation and example surface."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pykokoro

ROOT = Path(__file__).parents[1]
MAINTAINED_EXAMPLES = tuple(
    path for path in sorted((ROOT / "examples").glob("*.py")) if path.name != "__init__.py"
)
PYTHON_FENCE = re.compile(r"(?ms)^```python[ \t]*\n(.*?)^```[ \t]*$")


def test_current_docs_python_fences_parse_and_use_public_names() -> None:
    public_names = set(pykokoro.__all__)
    sources = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for source in PYTHON_FENCE.findall(text):
            tree = ast.parse(source, filename=str(path))
            for node in ast.walk(tree):
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


def test_spokenform_example_imports_without_running_main() -> None:
    _import_example(ROOT / "examples" / "spokenform_phoneme_equivalence.py")


def test_removed_api_is_confined_to_archived_examples() -> None:
    current_files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
    current_files.extend(MAINTAINED_EXAMPLES)
    for path in current_files:
        text = path.read_text(encoding="utf-8")
        assert "Kokoro(" not in text, path
        assert ".create(" not in text, path


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
