from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).parents[1]
PYTHON_FENCE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
EXAMPLE_REFERENCE = re.compile(r"examples/[A-Za-z0-9_./-]+\.py")


def test_readme_example_paths_exist() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    refs = set(EXAMPLE_REFERENCE.findall(text))
    missing = sorted(ref for ref in refs if not (ROOT / ref).is_file())
    assert missing == []


def test_readme_python_fences_compile() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    fences = PYTHON_FENCE.findall(text)
    assert fences
    for index, source in enumerate(fences):
        compile(source, f"README.md Python fence {index}", "exec")


def test_readme_top_level_imports_are_public() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    public_names = set(__import__("pykokoro").__all__)

    for index, source in enumerate(PYTHON_FENCE.findall(text)):
        tree = ast.parse(source, filename=f"README.md Python fence {index}")
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "pykokoro":
                imported_names = {alias.name for alias in node.names if alias.name != "*"}
                assert imported_names <= public_names
