from __future__ import annotations

import ast
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
_EXCLUDED = {"__init__.py", "_output.py", "run_all.py"}
_TEXT_CALLS = {"run", "prepare_units", "play_streaming"}
_PIPELINE_CALLS = {"KokoroPipeline", "build_pipeline"}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def _has_explicit_language(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node)
        if name == "GenerationConfig" and any(keyword.arg == "lang" for keyword in node.keywords):
            return True
        if name in _TEXT_CALLS and any(keyword.arg == "lang" for keyword in node.keywords):
            return True
    return False


def test_maintained_examples_do_not_use_retired_af_alias() -> None:
    violations: list[str] = []
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        if path.name in _EXCLUDED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value == "af":
                violations.append(f"{path.name}:{node.lineno}")

    assert not violations, "retired bare af voice alias found: " + ", ".join(violations)


def test_text_synthesizing_examples_declare_document_language() -> None:
    missing: list[str] = []
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        if path.name in _EXCLUDED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        if names.intersection(_PIPELINE_CALLS | _TEXT_CALLS) and not _has_explicit_language(tree):
            missing.append(path.name)

    assert not missing, "examples without explicit document language: " + ", ".join(missing)
