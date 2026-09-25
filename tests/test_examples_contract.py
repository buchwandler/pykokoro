from __future__ import annotations

import ast
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parents[1] / "examples"
_EXCLUDED = {"__init__.py", "_output.py", "run_all.py"}
_RETIRED_NAMES = {"KokoroPipeline", "PipelineConfig", "build_pipeline", "AudioJob"}


def _call_name(call: ast.Call) -> str | None:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        return call.func.attr
    return None


def test_maintained_examples_use_the_request_engine_api() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        if path.name in _EXCLUDED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert not names.intersection(_RETIRED_NAMES), path.name
        assert "KokoroSynthesizer" in names, path.name


def test_examples_supply_explicit_language_for_each_request() -> None:
    for path in sorted(EXAMPLES_DIR.glob("*.py")):
        if path.name in _EXCLUDED:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name == "synthesize_text":
                assert any(keyword.arg == "language" for keyword in node.keywords), path.name
            elif name == "SynthesisSegment":
                assert len(node.args) >= 3 or any(
                    keyword.arg == "language" for keyword in node.keywords
                ), path.name
