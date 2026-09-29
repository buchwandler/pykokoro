#!/usr/bin/env python3
"""Run maintained PyKokoro examples by capability and cost category."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

try:
    from examples._output import ARTIFACT_DIR, PROJECT_ROOT
except ModuleNotFoundError:
    from _output import ARTIFACT_DIR, PROJECT_ROOT

_OUTPUT_ENV = "PYKOKORO_EXAMPLE_OUTPUT_DIR"
_GROUPS: dict[str, set[str]] = {
    "core": {
        "simple_synthesis.py",
        "request_batch.py",
        "pronunciation_overrides.py",
        "linguistic_tokens.py",
        "long_text.py",
        "voice_blend.py",
        "result_metadata.py",
        "models_and_languages.py",
    },
    "feature": {
        "english.py",
        "language_routing.py",
        "frontend_and_lexicons.py",
        "asset_progress.py",
        "error_handling.py",
    },
    "language-showcase": {
        "chinese.py",
        "contractions.py",
        "french.py",
        "italian.py",
        "japanese.py",
        "korean.py",
        "portuguese.py",
        "spanish.py",
    },
    "optional-heavy": {"all_voices.py", "short_sentence_demo.py"},
}
_GROUP_CHOICES = tuple(_GROUPS)
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _example_paths(*, group: str = "core", include_optional: bool = False) -> list[Path]:
    selected = set(_GROUPS[group])
    if include_optional and group != "optional-heavy":
        selected.update(_GROUPS["optional-heavy"])
    return [
        path
        for path in sorted(PROJECT_ROOT.joinpath("examples").glob("*.py"))
        if path.name in selected
    ]


def _label(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def run_examples(paths: Sequence[Path], *, runner: RunCommand = subprocess.run) -> int:
    """Run selected example scripts and return the number of failures."""
    failures = 0
    for path in paths:
        relative = path.relative_to(PROJECT_ROOT)
        output_name = "__".join(relative.with_suffix("").parts)
        output_dir = ARTIFACT_DIR / output_name
        output_dir.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment[_OUTPUT_ENV] = str(output_dir)
        print(f"\n=== {_label(path)} ===")
        result = runner(
            [sys.executable, str(relative)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=False,
            text=True,
        )
        if result.returncode:
            failures += 1
            print(f"FAILED ({result.returncode}): {_label(path)}")
        else:
            print(f"PASSED: {_label(path)}")

    print(f"\nCompleted {len(paths)} examples: {len(paths) - failures} passed, {failures} failed.")
    return failures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=_GROUP_CHOICES, default="core")
    parser.add_argument(
        "--list", action="store_true", help="list selected examples without running"
    )
    parser.add_argument(
        "--include-optional",
        action="store_true",
        help="also include optional-heavy examples (unless that group is selected explicitly)",
    )
    args = parser.parse_args(argv)
    paths = _example_paths(group=args.group, include_optional=args.include_optional)
    if args.list:
        for path in paths:
            print(_label(path))
        return 0
    return min(run_examples(paths), 1)


if __name__ == "__main__":
    raise SystemExit(main())
