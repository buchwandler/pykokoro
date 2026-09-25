"""Run the maintained request-centric synthesis examples sequentially."""

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
_DEFAULT_EXAMPLES = {
    "linguistic_annotations.py",
    "models_and_languages.py",
    "pronunciation_overrides.py",
    "request_batch.py",
    "simple_synthesis.py",
}
_OPTIONAL_EXAMPLES = {"all_voices.py"}

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _example_paths(*, include_optional: bool = False) -> list[Path]:
    selected = _DEFAULT_EXAMPLES | (_OPTIONAL_EXAMPLES if include_optional else set())
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="list selected examples without running"
    )
    parser.add_argument(
        "--include-optional", action="store_true", help="include the all-voices showcase"
    )
    args = parser.parse_args()
    paths = _example_paths(include_optional=args.include_optional)
    if args.list:
        for path in paths:
            print(_label(path))
        return 0
    return min(run_examples(paths), 1)


if __name__ == "__main__":
    raise SystemExit(main())
