from __future__ import annotations

import subprocess
import sys


def test_cli_discovery() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.hard_cases", "--list-languages"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.splitlines() == ["en", "de"]


def test_cli_frontend_case_reproduction(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.hard_cases",
            "--case",
            "en_shared_001",
            "--limit",
            "1",
            "--results-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Evaluated 1 case" in result.stdout
    assert (tmp_path / "summary.json").exists()


def test_cli_rejects_invalid_locale() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "benchmarks.hard_cases", "--locale", "xx-YY"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unsupported locale" in result.stderr
